import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join, relative, sep } from "node:path";
import lockfile from "proper-lockfile";
import type { XlabManifestArtifact, XlabManifestEnvelope } from "../types.ts";
import {
	atomicWriteFile,
	canonicalDigest,
	canonicalJson,
	canonicalJsonBytes,
	codeReviewRoles,
	type ExperimentAcceptedResult,
	type ExperimentPlan,
	type ExperimentProtocolStore,
	type ExperimentReviewReport,
	type ExperimentReviewValidationContext,
	resolveContainedPath,
	SCIENCE_REVIEW_ROLES,
	sha256,
	validateCodePlan,
	validatePreparePlan,
	validateReviewMatrix,
	validateSciencePlan,
} from "./protocol.ts";

const FINAL_RESULTS = new Set(["positive", "negative", "neutral", "inconclusive"]);
const COMPONENT_FIELDS = ["result", "metric", "value", "confidence", "analysis", "method_context"] as const;

export type ExperimentMaterializerFaultPoint =
	| "after_symbolic_memory_cas"
	| "after_symbolic_memory_receipt"
	| "after_final_audit"
	| "after_manifest";

export interface ExperimentMaterializerOptions {
	runRoot: string;
	sharedRoot: string;
	protocol: ExperimentProtocolStore;
	skillName?: string;
	skillVersion?: string;
	faultInjector?: (point: ExperimentMaterializerFaultPoint) => void;
}

export interface PublishedExperimentArtifact {
	path: string;
	sha256: string;
	bytes: number;
}

export interface ExperimentPublication {
	latest: PublishedExperimentArtifact;
	attempt: PublishedExperimentArtifact;
}

export interface MaterializeReviewMatrixOptions {
	stage: "code" | "science";
	workUnit: string;
	reviewRound: number;
	assignmentIds: string[];
	validationContext?: ExperimentReviewValidationContext;
}

export interface FinalComponentDefinition {
	component: string;
	method_context: string;
}

export interface ExperimentFinalizationProvenance {
	idea_digest: string;
	policy_digest: string;
	final_review: { assignment_id: string; result_digest: string };
	science: {
		plan: { assignment_id: string; result_digest: string };
		cohort: Array<{ assignment_id: string; work_unit: string; result_digest: string }>;
		reviews: Array<{ assignment_id: string; role: string; result_digest: string }>;
		input_digest: string;
		target_digest: string;
	};
}

export interface MaterializeFinalOptions {
	componentDefinitions: FinalComponentDefinition[];
	componentResults: Record<string, unknown>;
	summary: Record<string, unknown>;
	provenance: ExperimentFinalizationProvenance;
	symbolicMemoryPath: string;
	expectedSymbolicMemoryDigest: string | null;
	audit?: Record<string, unknown>;
}

export interface SymbolicMemoryCasResult {
	path: string;
	operation_id: string;
	previous_digest: string;
	digest: string;
	record_ids: string[];
	replayed: boolean;
}

interface SymbolicMemoryOperation {
	operation_id: string;
	payload_digest: string;
	status: "pending" | "applied";
	previous_digest: string;
	digest: string;
	record_ids: string[];
}

interface SymbolicMemoryLedger {
	schema_version: "xlab.symbolic_memory_operations.v1";
	operations: Record<string, SymbolicMemoryOperation>;
}

function objectResult(accepted: ExperimentAcceptedResult): Record<string, unknown> {
	const result = accepted.submission.result;
	if (typeof result !== "object" || result === null || Array.isArray(result))
		throw new Error(`Accepted assignment ${accepted.accepted.assignment_id} did not submit an object result.`);
	if (canonicalDigest(result) !== accepted.accepted.result_digest)
		throw new Error(`Accepted assignment ${accepted.accepted.assignment_id} has a mismatched result digest.`);
	return result;
}

function slug(value: string): string {
	const normalized = value
		.trim()
		.replace(/[^A-Za-z0-9_.-]+/g, "_")
		.replace(/^_+|_+$/g, "");
	if (!normalized) throw new Error("Experiment artifact path component must be non-empty.");
	return normalized;
}

function stepParts(stage: "code" | "science", workUnit: string): string[] {
	if (stage === "science" && workUnit.includes("::")) {
		const [prefix, suffix] = workUnit.split("::", 2);
		return [slug(prefix), slug(suffix)];
	}
	return [slug(workUnit)];
}

function attemptName(attempt: number): string {
	if (!Number.isInteger(attempt) || attempt <= 0)
		throw new Error("Experiment publication attempt must be a positive integer.");
	return `${String(attempt).padStart(3, "0")}.json`;
}

function relativePath(root: string, path: string): string {
	const value = relative(root, path);
	if (!value || value === ".." || value.startsWith(`..${sep}`))
		throw new Error(`Experiment artifact path escapes publication root: ${path}.`);
	return value.split(sep).join("/");
}

function publishPair(root: string, base: string, attempt: number, value: unknown): ExperimentPublication {
	const attemptPath = resolveContainedPath(root, join(base, "attempts", attemptName(attempt)), false);
	const latestPath = resolveContainedPath(root, join(base, "latest.json"), false);
	const bytes = canonicalJsonBytes(value);
	atomicWriteFile(attemptPath, bytes);
	atomicWriteFile(latestPath, bytes);
	const digest = sha256(bytes);
	return {
		attempt: { path: relativePath(root, attemptPath), sha256: digest, bytes: bytes.byteLength },
		latest: { path: relativePath(root, latestPath), sha256: digest, bytes: bytes.byteLength },
	};
}

function acceptedFor(protocol: ExperimentProtocolStore, assignmentId: string): ExperimentAcceptedResult {
	const accepted = protocol.readAcceptedResult(assignmentId);
	if (!accepted) throw new Error(`Experiment materialization requires accepted assignment ${assignmentId}.`);
	return accepted;
}

function assertAcceptedMetadata(
	accepted: ExperimentAcceptedResult,
	expected: { role: string; kind: string; stage: string; workUnit?: string },
): void {
	const submission = accepted.submission;
	if (
		submission.role !== expected.role ||
		submission.kind !== expected.kind ||
		submission.stage !== expected.stage ||
		submission.work_unit !== expected.workUnit
	) {
		throw new Error(
			`Accepted assignment ${submission.assignment_id} does not match the requested materialization lineage.`,
		);
	}
}

function stringField(value: unknown, label: string): string {
	if (typeof value !== "string" || !value.trim()) throw new Error(`${label} must be a non-empty string.`);
	return value.trim();
}

function confidence(value: unknown, label: string): number {
	if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1)
		throw new Error(`${label} must be a finite number in [0, 1].`);
	return value;
}

function normalizeFinal(
	definitions: FinalComponentDefinition[],
	results: Record<string, unknown>,
	summary: Record<string, unknown>,
	baseline = false,
): Record<string, unknown> {
	if (definitions.length === 0 && !baseline) throw new Error("Final materialization requires canonical components.");
	const names = definitions.map((definition) => stringField(definition.component, "Canonical component name"));
	if (new Set(names).size !== names.length) throw new Error("Canonical component names must be unique.");
	if (Object.keys(results).length !== names.length || Object.keys(results).some((name) => !names.includes(name)))
		throw new Error("Final component results must exactly cover canonical components.");
	const components: Record<string, unknown> = {};
	for (const definition of definitions) {
		const name = definition.component.trim();
		const raw = results[name];
		if (typeof raw !== "object" || raw === null || Array.isArray(raw))
			throw new Error(`Final component ${name} must be an object.`);
		const source = raw as Record<string, unknown>;
		const result = stringField(source.result, `Component ${name} result`).toLowerCase();
		if (!FINAL_RESULTS.has(result)) throw new Error(`Component ${name} has invalid result ${result}.`);
		if (source.follow_up_required === true) throw new Error(`Component ${name} still requires follow-up.`);
		components[name] = {
			result,
			metric: stringField(source.metric, `Component ${name} metric`),
			value: stringField(source.value, `Component ${name} value`),
			confidence: confidence(source.confidence, `Component ${name} confidence`),
			analysis: stringField(source.analysis, `Component ${name} analysis`),
			method_context: stringField(definition.method_context, `Component ${name} method context`),
		};
	}
	if (typeof summary.feasible !== "boolean") throw new Error("Final summary feasible must be boolean.");
	if (!Array.isArray(summary.key_findings) || summary.key_findings.length === 0)
		throw new Error("Final summary key_findings must be non-empty.");
	return {
		...(baseline ? { ablation_status: "not_applicable", experiment_kind: "baseline" } : {}),
		components,
		summary: {
			feasible: summary.feasible,
			confidence: confidence(summary.confidence, "Final summary confidence"),
			key_findings: summary.key_findings.map((item) => stringField(item, "Final key finding")),
		},
	};
}

interface SymbolicMemoryState {
	next_id: number;
	fidmap2mid: Record<string, string>;
	records: Record<string, Record<string, unknown>>;
	config: { upsert_threshold: number };
}

const EMPTY_MEMORY: SymbolicMemoryState = {
	next_id: 0,
	fidmap2mid: {},
	records: {},
	config: { upsert_threshold: 0.82 },
};

function readMemoryBytes(bytes: Buffer): SymbolicMemoryState {
	const parsed = JSON.parse(bytes.toString("utf-8")) as SymbolicMemoryState;
	if (!parsed || typeof parsed !== "object" || !parsed.records || typeof parsed.records !== "object")
		throw new Error("Symbolic memory is not a valid state object.");
	return parsed;
}

function readLedger(path: string): SymbolicMemoryLedger {
	if (!existsSync(path)) return { schema_version: "xlab.symbolic_memory_operations.v1", operations: {} };
	const parsed = JSON.parse(readFileSync(path, "utf-8")) as SymbolicMemoryLedger;
	if (
		parsed?.schema_version !== "xlab.symbolic_memory_operations.v1" ||
		!parsed.operations ||
		typeof parsed.operations !== "object"
	)
		throw new Error("Symbolic memory operation ledger is invalid.");
	return parsed;
}

function identity(record: Record<string, unknown>): string {
	return [record.component, record.component_family, record.metric]
		.map((value) =>
			String(value ?? "")
				.trim()
				.toLowerCase(),
		)
		.join("|");
}

function family(component: string): string {
	const value = component
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "_")
		.replace(/^_+|_+$/g, "");
	return `component.${value || "unknown"}`;
}

function mergeResult(left: string, right: string): string {
	if (left === right) return left;
	const weak = new Set(["neutral", "inconclusive"]);
	if (weak.has(left) && weak.has(right)) return right === "inconclusive" ? left : right;
	if (weak.has(left) || weak.has(right)) return weak.has(left) ? right : left;
	return "inconclusive";
}

function mergedMemory(
	currentBytes: Buffer,
	components: Record<string, unknown>,
): { bytes: Buffer; recordIds: string[] } {
	const state = readMemoryBytes(currentBytes);
	const recordIds: string[] = [];
	for (const [component, raw] of Object.entries(components)) {
		const value = raw as Record<string, unknown>;
		const candidate: Record<string, unknown> = {
			component,
			component_family: family(component),
			result: value.result,
			metric: value.metric,
			value: value.value,
			analysis: value.analysis,
			method_context: value.method_context,
			confidence: value.confidence,
		};
		const existingEntry = Object.entries(state.records).find(
			([, record]) => identity(record) === identity(candidate),
		);
		if (existingEntry) {
			const [fid, existing] = existingEntry;
			candidate.id = existing.id;
			candidate.result = mergeResult(String(existing.result ?? ""), String(candidate.result ?? ""));
			candidate.confidence = (Number(existing.confidence) + Number(candidate.confidence)) / 2;
			state.records[fid] = candidate;
			recordIds.push(String(candidate.id));
		} else {
			const fid = String(state.next_id);
			const id = `memory_${fid}`;
			state.next_id += 1;
			candidate.id = id;
			state.records[fid] = candidate;
			state.fidmap2mid[fid] = id;
			recordIds.push(id);
		}
	}
	return { bytes: canonicalJsonBytes(state), recordIds };
}

export function currentSymbolicMemoryDigest(path: string): string {
	const bytes = existsSync(path) ? readFileSync(path) : canonicalJsonBytes(EMPTY_MEMORY);
	readMemoryBytes(bytes);
	return sha256(bytes);
}

export function compareAndSwapSymbolicMemory(options: {
	path: string;
	expectedDigest: string | null;
	operationId: string;
	payloadDigest: string;
	components: Record<string, unknown>;
}): SymbolicMemoryCasResult {
	mkdirSync(dirname(options.path), { recursive: true });
	const lockTarget = `${options.path}.lock-target`;
	if (!existsSync(lockTarget)) atomicWriteFile(lockTarget, Buffer.from("lock\n"));
	const release = lockfile.lockSync(lockTarget, { realpath: false });
	try {
		const ledgerPath = `${options.path}.operations.json`;
		const ledger = readLedger(ledgerPath);
		const existing = ledger.operations[options.operationId];
		const currentBytes = existsSync(options.path) ? readFileSync(options.path) : canonicalJsonBytes(EMPTY_MEMORY);
		const currentDigest = sha256(currentBytes);
		if (existing) {
			if (existing.payload_digest !== options.payloadDigest)
				throw new Error(
					`Symbolic memory operation ${options.operationId} was already recorded with different provenance or payload.`,
				);
			if (currentDigest === existing.digest) {
				if (existing.status !== "applied") {
					existing.status = "applied";
					atomicWriteFile(ledgerPath, canonicalJsonBytes(ledger));
				}
				return {
					path: options.path,
					operation_id: options.operationId,
					previous_digest: existing.previous_digest,
					digest: existing.digest,
					record_ids: [...existing.record_ids],
					replayed: true,
				};
			}
			if (existing.status !== "pending" || currentDigest !== existing.previous_digest)
				throw new Error(
					`Symbolic memory operation ${options.operationId} cannot be recovered from digest ${currentDigest}.`,
				);
			const recovered = mergedMemory(currentBytes, options.components);
			if (
				sha256(recovered.bytes) !== existing.digest ||
				canonicalDigest(recovered.recordIds) !== canonicalDigest(existing.record_ids)
			)
				throw new Error(`Symbolic memory operation ${options.operationId} recovery payload changed.`);
			atomicWriteFile(options.path, recovered.bytes);
			if (sha256(readFileSync(options.path)) !== existing.digest)
				throw new Error("Symbolic memory atomic write verification failed.");
			existing.status = "applied";
			atomicWriteFile(ledgerPath, canonicalJsonBytes(ledger));
			return {
				path: options.path,
				operation_id: options.operationId,
				previous_digest: existing.previous_digest,
				digest: existing.digest,
				record_ids: [...existing.record_ids],
				replayed: true,
			};
		}
		const expected = options.expectedDigest ?? sha256(canonicalJsonBytes(EMPTY_MEMORY));
		if (currentDigest !== expected)
			throw new Error(
				`Symbolic memory compare-and-swap conflict: expected ${options.expectedDigest ?? "empty"}, found ${currentDigest}.`,
			);
		const merged = mergedMemory(currentBytes, options.components);
		const digest = sha256(merged.bytes);
		ledger.operations[options.operationId] = {
			operation_id: options.operationId,
			payload_digest: options.payloadDigest,
			status: "pending",
			previous_digest: currentDigest,
			digest,
			record_ids: merged.recordIds,
		};
		atomicWriteFile(ledgerPath, canonicalJsonBytes(ledger));
		atomicWriteFile(options.path, merged.bytes);
		if (sha256(readFileSync(options.path)) !== digest)
			throw new Error("Symbolic memory atomic write verification failed.");
		ledger.operations[options.operationId].status = "applied";
		atomicWriteFile(ledgerPath, canonicalJsonBytes(ledger));
		return {
			path: options.path,
			operation_id: options.operationId,
			previous_digest: currentDigest,
			digest,
			record_ids: merged.recordIds,
			replayed: false,
		};
	} finally {
		release();
	}
}

export class ExperimentMaterializer {
	private readonly runRoot: string;
	private readonly sharedRoot: string;
	private readonly protocol: ExperimentProtocolStore;
	private readonly skillName: string;
	private readonly skillVersion: string;
	private readonly faultInjector?: (point: ExperimentMaterializerFaultPoint) => void;

	constructor(options: ExperimentMaterializerOptions) {
		this.runRoot = resolveContainedPath(options.runRoot, ".", true);
		this.sharedRoot = resolveContainedPath(options.sharedRoot, ".", true);
		this.protocol = options.protocol;
		this.skillName = options.skillName ?? "run_experiment";
		this.skillVersion = options.skillVersion ?? "2.0.0";
		this.faultInjector = options.faultInjector;
	}

	materializePlan(assignmentId: string, componentNames: string[] = []): ExperimentPublication {
		const accepted = acceptedFor(this.protocol, assignmentId);
		assertAcceptedMetadata(accepted, { role: "planner", kind: "plan", stage: accepted.submission.stage });
		if (!(["prepare", "code", "science"] as string[]).includes(accepted.submission.stage))
			throw new Error(`Assignment ${assignmentId} is not a materializable plan.`);
		const plan = objectResult(accepted) as ExperimentPlan;
		const stage = accepted.submission.stage as "prepare" | "code" | "science";
		const profile = this.protocol.read().assignments[assignmentId].result_validation?.execution_profile;
		const error =
			stage === "prepare"
				? validatePreparePlan(plan)
				: stage === "code"
					? validateCodePlan(plan, componentNames, profile)
					: validateSciencePlan(plan, componentNames, profile);
		if (error) throw new Error(error);
		return publishPair(
			this.runRoot,
			join("agent_reports", stage, "plan"),
			accepted.submission.execution_attempt,
			plan,
		);
	}

	materializeWorker(assignmentId: string): ExperimentPublication {
		const accepted = acceptedFor(this.protocol, assignmentId);
		assertAcceptedMetadata(accepted, {
			role: "worker",
			kind: "worker_result",
			stage: accepted.submission.stage,
			workUnit: accepted.submission.work_unit,
		});
		const result = objectResult(accepted);
		const workUnit = accepted.submission.work_unit;
		if (!workUnit) throw new Error(`Worker assignment ${assignmentId} is missing its work unit.`);
		return publishPair(
			this.runRoot,
			join(
				"agent_reports",
				accepted.submission.stage,
				"worker",
				...stepParts(accepted.submission.stage as "code" | "science", workUnit),
			),
			accepted.submission.execution_attempt,
			result,
		);
	}

	materializeReviewMatrix(options: MaterializeReviewMatrixOptions): { reports: ExperimentPublication[] } {
		const roles =
			options.stage === "code"
				? codeReviewRoles(options.validationContext?.execution_profile)
				: SCIENCE_REVIEW_ROLES;
		if (options.assignmentIds.length !== roles.length)
			throw new Error(`Review matrix must contain exactly ${roles.length} assignments.`);
		const accepted = options.assignmentIds.map((id) => acceptedFor(this.protocol, id));
		const reports = accepted.map((item, index) => {
			assertAcceptedMetadata(item, {
				role: "reviewer",
				kind: "review",
				stage: options.stage,
				workUnit: options.workUnit,
			});
			if (item.submission.review_round !== options.reviewRound)
				throw new Error("Reviewer assignment has the wrong review round.");
			const report = objectResult(item) as unknown as ExperimentReviewReport;
			if (report.role !== roles[index])
				throw new Error(`Review matrix role order mismatch: expected ${roles[index]}, got ${report.role}.`);
			return report;
		});
		const validationError = validateReviewMatrix(options.stage, reports, options.validationContext);
		if (validationError) throw new Error(validationError);
		return {
			reports: reports.map((report, index) =>
				publishPair(
					this.runRoot,
					join(
						"agent_reports",
						options.stage,
						"review",
						...stepParts(options.stage, options.workUnit),
						slug(roles[index]),
					),
					options.reviewRound,
					report,
				),
			),
		};
	}

	materializeFinal(
		options: MaterializeFinalOptions,
	): Record<string, PublishedExperimentArtifact | SymbolicMemoryCasResult> {
		const snapshot = this.protocol.read();
		if (canonicalDigest(options.provenance) !== canonicalDigest(canonicalJson(options.provenance)))
			throw new Error("Finalization provenance must be canonical JSON.");
		if (
			options.provenance.idea_digest !== snapshot.idea_digest ||
			options.provenance.policy_digest !== snapshot.policy_digest
		)
			throw new Error("Finalization provenance does not match the run Idea or policy.");
		const finalAccepted = acceptedFor(this.protocol, options.provenance.final_review.assignment_id);
		assertAcceptedMetadata(finalAccepted, { role: "final_reviewer", kind: "final_review", stage: "finalize" });
		if (finalAccepted.accepted.result_digest !== options.provenance.final_review.result_digest)
			throw new Error("Final-review provenance digest is not authoritative.");
		const sciencePlan = acceptedFor(this.protocol, options.provenance.science.plan.assignment_id);
		assertAcceptedMetadata(sciencePlan, { role: "planner", kind: "plan", stage: "science" });
		if (sciencePlan.accepted.result_digest !== options.provenance.science.plan.result_digest)
			throw new Error("Science-plan provenance digest is not authoritative.");
		const plan = objectResult(sciencePlan) as ExperimentPlan;
		const componentNames = options.componentDefinitions.map((item) => item.component);
		const profile = snapshot.assignments[sciencePlan.accepted.assignment_id].result_validation?.execution_profile;
		const planError = validateSciencePlan(plan, componentNames, profile);
		if (planError) throw new Error(planError);
		if (options.provenance.science.cohort.length !== plan.work_units.length)
			throw new Error("Finalization provenance requires the complete science worker cohort.");
		for (const [index, item] of options.provenance.science.cohort.entries()) {
			if (item.work_unit !== plan.work_units[index].id)
				throw new Error("Finalization science worker provenance is not in canonical plan order.");
			const worker = acceptedFor(this.protocol, item.assignment_id);
			assertAcceptedMetadata(worker, {
				role: "worker",
				kind: "worker_result",
				stage: "science",
				workUnit: item.work_unit,
			});
			if (worker.accepted.result_digest !== item.result_digest)
				throw new Error(`Science worker provenance for ${item.work_unit} is not authoritative.`);
		}
		if (options.provenance.science.reviews.length !== SCIENCE_REVIEW_ROLES.length)
			throw new Error("Finalization provenance requires the complete science review matrix.");
		const reports: ExperimentReviewReport[] = [];
		for (const [index, item] of options.provenance.science.reviews.entries()) {
			if (item.role !== SCIENCE_REVIEW_ROLES[index])
				throw new Error("Finalization science review provenance is not in canonical role order.");
			const review = acceptedFor(this.protocol, item.assignment_id);
			assertAcceptedMetadata(review, {
				role: "reviewer",
				kind: "review",
				stage: "science",
				workUnit: "science-cohort",
			});
			if (
				review.accepted.result_digest !== item.result_digest ||
				review.submission.input_digest !== options.provenance.science.input_digest ||
				review.submission.target_digest !== options.provenance.science.target_digest
			)
				throw new Error(`Science review provenance for ${item.role} is not authoritative.`);
			reports.push(objectResult(review) as unknown as ExperimentReviewReport);
		}
		const validationContext: ExperimentReviewValidationContext = {
			...(profile ? { execution_profile: profile } : {}),
			component_names: componentNames,
			science_conditions: plan.work_units.map((unit) => ({
				id: unit.id,
				enabled_components: unit.enabled_components ?? [],
				disabled_components: unit.disabled_components ?? [],
				kind: unit.kind as "all_components_reference" | "component_disabled",
			})),
		};
		const reviewError = validateReviewMatrix("science", reports, validationContext);
		if (reviewError) throw new Error(reviewError);
		const authoritativeResults = reports.find((report) => report.role === "statistical_interpretation")
			?.structured_findings.component_results;
		if (!authoritativeResults || typeof authoritativeResults !== "object" || Array.isArray(authoritativeResults))
			throw new Error("Authoritative science review matrix is missing component results.");
		const resultProjection = (results: Record<string, unknown>) =>
			Object.fromEntries(
				Object.entries(results).map(([name, raw]) => {
					const value = raw as Record<string, unknown>;
					return [
						name,
						{
							result: value.result,
							metric: value.metric,
							value: value.value,
							confidence: value.confidence,
							analysis: value.analysis,
							follow_up_required: value.follow_up_required,
						},
					];
				}),
			);
		if (
			canonicalDigest(resultProjection(authoritativeResults as Record<string, unknown>)) !==
			canonicalDigest(resultProjection(options.componentResults))
		)
			throw new Error("Final component results do not match the authoritative science review matrix.");
		const ablation = normalizeFinal(
			options.componentDefinitions,
			options.componentResults,
			options.summary,
			profile?.kind === "baseline",
		);
		const operationPayload = canonicalJson({ run_id: snapshot.run_id, provenance: options.provenance, ablation });
		const payloadDigest = canonicalDigest(operationPayload);
		const operationId = `experiment-finalization-${payloadDigest}`;
		const finalDir = join("agent_reports", "ablation", "final");
		const writePath = (pathValue: string, value: unknown): PublishedExperimentArtifact => {
			const path = resolveContainedPath(this.runRoot, pathValue, false);
			const bytes = canonicalJsonBytes(value);
			atomicWriteFile(path, bytes);
			if (!readFileSync(path).equals(bytes))
				throw new Error(`Experiment artifact atomic write verification failed: ${pathValue}.`);
			return { path: relativePath(this.runRoot, path), sha256: sha256(bytes), bytes: bytes.byteLength };
		};
		const write = (name: string, value: unknown): PublishedExperimentArtifact =>
			writePath(join(finalDir, name), value);
		const intent = write("finalization_intent.json", {
			schema_version: "xlab.experiment_finalization_intent.v1",
			operation_id: operationId,
			payload_digest: payloadDigest,
			run_id: snapshot.run_id,
			provenance: options.provenance,
			ablation_digest: canonicalDigest(ablation),
		});
		const results = write("ablation_results.json", ablation);
		const ablationManifest = write("ablation_results_manifest.json", {
			artifact_name: "ablation_results",
			artifact_role: "final_result",
			top_level_keys: ["components", "summary"],
			component_order_source: "idea.json.components",
			component_required_fields: [...COMPONENT_FIELDS],
			summary_required_fields: ["feasible", "confidence", "key_findings"],
			canonical_components: options.componentDefinitions.map((item) => item.component),
			provenance: options.provenance,
		});
		const materialization = write("materialization_report.json", {
			valid: true,
			mode: "deterministic",
			operation_id: operationId,
			artifact_role: "final_result",
			ablation_results_path: results.path,
			ablation_results_sha256: results.sha256,
			finalization_intent_path: intent.path,
			finalization_intent_sha256: intent.sha256,
		});
		const memory = compareAndSwapSymbolicMemory({
			path: resolveContainedPath(this.sharedRoot, options.symbolicMemoryPath, false),
			expectedDigest: options.expectedSymbolicMemoryDigest,
			operationId,
			payloadDigest,
			components: ablation.components as Record<string, unknown>,
		});
		this.faultInjector?.("after_symbolic_memory_cas");
		const receipt = write("symbolic_memory_receipt.json", {
			status: "PASS",
			hook: "final_science_prefinish",
			operation_id: operationId,
			symbolic_memory_file_path: relativePath(this.sharedRoot, memory.path),
			previous_digest: memory.previous_digest,
			digest: memory.digest,
			record_ids: memory.record_ids,
		});
		this.faultInjector?.("after_symbolic_memory_receipt");
		const audit = write(
			"final_audit.json",
			options.audit ?? {
				status: "PASS",
				checked_artifacts: [intent.path, results.path, ablationManifest.path, materialization.path, receipt.path],
				provenance: options.provenance,
				issues: [],
			},
		);
		this.faultInjector?.("after_final_audit");
		const artifacts: XlabManifestArtifact[] = [
			{
				type: "ablation_results",
				schema_version: "1",
				path: results.path,
				bytes: results.bytes,
				digest: `sha256:${results.sha256}`,
				metadata: { sha256: results.sha256, operation_id: operationId },
			},
			{
				type: "final_audit",
				schema_version: "1",
				path: audit.path,
				bytes: audit.bytes,
				digest: `sha256:${audit.sha256}`,
				metadata: { sha256: audit.sha256, operation_id: operationId },
			},
			{
				type: "symbolic_memory_receipt",
				schema_version: "1",
				path: receipt.path,
				bytes: receipt.bytes,
				digest: `sha256:${receipt.sha256}`,
				metadata: { sha256: receipt.sha256, operation_id: operationId },
			},
			{
				type: "ablation_results_manifest",
				schema_version: "1",
				path: ablationManifest.path,
				bytes: ablationManifest.bytes,
				digest: `sha256:${ablationManifest.sha256}`,
				metadata: { sha256: ablationManifest.sha256 },
			},
			{
				type: "materialization_report",
				schema_version: "1",
				path: materialization.path,
				bytes: materialization.bytes,
				digest: `sha256:${materialization.sha256}`,
				metadata: { sha256: materialization.sha256 },
			},
			{
				type: "finalization_intent",
				schema_version: "1",
				path: intent.path,
				bytes: intent.bytes,
				digest: `sha256:${intent.sha256}`,
				metadata: { sha256: intent.sha256 },
			},
		];
		const manifestValue: XlabManifestEnvelope = {
			schema_version: "2",
			run_id: snapshot.run_id,
			skill_name: this.skillName,
			skill_version: this.skillVersion,
			status: "success",
			created_at: snapshot.created_at,
			inputs: {
				idea_digest: snapshot.idea_digest,
				policy_digest: snapshot.policy_digest,
				finalization_provenance: options.provenance,
			},
			outputs: {
				operation_id: operationId,
				symbolic_memory_digest: memory.digest,
				ablation_results_sha256: results.sha256,
			},
			validation: { valid: true, final_review: options.provenance.final_review, symbolic_memory_verified: true },
			artifacts,
		};
		const manifest = writePath("manifest.json", manifestValue);
		this.faultInjector?.("after_manifest");
		return {
			ablation_results: results,
			ablation_results_manifest: ablationManifest,
			materialization_report: materialization,
			finalization_intent: intent,
			symbolic_memory_receipt: receipt,
			final_audit: audit,
			manifest,
			symbolic_memory: memory,
		};
	}
}
