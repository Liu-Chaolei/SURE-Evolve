import { createHash, randomUUID, timingSafeEqual } from "node:crypto";
import {
	closeSync,
	existsSync,
	fsyncSync,
	mkdirSync,
	openSync,
	readFileSync,
	realpathSync,
	renameSync,
	statSync,
	unlinkSync,
	writeFileSync,
	writeSync,
} from "node:fs";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import lockfile from "proper-lockfile";

export const EXPERIMENT_STAGES = ["prepare", "code", "science", "finalize"] as const;
export const PREPARE_WORK_UNITS = ["repos", "dataset", "model", "env", "synthesis"] as const;
export const CODE_REVIEW_ROLES = [
	"idea_alignment",
	"implementation_correctness",
	"scientific_invariants",
	"protocol_semantics",
	"integration_smoke",
	"reproducibility",
	"code_cleanliness",
] as const;
export const SCIENCE_REVIEW_ROLES = [
	"protocol_compliance",
	"protocol_semantics",
	"condition_toggle",
	"evidence_plausibility",
	"statistical_interpretation",
	"idea_alignment",
] as const;

export type ExperimentStage = (typeof EXPERIMENT_STAGES)[number];
export interface ExperimentExecutionProfile {
	mode: "standard" | "direct_formal";
	kind: "research" | "baseline";
}

export function codeReviewRoles(profile?: ExperimentExecutionProfile): readonly string[] {
	return profile?.mode === "direct_formal"
		? CODE_REVIEW_ROLES.map((role) => (role === "integration_smoke" ? "static_integration" : role))
		: CODE_REVIEW_ROLES;
}
export type ExperimentChildRole = "planner" | "worker" | "reviewer" | "final_reviewer";
export type ExperimentProtocolStatus =
	| "pending"
	| "running"
	| "paused"
	| "success"
	| "incomplete"
	| "failed"
	| "cancelled";
export type ExperimentAssignmentStatus = "active" | "submitted" | "accepted" | "superseded" | "cancelled";
export type ExperimentSubmissionKind = "plan" | "worker_result" | "review" | "final_review";

export type CanonicalJson = null | boolean | number | string | CanonicalJson[] | { [key: string]: CanonicalJson };

export interface ExperimentOwnerLease {
	owner_id: string;
	generation: number;
	acquired_at: string;
	lease_expires_at: string;
}

export interface ExperimentOwnerFence {
	owner_id: string;
	generation: number;
}

export interface ExperimentChildIdentity {
	child_id: string;
	session_id: string;
	session_file: string;
	parent_run_id: string;
	role: ExperimentChildRole;
	stage: ExperimentStage;
	work_unit?: string;
	execution_attempt: number;
	generation: number;
	review_round: number;
}

export interface ExperimentResultValidationContext {
	execution_profile?: ExperimentExecutionProfile;
	component_names?: string[];
	reviewer_role?: string;
	review_context?: ExperimentReviewValidationContext;
}

export interface ExperimentAssignment {
	schema_version: "xlab.experiment_assignment.v1";
	assignment_id: string;
	assignment_token_digest: string;
	child: ExperimentChildIdentity;
	kind: ExperimentSubmissionKind;
	input_digest: string;
	target_digest: string;
	policy_digest: string;
	read_scope: string[];
	write_scope: string[];
	result_validation?: ExperimentResultValidationContext;
	status: ExperimentAssignmentStatus;
	created_at: string;
	updated_at: string;
	accepted_result_digest?: string;
}

export interface ExperimentSubmission {
	schema_version: "xlab.experiment_submission.v1";
	submission_id: string;
	assignment_id: string;
	assignment_token: string;
	kind: ExperimentSubmissionKind;
	child_id: string;
	role: ExperimentChildRole;
	stage: ExperimentStage;
	work_unit?: string;
	execution_attempt: number;
	generation: number;
	review_round: number;
	input_digest: string;
	target_digest: string;
	policy_digest: string;
	result: CanonicalJson;
	output_paths: string[];
	submitted_at: string;
}

export interface AcceptedExperimentSubmission {
	schema_version: "xlab.experiment_accepted_submission.v1";
	submission_id: string;
	assignment_id: string;
	result_digest: string;
	output_digests: Record<string, string>;
	accepted_at: string;
}

export interface ExperimentAcceptedResult {
	accepted: AcceptedExperimentSubmission;
	submission: Omit<ExperimentSubmission, "assignment_token"> & { assignment_token: "[redacted]" };
}

export interface ExperimentProtocolEvent {
	schema_version: "xlab.experiment_event.v1";
	sequence: number;
	type: string;
	timestamp: string;
	owner_generation: number;
	previous_digest: string | null;
	data: CanonicalJson;
	projection: CanonicalJson;
	digest: string;
}

export interface ExperimentCancellationIntent {
	schema_version: "xlab.experiment_cancellation.v1";
	run_id: string;
	terminal_generation: number;
	requested_at: string;
	reason: string;
}

export interface ExperimentProtocolSnapshot {
	schema_version: "xlab.experiment_protocol.v1";
	run_id: string;
	idea_digest: string;
	policy_digest: string;
	status: ExperimentProtocolStatus;
	publication_recovery_reason?: string;
	owner?: ExperimentOwnerLease;
	terminal_generation?: number;
	current_stage?: ExperimentStage;
	completed_stages: ExperimentStage[];
	assignments: Record<string, ExperimentAssignment>;
	children: Record<string, ExperimentChildIdentity>;
	last_event_sequence: number;
	last_event_digest: string | null;
	created_at: string;
	updated_at: string;
}

export interface ExperimentCodeArtifact {
	path: string;
	artifact_type: string;
	symbols: string[];
	responsibility: string;
	dependencies: string[];
	config_keys: string[];
	entrypoint_role: string;
}

export interface ExperimentWorkUnit {
	id: string;
	needs?: string[];
	kind?: string;
	goal?: string;
	input_paths?: Record<string, unknown>;
	repos_policy?: "reference_or_copy";
	project_must_be_self_contained?: boolean;
	research_required?: boolean;
	acquisition_required?: boolean;
	existing_local_hints?: unknown[];
	done_condition?: string;
	artifact_ids?: string[];
	component_scope?: string[];
	code_artifacts?: ExperimentCodeArtifact[];
	interface_contract?: Record<string, unknown>;
	implementation_requirements?: Record<string, unknown>;
	experiment_bindings?: Record<string, unknown>;
	component_disable_hooks?: Array<Record<string, unknown>>;
	write_scope?: string;
	verify_command?: string;
	component?: string;
	enabled_components?: string[];
	disabled_components?: string[];
	disabled_component?: string;
	reference_condition_id?: string;
	output_dir?: string;
	command?: string;
	full_run?: boolean;
	run_level?: string;
	setup_rationale?: string;
	runtime_probe_summary?: string;
	source_basis?: unknown[];
	training_protocol?: Record<string, unknown>;
	evaluation_protocol?: Record<string, unknown>;
	train_dataset_binding?: unknown;
	evaluation_dataset_bindings?: unknown[];
	metric_bindings?: unknown[];
	component_set_description?: unknown;
	result_interpretation_rule?: unknown;
	raw_evidence?: unknown[];
	pass_condition?: string;
	evidence?: string[];
	[key: string]: unknown;
}

export interface ExperimentPlan {
	stage: "prepare" | "code" | "science";
	work_units: ExperimentWorkUnit[];
	[key: string]: unknown;
}

export interface ExperimentReviewIssue {
	code: string;
	message: string;
	required_fix: string;
	evidence: string;
}

export interface ExperimentReviewReport {
	artifact_role: "reviewer_report";
	role: string;
	reviewer_kind: "agent";
	verdict: "PASS" | "FAIL";
	blocking: boolean;
	summary: string;
	checked_artifacts: string[];
	issues: ExperimentReviewIssue[];
	structured_findings: Record<string, unknown>;
	[key: string]: unknown;
}

export type SubmissionAcceptance =
	| { ok: true; disposition: "accepted" | "idempotent"; accepted: AcceptedExperimentSubmission }
	| { ok: false; error: string };

export type ExperimentProtocolFaultPoint =
	| "after_assignment_record"
	| "after_submission_record"
	| "after_accepted_record"
	| "after_assignment_acceptance_record"
	| "after_cancellation_intent"
	| "after_cancellation_assignments"
	| "after_journal_append"
	| "after_projection_write";

function isPlainObject(value: unknown): value is Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		return false;
	}
	const prototype = Object.getPrototypeOf(value);
	return prototype === Object.prototype || prototype === null;
}

function normalizeCanonicalJson(value: unknown, seen: Set<object>): CanonicalJson {
	if (value === null || typeof value === "string" || typeof value === "boolean") {
		return value;
	}
	if (typeof value === "number") {
		if (!Number.isFinite(value)) {
			throw new TypeError("Canonical JSON does not permit non-finite numbers.");
		}
		return Object.is(value, -0) ? 0 : value;
	}
	if (Array.isArray(value)) {
		if (seen.has(value)) {
			throw new TypeError("Canonical JSON does not permit cyclic values.");
		}
		seen.add(value);
		const normalized = value.map((item, index) => {
			if (!(index in value) || item === undefined) {
				throw new TypeError("Canonical JSON does not permit sparse arrays or undefined values.");
			}
			return normalizeCanonicalJson(item, seen);
		});
		seen.delete(value);
		return normalized;
	}
	if (!isPlainObject(value)) {
		throw new TypeError("Canonical JSON only permits plain objects, arrays, and JSON primitives.");
	}
	if (seen.has(value)) {
		throw new TypeError("Canonical JSON does not permit cyclic values.");
	}
	seen.add(value);
	const normalized: Record<string, CanonicalJson> = {};
	for (const key of Object.keys(value).sort()) {
		const child = value[key];
		if (child === undefined) {
			throw new TypeError(`Canonical JSON does not permit undefined at key ${key}.`);
		}
		normalized[key] = normalizeCanonicalJson(child, seen);
	}
	seen.delete(value);
	return normalized;
}

export function canonicalJson(value: unknown): CanonicalJson {
	return normalizeCanonicalJson(value, new Set());
}

export function canonicalJsonBytes(value: unknown): Buffer {
	return Buffer.from(`${JSON.stringify(canonicalJson(value), null, 2)}\n`, "utf-8");
}

function canonicalJsonLine(value: unknown): Buffer {
	return Buffer.from(`${JSON.stringify(canonicalJson(value))}\n`, "utf-8");
}

export function sha256(value: Uint8Array | string): string {
	return createHash("sha256").update(value).digest("hex");
}

export function canonicalDigest(value: unknown): string {
	return sha256(canonicalJsonBytes(value));
}

function ensureDigest(value: string, field: string): string | undefined {
	return /^[a-f0-9]{64}$/.test(value) ? undefined : `${field} must be a lowercase SHA-256 digest.`;
}

function isContained(basePath: string, candidatePath: string): boolean {
	const rel = relative(basePath, candidatePath);
	return rel === "" || (rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel));
}

function realExistingAncestor(path: string): string {
	let current = path;
	while (!existsSync(current)) {
		const parent = dirname(current);
		if (parent === current) {
			throw new Error(`No existing ancestor for ${path}.`);
		}
		current = parent;
	}
	return realpathSync(current);
}

export function resolveContainedPath(baseDir: string, pathValue: string, requireExisting = true): string {
	const base = realpathSync(baseDir);
	const candidate = resolve(baseDir, pathValue);
	const resolvedCandidate = existsSync(candidate) ? realpathSync(candidate) : realExistingAncestor(candidate);
	if (!isContained(base, resolvedCandidate) || !isContained(resolve(baseDir), candidate)) {
		throw new Error(`Path escapes allowed scope: ${pathValue}`);
	}
	if (requireExisting && !existsSync(candidate)) {
		throw new Error(`Path does not exist: ${pathValue}`);
	}
	return candidate;
}

function fsyncDirectory(path: string): void {
	const fd = openSync(path, "r");
	try {
		fsyncSync(fd);
	} finally {
		closeSync(fd);
	}
}

export function atomicWriteFile(path: string, bytes: Uint8Array): void {
	const parent = dirname(path);
	mkdirSync(parent, { recursive: true });
	const temporary = join(parent, `.${path.split(sep).at(-1)}.${process.pid}.${randomUUID()}.tmp`);
	let fd: number | undefined;
	try {
		fd = openSync(temporary, "wx", 0o600);
		writeFileSync(fd, bytes);
		fsyncSync(fd);
		closeSync(fd);
		fd = undefined;
		renameSync(temporary, path);
		fsyncDirectory(parent);
	} catch (error) {
		if (fd !== undefined) {
			try {
				closeSync(fd);
			} catch {
				// Preserve the original publication error.
			}
		}
		try {
			if (existsSync(temporary)) {
				unlinkSync(temporary);
			}
		} catch {
			// Preserve the original publication error.
		}
		throw error;
	}
}

export function atomicWriteJson(path: string, value: unknown): void {
	atomicWriteFile(path, canonicalJsonBytes(value));
}

export function appendDurableJsonLine(path: string, value: unknown): void {
	mkdirSync(dirname(path), { recursive: true });
	const fd = openSync(path, "a", 0o600);
	try {
		writeSync(fd, canonicalJsonLine(value));
		fsyncSync(fd);
	} finally {
		closeSync(fd);
	}
}

function tokenDigest(token: string): Buffer {
	return createHash("sha256").update(token, "utf-8").digest();
}

export function assignmentTokenMatches(expectedDigest: string, providedToken: string): boolean {
	if (!/^[a-f0-9]{64}$/.test(expectedDigest)) {
		return false;
	}
	return timingSafeEqual(Buffer.from(expectedDigest, "hex"), tokenDigest(providedToken));
}

function validateChildIdentity(actual: ExperimentSubmission, expected: ExperimentAssignment): string | undefined {
	const child = expected.child;
	if (actual.child_id !== child.child_id || actual.role !== child.role) {
		return "Submission child identity or role does not match its assignment.";
	}
	if (actual.stage !== child.stage || actual.work_unit !== child.work_unit) {
		return "Submission stage or work unit does not match its assignment.";
	}
	if (
		actual.execution_attempt !== child.execution_attempt ||
		actual.generation !== child.generation ||
		actual.review_round !== child.review_round
	) {
		return "Submission attempt, generation, or review round does not match its assignment.";
	}
	return undefined;
}

export function validateSubmission(
	assignment: ExperimentAssignment,
	submission: ExperimentSubmission,
	terminalGeneration?: number,
): string | undefined {
	if (assignment.status !== "active" && assignment.status !== "submitted" && assignment.status !== "accepted") {
		return `Assignment ${assignment.assignment_id} is ${assignment.status}.`;
	}
	if (terminalGeneration !== undefined) {
		return `Run is terminal at generation ${terminalGeneration}; late submissions are rejected.`;
	}
	if (submission.assignment_id !== assignment.assignment_id || submission.kind !== assignment.kind) {
		return "Submission assignment ID or kind does not match its assignment.";
	}
	if (!assignmentTokenMatches(assignment.assignment_token_digest, submission.assignment_token)) {
		return "Submission assignment token is invalid.";
	}
	const childError = validateChildIdentity(submission, assignment);
	if (childError) {
		return childError;
	}
	for (const [field, actual, expected] of [
		["input_digest", submission.input_digest, assignment.input_digest],
		["target_digest", submission.target_digest, assignment.target_digest],
		["policy_digest", submission.policy_digest, assignment.policy_digest],
	] as const) {
		if (actual !== expected) {
			return `Submission ${field} does not match its assignment.`;
		}
	}
	return undefined;
}

const PREPARE_ARTIFACT_IDS: Record<(typeof PREPARE_WORK_UNITS)[number], string[]> = {
	repos: ["prepare.discovery", "prepare.repos"],
	dataset: ["prepare.dataset"],
	model: ["prepare.model"],
	env: ["prepare.env"],
	synthesis: ["prepare.idea", "prepare.target_inventory"],
};

function nonEmptyString(value: unknown): value is string {
	return typeof value === "string" && value.trim().length > 0;
}

function nonEmptyRecord(value: unknown): value is Record<string, unknown> {
	return isPlainObject(value) && Object.keys(value).length > 0;
}

function stringList(value: unknown, allowEmpty = false): value is string[] {
	return Array.isArray(value) && (allowEmpty || value.length > 0) && value.every((item) => nonEmptyString(item));
}

function validateManagedCompletion(unit: ExperimentWorkUnit): string | undefined {
	if (!nonEmptyString(unit.done_condition)) {
		return `Work unit ${unit.id} must provide a concrete done_condition.`;
	}
	const done = unit.done_condition.toLowerCase();
	if (!done.includes("artifact ledger") || !/(managed artifact|artifact tool)/.test(done)) {
		return `Work unit ${unit.id} done_condition must require managed artifact tools and artifact ledger proof.`;
	}
	for (const artifactId of unit.artifact_ids ?? []) {
		if (!done.includes(artifactId.toLowerCase())) {
			return `Work unit ${unit.id} done_condition must name required artifact ${artifactId}.`;
		}
	}
	return undefined;
}

function hasForbiddenExecutionShortcut(value: string): boolean {
	return /(?:^|[\s_=-])(mock|random[-_ ]?only|import[-_ ]?only|dry[-_ ]?run|timeout[-_ ]?as[-_ ]?success|debug|quick|pilot)(?:$|[\s_=-])/i.test(
		value,
	);
}

function validateProjectRelativePath(pathValue: unknown, field: string): string | undefined {
	if (!nonEmptyString(pathValue)) {
		return `${field} must be a non-empty project-relative path.`;
	}
	if (isAbsolute(pathValue) || pathValue === ".." || pathValue.startsWith(`..${sep}`)) {
		return `${field} must stay within the managed project.`;
	}
	return undefined;
}

function sameStringSet(actual: unknown, expected: string[]): boolean {
	return (
		stringList(actual, expected.length === 0) &&
		actual.length === expected.length &&
		new Set(actual).size === expected.length &&
		expected.every((value) => actual.includes(value))
	);
}

export function validatePreparePlan(plan: ExperimentPlan): string | undefined {
	if (plan.stage !== "prepare") {
		return "Prepare plan must declare stage prepare.";
	}
	const ids = plan.work_units.map((unit) => unit.id);
	if (JSON.stringify(ids) !== JSON.stringify(PREPARE_WORK_UNITS)) {
		return `Prepare plan work units must be exactly: ${PREPARE_WORK_UNITS.join(", ")}.`;
	}
	for (const unit of plan.work_units) {
		if (!nonEmptyString(unit.goal) || !nonEmptyRecord(unit.input_paths)) {
			return `Prepare work unit ${unit.id} must provide goal and input_paths.`;
		}
		if (
			unit.repos_policy !== "reference_or_copy" ||
			unit.project_must_be_self_contained !== true ||
			typeof unit.research_required !== "boolean" ||
			typeof unit.acquisition_required !== "boolean" ||
			!Array.isArray(unit.existing_local_hints)
		) {
			return `Prepare work unit ${unit.id} has an invalid repository, research, or acquisition policy.`;
		}
		const expectedArtifacts = PREPARE_ARTIFACT_IDS[unit.id as (typeof PREPARE_WORK_UNITS)[number]];
		if (JSON.stringify(unit.artifact_ids) !== JSON.stringify(expectedArtifacts)) {
			return `Prepare work unit ${unit.id} artifact_ids must be exactly: ${expectedArtifacts.join(", ")}.`;
		}
		const completionError = validateManagedCompletion(unit);
		if (completionError) {
			return completionError;
		}
	}
	return undefined;
}

function validateUniqueWorkUnitIds(workUnits: ExperimentWorkUnit[]): string | undefined {
	if (workUnits.length === 0) {
		return "Plan must contain at least one work unit.";
	}
	const ids = new Set<string>();
	for (const unit of workUnits) {
		if (!unit.id.trim()) {
			return "Plan work unit IDs must be non-empty.";
		}
		if (ids.has(unit.id)) {
			return `Plan contains duplicate work unit ${unit.id}.`;
		}
		ids.add(unit.id);
	}
	for (const unit of workUnits) {
		for (const dependency of unit.needs ?? []) {
			if (!ids.has(dependency) || dependency === unit.id) {
				return `Work unit ${unit.id} has invalid dependency ${dependency}.`;
			}
		}
	}
	return undefined;
}

export function validateCodePlan(
	plan: ExperimentPlan,
	componentNames: string[] = [],
	profile?: ExperimentExecutionProfile,
): string | undefined {
	if (plan.stage !== "code") {
		return "Code plan must declare stage code.";
	}
	const idError = validateUniqueWorkUnitIds(plan.work_units);
	if (idError) {
		return idError;
	}
	if (!stringList(componentNames, profile?.kind === "baseline")) {
		return "Code plan validation requires canonical Idea components.";
	}
	const terminalId = profile?.mode === "direct_formal" ? "final_static_integration" : "final_integration_smoke";
	const smoke = plan.work_units.filter((unit) => unit.id === terminalId);
	if (smoke.length !== 1 || plan.work_units.at(-1)?.id !== terminalId) {
		return `Code plan must contain one terminal ${terminalId} work unit.`;
	}
	const terminal = smoke[0];
	const otherIds = plan.work_units.slice(0, -1).map((unit) => unit.id);
	if (otherIds.some((id) => !(terminal.needs ?? []).includes(id))) {
		return `${terminalId} must depend on every preceding code work unit.`;
	}
	for (const unit of plan.work_units) {
		if (!nonEmptyString(unit.goal) || !nonEmptyRecord(unit.input_paths)) {
			return `Code work unit ${unit.id} must provide goal and input_paths.`;
		}
		if (
			unit.repos_policy !== "reference_or_copy" ||
			unit.project_must_be_self_contained !== true ||
			unit.write_scope !== "project"
		) {
			return `Code work unit ${unit.id} must use the self-contained managed project policy.`;
		}
		if (!sameStringSet(unit.component_scope, componentNames)) {
			return `Code work unit ${unit.id} component_scope must exactly cover canonical Idea components.`;
		}
		if (!Array.isArray(unit.code_artifacts) || unit.code_artifacts.length === 0) {
			return `Code work unit ${unit.id} must declare structured code_artifacts.`;
		}
		for (const artifact of unit.code_artifacts) {
			const pathError = validateProjectRelativePath(artifact.path, `Code artifact path for ${unit.id}`);
			if (pathError) {
				return pathError;
			}
			if (
				!nonEmptyString(artifact.artifact_type) ||
				!stringList(artifact.symbols) ||
				!nonEmptyString(artifact.responsibility) ||
				!stringList(artifact.dependencies, true) ||
				!stringList(artifact.config_keys, true) ||
				!nonEmptyString(artifact.entrypoint_role)
			) {
				return `Code work unit ${unit.id} contains an incomplete code artifact contract.`;
			}
		}
		if (
			!nonEmptyRecord(unit.interface_contract) ||
			!nonEmptyRecord(unit.implementation_requirements) ||
			!nonEmptyRecord(unit.experiment_bindings) ||
			!Array.isArray(unit.component_disable_hooks) ||
			(unit.component_disable_hooks.length === 0 && profile?.kind !== "baseline") ||
			unit.component_disable_hooks.some((hook) => !nonEmptyRecord(hook))
		) {
			return `Code work unit ${unit.id} must define interfaces, implementation bindings, and component-disable hooks.`;
		}
		if (!nonEmptyString(unit.verify_command) || hasForbiddenExecutionShortcut(unit.verify_command)) {
			return `Code work unit ${unit.id} must provide a concrete evidentiary verify_command.`;
		}
		const commandContract = `${unit.verify_command}\n${unit.done_condition ?? ""}`;
		if (/sys\.path|pip\s+install\s+-e|editable install|outside[- ]project import/i.test(commandContract)) {
			return `Code work unit ${unit.id} may not depend on outside-project import shortcuts.`;
		}
		if (!Array.isArray(unit.artifact_ids) || !unit.artifact_ids.includes(`code.${unit.id}.handoff`)) {
			return `Code work unit ${unit.id} must publish code.${unit.id}.handoff.`;
		}
		const completionError = validateManagedCompletion(unit);
		if (completionError) {
			return completionError;
		}
		if (hasForbiddenExecutionShortcut(unit.kind ?? "") || hasForbiddenExecutionShortcut(unit.id)) {
			return "Code plan contains a non-evidentiary smoke strategy.";
		}
	}
	const evidence = terminal.evidence ?? [];
	if (profile?.mode === "direct_formal") {
		if (plan.work_units.some((unit) => /smoke/i.test(unit.id)))
			return "direct_formal cannot schedule smoke work units.";
		return evidence.includes("syntax") && evidence.includes("component_contract")
			? undefined
			: "Direct formal code requires syntax and component_contract evidence; runtime evidence is deferred to science.";
	}
	if (!evidence.includes("integrated") || !evidence.includes("component_disabled")) {
		return "final_integration_smoke must require integrated and component_disabled evidence.";
	}
	if (!/dataset/i.test(terminal.done_condition ?? "") || !/(metric|evaluation)/i.test(terminal.done_condition ?? "")) {
		return "final_integration_smoke must require real dataset and metric or evaluation evidence.";
	}
	return undefined;
}

export function validateSciencePlan(
	plan: ExperimentPlan,
	componentNames: string[],
	profile?: ExperimentExecutionProfile,
): string | undefined {
	if (plan.stage !== "science") {
		return "Science plan must declare stage science.";
	}
	const idError = validateUniqueWorkUnitIds(plan.work_units);
	if (idError) {
		return idError;
	}
	if (!stringList(componentNames, profile?.kind === "baseline")) {
		return "Science plan validation requires canonical Idea components.";
	}
	const references = plan.work_units.filter((unit) => unit.kind === "all_components_reference");
	if (references.length !== 1 || references[0].full_run !== true || plan.work_units[0] !== references[0]) {
		return "Science plan must begin with exactly one full-run all-components reference condition.";
	}
	const reference = references[0];
	const disabled = plan.work_units.filter((unit) => unit.kind === "component_disabled");
	const actual = disabled
		.map((unit) => unit.disabled_component)
		.filter((value): value is string => typeof value === "string");
	if (
		actual.length !== componentNames.length ||
		new Set(actual).size !== componentNames.length ||
		componentNames.some((component) => !actual.includes(component))
	) {
		return "Science plan must contain exactly one disabled full-run condition for each canonical Idea component.";
	}
	if (plan.work_units.length !== componentNames.length + 1) {
		return "Science plan may not contain extra formal conditions.";
	}
	for (const unit of plan.work_units) {
		if (!nonEmptyString(unit.goal)) {
			return `Science condition ${unit.id} must provide a goal.`;
		}
		if (unit.full_run !== true || unit.run_level !== "full") {
			return `Science condition ${unit.id} must be a full run.`;
		}
		const expectedDisabled = unit === reference ? [] : [unit.disabled_component as string];
		const expectedEnabled = componentNames.filter((component) => !expectedDisabled.includes(component));
		if (
			!sameStringSet(unit.disabled_components, expectedDisabled) ||
			!sameStringSet(unit.enabled_components, expectedEnabled)
		) {
			return `Science condition ${unit.id} must exactly partition enabled and disabled Idea components.`;
		}
		if (unit !== reference && unit.reference_condition_id !== reference.id) {
			return `Science condition ${unit.id} must bind to reference condition ${reference.id}.`;
		}
		const outputError = validateProjectRelativePath(unit.output_dir, `Science output_dir for ${unit.id}`);
		if (outputError || unit.output_dir !== `results/science/${unit.id}`) {
			return `Science condition ${unit.id} output_dir must be results/science/${unit.id}.`;
		}
		if (!nonEmptyString(unit.command) || hasForbiddenExecutionShortcut(unit.command)) {
			return `Science condition ${unit.id} must provide a concrete full-run command.`;
		}
		if (unit !== reference && !unit.command.includes(unit.disabled_component as string)) {
			return `Science condition ${unit.id} command must expose its component-disable toggle.`;
		}
		if (
			!nonEmptyString(unit.setup_rationale) ||
			!nonEmptyString(unit.runtime_probe_summary) ||
			!Array.isArray(unit.source_basis) ||
			unit.source_basis.length === 0 ||
			!nonEmptyRecord(unit.training_protocol) ||
			!nonEmptyRecord(unit.evaluation_protocol) ||
			unit.train_dataset_binding === undefined ||
			!Array.isArray(unit.evaluation_dataset_bindings) ||
			unit.evaluation_dataset_bindings.length === 0 ||
			!Array.isArray(unit.metric_bindings) ||
			unit.metric_bindings.length === 0 ||
			!Array.isArray(unit.raw_evidence) ||
			unit.raw_evidence.length === 0
		) {
			return `Science condition ${unit.id} lacks protocol, source, dataset, metric, or raw-evidence bindings.`;
		}
		if (!Array.isArray(unit.artifact_ids) || !unit.artifact_ids.includes(`science.${unit.id}.evidence`)) {
			return `Science condition ${unit.id} must publish science.${unit.id}.evidence.`;
		}
		const pass = unit.pass_condition?.toLowerCase() ?? "";
		if (!pass.includes("metric") || !pass.includes("log") || !pass.includes("evidence")) {
			return `Science condition ${unit.id} pass_condition must require metrics, logs, and evidence.`;
		}
		const completionError = validateManagedCompletion({ ...unit, done_condition: unit.pass_condition });
		if (completionError) {
			return completionError.replace("done_condition", "pass_condition");
		}
	}
	return undefined;
}

export interface ExperimentReviewValidationContext {
	execution_profile?: ExperimentExecutionProfile;
	component_names: string[];
	science_conditions: Array<{
		id: string;
		enabled_components: string[];
		disabled_components: string[];
		kind: "all_components_reference" | "component_disabled";
	}>;
}

export function validateReviewMatrix(
	stage: "code" | "science",
	reports: ExperimentReviewReport[],
	context?: ExperimentReviewValidationContext,
): string | undefined {
	const expected = stage === "code" ? codeReviewRoles(context?.execution_profile) : SCIENCE_REVIEW_ROLES;
	if (reports.length !== expected.length) {
		return `${stage} review matrix must contain ${expected.length} reports.`;
	}
	for (let index = 0; index < expected.length; index += 1) {
		const report = reports[index] as unknown;
		if (!isPlainObject(report)) {
			return `${stage} reviewer ${index + 1} must provide an object-shaped report.`;
		}
		if (report.role !== expected[index]) {
			return `${stage} reviewer ${index + 1} must be ${expected[index]}.`;
		}
		if (
			report.artifact_role !== "reviewer_report" ||
			report.reviewer_kind !== "agent" ||
			(report.verdict !== "PASS" && report.verdict !== "FAIL") ||
			report.blocking !== true ||
			!nonEmptyString(report.summary) ||
			!stringList(report.checked_artifacts) ||
			!nonEmptyRecord(report.structured_findings)
		) {
			return `Reviewer ${report.role} has an incomplete reviewer report contract.`;
		}
		if (
			!Array.isArray(report.issues) ||
			report.issues.some(
				(issue) =>
					!isPlainObject(issue) ||
					![issue.code, issue.message, issue.required_fix, issue.evidence].every(nonEmptyString),
			)
		) {
			return `Reviewer ${report.role} must provide typed structured issues.`;
		}
		if (
			(report.verdict === "PASS" && report.issues.length > 0) ||
			(report.verdict === "FAIL" && report.issues.length === 0)
		) {
			return `Reviewer ${report.role} verdict and issues are inconsistent.`;
		}
	}
	if (stage === "science") {
		if (
			!context ||
			!stringList(context.component_names, context.execution_profile?.kind === "baseline") ||
			context.science_conditions.length === 0
		) {
			return "Science review validation requires canonical condition and component context.";
		}
		const statistical = reports.find((report) => report.role === "statistical_interpretation");
		const componentResults = statistical?.structured_findings.component_results;
		if (!isPlainObject(componentResults)) {
			return "Science statistical_interpretation review must provide object-shaped component_results.";
		}
		if (!sameStringSet(Object.keys(componentResults), context.component_names)) {
			return "Science statistical component_results must exactly cover canonical Idea components.";
		}
		for (const condition of context.science_conditions) {
			if (condition.kind === "all_components_reference") {
				continue;
			}
			if (condition.disabled_components.length !== 1) {
				return `Science condition ${condition.id} must disable exactly one component.`;
			}
			const component = condition.disabled_components[0];
			const result = componentResults[component];
			if (!isPlainObject(result) || result.condition_id !== condition.id) {
				return `Statistical result for ${component} must bind to condition ${condition.id}.`;
			}
			if (
				!sameStringSet(result.enabled_components, condition.enabled_components) ||
				!sameStringSet(result.disabled_components, condition.disabled_components)
			) {
				return `Statistical result for ${component} has stale condition component lineage.`;
			}
		}
	}
	return undefined;
}

export function validateExperimentResult(assignment: ExperimentAssignment, result: CanonicalJson): string | undefined {
	if (!isPlainObject(result)) {
		return `Assignment ${assignment.assignment_id} must submit an object-shaped result.`;
	}
	if (assignment.kind === "plan") {
		const plan = result as unknown as ExperimentPlan;
		if (!Array.isArray(plan.work_units)) {
			return `${assignment.child.stage} planner must submit a plan with work units.`;
		}
		const components = assignment.result_validation?.component_names ?? [];
		const profile = assignment.result_validation?.execution_profile;
		return assignment.child.stage === "prepare"
			? validatePreparePlan(plan)
			: assignment.child.stage === "code"
				? validateCodePlan(plan, components, profile)
				: assignment.child.stage === "science"
					? validateSciencePlan(plan, components, profile)
					: "Finalize assignments cannot submit plans.";
	}
	if (assignment.kind === "worker_result") {
		if (result.artifact_role !== "worker_result") {
			return `${assignment.child.stage} worker ${assignment.child.work_unit ?? ""} result must declare artifact_role worker_result.`;
		}
		if (result.status !== undefined && result.status !== "success") {
			return `${assignment.child.stage} worker result status must be success when provided.`;
		}
		return undefined;
	}
	if (assignment.kind === "review") {
		const role = assignment.result_validation?.reviewer_role;
		const report = result as unknown as ExperimentReviewReport;
		if (!role || report.role !== role) {
			return `Reviewer result must contain the JSON field "role": ${JSON.stringify(role)}. Do not use exact_assigned_role or a descriptive role sentence.`;
		}
		if (
			report.artifact_role !== "reviewer_report" ||
			report.reviewer_kind !== "agent" ||
			(report.verdict !== "PASS" && report.verdict !== "FAIL") ||
			report.blocking !== true ||
			!nonEmptyString(report.summary) ||
			!stringList(report.checked_artifacts) ||
			!nonEmptyRecord(report.structured_findings) ||
			!Array.isArray(report.issues) ||
			report.issues.some(
				(issue) =>
					!isPlainObject(issue) ||
					![issue.code, issue.message, issue.required_fix, issue.evidence].every(nonEmptyString),
			) ||
			(report.verdict === "PASS" ? report.issues.length !== 0 : report.issues.length === 0)
		) {
			return `Reviewer ${role} has an invalid reviewer report contract.`;
		}
		if (assignment.child.stage === "science" && role === "statistical_interpretation") {
			const context = assignment.result_validation?.review_context;
			const componentResults = report.structured_findings.component_results;
			if (
				!context ||
				!isPlainObject(componentResults) ||
				!sameStringSet(Object.keys(componentResults), context.component_names)
			) {
				return "Science statistical component_results must exactly cover canonical Idea components.";
			}
			for (const condition of context.science_conditions.filter((item) => item.kind === "component_disabled")) {
				const component = condition.disabled_components[0];
				const value = componentResults[component];
				if (
					!component ||
					!isPlainObject(value) ||
					value.condition_id !== condition.id ||
					!sameStringSet(value.enabled_components, condition.enabled_components) ||
					!sameStringSet(value.disabled_components, condition.disabled_components)
				) {
					return `Statistical result for ${component ?? condition.id} has stale condition lineage.`;
				}
			}
		}
		return undefined;
	}
	if (
		result.artifact_role !== "final_review" ||
		result.verdict !== "PASS" ||
		result.blocking !== true ||
		!stringList(result.checked_artifacts) ||
		!Array.isArray(result.issues) ||
		result.issues.length !== 0 ||
		!isPlainObject(result.summary) ||
		"component_results" in result
	) {
		return "Final reviewer must submit a blocking PASS final_review with checked artifacts, no issues, summary, and no component_results.";
	}
	return undefined;
}

export class ExperimentProtocolStore {
	readonly root: string;
	readonly protocolPath: string;
	readonly eventsPath: string;
	readonly cancellationPath: string;
	readonly assignmentsDir: string;
	readonly submissionsDir: string;
	readonly acceptedDir: string;
	private readonly clock: () => string;
	private readonly faultInjector?: (point: ExperimentProtocolFaultPoint) => void;
	private mutationDepth = 0;

	constructor(
		runDir: string,
		clock: () => string = () => new Date().toISOString(),
		faultInjector?: (point: ExperimentProtocolFaultPoint) => void,
	) {
		this.root = join(runDir, "experiment");
		this.protocolPath = join(this.root, "protocol.json");
		this.eventsPath = join(this.root, "events.jsonl");
		this.cancellationPath = join(this.root, "cancellation.json");
		this.assignmentsDir = join(this.root, "assignments");
		this.submissionsDir = join(this.root, "submissions");
		this.acceptedDir = join(this.root, "accepted");
		this.clock = clock;
		this.faultInjector = faultInjector;
		const lockedMethods = new Set([
			"initialize",
			"read",
			"readAcceptedResult",
			"listAcceptedResults",
			"reconcile",
			"acquireOwner",
			"start",
			"completeStage",
			"finish",
			"registerChild",
			"advanceChild",
			"createAssignment",
			"acceptSubmission",
			"supersedeAssignment",
			"resume",
			"pause",
			"cancel",
		]);
		// biome-ignore lint/correctness/noConstructorReturn: The proxy applies the durable store lock uniformly to public operations.
		return new Proxy(this, {
			get: (target, property, receiver) => {
				const value = Reflect.get(target, property, receiver);
				if (typeof property !== "string" || !lockedMethods.has(property) || typeof value !== "function") {
					return value;
				}
				return (...args: unknown[]) => target.withMutationLock(() => Reflect.apply(value, receiver, args));
			},
		});
	}

	initialize(params: { runId: string; ideaDigest: string; policyDigest: string }): ExperimentProtocolSnapshot {
		mkdirSync(this.root, { recursive: true });
		return this.withMutationLock(() => this.initializeUnlocked(params));
	}

	private initializeUnlocked(params: {
		runId: string;
		ideaDigest: string;
		policyDigest: string;
	}): ExperimentProtocolSnapshot {
		if (existsSync(this.protocolPath) || existsSync(this.eventsPath)) {
			const existing = this.reconcile();
			if (
				existing.run_id !== params.runId ||
				existing.idea_digest !== params.ideaDigest ||
				existing.policy_digest !== params.policyDigest
			) {
				throw new Error("Existing experiment protocol does not match immutable run inputs.");
			}
			return existing;
		}
		for (const path of [this.assignmentsDir, this.submissionsDir, this.acceptedDir]) {
			mkdirSync(path, { recursive: true });
		}
		const now = this.clock();
		const snapshot: ExperimentProtocolSnapshot = {
			schema_version: "xlab.experiment_protocol.v1",
			run_id: params.runId,
			idea_digest: params.ideaDigest,
			policy_digest: params.policyDigest,
			status: "pending",
			completed_stages: [],
			assignments: {},
			children: {},
			last_event_sequence: 0,
			last_event_digest: null,
			created_at: now,
			updated_at: now,
		};
		atomicWriteJson(this.protocolPath, snapshot);
		return snapshot;
	}

	read(): ExperimentProtocolSnapshot {
		return this.withMutationLock(() => this.reconcileUnlocked());
	}

	readAcceptedResult(assignmentId: string): ExperimentAcceptedResult | undefined {
		const snapshot = this.read();
		const assignment = snapshot.assignments[assignmentId];
		if (!assignment?.accepted_result_digest) {
			return undefined;
		}
		const events = this.readEvents();
		let event: ExperimentProtocolEvent | undefined;
		for (let index = events.length - 1; index >= 0; index -= 1) {
			const candidate = events[index];
			if (
				candidate.type === "submission_accepted" &&
				isPlainObject(candidate.data) &&
				candidate.data.assignment_id === assignmentId
			) {
				event = candidate;
				break;
			}
		}
		if (!event || !isPlainObject(event.data)) {
			throw new Error(`Accepted assignment ${assignmentId} is missing its journal record.`);
		}
		const accepted = event.data.accepted;
		const submission = event.data.submission;
		if (!isPlainObject(accepted) || !isPlainObject(submission)) {
			throw new Error(`Accepted assignment ${assignmentId} has an invalid journal record.`);
		}
		const result = {
			accepted: accepted as unknown as AcceptedExperimentSubmission,
			submission: submission as unknown as ExperimentAcceptedResult["submission"],
		};
		if (
			result.accepted.assignment_id !== assignmentId ||
			result.accepted.result_digest !== assignment.accepted_result_digest ||
			canonicalDigest(result.submission.result) !== result.accepted.result_digest
		) {
			throw new Error(`Accepted assignment ${assignmentId} failed result lineage validation.`);
		}
		return result;
	}

	listAcceptedResults(): ExperimentAcceptedResult[] {
		const snapshot = this.read();
		return Object.keys(snapshot.assignments)
			.sort()
			.flatMap((assignmentId) => {
				const result = this.readAcceptedResult(assignmentId);
				return result ? [result] : [];
			});
	}

	reconcile(): ExperimentProtocolSnapshot {
		return this.withMutationLock(() => this.reconcileUnlocked());
	}

	private reconcileUnlocked(): ExperimentProtocolSnapshot {
		const events = this.readEvents();
		let snapshot: ExperimentProtocolSnapshot | undefined;
		let projectionError: unknown;
		try {
			snapshot = this.readProjection();
		} catch (error) {
			projectionError = error;
		}
		if (events.length === 0) {
			if (!snapshot) {
				throw projectionError;
			}
			if (snapshot.last_event_sequence !== 0 || snapshot.last_event_digest !== null) {
				throw new Error("Experiment protocol projection references a missing journal.");
			}
			return this.reconcileCancellation(snapshot);
		}
		const latest = events.at(-1);
		if (!latest) {
			throw new Error("Experiment protocol journal is empty.");
		}
		const journalProjection = this.projectionFromEvent(latest);
		if (!snapshot) {
			atomicWriteJson(this.protocolPath, journalProjection);
			this.materializeRecords(events, journalProjection);
			return this.reconcileCancellation(journalProjection);
		}
		if (snapshot.run_id !== journalProjection.run_id) {
			throw new Error("Experiment protocol projection and journal identify different runs.");
		}
		if (snapshot.last_event_sequence > latest.sequence) {
			throw new Error("Experiment protocol projection is ahead of its journal.");
		}
		if (snapshot.last_event_sequence === latest.sequence) {
			if (
				snapshot.last_event_digest !== latest.digest ||
				canonicalDigest(snapshot) !== canonicalDigest(journalProjection)
			) {
				throw new Error("Experiment protocol projection does not match its journal.");
			}
			this.materializeRecords(events, snapshot);
			return this.reconcileCancellation(snapshot);
		}
		atomicWriteJson(this.protocolPath, journalProjection);
		this.materializeRecords(events, journalProjection);
		return this.reconcileCancellation(journalProjection);
	}

	private reconcileCancellation(snapshot: ExperimentProtocolSnapshot): ExperimentProtocolSnapshot {
		if (!existsSync(this.cancellationPath)) {
			return snapshot;
		}
		const intent = JSON.parse(readFileSync(this.cancellationPath, "utf-8")) as unknown;
		if (
			!isPlainObject(intent) ||
			intent.schema_version !== "xlab.experiment_cancellation.v1" ||
			intent.run_id !== snapshot.run_id ||
			!Number.isInteger(intent.terminal_generation) ||
			(intent.terminal_generation as number) <= 0 ||
			!nonEmptyString(intent.requested_at) ||
			!nonEmptyString(intent.reason)
		) {
			throw new Error("Experiment cancellation intent is malformed or belongs to another run.");
		}
		if (snapshot.terminal_generation !== undefined) {
			if (snapshot.status !== "cancelled" || snapshot.terminal_generation !== intent.terminal_generation) {
				throw new Error("Experiment cancellation intent conflicts with terminal protocol state.");
			}
			return snapshot;
		}
		if (["success", "incomplete", "failed"].includes(snapshot.status)) {
			throw new Error(`Experiment cancellation intent conflicts with terminal status ${snapshot.status}.`);
		}
		const now = intent.requested_at as string;
		const assignments = Object.fromEntries(
			Object.entries(snapshot.assignments).map(([id, assignment]) => [
				id,
				assignment.status === "active" || assignment.status === "submitted"
					? { ...assignment, status: "cancelled" as const, updated_at: now }
					: assignment,
			]),
		);
		const cancelled = this.commit(
			snapshot,
			{
				...snapshot,
				status: "cancelled",
				terminal_generation: intent.terminal_generation as number,
				assignments,
				updated_at: now,
			},
			"cancelled",
			intent as unknown as CanonicalJson,
		);
		this.materializeRecords(this.readEvents(), cancelled);
		return cancelled;
	}

	private materializeRecords(events: ExperimentProtocolEvent[], snapshot: ExperimentProtocolSnapshot): void {
		const ensureRecord = (path: string, value: unknown): void => {
			const bytes = canonicalJsonBytes(value);
			if (existsSync(path) && readFileSync(path).equals(bytes)) {
				return;
			}
			atomicWriteFile(path, bytes);
		};
		for (const assignment of Object.values(snapshot.assignments)) {
			ensureRecord(join(this.assignmentsDir, `${assignment.assignment_id}.json`), assignment);
		}
		for (const event of events) {
			if (event.type !== "submission_accepted" || !isPlainObject(event.data)) {
				continue;
			}
			const assignmentId = event.data.assignment_id;
			if (typeof assignmentId !== "string") {
				throw new Error(`Experiment protocol event ${event.sequence} is missing its assignment ID.`);
			}
			const accepted = event.data.accepted;
			if (!isPlainObject(accepted)) {
				throw new Error(`Experiment protocol event ${event.sequence} is missing its accepted result.`);
			}
			ensureRecord(join(this.acceptedDir, `${assignmentId}.json`), accepted);
			const rawSubmission = event.data.submission;
			if (!isPlainObject(rawSubmission) || typeof rawSubmission.submission_id !== "string") {
				throw new Error(`Experiment protocol event ${event.sequence} is missing its submission.`);
			}
			ensureRecord(join(this.submissionsDir, `${rawSubmission.submission_id}.json`), rawSubmission);
		}
	}

	private readProjection(): ExperimentProtocolSnapshot {
		const snapshot = JSON.parse(readFileSync(this.protocolPath, "utf-8")) as ExperimentProtocolSnapshot;
		this.validateProjection(snapshot);
		return snapshot;
	}

	private validateProjection(snapshot: ExperimentProtocolSnapshot): void {
		if (!isPlainObject(snapshot) || snapshot.schema_version !== "xlab.experiment_protocol.v1") {
			throw new Error("Unsupported experiment protocol snapshot.");
		}
		if (
			typeof snapshot.run_id !== "string" ||
			!snapshot.run_id ||
			ensureDigest(snapshot.idea_digest, "idea_digest") ||
			ensureDigest(snapshot.policy_digest, "policy_digest") ||
			!Number.isInteger(snapshot.last_event_sequence) ||
			snapshot.last_event_sequence < 0 ||
			(snapshot.last_event_digest !== null && ensureDigest(snapshot.last_event_digest, "last_event_digest")) ||
			!isPlainObject(snapshot.assignments) ||
			!isPlainObject(snapshot.children) ||
			!Array.isArray(snapshot.completed_stages) ||
			snapshot.completed_stages.some((stage) => !EXPERIMENT_STAGES.includes(stage)) ||
			new Set(snapshot.completed_stages).size !== snapshot.completed_stages.length ||
			(snapshot.current_stage !== undefined && !EXPERIMENT_STAGES.includes(snapshot.current_stage)) ||
			(snapshot.publication_recovery_reason !== undefined &&
				(typeof snapshot.publication_recovery_reason !== "string" ||
					!snapshot.publication_recovery_reason.trim() ||
					snapshot.status !== "incomplete"))
		) {
			throw new Error("Experiment protocol snapshot is malformed.");
		}
	}

	private projectionFromEvent(event: ExperimentProtocolEvent): ExperimentProtocolSnapshot {
		if (!isPlainObject(event.projection)) {
			throw new Error(`Experiment protocol event ${event.sequence} has an invalid projection.`);
		}
		const embedded = event.projection as unknown as ExperimentProtocolSnapshot;
		if (embedded.last_event_sequence !== event.sequence || embedded.last_event_digest !== null) {
			throw new Error(`Experiment protocol event ${event.sequence} has an inconsistent projection.`);
		}
		const projection = {
			...embedded,
			last_event_digest: event.digest,
		};
		this.validateProjection(projection);
		return projection;
	}

	private readEvents(): ExperimentProtocolEvent[] {
		if (!existsSync(this.eventsPath)) {
			return [];
		}
		const text = readFileSync(this.eventsPath, "utf-8");
		if (!text) {
			return [];
		}
		if (!text.endsWith("\n")) {
			throw new Error("Experiment protocol journal contains a torn final record.");
		}
		const events = text
			.trimEnd()
			.split("\n")
			.map((line) => JSON.parse(line) as ExperimentProtocolEvent);
		let previousDigest: string | null = null;
		for (let index = 0; index < events.length; index += 1) {
			const event = events[index];
			if (!isPlainObject(event)) {
				throw new Error("Experiment protocol journal record is invalid.");
			}
			if (event.schema_version !== "xlab.experiment_event.v1" || event.sequence !== index + 1) {
				throw new Error("Experiment protocol journal sequence is invalid.");
			}
			if (event.previous_digest !== previousDigest) {
				throw new Error("Experiment protocol journal digest chain is invalid.");
			}
			const { digest, ...eventWithoutDigest } = event;
			if (digest !== canonicalDigest(eventWithoutDigest)) {
				throw new Error(`Experiment protocol event ${event.sequence} failed digest validation.`);
			}
			this.projectionFromEvent(event);
			previousDigest = digest;
		}
		return events;
	}

	acquireOwner(ownerId: string, leaseExpiresAt: string): ExperimentProtocolSnapshot {
		const snapshot = this.reconcileUnlocked();
		const now = this.clock();
		if (!ownerId.trim()) {
			throw new Error("Experiment owner ID must be non-empty.");
		}
		if (!Number.isFinite(Date.parse(now)) || !Number.isFinite(Date.parse(leaseExpiresAt)) || leaseExpiresAt <= now) {
			throw new Error("Experiment owner lease must have a valid future expiry.");
		}
		if (snapshot.owner && snapshot.owner.owner_id !== ownerId && snapshot.owner.lease_expires_at > now) {
			throw new Error(`Experiment protocol is owned by ${snapshot.owner.owner_id}.`);
		}
		const currentOwner = snapshot.owner;
		const renewing = currentOwner?.owner_id === ownerId && currentOwner.lease_expires_at > now;
		const generation = renewing ? currentOwner.generation : (currentOwner?.generation ?? 0) + 1;
		const owner: ExperimentOwnerLease = {
			owner_id: ownerId,
			generation,
			acquired_at: renewing ? currentOwner.acquired_at : now,
			lease_expires_at: leaseExpiresAt,
		};
		return this.commit(
			snapshot,
			{ ...snapshot, owner, updated_at: now },
			renewing ? "owner_renewed" : "owner_acquired",
			owner as unknown as CanonicalJson,
		);
	}

	start(stage: ExperimentStage, fence?: ExperimentOwnerFence): ExperimentProtocolSnapshot {
		const snapshot = this.read();
		this.assertOwner(snapshot, fence);
		if (snapshot.terminal_generation !== undefined) {
			throw new Error("Terminal experiment runs cannot be started.");
		}
		const expectedIndex = snapshot.completed_stages.length;
		if (EXPERIMENT_STAGES[expectedIndex] !== stage) {
			throw new Error(`Experiment stage ${stage} is not the next macro stage.`);
		}
		if (snapshot.current_stage === stage && snapshot.status === "running") {
			return snapshot;
		}
		if (snapshot.current_stage !== undefined && snapshot.current_stage !== stage) {
			throw new Error(`Experiment stage ${snapshot.current_stage} is still active.`);
		}
		const now = this.clock();
		return this.commit(
			snapshot,
			{ ...snapshot, status: "running", current_stage: stage, updated_at: now },
			"stage_started",
			{ stage },
		);
	}

	completeStage(stage: ExperimentStage, fence?: ExperimentOwnerFence): ExperimentProtocolSnapshot {
		const snapshot = this.read();
		this.assertOwner(snapshot, fence);
		if (snapshot.current_stage !== stage || snapshot.status !== "running") {
			throw new Error(`Experiment stage ${stage} is not running.`);
		}
		const expectedIndex = snapshot.completed_stages.length;
		if (EXPERIMENT_STAGES[expectedIndex] !== stage) {
			throw new Error(`Experiment stage ${stage} cannot complete out of order.`);
		}
		const completedStages = [...snapshot.completed_stages, stage];
		const now = this.clock();
		const { current_stage: _currentStage, ...withoutCurrentStage } = snapshot;
		return this.commit(
			snapshot,
			{ ...withoutCurrentStage, completed_stages: completedStages, updated_at: now },
			"stage_completed",
			{ stage },
		);
	}

	finish(
		status: Extract<ExperimentProtocolStatus, "success" | "incomplete" | "failed">,
		reason: string,
		fence?: ExperimentOwnerFence,
		publicationRecovery = false,
	): ExperimentProtocolSnapshot {
		const snapshot = this.read();
		this.assertOwner(snapshot, fence);
		if (snapshot.terminal_generation !== undefined) {
			if (snapshot.status === status) {
				return snapshot;
			}
			const markingPublicationIncomplete =
				publicationRecovery &&
				snapshot.status === "success" &&
				status === "incomplete" &&
				snapshot.completed_stages.length === EXPERIMENT_STAGES.length &&
				Boolean(reason.trim());
			const recoveringPublication =
				publicationRecovery &&
				snapshot.status === "incomplete" &&
				status === "success" &&
				snapshot.completed_stages.length === EXPERIMENT_STAGES.length &&
				Boolean(snapshot.publication_recovery_reason?.trim());
			if (!markingPublicationIncomplete && !recoveringPublication) {
				throw new Error(`Experiment run is already terminal with status ${snapshot.status}.`);
			}
			const now = this.clock();
			const next = markingPublicationIncomplete
				? { ...snapshot, status, publication_recovery_reason: reason, updated_at: now }
				: (() => {
						const { publication_recovery_reason: _reason, ...withoutRecoveryReason } = snapshot;
						return { ...withoutRecoveryReason, status, updated_at: now };
					})();
			return this.commit(
				snapshot,
				next,
				markingPublicationIncomplete ? "publication_incomplete" : "publication_recovered",
				{ status, reason, terminal_generation: snapshot.terminal_generation },
			);
		}
		if (status === "success" && snapshot.completed_stages.length !== EXPERIMENT_STAGES.length) {
			throw new Error("Successful experiment runs must complete every macro stage.");
		}
		const terminalGeneration = snapshot.owner?.generation ?? 0;
		const now = this.clock();
		const { current_stage: _currentStage, ...withoutCurrentStage } = snapshot;
		return this.commit(
			snapshot,
			{ ...withoutCurrentStage, status, terminal_generation: terminalGeneration, updated_at: now },
			"run_finished",
			{ status, reason, terminal_generation: terminalGeneration },
		);
	}

	registerChild(child: ExperimentChildIdentity, fence?: ExperimentOwnerFence): ExperimentProtocolSnapshot {
		const snapshot = this.read();
		this.assertOwner(snapshot, fence);
		const existing = snapshot.children[child.child_id];
		if (existing && canonicalDigest(existing) !== canonicalDigest(child)) {
			throw new Error(`Conflicting child identity ${child.child_id}.`);
		}
		if (existing) {
			return snapshot;
		}
		return this.commit(
			snapshot,
			{ ...snapshot, children: { ...snapshot.children, [child.child_id]: child }, updated_at: this.clock() },
			"child_registered",
			child as unknown as CanonicalJson,
		);
	}

	advanceChild(child: ExperimentChildIdentity, fence?: ExperimentOwnerFence): ExperimentProtocolSnapshot {
		const snapshot = this.read();
		this.assertOwner(snapshot, fence);
		const current = snapshot.children[child.child_id];
		if (!current) {
			throw new Error(`Unknown child identity ${child.child_id}.`);
		}
		const stableCurrent = {
			...current,
			execution_attempt: child.execution_attempt,
			generation: child.generation,
			review_round: child.review_round,
		};
		if (canonicalDigest(stableCurrent) !== canonicalDigest(child)) {
			throw new Error(`Child ${child.child_id} cannot change its durable session identity or assignment role.`);
		}
		if (
			child.execution_attempt < current.execution_attempt ||
			child.generation < current.generation ||
			child.review_round < current.review_round
		) {
			throw new Error(`Child identity cannot move backwards: ${child.child_id}.`);
		}
		if (canonicalDigest(current) === canonicalDigest(child)) {
			return snapshot;
		}
		return this.commit(
			snapshot,
			{ ...snapshot, children: { ...snapshot.children, [child.child_id]: child }, updated_at: this.clock() },
			"child_advanced",
			child as unknown as CanonicalJson,
		);
	}

	createAssignment(
		assignment: Omit<ExperimentAssignment, "assignment_token_digest" | "status" | "created_at" | "updated_at">,
		token: string,
		fence?: ExperimentOwnerFence,
	): ExperimentAssignment {
		const snapshot = this.read();
		this.assertOwner(snapshot, fence);
		if (snapshot.terminal_generation !== undefined) {
			throw new Error("Cannot create an assignment for a terminal experiment run.");
		}
		for (const [field, value] of [
			["input_digest", assignment.input_digest],
			["target_digest", assignment.target_digest],
			["policy_digest", assignment.policy_digest],
		] as const) {
			const error = ensureDigest(value, field);
			if (error) {
				throw new Error(error);
			}
		}
		if (assignment.policy_digest !== snapshot.policy_digest) {
			throw new Error("Assignment policy digest does not match the protocol snapshot.");
		}
		const registeredChild = snapshot.children[assignment.child.child_id];
		if (!registeredChild || canonicalDigest(registeredChild) !== canonicalDigest(assignment.child)) {
			throw new Error(`Assignment child ${assignment.child.child_id} does not match its durable identity.`);
		}
		if (assignment.child.parent_run_id !== snapshot.run_id) {
			throw new Error("Assignment child belongs to a different experiment run.");
		}
		if (snapshot.current_stage !== undefined && assignment.child.stage !== snapshot.current_stage) {
			throw new Error(`Assignment stage ${assignment.child.stage} is not the active macro stage.`);
		}
		const assignmentTokenDigest = tokenDigest(token).toString("hex");
		const existing = snapshot.assignments[assignment.assignment_id];
		if (existing) {
			const immutableExisting: typeof assignment = {
				schema_version: existing.schema_version,
				assignment_id: existing.assignment_id,
				child: existing.child,
				kind: existing.kind,
				input_digest: existing.input_digest,
				target_digest: existing.target_digest,
				policy_digest: existing.policy_digest,
				read_scope: existing.read_scope,
				write_scope: existing.write_scope,
				...(existing.result_validation === undefined ? {} : { result_validation: existing.result_validation }),
			};
			if (
				existing.assignment_token_digest !== assignmentTokenDigest ||
				canonicalDigest(immutableExisting) !== canonicalDigest(assignment)
			) {
				throw new Error(`Conflicting assignment ${assignment.assignment_id}.`);
			}
			return existing;
		}
		const now = this.clock();
		const record: ExperimentAssignment = {
			...assignment,
			assignment_token_digest: assignmentTokenDigest,
			status: "active",
			created_at: now,
			updated_at: now,
		};
		atomicWriteJson(join(this.assignmentsDir, `${record.assignment_id}.json`), record);
		this.injectFault("after_assignment_record");
		this.commit(
			snapshot,
			{
				...snapshot,
				assignments: { ...snapshot.assignments, [record.assignment_id]: record },
				updated_at: now,
			},
			"assignment_created",
			{ assignment_id: record.assignment_id, child_id: record.child.child_id },
		);
		return record;
	}

	acceptSubmission(submission: ExperimentSubmission, fence?: ExperimentOwnerFence): SubmissionAcceptance {
		const snapshot = this.read();
		try {
			this.assertOwner(snapshot, fence);
		} catch (error) {
			return { ok: false, error: error instanceof Error ? error.message : String(error) };
		}
		const assignment = snapshot.assignments[submission.assignment_id];
		if (!assignment) {
			return { ok: false, error: `Unknown assignment ${submission.assignment_id}.` };
		}
		const validationError = validateSubmission(assignment, submission, snapshot.terminal_generation);
		if (validationError) {
			return { ok: false, error: validationError };
		}
		let resultDigest: string;
		try {
			resultDigest = canonicalDigest(submission.result);
		} catch (error) {
			return { ok: false, error: error instanceof Error ? error.message : String(error) };
		}
		const resultError = validateExperimentResult(assignment, submission.result);
		if (resultError) {
			return { ok: false, error: resultError };
		}
		if (assignment.accepted_result_digest) {
			if (assignment.accepted_result_digest !== resultDigest) {
				return { ok: false, error: `Conflicting replay for assignment ${assignment.assignment_id}.` };
			}
			const accepted = JSON.parse(
				readFileSync(join(this.acceptedDir, `${assignment.assignment_id}.json`), "utf-8"),
			) as AcceptedExperimentSubmission;
			return { ok: true, disposition: "idempotent", accepted };
		}
		const outputDigests: Record<string, string> = {};
		try {
			for (const pathValue of submission.output_paths) {
				const scope = assignment.write_scope.find((allowed) => {
					try {
						const base = realpathSync(allowed);
						const candidate = realpathSync(pathValue);
						return isContained(base, candidate);
					} catch {
						return false;
					}
				});
				if (!scope || !statSync(pathValue).isFile()) {
					return { ok: false, error: `Submission output is outside its write scope: ${pathValue}` };
				}
				outputDigests[pathValue] = sha256(readFileSync(pathValue));
			}
		} catch (error) {
			return { ok: false, error: error instanceof Error ? error.message : String(error) };
		}
		const accepted: AcceptedExperimentSubmission = {
			schema_version: "xlab.experiment_accepted_submission.v1",
			submission_id: submission.submission_id,
			assignment_id: assignment.assignment_id,
			result_digest: resultDigest,
			output_digests: outputDigests,
			accepted_at: this.clock(),
		};
		const redactedSubmission = {
			...submission,
			assignment_token: "[redacted]",
		};
		atomicWriteJson(join(this.submissionsDir, `${submission.submission_id}.json`), redactedSubmission);
		this.injectFault("after_submission_record");
		atomicWriteJson(join(this.acceptedDir, `${assignment.assignment_id}.json`), accepted);
		this.injectFault("after_accepted_record");
		const nextAssignment: ExperimentAssignment = {
			...assignment,
			status: "accepted",
			accepted_result_digest: resultDigest,
			updated_at: accepted.accepted_at,
		};
		atomicWriteJson(join(this.assignmentsDir, `${assignment.assignment_id}.json`), nextAssignment);
		this.injectFault("after_assignment_acceptance_record");
		this.commit(
			snapshot,
			{
				...snapshot,
				assignments: { ...snapshot.assignments, [assignment.assignment_id]: nextAssignment },
				updated_at: accepted.accepted_at,
			},
			"submission_accepted",
			{
				assignment_id: assignment.assignment_id,
				result_digest: resultDigest,
				accepted: accepted as unknown as CanonicalJson,
				submission: redactedSubmission as unknown as CanonicalJson,
			},
		);
		return { ok: true, disposition: "accepted", accepted };
	}

	supersedeAssignment(assignmentId: string, fence?: ExperimentOwnerFence): ExperimentProtocolSnapshot {
		return this.updateAssignmentStatus(assignmentId, "superseded", "assignment_superseded", fence);
	}

	resume(fence?: ExperimentOwnerFence): ExperimentProtocolSnapshot {
		const snapshot = this.read();
		this.assertOwner(snapshot, fence);
		if (snapshot.status !== "paused") {
			return snapshot;
		}
		return this.commit(snapshot, { ...snapshot, status: "running", updated_at: this.clock() }, "resumed", {});
	}

	pause(reason: string, fence?: ExperimentOwnerFence): ExperimentProtocolSnapshot {
		const snapshot = this.read();
		this.assertOwner(snapshot, fence);
		if (snapshot.terminal_generation !== undefined) {
			throw new Error("Terminal experiment runs cannot be paused.");
		}
		return this.commit(snapshot, { ...snapshot, status: "paused", updated_at: this.clock() }, "paused", { reason });
	}

	cancel(reason: string): ExperimentProtocolSnapshot {
		const snapshot = this.read();
		if (snapshot.status === "cancelled" && snapshot.terminal_generation !== undefined) {
			return snapshot;
		}
		if (["success", "incomplete", "failed"].includes(snapshot.status)) {
			throw new Error(`Cannot cancel terminal experiment run with status ${snapshot.status}.`);
		}
		const terminalGeneration = (snapshot.owner?.generation ?? snapshot.terminal_generation ?? 0) + 1;
		const now = this.clock();
		const intent: ExperimentCancellationIntent = {
			schema_version: "xlab.experiment_cancellation.v1",
			run_id: snapshot.run_id,
			terminal_generation: terminalGeneration,
			requested_at: now,
			reason,
		};
		atomicWriteJson(this.cancellationPath, intent);
		this.injectFault("after_cancellation_intent");
		const assignments = Object.fromEntries(
			Object.entries(snapshot.assignments).map(([id, assignment]) => {
				const nextAssignment =
					assignment.status === "active" || assignment.status === "submitted"
						? { ...assignment, status: "cancelled" as const, updated_at: now }
						: assignment;
				if (nextAssignment !== assignment) {
					atomicWriteJson(join(this.assignmentsDir, `${id}.json`), nextAssignment);
				}
				return [id, nextAssignment];
			}),
		);
		this.injectFault("after_cancellation_assignments");
		return this.commit(
			snapshot,
			{ ...snapshot, status: "cancelled", terminal_generation: terminalGeneration, assignments, updated_at: now },
			"cancelled",
			intent as unknown as CanonicalJson,
		);
	}

	private updateAssignmentStatus(
		assignmentId: string,
		status: ExperimentAssignmentStatus,
		eventType: string,
		fence?: ExperimentOwnerFence,
	): ExperimentProtocolSnapshot {
		const snapshot = this.read();
		this.assertOwner(snapshot, fence);
		const assignment = snapshot.assignments[assignmentId];
		if (!assignment) {
			throw new Error(`Unknown assignment ${assignmentId}.`);
		}
		if (assignment.status === status) {
			return snapshot;
		}
		if (assignment.status === "accepted") {
			throw new Error(`Accepted assignment ${assignmentId} cannot be ${status}.`);
		}
		const nextAssignment = { ...assignment, status, updated_at: this.clock() };
		atomicWriteJson(join(this.assignmentsDir, `${assignmentId}.json`), nextAssignment);
		return this.commit(
			snapshot,
			{
				...snapshot,
				assignments: { ...snapshot.assignments, [assignmentId]: nextAssignment },
				updated_at: nextAssignment.updated_at,
			},
			eventType,
			{ assignment_id: assignmentId },
		);
	}

	private withMutationLock<T>(operation: () => T): T {
		if (this.mutationDepth > 0) {
			return operation();
		}
		mkdirSync(this.root, { recursive: true });
		let release: () => void;
		const deadline = Date.now() + 10_000;
		while (true) {
			try {
				release = lockfile.lockSync(this.root, { realpath: false });
				break;
			} catch (error) {
				if (
					!(error instanceof Error) ||
					(error as NodeJS.ErrnoException).code !== "ELOCKED" ||
					Date.now() >= deadline
				) {
					throw error;
				}
				Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10);
			}
		}
		this.mutationDepth += 1;
		try {
			return operation();
		} finally {
			this.mutationDepth -= 1;
			release();
		}
	}

	private assertOwner(snapshot: ExperimentProtocolSnapshot, fence?: ExperimentOwnerFence): void {
		if (!fence) {
			return;
		}
		const owner = snapshot.owner;
		if (!owner || owner.owner_id !== fence.owner_id || owner.generation !== fence.generation) {
			throw new Error("Experiment coordinator ownership fence is stale.");
		}
		if (owner.lease_expires_at <= this.clock()) {
			throw new Error("Experiment coordinator ownership lease has expired.");
		}
	}

	private injectFault(point: ExperimentProtocolFaultPoint): void {
		this.faultInjector?.(point);
	}

	private commit(
		previous: ExperimentProtocolSnapshot,
		next: ExperimentProtocolSnapshot,
		type: string,
		data: CanonicalJson,
	): ExperimentProtocolSnapshot {
		const projection = {
			...next,
			last_event_sequence: previous.last_event_sequence + 1,
			last_event_digest: null,
		};
		const eventWithoutDigest = {
			schema_version: "xlab.experiment_event.v1" as const,
			sequence: projection.last_event_sequence,
			type,
			timestamp: this.clock(),
			owner_generation: next.owner?.generation ?? previous.owner?.generation ?? 0,
			previous_digest: previous.last_event_digest,
			data,
			projection: projection as unknown as CanonicalJson,
		};
		const event: ExperimentProtocolEvent = {
			...eventWithoutDigest,
			digest: canonicalDigest(eventWithoutDigest),
		};
		const committed = {
			...projection,
			last_event_digest: event.digest,
		};
		appendDurableJsonLine(this.eventsPath, event);
		this.injectFault("after_journal_append");
		atomicWriteJson(this.protocolPath, committed);
		this.injectFault("after_projection_write");
		return committed;
	}
}
