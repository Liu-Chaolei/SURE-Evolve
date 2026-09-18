import {
	acceptedPlan,
	type ExperimentAssignmentSpec,
	type ExperimentCoordinator,
	type ExperimentReviewMatrixResult,
	type ExperimentStageDriver,
	type ExperimentWorkUnitDriver,
} from "./coordinator.ts";
import type { ExperimentFinalizationProvenance } from "./materializer.ts";
import {
	type CanonicalJson,
	canonicalDigest,
	canonicalJson,
	EXPERIMENT_STAGES,
	type ExperimentAcceptedResult,
	type ExperimentExecutionProfile,
	type ExperimentPlan,
	type ExperimentReviewReport,
	type ExperimentReviewValidationContext,
	type ExperimentStage,
	type ExperimentWorkUnit,
	SCIENCE_REVIEW_ROLES,
	validateCodePlan,
	validateExperimentResult,
	validatePreparePlan,
	validateReviewMatrix,
	validateSciencePlan,
} from "./protocol.ts";

export interface CanonicalExperimentIdeaComponent {
	component: string;
	explanation: string;
	index: string;
}

export interface ExperimentDriverPublication {
	publishPlan(params: {
		stage: "prepare" | "code" | "science";
		assignmentId: string;
		plan: ExperimentPlan;
	}): void | Promise<void>;
	publishWorker(params: {
		stage: "prepare" | "code" | "science";
		workUnit: ExperimentWorkUnit;
		accepted: ExperimentAcceptedResult;
	}): void | Promise<void>;
	publishReview(params: {
		stage: "code" | "science";
		workUnit: ExperimentWorkUnit;
		reviewRound: number;
		assignmentIds: string[];
		validationContext?: ExperimentReviewValidationContext;
		result: ExperimentReviewMatrixResult;
	}): void | Promise<void>;
	publishFinal(params: {
		accepted: ExperimentAcceptedResult;
		componentDefinitions: CanonicalExperimentIdeaComponent[];
		componentResults: Record<string, CanonicalJson>;
		summary: Record<string, CanonicalJson>;
		provenance: ExperimentFinalizationProvenance;
	}): void | Promise<void>;
}

export interface ExperimentDriverOptions {
	executionProfile?: ExperimentExecutionProfile;
	planTemplates?: Partial<Record<"prepare" | "code" | "science", ExperimentPlan>>;
	idea: Record<string, unknown>;
	ideaDigest: string;
	policyDigest: string;
	readScope: string[];
	projectWriteScope: string[];
	maxReviewRounds: number;
	publication: ExperimentDriverPublication;
}

interface FinalReviewResult {
	artifact_role: "final_review";
	verdict: "PASS";
	blocking: true;
	summary: Record<string, CanonicalJson>;
	checked_artifacts: string[];
	issues: [];
}

function objectResult(accepted: ExperimentAcceptedResult, label: string): Record<string, unknown> {
	const result = accepted.submission.result;
	if (typeof result !== "object" || result === null || Array.isArray(result)) {
		throw new Error(`${label} must submit an object-shaped result.`);
	}
	return result;
}

function nonEmptyString(value: unknown): value is string {
	return typeof value === "string" && value.trim().length > 0;
}

function stringList(value: unknown): value is string[] {
	return Array.isArray(value) && value.every(nonEmptyString);
}

function canonicalComponents(
	idea: Record<string, unknown>,
	profile?: ExperimentExecutionProfile,
): CanonicalExperimentIdeaComponent[] {
	if (profile?.kind === "baseline") {
		if (
			idea.schema_version !== "xlab.experiment_baseline.v1" ||
			!Array.isArray(idea.components) ||
			idea.components.length !== 0
		) {
			throw new Error("Baseline profile requires an explicit baseline input with no scientific components.");
		}
		return [];
	}
	if (
		idea.schema_version !== "xlab.experiment_idea.v1" ||
		idea.status !== "success" ||
		!Array.isArray(idea.blockers) ||
		idea.blockers.length !== 0 ||
		!Array.isArray(idea.components) ||
		idea.components.length === 0
	) {
		throw new Error(
			"Experiment drivers require a successful canonical Idea with no blockers and non-empty components.",
		);
	}
	const components = idea.components.map((value, offset) => {
		if (typeof value !== "object" || value === null || Array.isArray(value)) {
			throw new Error(`Canonical Idea component ${offset + 1} must be an object.`);
		}
		const item = value as Record<string, unknown>;
		if (
			!nonEmptyString(item.component) ||
			typeof item.explanation !== "string" ||
			item.index !== String(offset + 1)
		) {
			throw new Error(`Canonical Idea component ${offset + 1} has an invalid component, explanation, or index.`);
		}
		return {
			component: item.component.trim(),
			explanation: item.explanation,
			index: item.index,
		};
	});
	if (new Set(components.map((item) => item.component)).size !== components.length) {
		throw new Error("Canonical Idea component names must be unique.");
	}
	return components;
}

function requireDigest(value: string, label: string): void {
	if (!/^[a-f0-9]{64}$/.test(value)) {
		throw new Error(`${label} must be a lowercase SHA-256 digest.`);
	}
}

function stablePart(value: string): string {
	const readable =
		value
			.trim()
			.replace(/[^A-Za-z0-9_.-]+/g, "-")
			.replace(/^-+|-+$/g, "")
			.slice(0, 48) || "unit";
	return `${readable}-${canonicalDigest(value).slice(0, 16)}`;
}

function plannerSpec(
	stage: "prepare" | "code" | "science",
	options: ExperimentDriverOptions,
	components: CanonicalExperimentIdeaComponent[],
): ExperimentAssignmentSpec {
	const target = {
		stage,
		idea_digest: options.ideaDigest,
		policy_digest: options.policyDigest,
		components,
	};
	return {
		assignmentId: `assignment-planner-${stage}`,
		childId: `planner-${stage}`,
		role: "planner",
		stage,
		kind: "plan",
		executionAttempt: 1,
		generation: 1,
		reviewRound: 0,
		inputDigest: options.ideaDigest,
		targetDigest: canonicalDigest(target),
		readScope: [...options.readScope],
		writeScope: [],
		resultValidation: {
			component_names: components.map((item) => item.component),
			...(options.executionProfile ? { execution_profile: options.executionProfile } : {}),
		},
		prompt: [
			`Produce the canonical ${stage} experiment plan and submit it with submit_plan.`,
			"The plan must satisfy the native protocol validator exactly; do not use mock, quick, pilot, dry-run, or timeout-as-success shortcuts.",
			`Canonical Idea components, in order: ${JSON.stringify(components)}.`,
			`Immutable target: ${JSON.stringify(canonicalJson(target))}.`,
			...(options.planTemplates?.[stage]
				? [
						`Use this executor-bound plan. Preserve command bindings, identifiers, dependencies, and evidence contracts: ${JSON.stringify(options.planTemplates[stage])}`,
					]
				: []),
		].join("\n\n"),
	};
}

function validatePlanShape(plan: ExperimentPlan, stage: "prepare" | "code" | "science"): void {
	if (
		typeof plan !== "object" ||
		plan === null ||
		plan.stage !== stage ||
		!Array.isArray(plan.work_units) ||
		plan.work_units.some(
			(unit) => typeof unit !== "object" || unit === null || Array.isArray(unit) || !nonEmptyString(unit.id),
		)
	) {
		throw new Error(`${stage} planner must submit a plan with object-shaped, named work units.`);
	}
}

function validateWorkerResult(accepted: ExperimentAcceptedResult, stage: ExperimentStage, workUnit: string): void {
	const result = objectResult(accepted, `${stage} worker ${workUnit}`);
	if (result.artifact_role !== "worker_result") {
		throw new Error(`${stage} worker ${workUnit} result must declare artifact_role worker_result.`);
	}
	if (result.status !== undefined && result.status !== "success") {
		throw new Error(`${stage} worker ${workUnit} result status must be success when provided.`);
	}
}

function workerSpec(params: {
	stage: "prepare" | "code" | "science";
	workUnit: ExperimentWorkUnit;
	planDigest: string;
	readScope: string[];
	writeScope: string[];
	reviewRound?: number;
	previousResultDigest?: string;
}): ExperimentAssignmentSpec {
	const round = params.reviewRound ?? 0;
	const part = stablePart(params.workUnit.id);
	const childId = `worker-${params.stage}-${part}`;
	const targetDigest = canonicalDigest(params.workUnit);
	return {
		assignmentId: `assignment-${childId}-r${round}`,
		childId,
		role: "worker",
		stage: params.stage,
		workUnit: params.workUnit.id,
		kind: "worker_result",
		executionAttempt: round + 1,
		generation: round + 1,
		reviewRound: round,
		inputDigest:
			params.previousResultDigest ??
			canonicalDigest({ plan_digest: params.planDigest, work_unit: params.workUnit.id }),
		targetDigest,
		readScope: [...params.readScope],
		writeScope: [...params.writeScope],
		prompt: [
			`Execute ${params.stage} work unit ${params.workUnit.id} and submit an object with artifact_role "worker_result" using submit_worker_result.`,
			"Complete the exact validated work-unit contract and report only evidence from the permitted scopes.",
			`Work-unit contract: ${JSON.stringify(canonicalJson(params.workUnit), null, 2)}.`,
		].join("\n\n"),
	};
}

function reviewContext(
	plan: ExperimentPlan,
	componentNames: string[],
	profile?: ExperimentExecutionProfile,
): ExperimentReviewValidationContext {
	return {
		...(profile ? { execution_profile: profile } : {}),
		component_names: componentNames,
		science_conditions: (plan.stage === "science" ? plan.work_units : []).map((unit) => ({
			id: unit.id,
			enabled_components: unit.enabled_components ?? [],
			disabled_components: unit.disabled_components ?? [],
			kind: unit.kind as "all_components_reference" | "component_disabled",
		})),
	};
}

async function runPlanner(
	coordinator: ExperimentCoordinator,
	stage: "prepare" | "code" | "science",
	options: ExperimentDriverOptions,
	components: CanonicalExperimentIdeaComponent[],
): Promise<{ plan: ExperimentPlan; planDigest: string }> {
	const spec = plannerSpec(stage, options, components);
	const accepted = await coordinator.executeAssignment(spec);
	const plan = acceptedPlan(accepted);
	validatePlanShape(plan, stage);
	const componentNames = components.map((item) => item.component);
	const validationError =
		stage === "prepare"
			? validatePreparePlan(plan)
			: stage === "code"
				? validateCodePlan(plan, componentNames, options.executionProfile)
				: validateSciencePlan(plan, componentNames, options.executionProfile);
	if (validationError) throw new Error(validationError);
	await options.publication.publishPlan({ stage, assignmentId: accepted.accepted.assignment_id, plan });
	return { plan, planDigest: accepted.accepted.result_digest };
}

async function runPrepare(
	coordinator: ExperimentCoordinator,
	options: ExperimentDriverOptions,
	components: CanonicalExperimentIdeaComponent[],
): Promise<void> {
	const { plan, planDigest } = await runPlanner(coordinator, "prepare", options, components);
	const drivers: ExperimentWorkUnitDriver[] = plan.work_units.map((workUnit) => ({
		workUnit,
		run: async (activeCoordinator) => {
			const accepted = await activeCoordinator.executeAssignment(
				workerSpec({
					stage: "prepare",
					workUnit,
					planDigest,
					readScope: options.readScope,
					writeScope: options.projectWriteScope,
				}),
			);
			validateWorkerResult(accepted, "prepare", workUnit.id);
			await options.publication.publishWorker({ stage: "prepare", workUnit, accepted });
		},
	}));
	await coordinator.runPlanWorkUnits(plan, drivers);
}

async function runCode(
	coordinator: ExperimentCoordinator,
	options: ExperimentDriverOptions,
	components: CanonicalExperimentIdeaComponent[],
): Promise<void> {
	const { plan, planDigest } = await runPlanner(coordinator, "code", options, components);
	const drivers: ExperimentWorkUnitDriver[] = plan.work_units.map((workUnit) => ({
		workUnit,
		run: async (activeCoordinator) => {
			const initial = workerSpec({
				stage: "code",
				workUnit,
				planDigest,
				readScope: options.readScope,
				writeScope: options.projectWriteScope,
			});
			const reviewed = await activeCoordinator.runReviewedWorker({
				worker: initial,
				maxReviewRounds: options.maxReviewRounds,
				review: (accepted) => ({
					validationContext: reviewContext(
						plan,
						components.map((item) => item.component),
						options.executionProfile,
					),
					stage: "code",
					workUnit: workUnit.id,
					inputDigest: canonicalDigest({
						plan_digest: planDigest,
						worker_result_digest: accepted.accepted.result_digest,
					}),
					readScope: [...options.readScope, ...options.projectWriteScope],
					prompt: (role) =>
						[
							`Act as the blocking ${role} reviewer for code work unit ${workUnit.id}.`,
							"Submit the exact native reviewer_report shape with submit_review. PASS must have no issues; FAIL must have typed blocking issues.",
							`Worker result digest: ${accepted.accepted.result_digest}.`,
						].join("\n\n"),
				}),
				repair: (previous, _issues, reviewRound) =>
					workerSpec({
						stage: "code",
						workUnit,
						planDigest,
						readScope: options.readScope,
						writeScope: options.projectWriteScope,
						reviewRound,
						previousResultDigest: previous.accepted.result_digest,
					}),
			});
			validateWorkerResult(reviewed.worker, "code", workUnit.id);
			await options.publication.publishWorker({ stage: "code", workUnit, accepted: reviewed.worker });
			await options.publication.publishReview({
				stage: "code",
				validationContext: reviewContext(
					plan,
					components.map((item) => item.component),
					options.executionProfile,
				),
				workUnit,
				reviewRound: reviewed.reviewRound,
				assignmentIds: reviewed.review.assignmentIds,
				result: reviewed.review,
			});
		},
	}));
	await coordinator.runPlanWorkUnits(plan, drivers);
}

interface ScienceAuthority {
	componentResults: Record<string, CanonicalJson>;
	review: ExperimentReviewMatrixResult;
	provenance: ExperimentFinalizationProvenance["science"];
}

function scienceComponentResults(
	review: ExperimentReviewMatrixResult,
	componentNames: string[],
): Record<string, CanonicalJson> {
	if (review.blockingIssues.length > 0) {
		throw new Error("Finalize requires a complete current PASS science review matrix.");
	}
	const report = review.reports.find((item) => item.role === "statistical_interpretation");
	const results = report?.structured_findings.component_results;
	if (
		typeof results !== "object" ||
		results === null ||
		Array.isArray(results) ||
		Object.keys(results).length !== componentNames.length ||
		componentNames.some((name) => !(name in results))
	) {
		throw new Error("Current science review matrix is missing canonical component_results.");
	}
	return canonicalJson(results) as Record<string, CanonicalJson>;
}

async function runScience(
	coordinator: ExperimentCoordinator,
	options: ExperimentDriverOptions,
	components: CanonicalExperimentIdeaComponent[],
): Promise<ScienceAuthority> {
	const { plan, planDigest } = await runPlanner(coordinator, "science", options, components);
	const acceptedWorkers: ExperimentAcceptedResult[] = [];
	const drivers: ExperimentWorkUnitDriver[] = plan.work_units.map((workUnit) => ({
		workUnit,
		run: async (activeCoordinator) => {
			const accepted = await activeCoordinator.executeAssignment(
				workerSpec({
					stage: "science",
					workUnit,
					planDigest,
					readScope: options.readScope,
					writeScope: options.projectWriteScope,
				}),
			);
			validateWorkerResult(accepted, "science", workUnit.id);
			acceptedWorkers.push(accepted);
			await options.publication.publishWorker({ stage: "science", workUnit, accepted });
		},
	}));
	await coordinator.runPlanWorkUnits(plan, drivers);
	const cohort = plan.work_units.map((workUnit) => {
		const accepted = acceptedWorkers.find((item) => item.submission.work_unit === workUnit.id);
		if (!accepted) throw new Error(`Science cohort is missing accepted condition ${workUnit.id}.`);
		return { work_unit: workUnit.id, result_digest: accepted.accepted.result_digest };
	});
	const validationContext = reviewContext(
		plan,
		components.map((item) => item.component),
		options.executionProfile,
	);
	const inputDigest = canonicalDigest({ plan_digest: planDigest, cohort });
	const targetDigest = canonicalDigest({ plan_digest: planDigest, conditions: plan.work_units, cohort });
	const workUnit: ExperimentWorkUnit = { id: "science-cohort" };
	const review = await coordinator.runReviewMatrix({
		stage: "science",
		workUnit: workUnit.id,
		reviewRound: 1,
		inputDigest,
		targetDigest,
		readScope: [...options.readScope, ...options.projectWriteScope],
		validationContext,
		prompt: (role) =>
			[
				`Act as the blocking ${role} reviewer for the complete science condition cohort.`,
				"Review every accepted condition together and submit the exact native reviewer_report shape with submit_review.",
				`Complete cohort: ${JSON.stringify(canonicalJson(cohort))}.`,
			].join("\n\n"),
	});
	if (review.blockingIssues.length > 0) {
		throw new Error("Complete science review matrix did not pass.");
	}
	await options.publication.publishReview({
		stage: "science",
		workUnit,
		reviewRound: review.reviewRound,
		assignmentIds: review.assignmentIds,
		validationContext,
		result: review,
	});
	return {
		review,
		componentResults: scienceComponentResults(review, validationContext.component_names),
		provenance: {
			plan: { assignment_id: "assignment-planner-science", result_digest: planDigest },
			cohort: cohort.map((item) => {
				const accepted = acceptedWorkers.find((value) => value.submission.work_unit === item.work_unit)!;
				return {
					assignment_id: accepted.accepted.assignment_id,
					work_unit: item.work_unit,
					result_digest: item.result_digest,
				};
			}),
			reviews: review.reports.map((report, index) => ({
				assignment_id: review.assignmentIds[index],
				role: report.role,
				result_digest: canonicalDigest(report),
			})),
			input_digest: review.inputDigest,
			target_digest: review.targetDigest,
		},
	};
}

function finalReviewResult(accepted: ExperimentAcceptedResult): FinalReviewResult {
	const result = objectResult(accepted, "Final reviewer");
	if (
		result.artifact_role !== "final_review" ||
		result.verdict !== "PASS" ||
		result.blocking !== true ||
		!stringList(result.checked_artifacts) ||
		!Array.isArray(result.issues) ||
		result.issues.length !== 0 ||
		typeof result.summary !== "object" ||
		result.summary === null ||
		Array.isArray(result.summary)
	) {
		throw new Error(
			"Final reviewer must submit a blocking PASS final_review with checked artifacts, no issues, and summary.",
		);
	}
	if ("component_results" in result) {
		throw new Error(
			"Final reviewer must not supply component_results; accepted science statistical_interpretation reviews are authoritative.",
		);
	}
	return canonicalJson(result) as unknown as FinalReviewResult;
}

async function currentScienceAuthority(
	coordinator: ExperimentCoordinator,
	components: CanonicalExperimentIdeaComponent[],
	profile?: ExperimentExecutionProfile,
): Promise<ScienceAuthority> {
	const snapshot = coordinator.read();
	if (!snapshot.assignments["assignment-planner-science"]) {
		throw new Error("Finalize requires the current complete PASS science review matrix.");
	}
	const acceptedResults = new Map(
		coordinator.listAcceptedResults().map((item) => [item.accepted.assignment_id, item]),
	);
	const requireAccepted = (assignmentId: string): ExperimentAcceptedResult => {
		const assignment = snapshot.assignments[assignmentId];
		const accepted = acceptedResults.get(assignmentId);
		if (
			!assignment ||
			assignment.status !== "accepted" ||
			!accepted ||
			assignment.accepted_result_digest !== accepted.accepted.result_digest
		) {
			throw new Error(`Current science authority is missing accepted assignment ${assignmentId}.`);
		}
		const validationError = validateExperimentResult(assignment, accepted.submission.result);
		if (validationError) {
			throw new Error(`Current science authority assignment ${assignmentId} is invalid: ${validationError}`);
		}
		return accepted;
	};

	const planner = requireAccepted("assignment-planner-science");
	const plan = acceptedPlan(planner);
	validatePlanShape(plan, "science");
	const componentNames = components.map((item) => item.component);
	const planError = validateSciencePlan(plan, componentNames, profile);
	if (planError) throw new Error(planError);
	const planDigest = planner.accepted.result_digest;
	const cohort = plan.work_units.map((workUnit) => {
		const assignmentId = `assignment-worker-science-${stablePart(workUnit.id)}-r0`;
		const accepted = requireAccepted(assignmentId);
		if (
			accepted.submission.work_unit !== workUnit.id ||
			accepted.submission.target_digest !== canonicalDigest(workUnit) ||
			accepted.submission.input_digest !== canonicalDigest({ plan_digest: planDigest, work_unit: workUnit.id })
		) {
			throw new Error(`Current science authority has stale condition ${workUnit.id}.`);
		}
		return { work_unit: workUnit.id, result_digest: accepted.accepted.result_digest };
	});
	const validationContext = reviewContext(plan, componentNames, profile);
	const inputDigest = canonicalDigest({ plan_digest: planDigest, cohort });
	const targetDigest = canonicalDigest({ plan_digest: planDigest, conditions: plan.work_units, cohort });
	const lineage = canonicalDigest({ input_digest: inputDigest, target_digest: targetDigest });
	const reports: ExperimentReviewReport[] = [];
	const assignmentIds: string[] = [];
	for (const role of SCIENCE_REVIEW_ROLES) {
		const assignmentId = `assignment-reviewer-science-${stablePart("science-cohort")}-${role}-r1-${lineage}`;
		const accepted = requireAccepted(assignmentId);
		const assignment = snapshot.assignments[assignmentId];
		if (
			assignment.child.work_unit !== "science-cohort" ||
			assignment.child.review_round !== 1 ||
			assignment.input_digest !== inputDigest ||
			assignment.target_digest !== targetDigest
		) {
			throw new Error(`Current science authority has stale review ${role}.`);
		}
		reports.push(objectResult(accepted, `Science reviewer ${role}`) as unknown as ExperimentReviewReport);
		assignmentIds.push(assignmentId);
	}
	const matrixError = validateReviewMatrix("science", reports, validationContext);
	if (matrixError) throw new Error(matrixError);
	const review: ExperimentReviewMatrixResult = {
		reports,
		assignmentIds,
		inputDigest,
		targetDigest,
		reviewRound: 1,
		blockingIssues: reports
			.filter((report) => report.blocking && report.verdict === "FAIL")
			.flatMap((report) => report.issues),
	};
	return {
		review,
		componentResults: scienceComponentResults(review, componentNames),
		provenance: {
			plan: { assignment_id: planner.accepted.assignment_id, result_digest: planDigest },
			cohort: cohort.map((item) => {
				const assignmentId = `assignment-worker-science-${stablePart(item.work_unit)}-r0`;
				return {
					assignment_id: assignmentId,
					work_unit: item.work_unit,
					result_digest: requireAccepted(assignmentId).accepted.result_digest,
				};
			}),
			reviews: reports.map((report, index) => ({
				assignment_id: assignmentIds[index],
				role: report.role,
				result_digest: requireAccepted(assignmentIds[index]).accepted.result_digest,
			})),
			input_digest: inputDigest,
			target_digest: targetDigest,
		},
	};
}

async function runFinalize(
	coordinator: ExperimentCoordinator,
	options: ExperimentDriverOptions,
	components: CanonicalExperimentIdeaComponent[],
	currentAuthority?: ScienceAuthority,
): Promise<void> {
	const authority =
		currentAuthority ?? (await currentScienceAuthority(coordinator, components, options.executionProfile));
	const componentResults = authority.componentResults;
	const acceptedResults = coordinator.read().assignments;
	const target = {
		idea_digest: options.ideaDigest,
		policy_digest: options.policyDigest,
		components,
		accepted_assignments: Object.entries(acceptedResults)
			.filter(([, assignment]) => assignment.status === "accepted")
			.map(([assignmentId, assignment]) => ({
				assignment_id: assignmentId,
				result_digest: assignment.accepted_result_digest,
			}))
			.sort((left, right) => left.assignment_id.localeCompare(right.assignment_id)),
	};
	const accepted = await coordinator.executeAssignment({
		assignmentId: "assignment-final-reviewer-finalize",
		childId: "final-reviewer-finalize",
		role: "final_reviewer",
		stage: "finalize",
		kind: "final_review",
		executionAttempt: 1,
		generation: 1,
		reviewRound: 0,
		inputDigest: canonicalDigest(target.accepted_assignments),
		targetDigest: canonicalDigest(target),
		readScope: [...options.readScope, ...options.projectWriteScope],
		writeScope: [],
		prompt: [
			"Perform the final blocking audit of the accepted native experiment evidence and submit it with submit_final_review.",
			"Return only artifact_role final_review, verdict PASS, blocking true, checked_artifacts, an empty issues array, and a summary. Do not supply component_results or pass incomplete or unsupported evidence.",
			`Final target: ${JSON.stringify(canonicalJson(target), null, 2)}.`,
		].join("\n\n"),
	});
	const final = finalReviewResult(accepted);
	await options.publication.publishFinal({
		accepted,
		componentDefinitions: components,
		componentResults,
		summary: final.summary,
		provenance: {
			idea_digest: options.ideaDigest,
			policy_digest: options.policyDigest,
			final_review: {
				assignment_id: accepted.accepted.assignment_id,
				result_digest: accepted.accepted.result_digest,
			},
			science: authority.provenance,
		},
	});
}

export function createExperimentStageDrivers(options: ExperimentDriverOptions): ExperimentStageDriver[] {
	requireDigest(options.ideaDigest, "Experiment Idea digest");
	requireDigest(options.policyDigest, "Experiment policy digest");
	if (!Number.isInteger(options.maxReviewRounds) || options.maxReviewRounds <= 0) {
		throw new Error("Experiment drivers require a positive review-round budget.");
	}
	if (!stringList(options.readScope) || !stringList(options.projectWriteScope)) {
		throw new Error("Experiment driver scopes must contain non-empty paths.");
	}
	const components = canonicalComponents(options.idea, options.executionProfile);
	let scienceAuthority: ScienceAuthority | undefined;
	const drivers: ExperimentStageDriver[] = [
		{ stage: "prepare", run: (coordinator) => runPrepare(coordinator, options, components) },
		{ stage: "code", run: (coordinator) => runCode(coordinator, options, components) },
		{
			stage: "science",
			run: async (coordinator) => {
				scienceAuthority = await runScience(coordinator, options, components);
			},
		},
		{ stage: "finalize", run: (coordinator) => runFinalize(coordinator, options, components, scienceAuthority) },
	];
	if (drivers.some((driver, index) => driver.stage !== EXPERIMENT_STAGES[index])) {
		throw new Error("Experiment stage drivers are not in canonical order.");
	}
	return drivers;
}
