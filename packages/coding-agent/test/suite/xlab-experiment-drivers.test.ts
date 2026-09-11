import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import type {
	ExperimentAssignmentSpec,
	ExperimentCoordinator,
	ExperimentReviewedWorkerSpec,
	ExperimentReviewMatrixResult,
	ExperimentWorkUnitDriver,
} from "../../src/core/xlab/experiment/coordinator.ts";
import {
	createExperimentStageDrivers,
	type ExperimentDriverPublication,
} from "../../src/core/xlab/experiment/drivers.ts";
import {
	type CanonicalJson,
	CODE_REVIEW_ROLES,
	canonicalDigest,
	EXPERIMENT_STAGES,
	type ExperimentAcceptedResult,
	type ExperimentPlan,
	type ExperimentReviewReport,
	type ExperimentWorkUnit,
	PREPARE_WORK_UNITS,
	SCIENCE_REVIEW_ROLES,
} from "../../src/core/xlab/experiment/protocol.ts";

const COMPONENTS = ["Evidence memory", "Verifier"];

function digest(value: string): string {
	return createHash("sha256").update(value).digest("hex");
}

function idea() {
	return {
		schema_version: "xlab.experiment_idea.v1",
		status: "success",
		blockers: [],
		components: COMPONENTS.map((component, index) => ({
			component,
			explanation: `${component} explanation`,
			index: String(index + 1),
		})),
	};
}

function managedCompletion(ids: string[]): string {
	return `Use managed artifact tools and artifact ledger proof for ${ids.join(", ")}.`;
}

function preparePlan(): ExperimentPlan {
	const artifacts: Record<string, string[]> = {
		repos: ["prepare.discovery", "prepare.repos"],
		dataset: ["prepare.dataset"],
		model: ["prepare.model"],
		env: ["prepare.env"],
		synthesis: ["prepare.idea", "prepare.target_inventory"],
	};
	return {
		stage: "prepare",
		work_units: PREPARE_WORK_UNITS.map((id) => ({
			id,
			goal: `Prepare ${id}`,
			input_paths: { idea: "inputs/idea.json" },
			repos_policy: "reference_or_copy",
			project_must_be_self_contained: true,
			research_required: false,
			acquisition_required: false,
			existing_local_hints: [],
			artifact_ids: artifacts[id],
			done_condition: managedCompletion(artifacts[id]),
		})),
	};
}

function codeUnit(id: string): ExperimentWorkUnit {
	const artifactId = `code.${id}.handoff`;
	return {
		id,
		goal: `Implement ${id}`,
		input_paths: { idea: "inputs/idea.json" },
		repos_policy: "reference_or_copy",
		project_must_be_self_contained: true,
		component_scope: COMPONENTS,
		code_artifacts: [
			{
				path: `src/${id}.ts`,
				artifact_type: "source",
				symbols: [id],
				responsibility: `Implement ${id}`,
				dependencies: [],
				config_keys: [],
				entrypoint_role: "runtime",
			},
		],
		interface_contract: { api: id },
		implementation_requirements: { executable: true },
		experiment_bindings: { components: COMPONENTS },
		component_disable_hooks: [{ config_key: "disabled_component" }],
		write_scope: "project",
		verify_command: `node project/${id}.js`,
		artifact_ids: [artifactId],
		done_condition: `${managedCompletion([artifactId])} Verify a real dataset with metric evaluation.`,
	};
}

function codePlan(): ExperimentPlan {
	const implementation = codeUnit("implementation");
	const smoke = {
		...codeUnit("final_integration_smoke"),
		needs: [implementation.id],
		evidence: ["integrated", "component_disabled"],
	};
	return { stage: "code", work_units: [implementation, smoke] };
}

function scienceCondition(id: string, disabled: string[] = []): ExperimentWorkUnit {
	const artifactId = `science.${id}.evidence`;
	return {
		id,
		kind: disabled.length === 0 ? "all_components_reference" : "component_disabled",
		goal: `Run ${id}`,
		full_run: true,
		run_level: "full",
		...(disabled.length === 0 ? {} : { disabled_component: disabled[0], reference_condition_id: "reference" }),
		enabled_components: COMPONENTS.filter((component) => !disabled.includes(component)),
		disabled_components: disabled,
		output_dir: `results/science/${id}`,
		command: `node project/run.js --condition ${id}${disabled.length ? ` --disabled-component "${disabled[0]}"` : ""}`,
		setup_rationale: "Canonical full comparison",
		runtime_probe_summary: "Runtime and resources verified",
		source_basis: [{ source: "idea" }],
		training_protocol: { epochs: 1 },
		evaluation_protocol: { metric: "accuracy" },
		train_dataset_binding: { id: "train" },
		evaluation_dataset_bindings: [{ id: "test" }],
		metric_bindings: [{ id: "accuracy" }],
		raw_evidence: [{ path: `${id}/metrics.json` }],
		artifact_ids: [artifactId],
		pass_condition: `${managedCompletion([artifactId])} Require metric, log, and evidence files.`,
	};
}

function sciencePlan(): ExperimentPlan {
	return {
		stage: "science",
		work_units: [
			scienceCondition("reference"),
			...COMPONENTS.map((component, index) => scienceCondition(`disabled_${index + 1}`, [component])),
		],
	};
}

function accepted(spec: ExperimentAssignmentSpec, result: Record<string, unknown>): ExperimentAcceptedResult {
	const canonicalResult = result as CanonicalJson;
	const resultDigest = canonicalDigest(canonicalResult);
	return {
		accepted: {
			schema_version: "xlab.experiment_accepted_submission.v1",
			submission_id: `${spec.assignmentId}-submission`,
			assignment_id: spec.assignmentId,
			result_digest: resultDigest,
			output_digests: {},
			accepted_at: "2026-08-08T00:00:00.000Z",
		},
		submission: {
			schema_version: "xlab.experiment_submission.v1",
			submission_id: `${spec.assignmentId}-submission`,
			assignment_id: spec.assignmentId,
			assignment_token: "[redacted]",
			kind: spec.kind,
			child_id: spec.childId,
			role: spec.role,
			stage: spec.stage,
			...(spec.workUnit === undefined ? {} : { work_unit: spec.workUnit }),
			execution_attempt: spec.executionAttempt,
			generation: spec.generation,
			review_round: spec.reviewRound,
			input_digest: spec.inputDigest,
			target_digest: spec.targetDigest,
			policy_digest: digest("policy"),
			result: canonicalResult,
			output_paths: [],
			submitted_at: "2026-08-08T00:00:00.000Z",
		},
	};
}

function reviewResult(stage: "code" | "science", workUnit = "unit", reviewRound = 1): ExperimentReviewMatrixResult {
	const roles = stage === "code" ? CODE_REVIEW_ROLES : SCIENCE_REVIEW_ROLES;
	const reports = roles.map(
		(role): ExperimentReviewReport => ({
			artifact_role: "reviewer_report",
			role,
			reviewer_kind: "agent",
			verdict: "PASS",
			blocking: true,
			summary: `${role} passed`,
			checked_artifacts: ["result.json"],
			issues: [],
			structured_findings: { checked: true },
		}),
	);
	if (stage === "science") {
		const statistical = reports.find((report) => report.role === "statistical_interpretation")!;
		statistical.structured_findings = {
			component_results: Object.fromEntries(
				COMPONENTS.map((component, index) => [
					component,
					{
						result: "positive",
						condition_id: `disabled_${index + 1}`,
						enabled_components: COMPONENTS.filter((item) => item !== component),
						disabled_components: [component],
					},
				]),
			),
		};
	}
	return {
		reports,
		assignmentIds: roles.map((role) => `actual-${stage}-${workUnit}-${role}-${reviewRound}`),
		inputDigest: digest(`${stage}-review-input`),
		targetDigest: digest(`${stage}-review-target`),
		reviewRound,
		blockingIssues: [],
	};
}

function publication() {
	return {
		publishPlan: vi.fn(),
		publishWorker: vi.fn(),
		publishReview: vi.fn(),
		publishFinal: vi.fn(),
	} satisfies ExperimentDriverPublication;
}

function options(publicationValue: ExperimentDriverPublication, ideaValue: Record<string, unknown> = idea()) {
	return {
		idea: ideaValue,
		ideaDigest: digest("idea"),
		policyDigest: digest("policy"),
		readScope: ["inputs", "project"],
		projectWriteScope: ["project"],
		maxReviewRounds: 2,
		publication: publicationValue,
	};
}

describe("native experiment stage drivers", () => {
	it("validates the canonical Idea and returns canonical stage order", () => {
		const drivers = createExperimentStageDrivers(options(publication()));
		expect(drivers.map((driver) => driver.stage)).toEqual(EXPERIMENT_STAGES);

		for (const invalid of [
			{ ...idea(), schema_version: "wrong" },
			{ ...idea(), status: "blocked" },
			{ ...idea(), blockers: ["missing"] },
			{ ...idea(), components: [] },
			{ ...idea(), components: [{ component: "A", explanation: "A", index: "2" }] },
			{
				...idea(),
				components: [
					{ component: "A", explanation: "A", index: "1" },
					{ component: "A", explanation: "B", index: "2" },
				],
			},
		]) {
			expect(() => createExperimentStageDrivers(options(publication(), invalid))).toThrow();
		}
	});

	it("constructs durable assignments, schedules plans, and publishes only accepted evidence", async () => {
		const published = publication();
		const executed: ExperimentAssignmentSpec[] = [];
		const reviewed: ExperimentReviewedWorkerSpec[] = [];
		const acceptedScienceReviews: ExperimentAcceptedResult[] = [];
		const snapshotAssignments: Record<string, { status: string; accepted_result_digest?: string }> = {};
		const plans = { prepare: preparePlan(), code: codePlan(), science: sciencePlan() };
		const coordinator = {
			executeAssignment: vi.fn(async (spec: ExperimentAssignmentSpec) => {
				executed.push(spec);
				let result: Record<string, unknown>;
				if (spec.role === "planner")
					result = plans[spec.stage as keyof typeof plans] as unknown as Record<string, unknown>;
				else if (spec.role === "final_reviewer") {
					result = {
						artifact_role: "final_review",
						verdict: "PASS",
						blocking: true,
						checked_artifacts: ["science/results.json"],
						issues: [],
						summary: { feasible: true },
					};
				} else result = { artifact_role: "worker_result", status: "success" };
				const value = accepted(spec, result);
				snapshotAssignments[spec.assignmentId] = {
					status: "accepted",
					accepted_result_digest: value.accepted.result_digest,
				};
				return value;
			}),
			runPlanWorkUnits: vi.fn(async (plan: ExperimentPlan, unitDrivers: ExperimentWorkUnitDriver[]) => {
				for (const unit of plan.work_units)
					await unitDrivers
						.find((driver) => driver.workUnit.id === unit.id)!
						.run(coordinator as unknown as ExperimentCoordinator, unit);
			}),
			runReviewedWorker: vi.fn(async (spec: ExperimentReviewedWorkerSpec) => {
				reviewed.push(spec);
				const worker = accepted(spec.worker, { artifact_role: "worker_result", status: "success" });
				return { worker, review: reviewResult("code", spec.worker.workUnit, 1), reviewRound: 1 };
			}),
			runReviewMatrix: vi.fn(async (spec) => {
				const review = reviewResult("science", spec.workUnit, spec.reviewRound);
				for (const report of review.reports) {
					acceptedScienceReviews.push(
						accepted(
							{
								assignmentId: review.assignmentIds[review.reports.indexOf(report)],
								childId: `review-${report.role}`,
								role: "reviewer",
								stage: "science",
								kind: "review",
								executionAttempt: 1,
								generation: 1,
								reviewRound: spec.reviewRound,
								inputDigest: spec.inputDigest,
								targetDigest: spec.targetDigest,
								readScope: spec.readScope,
								writeScope: [],
								prompt: "review",
							},
							report as unknown as Record<string, unknown>,
						),
					);
				}
				return review;
			}),
			read: vi.fn(() => ({ assignments: snapshotAssignments })),
			listAcceptedResults: vi.fn(() => acceptedScienceReviews),
		};

		for (const driver of createExperimentStageDrivers(options(published))) {
			await driver.run(coordinator as unknown as ExperimentCoordinator);
		}

		expect(executed.filter((spec) => spec.role === "planner").map((spec) => spec.assignmentId)).toEqual([
			"assignment-planner-prepare",
			"assignment-planner-code",
			"assignment-planner-science",
		]);
		expect(
			executed.filter((spec) => spec.stage === "prepare" && spec.role === "worker").map((spec) => spec.workUnit),
		).toEqual(PREPARE_WORK_UNITS);
		expect(reviewed.map((spec) => spec.worker.stage)).toEqual(["code", "code"]);
		for (const spec of reviewed) {
			expect(spec.worker.readScope).toEqual(["inputs", "project"]);
			expect(spec.worker.writeScope).toEqual(["project"]);
			const initial = accepted(spec.worker, { artifact_role: "worker_result" });
			const reviewSpec = spec.review(initial, 1);
			expect(reviewSpec.readScope).toEqual(["inputs", "project", "project"]);
			if (spec.worker.stage === "science") expect(reviewSpec.validationContext?.component_names).toEqual(COMPONENTS);
			const repaired = spec.repair(
				initial,
				[{ code: "fix", message: "fix", required_fix: "fix", evidence: "result" }],
				1,
			);
			expect(repaired).toMatchObject({
				childId: spec.worker.childId,
				stage: spec.worker.stage,
				workUnit: spec.worker.workUnit,
				targetDigest: spec.worker.targetDigest,
				executionAttempt: 2,
				generation: 2,
				reviewRound: 1,
			});
		}
		expect(executed.at(-1)).toMatchObject({
			assignmentId: "assignment-final-reviewer-finalize",
			childId: "final-reviewer-finalize",
			role: "final_reviewer",
			stage: "finalize",
			kind: "final_review",
			writeScope: [],
		});
		expect(published.publishPlan).toHaveBeenCalledTimes(3);
		expect(published.publishWorker).toHaveBeenCalledTimes(10);
		expect(published.publishReview).toHaveBeenCalledTimes(3);
		expect(published.publishFinal).toHaveBeenCalledOnce();
		expect(
			published.publishFinal.mock.calls[0][0].componentDefinitions.map(
				(item: { component: string }) => item.component,
			),
		).toEqual(COMPONENTS);
		expect(published.publishFinal.mock.calls[0][0].componentResults).toEqual(
			reviewResult("science").reports.find((report) => report.role === "statistical_interpretation")!
				.structured_findings.component_results,
		);
		expect(published.publishFinal.mock.calls[0][0].provenance).toMatchObject({
			idea_digest: digest("idea"),
			policy_digest: digest("policy"),
			final_review: { assignment_id: "assignment-final-reviewer-finalize" },
			science: {
				plan: { assignment_id: "assignment-planner-science" },
				cohort: expect.arrayContaining([expect.objectContaining({ work_unit: "reference" })]),
				reviews: SCIENCE_REVIEW_ROLES.map((role) => expect.objectContaining({ role })),
			},
		});
		expect(executed.at(-1)?.prompt).toContain("Do not supply component_results");
	});

	it("rejects malformed accepted worker and final-review results before publication", async () => {
		const workerPublished = publication();
		const prepare = createExperimentStageDrivers(options(workerPublished))[0];
		const workerCoordinator = {
			executeAssignment: vi.fn(async (spec: ExperimentAssignmentSpec) =>
				accepted(
					spec,
					spec.role === "planner" ? (preparePlan() as unknown as Record<string, unknown>) : { status: "success" },
				),
			),
			runPlanWorkUnits: vi.fn(async (plan: ExperimentPlan, drivers: ExperimentWorkUnitDriver[]) =>
				drivers[0].run(workerCoordinator as unknown as ExperimentCoordinator, plan.work_units[0]),
			),
		};
		await expect(prepare.run(workerCoordinator as unknown as ExperimentCoordinator)).rejects.toThrow(
			"artifact_role worker_result",
		);
		expect(workerPublished.publishWorker).not.toHaveBeenCalled();

		const finalPublished = publication();
		const finalize = createExperimentStageDrivers(options(finalPublished))[3];
		const finalSpecCoordinator = {
			read: vi.fn(() => ({ assignments: {} })),
			listAcceptedResults: vi.fn(() => []),
			executeAssignment: vi.fn(async (spec: ExperimentAssignmentSpec) =>
				accepted(spec, {
					artifact_role: "final_review",
					verdict: "PASS",
					blocking: true,
					checked_artifacts: ["result.json"],
					issues: [],
					summary: { feasible: true },
				}),
			),
		};
		await expect(finalize.run(finalSpecCoordinator as unknown as ExperimentCoordinator)).rejects.toThrow(
			"requires the current complete PASS science review matrix",
		);
		expect(finalSpecCoordinator.executeAssignment).not.toHaveBeenCalled();
		expect(finalPublished.publishFinal).not.toHaveBeenCalled();
	});
});
