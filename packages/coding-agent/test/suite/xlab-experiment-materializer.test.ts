import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	compareAndSwapSymbolicMemory,
	currentSymbolicMemoryDigest,
	type ExperimentFinalizationProvenance,
	ExperimentMaterializer,
} from "../../src/core/xlab/experiment/materializer.ts";
import {
	CODE_REVIEW_ROLES,
	canonicalDigest,
	type ExperimentAssignment,
	type ExperimentChildIdentity,
	ExperimentProtocolStore,
	type ExperimentReviewReport,
	type ExperimentSubmission,
	SCIENCE_REVIEW_ROLES,
} from "../../src/core/xlab/experiment/protocol.ts";

function digest(value: string): string {
	return createHash("sha256").update(value).digest("hex");
}

function review(role: string): ExperimentReviewReport {
	return {
		artifact_role: "reviewer_report",
		role,
		reviewer_kind: "agent",
		verdict: "PASS",
		blocking: true,
		summary: `${role} passed`,
		checked_artifacts: ["result.json"],
		issues: [],
		structured_findings: { checked: true },
	};
}

function sciencePlan() {
	const base = {
		full_run: true,
		run_level: "full",
		goal: "Run condition",
		setup_rationale: "Canonical comparison",
		runtime_probe_summary: "Runtime verified",
		source_basis: [{ source: "idea" }],
		training_protocol: { epochs: 1 },
		evaluation_protocol: { metric: "accuracy" },
		train_dataset_binding: { id: "train" },
		evaluation_dataset_bindings: [{ id: "test" }],
		metric_bindings: [{ id: "accuracy" }],
		raw_evidence: [{ path: "metrics.json" }],
	};
	return {
		stage: "science",
		work_units: [
			{
				...base,
				id: "reference",
				kind: "all_components_reference",
				enabled_components: ["Verifier"],
				disabled_components: [],
				output_dir: "results/science/reference",
				command: "node project/run.js --condition reference",
				artifact_ids: ["science.reference.evidence"],
				pass_condition:
					"Use managed artifact tools and artifact ledger proof for science.reference.evidence with metric, log, and evidence files.",
			},
			{
				...base,
				id: "disabled_verifier",
				kind: "component_disabled",
				disabled_component: "Verifier",
				reference_condition_id: "reference",
				enabled_components: [],
				disabled_components: ["Verifier"],
				output_dir: "results/science/disabled_verifier",
				command: "node project/run.js --condition disabled_verifier --disabled-component Verifier",
				artifact_ids: ["science.disabled_verifier.evidence"],
				pass_condition:
					"Use managed artifact tools and artifact ledger proof for science.disabled_verifier.evidence with metric, log, and evidence files.",
			},
		],
	};
}

interface AcceptedFixture {
	assignmentId: string;
	resultDigest: string;
}

describe("native experiment materializer", () => {
	const cleanups: Array<() => void> = [];
	afterEach(() => {
		while (cleanups.length) cleanups.pop()?.();
	});

	function fixture(runId = "run-1", existing?: { root: string; workspace: string }) {
		const root = existing?.root ?? mkdtempSync(join(tmpdir(), "xlab-materializer-"));
		if (!existing) cleanups.push(() => rmSync(root, { recursive: true, force: true }));
		const runDir = join(root, runId);
		const workspace = existing?.workspace ?? join(root, "workspace");
		if (!existsSync(workspace)) mkdirSync(workspace);
		const ideaDigest = digest("idea");
		const policyDigest = digest("policy");
		const protocol = new ExperimentProtocolStore(runDir, () => "2026-08-08T00:00:00.000Z");
		protocol.initialize({ runId, ideaDigest, policyDigest });
		protocol.start("prepare");
		const materializer = new ExperimentMaterializer({ runRoot: runDir, sharedRoot: workspace, protocol });
		return { root, runDir, workspace, protocol, ideaDigest, policyDigest, materializer };
	}

	function accept(
		protocol: ExperimentProtocolStore,
		policyDigest: string,
		params: {
			id: string;
			role: "planner" | "worker" | "reviewer" | "final_reviewer";
			stage: "prepare" | "code" | "science" | "finalize";
			kind?: "plan" | "worker_result" | "review" | "final_review";
			workUnit?: string;
			reviewRound?: number;
			inputDigest?: string;
			targetDigest?: string;
			resultValidation?: ExperimentAssignment["result_validation"];
			result: Record<string, unknown>;
		},
	): AcceptedFixture {
		const child: ExperimentChildIdentity = {
			child_id: `child-${params.id}`,
			session_id: `session-${params.id}`,
			session_file: `/sessions/${params.id}.jsonl`,
			parent_run_id: protocol.read().run_id,
			role: params.role,
			stage: params.stage,
			...(params.workUnit ? { work_unit: params.workUnit } : {}),
			execution_attempt: params.reviewRound ?? 1,
			generation: 1,
			review_round: params.reviewRound ?? 0,
		};
		protocol.registerChild(child);
		const kind =
			params.kind ??
			(params.role === "planner"
				? "plan"
				: params.role === "worker"
					? "worker_result"
					: params.role === "reviewer"
						? "review"
						: "final_review");
		const assignment = protocol.createAssignment(
			{
				schema_version: "xlab.experiment_assignment.v1",
				assignment_id: params.id,
				child,
				kind,
				input_digest: params.inputDigest ?? digest(`input-${params.id}`),
				target_digest: params.targetDigest ?? digest(`target-${params.id}`),
				policy_digest: policyDigest,
				read_scope: [],
				write_scope: [],
				...(params.resultValidation
					? { result_validation: params.resultValidation }
					: params.role === "reviewer"
						? { result_validation: { reviewer_role: params.result.role as string } }
						: {}),
			} satisfies Omit<ExperimentAssignment, "assignment_token_digest" | "status" | "created_at" | "updated_at">,
			"secret-token",
		);
		const submission: ExperimentSubmission = {
			schema_version: "xlab.experiment_submission.v1",
			submission_id: `submission-${params.id}`,
			assignment_id: params.id,
			assignment_token: "secret-token",
			kind,
			child_id: child.child_id,
			role: child.role,
			stage: child.stage,
			...(child.work_unit ? { work_unit: child.work_unit } : {}),
			execution_attempt: child.execution_attempt,
			generation: child.generation,
			review_round: child.review_round,
			input_digest: assignment.input_digest,
			target_digest: assignment.target_digest,
			policy_digest: assignment.policy_digest,
			result: params.result,
			output_paths: [],
			submitted_at: "2026-08-08T00:00:00.000Z",
		};
		const accepted = protocol.acceptSubmission(submission);
		if (!accepted.ok) throw new Error(accepted.error);
		return { assignmentId: params.id, resultDigest: accepted.accepted.result_digest };
	}

	function finalFixture(runId = "run-1", existing?: { root: string; workspace: string }) {
		const value = fixture(runId, existing);
		value.protocol.completeStage("prepare");
		value.protocol.start("code");
		value.protocol.completeStage("code");
		value.protocol.start("science");
		const plan = accept(value.protocol, value.policyDigest, {
			id: "science-plan",
			role: "planner",
			stage: "science",
			resultValidation: { component_names: ["Verifier"] },
			result: sciencePlan(),
		});
		const worker = accept(value.protocol, value.policyDigest, {
			id: "science-worker-reference",
			role: "worker",
			stage: "science",
			workUnit: "reference",
			result: { artifact_role: "worker_result", status: "success" },
		});
		const disabledWorker = accept(value.protocol, value.policyDigest, {
			id: "science-worker-disabled",
			role: "worker",
			stage: "science",
			workUnit: "disabled_verifier",
			result: { artifact_role: "worker_result", status: "success" },
		});
		const inputDigest = digest("science-input");
		const targetDigest = digest("science-target");
		const validationContext = {
			component_names: ["Verifier"],
			science_conditions: [
				{
					id: "reference",
					enabled_components: ["Verifier"],
					disabled_components: [],
					kind: "all_components_reference" as const,
				},
				{
					id: "disabled_verifier",
					enabled_components: [],
					disabled_components: ["Verifier"],
					kind: "component_disabled" as const,
				},
			],
		};
		const reviews = SCIENCE_REVIEW_ROLES.map((role) => {
			const report = review(role);
			if (role === "statistical_interpretation")
				report.structured_findings = {
					component_results: {
						Verifier: {
							result: "positive",
							metric: "accuracy",
							value: "90%",
							confidence: 0.8,
							analysis: "Improves accuracy",
							follow_up_required: false,
							condition_id: "disabled_verifier",
							enabled_components: [],
							disabled_components: ["Verifier"],
						},
					},
				};
			return accept(value.protocol, value.policyDigest, {
				id: `science-review-${role}`,
				role: "reviewer",
				stage: "science",
				workUnit: "science-cohort",
				reviewRound: 1,
				inputDigest,
				targetDigest,
				resultValidation: { reviewer_role: role, review_context: validationContext },
				result: report as unknown as Record<string, unknown>,
			});
		});
		value.protocol.completeStage("science");
		value.protocol.start("finalize");
		const final = accept(value.protocol, value.policyDigest, {
			id: "final-review",
			role: "final_reviewer",
			stage: "finalize",
			result: {
				artifact_role: "final_review",
				verdict: "PASS",
				blocking: true,
				summary: { feasible: true },
				checked_artifacts: ["result.json"],
				issues: [],
			},
		});
		const provenance: ExperimentFinalizationProvenance = {
			idea_digest: value.ideaDigest,
			policy_digest: value.policyDigest,
			final_review: { assignment_id: final.assignmentId, result_digest: final.resultDigest },
			science: {
				plan: { assignment_id: plan.assignmentId, result_digest: plan.resultDigest },
				cohort: [
					{ assignment_id: worker.assignmentId, work_unit: "reference", result_digest: worker.resultDigest },
					{
						assignment_id: disabledWorker.assignmentId,
						work_unit: "disabled_verifier",
						result_digest: disabledWorker.resultDigest,
					},
				],
				reviews: reviews.map((item, index) => ({
					assignment_id: item.assignmentId,
					role: SCIENCE_REVIEW_ROLES[index],
					result_digest: item.resultDigest,
				})),
				input_digest: inputDigest,
				target_digest: targetDigest,
			},
		};
		return { ...value, provenance };
	}

	it("publishes isolated reviewer attempts and byte-identical latest files in fixed role order", () => {
		const { runDir, workspace, protocol, policyDigest, materializer } = fixture();
		protocol.completeStage("prepare");
		protocol.start("code");
		const ids = CODE_REVIEW_ROLES.map((role) => `review-${role}`);
		for (let index = ids.length - 1; index >= 0; index--)
			accept(protocol, policyDigest, {
				id: ids[index],
				role: "reviewer",
				stage: "code",
				workUnit: "unit",
				reviewRound: 2,
				result: review(CODE_REVIEW_ROLES[index]) as unknown as Record<string, unknown>,
			});
		const publication = materializer.materializeReviewMatrix({
			stage: "code",
			workUnit: "unit",
			reviewRound: 2,
			assignmentIds: ids,
		});
		for (const [index, report] of publication.reports.entries()) {
			expect(report.attempt.path).toContain("attempts/002.json");
			expect(report.attempt.path).toContain(CODE_REVIEW_ROLES[index]);
			expect(readFileSync(join(runDir, report.latest.path))).toEqual(
				readFileSync(join(runDir, report.attempt.path)),
			);
			expect(report.latest.sha256).toBe(report.attempt.sha256);
		}
		expect(existsSync(join(workspace, "agent_reports"))).toBe(false);
	});

	it("rejects unaccepted inputs and invalid plans before publishing", () => {
		const { runDir, protocol, policyDigest, materializer } = fixture();
		protocol.completeStage("prepare");
		protocol.start("code");
		protocol.completeStage("code");
		protocol.start("science");
		accept(protocol, policyDigest, {
			id: "bad-plan",
			role: "planner",
			stage: "science",
			resultValidation: { component_names: ["Verifier"] },
			result: sciencePlan(),
		});
		expect(() => materializer.materializePlan("bad-plan", ["Other"])).toThrow();
		expect(() => materializer.materializePlan("missing")).toThrow("requires accepted assignment");
		expect(existsSync(join(runDir, "agent_reports/science/plan/latest.json"))).toBe(false);
	});

	it("keeps publications run-local, replays one memory operation, and emits schema 2 last", () => {
		const { runDir, workspace, materializer, provenance } = finalFixture();
		const componentDefinitions = [{ component: "Verifier", method_context: "Checks evidence" }];
		const componentResults = {
			Verifier: {
				result: "positive",
				metric: "accuracy",
				value: "90%",
				confidence: 0.8,
				analysis: "Improves accuracy",
				follow_up_required: false,
			},
		};
		const options = {
			componentDefinitions,
			componentResults,
			summary: { feasible: true, confidence: 0.8, key_findings: ["Works"] },
			provenance,
			symbolicMemoryPath: ".xlab/symbolic_memory/symbolic_memory.json",
			expectedSymbolicMemoryDigest: null,
		};
		const first = materializer.materializeFinal(options);
		const memoryPath = join(workspace, ".xlab/symbolic_memory/symbolic_memory.json");
		const firstMemory = readFileSync(memoryPath);
		const second = materializer.materializeFinal(options);
		expect(readFileSync(memoryPath)).toEqual(firstMemory);
		expect(second.symbolic_memory).toMatchObject({
			operation_id: (first.symbolic_memory as { operation_id: string }).operation_id,
			replayed: true,
		});
		expect(existsSync(join(workspace, "manifest.json"))).toBe(false);
		const manifest = JSON.parse(readFileSync(join(runDir, first.manifest.path), "utf-8"));
		expect(manifest).toMatchObject({
			schema_version: "2",
			run_id: "run-1",
			skill_name: "run_experiment",
			skill_version: "2.0.0",
			status: "success",
		});
		expect(manifest.artifacts.map((item: { type: string }) => item.type)).toEqual(
			expect.arrayContaining(["ablation_results", "final_audit", "symbolic_memory_receipt"]),
		);
		expect(
			manifest.artifacts.every(
				(item: { path: string }) => !item.path.startsWith("/") && existsSync(join(runDir, item.path)),
			),
		).toBe(true);
		expect(manifest.inputs.finalization_provenance).toEqual(provenance);
		const attempt = JSON.parse(
			readFileSync(join(runDir, "agent_reports/ablation/final/symbolic_memory_receipt.json"), "utf-8"),
		);
		expect(attempt.operation_id).toBe((first.symbolic_memory as { operation_id: string }).operation_id);
		expect(attempt).not.toHaveProperty("replayed");
	});

	it("supports sequential different runs over existing shared memory", () => {
		const first = finalFixture("run-a");
		const componentDefinitions = [{ component: "Verifier", method_context: "Checks evidence" }];
		const result = {
			Verifier: {
				result: "positive",
				metric: "accuracy",
				value: "90%",
				confidence: 0.8,
				analysis: "Improves accuracy",
				follow_up_required: false,
			},
		};
		const firstPublication = first.materializer.materializeFinal({
			componentDefinitions,
			componentResults: result,
			summary: { feasible: true, confidence: 0.8, key_findings: ["First"] },
			provenance: first.provenance,
			symbolicMemoryPath: ".xlab/symbolic_memory/symbolic_memory.json",
			expectedSymbolicMemoryDigest: null,
		});
		const second = finalFixture("run-b", { root: first.root, workspace: first.workspace });
		const currentDigest = (firstPublication.symbolic_memory as { digest: string }).digest;
		const secondPublication = second.materializer.materializeFinal({
			componentDefinitions,
			componentResults: result,
			summary: { feasible: true, confidence: 0.6, key_findings: ["Second"] },
			provenance: second.provenance,
			symbolicMemoryPath: ".xlab/symbolic_memory/symbolic_memory.json",
			expectedSymbolicMemoryDigest: currentDigest,
		});
		expect(secondPublication.symbolic_memory).toMatchObject({ previous_digest: currentDigest, replayed: false });
		expect(existsSync(join(first.runDir, "manifest.json"))).toBe(true);
		expect(existsSync(join(second.runDir, "manifest.json"))).toBe(true);
	});

	for (const faultPoint of [
		"after_symbolic_memory_cas",
		"after_symbolic_memory_receipt",
		"after_final_audit",
		"after_manifest",
	] as const) {
		it(`recovers byte-identical publication after ${faultPoint}`, () => {
			const baseline = finalFixture("run-crash-baseline");
			const crashed = finalFixture("run-crash-baseline");
			const componentDefinitions = [{ component: "Verifier", method_context: "Checks evidence" }];
			const componentResults = {
				Verifier: {
					result: "positive",
					metric: "accuracy",
					value: "90%",
					confidence: 0.8,
					analysis: "Improves accuracy",
					follow_up_required: false,
				},
			};
			const createOptions = (value: ReturnType<typeof finalFixture>) => ({
				componentDefinitions,
				componentResults,
				summary: { feasible: true, confidence: 0.8, key_findings: ["Works"] },
				provenance: value.provenance,
				symbolicMemoryPath: ".xlab/symbolic_memory/symbolic_memory.json",
				expectedSymbolicMemoryDigest: currentSymbolicMemoryDigest(
					join(value.workspace, ".xlab/symbolic_memory/symbolic_memory.json"),
				),
			});
			baseline.materializer.materializeFinal(createOptions(baseline));
			const crashing = new ExperimentMaterializer({
				runRoot: crashed.runDir,
				sharedRoot: crashed.workspace,
				protocol: crashed.protocol,
				faultInjector: (point) => {
					if (point === faultPoint) throw new Error(`fault:${faultPoint}`);
				},
			});

			expect(() => crashing.materializeFinal(createOptions(crashed))).toThrow(`fault:${faultPoint}`);
			const memoryPath = join(crashed.workspace, ".xlab/symbolic_memory/symbolic_memory.json");
			const memoryAfterCrash = readFileSync(memoryPath);
			const recovered = crashed.materializer.materializeFinal(createOptions(crashed));
			expect(readFileSync(memoryPath)).toEqual(memoryAfterCrash);
			expect(recovered.symbolic_memory).toMatchObject({ replayed: true });

			for (const relativePath of [
				"agent_reports/ablation/final/symbolic_memory_receipt.json",
				"agent_reports/ablation/final/final_audit.json",
				"manifest.json",
			]) {
				expect(readFileSync(join(crashed.runDir, relativePath))).toEqual(
					readFileSync(join(baseline.runDir, relativePath)),
				);
			}
			const receipt = JSON.parse(
				readFileSync(join(crashed.runDir, "agent_reports/ablation/final/symbolic_memory_receipt.json"), "utf-8"),
			);
			expect(receipt).not.toHaveProperty("replayed");
			const ledger = JSON.parse(readFileSync(`${memoryPath}.operations.json`, "utf-8"));
			expect(Object.keys(ledger.operations)).toHaveLength(1);
		});
	}

	it("computes the digest of empty and existing symbolic memory", () => {
		const root = mkdtempSync(join(tmpdir(), "xlab-memory-digest-"));
		cleanups.push(() => rmSync(root, { recursive: true, force: true }));
		const path = join(root, "memory.json");
		const emptyDigest = currentSymbolicMemoryDigest(path);
		const components = {
			Verifier: {
				result: "positive",
				metric: "accuracy",
				value: "90%",
				confidence: 0.8,
				analysis: "Works",
				method_context: "Checks",
			},
		};
		const applied = compareAndSwapSymbolicMemory({
			path,
			expectedDigest: emptyDigest,
			operationId: "op-1",
			payloadDigest: digest("payload"),
			components,
		});
		expect(currentSymbolicMemoryDigest(path)).toBe(applied.digest);
	});

	it("rejects stale CAS without changing existing memory", () => {
		const root = mkdtempSync(join(tmpdir(), "xlab-memory-"));
		cleanups.push(() => rmSync(root, { recursive: true, force: true }));
		const path = join(root, "memory.json");
		const components = {
			Verifier: {
				result: "positive",
				metric: "accuracy",
				value: "90%",
				confidence: 0.8,
				analysis: "Works",
				method_context: "Checks",
			},
		};
		const first = compareAndSwapSymbolicMemory({
			path,
			expectedDigest: null,
			operationId: "op-1",
			payloadDigest: digest("payload-1"),
			components,
		});
		const before = readFileSync(path);
		expect(() =>
			compareAndSwapSymbolicMemory({
				path,
				expectedDigest: digest("stale"),
				operationId: "op-2",
				payloadDigest: digest("payload-2"),
				components,
			}),
		).toThrow("compare-and-swap conflict");
		expect(readFileSync(path)).toEqual(before);
		expect(first.previous_digest).toMatch(/^[a-f0-9]{64}$/);
	});
});
