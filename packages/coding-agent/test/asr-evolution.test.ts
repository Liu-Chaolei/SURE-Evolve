import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { AgentSession } from "../src/core/agent-session.ts";
import type { AsrEvolutionConfig } from "../src/core/xlab/experiment/asr-execution.ts";
import { xiAgentOptions } from "../src/core/xlab/experiment/asr-experiment.ts";
import { asrPlans } from "../src/core/xlab/experiment/asr-plans.ts";
import type {
	CreateExperimentChildOptions,
	ExperimentChildRuntime,
} from "../src/core/xlab/experiment/child-runtime.ts";
import { ExperimentCoordinator } from "../src/core/xlab/experiment/coordinator.ts";
import {
	canonicalDigest,
	codeReviewRoles,
	type ExperimentAssignment,
	type ExperimentPlan,
	ExperimentProtocolStore,
	validateCodePlan,
	validatePreparePlan,
	validateSciencePlan,
} from "../src/core/xlab/experiment/protocol.ts";

const directories: string[] = [];
afterEach(() => {
	for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

const config = {
	runtime: "/runtime",
	recipe: "/recipe",
	data: "/data",
	resource_manifest: "/manifest",
	refs: { regular: "/regular" },
	pipeline_id: "asr.en.wer",
	training: { epochs: 30, world_size: 4, max_duration: 900, seed: 42, precision: "fp32" },
} as AsrEvolutionConfig;

function coordinator() {
	const directory = mkdtempSync(join(tmpdir(), "xlab-asr-test-"));
	directories.push(directory);
	const protocol = new ExperimentProtocolStore(directory);
	protocol.initialize({
		runId: "offline",
		ideaDigest: canonicalDigest("idea"),
		policyDigest: canonicalDigest("policy"),
	});
	return new ExperimentCoordinator({
		runId: "offline",
		runDir: directory,
		cwd: directory,
		agentDir: directory,
		ideaDigest: canonicalDigest("idea"),
		policyDigest: canonicalDigest("policy"),
		ownerId: "offline-owner",
		protocol,
	});
}

describe("native ASR full-condition contracts", () => {
	it("retries an interrupted provider turn in the same durable child without repeating accepted work", async () => {
		const directory = mkdtempSync(join(tmpdir(), "xlab-retry-"));
		directories.push(directory);
		const protocol = new ExperimentProtocolStore(directory);
		const digest = canonicalDigest("retry");
		let prompts = 0;
		let opened = 0;
		let released = 0;
		const runtime = {
			createIdentity(options: CreateExperimentChildOptions) {
				const identity = { ...options, session_id: options.child_id, session_file: join(directory, "child.jsonl") };
				protocol.registerChild(identity);
				return identity;
			},
			async openAssignment(assignment: ExperimentAssignment, token: string) {
				opened++;
				const session = {
					async prompt() {
						if (++prompts === 1) throw new Error("Stream ended without finish_reason");
						const child = assignment.child;
						const answer = protocol.acceptSubmission({
							schema_version: "xlab.experiment_submission.v1",
							submission_id: "retry-submission",
							assignment_id: assignment.assignment_id,
							assignment_token: token,
							kind: assignment.kind,
							child_id: child.child_id,
							role: child.role,
							stage: child.stage,
							work_unit: child.work_unit,
							execution_attempt: child.execution_attempt,
							generation: child.generation,
							review_round: child.review_round,
							input_digest: assignment.input_digest,
							target_digest: assignment.target_digest,
							policy_digest: assignment.policy_digest,
							result: { artifact_role: "worker_result", status: "success" },
							output_paths: [],
							submitted_at: new Date().toISOString(),
						});
						expect(answer.ok).toBe(true);
					},
				} as unknown as AgentSession;
				return { identity: assignment.child, assignment, session, resumed: false };
			},
			release() {
				released++;
			},
			async abortAll() {},
		} as unknown as ExperimentChildRuntime;
		const control = new ExperimentCoordinator({
			runId: "retry",
			runDir: directory,
			cwd: directory,
			agentDir: directory,
			ideaDigest: digest,
			policyDigest: digest,
			ownerId: "retry-owner",
			protocol,
			childRuntime: runtime,
			maxAssignmentPromptAttempts: 2,
			tokenFactory: () => "offline-token",
		});
		control.initialize();
		protocol.start("prepare");
		const spec = {
			assignmentId: "assignment-retry",
			childId: "retry-child",
			role: "worker" as const,
			stage: "prepare" as const,
			workUnit: "repos",
			kind: "worker_result" as const,
			executionAttempt: 1,
			generation: 1,
			reviewRound: 0,
			inputDigest: digest,
			targetDigest: digest,
			readScope: [directory],
			writeScope: [directory],
			prompt: "Complete the current work unit",
		};
		await control.executeAssignment(spec);
		await control.executeAssignment(spec);
		expect(prompts).toBe(2);
		expect(opened).toBe(1);
		expect(released).toBe(1);
	});
	it("registers XI credentials in memory without a provider request", async () => {
		const directory = mkdtempSync(join(tmpdir(), "xlab-xi-config-"));
		directories.push(directory);
		const env = join(directory, ".env");
		writeFileSync(env, "XI_API_KEY=offline-test-key\nXI_BASE_URL=https://xi.example.invalid\n");
		const options = xiAgentOptions({ ...config, env_file: env });
		expect(options.model.id).toBe("gpt-6-astra");
		expect(options.model.baseUrl).toBe("https://xi.example.invalid/v1");
		expect(await options.authStorage.getApiKey("xi")).toBe("offline-test-key");
	});
	it("retains every component above five and creates exactly one disabled full run each", () => {
		const names = Array.from({ length: 8 }, (_, index) => `component_${index}`);
		const plans = asrPlans(config, "/run", names);
		const profile = { mode: "direct_formal", kind: "research" } as const;
		expect(validatePreparePlan(plans.prepare)).toBeUndefined();
		expect(validateCodePlan(plans.code, names, profile)).toBeUndefined();
		expect(validateSciencePlan(plans.science, names, profile)).toBeUndefined();
		expect(plans.science.work_units).toHaveLength(9);
		expect(plans.science.work_units.slice(1).map((unit) => unit.disabled_components)).toEqual(
			names.map((name) => [name]),
		);
		plans.science.work_units.pop();
		expect(validateSciencePlan(plans.science, names, profile)).toMatch(/exactly one disabled/);
	});

	it("requires an explicit baseline profile and never manufactures smoke evidence", () => {
		const plans = asrPlans(config, "/run", []);
		const profile = { mode: "direct_formal", kind: "baseline" } as const;
		expect(validateCodePlan(plans.code, [], profile)).toBeUndefined();
		expect(validateSciencePlan(plans.science, [], profile)).toBeUndefined();
		expect(validateSciencePlan(plans.science, [])).toBeDefined();
		expect(plans.science.work_units).toHaveLength(1);
		expect(codeReviewRoles(profile)).toContain("static_integration");
		expect(codeReviewRoles(profile)).not.toContain("integration_smoke");
	});

	it("starts eight independent science work units without a three-task cap", async () => {
		const control = coordinator();
		const plan: ExperimentPlan = {
			stage: "science",
			work_units: Array.from({ length: 8 }, (_, index) => ({ id: String(index) })),
		};
		let started = 0;
		let release!: () => void;
		const gate = new Promise<void>((done) => {
			release = done;
		});
		await control.runPlanWorkUnits(
			plan,
			plan.work_units.map((workUnit) => ({
				workUnit,
				run: async () => {
					started++;
					if (started === 8) release();
					await gate;
				},
			})),
		);
		expect(started).toBe(8);
	});

	it("waits for sibling conditions and reports a failed cohort rather than publishing partial success", async () => {
		const control = coordinator();
		const plan: ExperimentPlan = { stage: "science", work_units: [{ id: "bad" }, { id: "good" }] };
		let preserved = false;
		await expect(
			control.runPlanWorkUnits(
				plan,
				plan.work_units.map((workUnit) => ({
					workUnit,
					run: async () => {
						if (workUnit.id === "bad") throw new Error("failed condition");
						await Promise.resolve();
						preserved = true;
					},
				})),
			),
		).rejects.toThrow("Science conditions remain incomplete");
		expect(preserved).toBe(true);
	});
});
