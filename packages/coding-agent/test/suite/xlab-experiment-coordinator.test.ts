import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentSession } from "../../src/core/agent-session.ts";
import type {
	AdvanceExperimentChildOptions,
	CreateExperimentChildOptions,
	ExperimentChildHandle,
	ExperimentChildRuntime,
} from "../../src/core/xlab/experiment/child-runtime.ts";
import {
	type ExperimentAssignmentSpec,
	ExperimentCoordinator,
	type ExperimentStageDriver,
	type ExperimentWorkUnitDriver,
} from "../../src/core/xlab/experiment/coordinator.ts";
import {
	CODE_REVIEW_ROLES,
	EXPERIMENT_STAGES,
	type ExperimentAssignment,
	type ExperimentChildIdentity,
	type ExperimentPlan,
	ExperimentProtocolStore,
	type ExperimentReviewReport,
	type ExperimentSubmission,
} from "../../src/core/xlab/experiment/protocol.ts";

function digest(value: string): string {
	return createHash("sha256").update(value).digest("hex");
}

class FakeChildRuntime {
	readonly opened: ExperimentAssignment[] = [];
	readonly released: string[] = [];
	readonly abortAll = vi.fn(async () => undefined);
	releaseError?: Error;
	onPrompt: (assignment: ExperimentAssignment, token: string, prompt: string) => Promise<void> = async () => undefined;

	constructor(private readonly protocol: ExperimentProtocolStore) {}

	createIdentity(options: CreateExperimentChildOptions): ExperimentChildIdentity {
		const identity: ExperimentChildIdentity = {
			child_id: options.child_id,
			session_id: options.child_id,
			session_file: join(options.cwd, `${options.child_id}.jsonl`),
			parent_run_id: options.parent_run_id,
			role: options.role,
			stage: options.stage,
			...(options.work_unit === undefined ? {} : { work_unit: options.work_unit }),
			execution_attempt: options.execution_attempt,
			generation: options.generation,
			review_round: options.review_round,
		};
		const owner = this.protocol.read().owner;
		this.protocol.registerChild(identity, owner && { owner_id: owner.owner_id, generation: owner.generation });
		return identity;
	}

	advanceIdentity(options: AdvanceExperimentChildOptions): ExperimentChildIdentity {
		const current = this.protocol.read().children[options.child_id];
		const identity = { ...current, ...options };
		const owner = this.protocol.read().owner;
		this.protocol.advanceChild(identity, owner && { owner_id: owner.owner_id, generation: owner.generation });
		return identity;
	}

	async openAssignment(assignment: ExperimentAssignment, token: string): Promise<ExperimentChildHandle> {
		this.opened.push(assignment);
		const session = {
			prompt: async (prompt: string) => this.onPrompt(assignment, token, prompt),
			abort: vi.fn(async () => undefined),
			dispose: vi.fn(),
		} as unknown as AgentSession;
		return { identity: assignment.child, assignment, session, resumed: this.opened.length > 1 };
	}

	release(childId: string): void {
		this.released.push(childId);
		if (this.releaseError) throw this.releaseError;
	}
}

function review(role: string, verdict: "PASS" | "FAIL" = "PASS"): ExperimentReviewReport {
	return {
		artifact_role: "reviewer_report",
		role,
		reviewer_kind: "agent",
		verdict,
		blocking: true,
		summary: `${role} ${verdict.toLowerCase()}`,
		checked_artifacts: ["result.json"],
		issues:
			verdict === "FAIL" ? [{ code: "repair", message: "repair", required_fix: "fix", evidence: "result" }] : [],
		structured_findings: { checked: true },
	};
}

function acceptedSubmission(
	assignment: ExperimentAssignment,
	token: string,
	result: Record<string, unknown>,
): ExperimentSubmission {
	return {
		schema_version: "xlab.experiment_submission.v1",
		submission_id: `${assignment.assignment_id}-submission`,
		assignment_id: assignment.assignment_id,
		assignment_token: token,
		kind: assignment.kind,
		child_id: assignment.child.child_id,
		role: assignment.child.role,
		stage: assignment.child.stage,
		...(assignment.child.work_unit === undefined ? {} : { work_unit: assignment.child.work_unit }),
		execution_attempt: assignment.child.execution_attempt,
		generation: assignment.child.generation,
		review_round: assignment.child.review_round,
		input_digest: assignment.input_digest,
		target_digest: assignment.target_digest,
		policy_digest: assignment.policy_digest,
		result,
		output_paths: [],
		submitted_at: "2026-08-08T00:00:00.000Z",
	};
}

describe("native experiment coordinator", () => {
	const cleanups: Array<() => void> = [];

	afterEach(() => {
		while (cleanups.length > 0) cleanups.pop()?.();
	});

	function fixture(tokenFactory: (assignmentId: string) => string = () => "token") {
		const root = mkdtempSync(join(tmpdir(), "xlab-coordinator-"));
		cleanups.push(() => rmSync(root, { recursive: true, force: true }));
		const runDir = join(root, "run");
		const cwd = join(root, "project");
		const agentDir = join(root, "agent");
		mkdirSync(cwd);
		mkdirSync(agentDir);
		const protocol = new ExperimentProtocolStore(runDir, () => "2026-08-08T00:00:00.000Z");
		const runtime = new FakeChildRuntime(protocol);
		const coordinator = new ExperimentCoordinator({
			runId: "run-1",
			runDir,
			cwd,
			agentDir,
			ideaDigest: digest("idea"),
			policyDigest: digest("policy"),
			ownerId: "owner-1",
			protocol,
			tokenFactory,
			childRuntime: runtime as unknown as ExperimentChildRuntime,
			clock: () => "2026-08-08T00:00:00.000Z",
		});
		return { runDir, protocol, runtime, coordinator };
	}

	function drivers(
		run: (stage: string, coordinator: ExperimentCoordinator) => Promise<void> = async () => undefined,
	): ExperimentStageDriver[] {
		return EXPERIMENT_STAGES.map((stage) => ({ stage, run: (coordinator) => run(stage, coordinator) }));
	}

	it("validates exact stage coverage before creating durable state", async () => {
		const { runDir, coordinator } = fixture();
		const duplicate = drivers();
		duplicate[3] = duplicate[2];
		await expect(coordinator.run(duplicate)).rejects.toThrow("exactly these stage drivers");
		expect(existsSync(join(runDir, "experiment"))).toBe(false);
	});

	it("runs shuffled drivers in canonical stage order", async () => {
		const { coordinator } = fixture();
		const order: string[] = [];
		const shuffled = drivers(async (stage) => {
			order.push(stage);
		}).reverse();
		const snapshot = await coordinator.run(shuffled);
		expect(order).toEqual(EXPERIMENT_STAGES);
		expect(snapshot.status).toBe("success");
		expect(snapshot.completed_stages).toEqual(EXPERIMENT_STAGES);
	});

	it("returns a pause or cancellation performed by a stage driver without completing the stage", async () => {
		const pausedFixture = fixture();
		const paused = await pausedFixture.coordinator.run(
			drivers(async (stage, coordinator) => {
				if (stage === "prepare") coordinator.pause("wait");
			}),
		);
		expect(paused.status).toBe("paused");
		expect(paused.completed_stages).toEqual([]);

		const cancelledFixture = fixture();
		const cancelled = await cancelledFixture.coordinator.run(
			drivers(async (stage, coordinator) => {
				if (stage === "prepare") await coordinator.cancel("stop");
			}),
		);
		expect(cancelled.status).toBe("cancelled");
		expect(cancelled.completed_stages).toEqual([]);
		expect(cancelledFixture.runtime.abortAll).toHaveBeenCalledOnce();
	});

	it("fences and aborts local children when a heartbeat loses ownership", async () => {
		vi.useFakeTimers();
		cleanups.push(() => vi.useRealTimers());
		const root = mkdtempSync(join(tmpdir(), "xlab-coordinator-heartbeat-"));
		cleanups.push(() => rmSync(root, { recursive: true, force: true }));
		const runDir = join(root, "run");
		const cwd = join(root, "project");
		const agentDir = join(root, "agent");
		mkdirSync(cwd);
		mkdirSync(agentDir);
		let now = "2026-08-08T00:00:00.000Z";
		const clock = () => now;
		const protocol = new ExperimentProtocolStore(runDir, clock);
		const runtime = new FakeChildRuntime(protocol);
		const coordinator = new ExperimentCoordinator({
			runId: "run-1",
			runDir,
			cwd,
			agentDir,
			ideaDigest: digest("idea"),
			policyDigest: digest("policy"),
			ownerId: "owner-1",
			protocol,
			childRuntime: runtime as unknown as ExperimentChildRuntime,
			clock,
			leaseDurationMs: 30,
		});
		let releaseStage!: () => void;
		const stageBlocked = new Promise<void>((resolve) => {
			releaseStage = resolve;
		});
		const execution = coordinator.run(
			drivers(async (stage) => {
				if (stage === "prepare") await stageBlocked;
			}),
		);
		await vi.advanceTimersByTimeAsync(0);
		now = "2026-08-08T00:00:00.031Z";
		protocol.acquireOwner("owner-2", "2026-08-08T00:00:01.000Z");
		await vi.advanceTimersByTimeAsync(10);
		releaseStage();
		await expect(execution).rejects.toThrow("lost ownership");
		expect(runtime.abortAll).toHaveBeenCalledOnce();
		expect(protocol.read().owner?.owner_id).toBe("owner-2");
	});

	it("returns durable cancellation when abort fan-out fails", async () => {
		const { coordinator, protocol, runtime } = fixture();
		coordinator.initialize();
		runtime.abortAll.mockRejectedValueOnce(new Error("abort failed"));
		const cancelled = await coordinator.cancel("stop");
		expect(cancelled.status).toBe("cancelled");
		expect(cancelled.terminal_generation).toBeDefined();
		expect(protocol.read()).toEqual(cancelled);
		expect(runtime.abortAll).toHaveBeenCalledOnce();
	});

	it("schedules validated work units deterministically after their dependencies", async () => {
		const { coordinator } = fixture();
		const plan: ExperimentPlan = {
			stage: "code",
			work_units: [{ id: "a" }, { id: "b", needs: ["a"] }, { id: "c", needs: ["a"] }],
		};
		const order: string[] = [];
		const unitDrivers: ExperimentWorkUnitDriver[] = plan.work_units.map((workUnit) => ({
			workUnit,
			run: async (_coordinator, unit) => {
				order.push(unit.id);
			},
		}));
		await coordinator.runPlanWorkUnits(plan, unitDrivers.reverse());
		expect(order).toEqual(["a", "b", "c"]);

		const cyclic = {
			...plan,
			work_units: [
				{ id: "a", needs: ["b"] },
				{ id: "b", needs: ["a"] },
			],
		};
		await expect(
			coordinator.runPlanWorkUnits(
				cyclic,
				cyclic.work_units.map((workUnit) => ({
					workUnit,
					run: async () => undefined,
				})),
			),
		).rejects.toThrow("cyclic or references an unknown unit");
	});

	it("runs a fresh full review matrix after repairing the exact worker", async () => {
		const { coordinator, protocol, runtime } = fixture();
		coordinator.initialize();
		coordinator.acquireOwnership();
		const owner = protocol.read().owner!;
		protocol.start("prepare", { owner_id: owner.owner_id, generation: owner.generation });
		protocol.completeStage("prepare", { owner_id: owner.owner_id, generation: owner.generation });
		protocol.start("code", { owner_id: owner.owner_id, generation: owner.generation });
		const worker = (reviewRound: number): ExperimentAssignmentSpec => ({
			assignmentId: `assignment-worker-code-unit-r${reviewRound}`,
			childId: "worker-code-unit",
			role: "worker",
			stage: "code",
			workUnit: "unit",
			kind: "worker_result",
			executionAttempt: reviewRound + 1,
			generation: reviewRound + 1,
			reviewRound,
			inputDigest: digest(`input-${reviewRound}`),
			targetDigest: digest("target"),
			readScope: [],
			writeScope: [],
			prompt: reviewRound === 0 ? "implement" : "repair",
		});
		const reviewerRounds = new Map<number, string[]>();
		runtime.onPrompt = async (assignment, token, prompt) => {
			if (assignment.child.role === "worker") {
				if (assignment.child.review_round > 0) {
					expect(assignment.child.session_id).toBe("worker-code-unit");
					expect(prompt).toContain("structured reviewer issues");
				}
				expect(
					protocol.acceptSubmission(
						acceptedSubmission(assignment, token, {
							artifact_role: "worker_result",
							review_round: assignment.child.review_round,
						}),
					),
				).toMatchObject({ ok: true });
				return;
			}
			const round = assignment.child.review_round;
			const role = CODE_REVIEW_ROLES.find((candidate) =>
				assignment.child.child_id.includes(`-${candidate}-r${round}-`),
			)!;
			reviewerRounds.set(round, [...(reviewerRounds.get(round) ?? []), role]);
			const verdict = round === 1 && role === "implementation_correctness" ? "FAIL" : "PASS";
			expect(
				protocol.acceptSubmission(
					acceptedSubmission(assignment, token, review(role, verdict) as unknown as Record<string, unknown>),
				),
			).toMatchObject({ ok: true });
		};

		const result = await coordinator.runReviewedWorker({
			worker: worker(0),
			maxReviewRounds: 2,
			review: (accepted, reviewRound) => ({
				stage: "code",
				workUnit: "unit",
				inputDigest: accepted.accepted.result_digest,
				readScope: [],
				prompt: (role) => `review ${role} round ${reviewRound}`,
			}),
			repair: (_previous, _issues, reviewRound) => worker(reviewRound),
		});

		expect(result.reviewRound).toBe(2);
		expect(result.worker.submission.review_round).toBe(1);
		expect(reviewerRounds.get(1)).toEqual(CODE_REVIEW_ROLES);
		expect(reviewerRounds.get(2)).toEqual(CODE_REVIEW_ROLES);
		const workerAssignments = runtime.opened.filter((assignment) => assignment.child.role === "worker");
		expect(workerAssignments).toHaveLength(2);
		expect(new Set(workerAssignments.map((assignment) => assignment.child.session_file)).size).toBe(1);
		const reviewerIds = runtime.opened
			.filter((assignment) => assignment.child.role === "reviewer")
			.map((assignment) => assignment.child.child_id);
		expect(new Set(reviewerIds).size).toBe(CODE_REVIEW_ROLES.length * 2);
	});

	it("rejects repair that does not advance the exact accepted worker", async () => {
		const { coordinator, protocol, runtime } = fixture();
		coordinator.initialize();
		coordinator.acquireOwnership();
		const owner = protocol.read().owner!;
		protocol.start("prepare", { owner_id: owner.owner_id, generation: owner.generation });
		protocol.completeStage("prepare", { owner_id: owner.owner_id, generation: owner.generation });
		protocol.start("code", { owner_id: owner.owner_id, generation: owner.generation });
		const initial: ExperimentAssignmentSpec = {
			assignmentId: "assignment-worker-code-unit-r0",
			childId: "worker-code-unit",
			role: "worker",
			stage: "code",
			workUnit: "unit",
			kind: "worker_result",
			executionAttempt: 1,
			generation: 1,
			reviewRound: 0,
			inputDigest: digest("input"),
			targetDigest: digest("target"),
			readScope: [],
			writeScope: [],
			prompt: "implement",
		};
		runtime.onPrompt = async (assignment, token) => {
			expect(
				protocol.acceptSubmission(
					acceptedSubmission(assignment, token, { artifact_role: "worker_result", status: "success" }),
				),
			).toMatchObject({ ok: true });
		};
		await coordinator.executeAssignment(initial);
		const issue = review("implementation_correctness", "FAIL").issues;
		await expect(
			coordinator.repairWorker(
				initial.assignmentId,
				{
					...initial,
					assignmentId: "assignment-worker-code-unit-r1-replacement",
					childId: "replacement-worker",
					executionAttempt: 2,
					generation: 2,
					reviewRound: 1,
				},
				issue,
			),
		).rejects.toThrow("exact prior worker identity");
		await expect(
			coordinator.repairWorker(
				initial.assignmentId,
				{
					...initial,
					assignmentId: "assignment-worker-code-unit-r1-stale",
					reviewRound: 1,
				},
				issue,
			),
		).rejects.toThrow("advance assignment, attempt, generation, and review round");
		await expect(
			coordinator.repairWorker(
				"unknown-assignment",
				{
					...initial,
					assignmentId: "assignment-worker-code-unit-r1-unknown",
					executionAttempt: 2,
					generation: 2,
					reviewRound: 1,
				},
				issue,
			),
		).rejects.toThrow("accepted prior assignment unknown-assignment");
	});

	it("resumes an active assignment with the same derived token after restart", async () => {
		const root = mkdtempSync(join(tmpdir(), "xlab-coordinator-restart-"));
		cleanups.push(() => rmSync(root, { recursive: true, force: true }));
		const runDir = join(root, "run");
		const cwd = join(root, "project");
		const agentDir = join(root, "agent");
		mkdirSync(cwd);
		mkdirSync(agentDir);
		const spec: ExperimentAssignmentSpec = {
			assignmentId: "assignment-worker-prepare-restart",
			childId: "worker-prepare-restart",
			role: "worker",
			stage: "prepare",
			workUnit: "restart",
			kind: "worker_result",
			executionAttempt: 1,
			generation: 1,
			reviewRound: 0,
			inputDigest: digest("input"),
			targetDigest: digest("target"),
			readScope: [],
			writeScope: [],
			prompt: "work",
		};
		const createCoordinator = () => {
			const protocol = new ExperimentProtocolStore(runDir, () => "2026-08-08T00:00:00.000Z");
			const runtime = new FakeChildRuntime(protocol);
			const coordinator = new ExperimentCoordinator({
				runId: "run-1",
				runDir,
				cwd,
				agentDir,
				ideaDigest: digest("idea"),
				policyDigest: digest("policy"),
				ownerId: "owner-1",
				protocol,
				childRuntime: runtime as unknown as ExperimentChildRuntime,
				clock: () => "2026-08-08T00:00:00.000Z",
			});
			return { protocol, runtime, coordinator };
		};

		const first = createCoordinator();
		first.coordinator.initialize();
		first.coordinator.acquireOwnership();
		first.protocol.start(
			"prepare",
			first.protocol.read().owner && {
				owner_id: first.protocol.read().owner!.owner_id,
				generation: first.protocol.read().owner!.generation,
			},
		);
		let firstToken = "";
		first.runtime.onPrompt = async (_assignment, token) => {
			firstToken = token;
		};
		await expect(first.coordinator.executeAssignment(spec)).rejects.toThrow("completed without submitting");

		const restarted = createCoordinator();
		let restartedToken = "";
		restarted.runtime.onPrompt = async (assignment, token) => {
			restartedToken = token;
			expect(
				restarted.protocol.acceptSubmission(
					acceptedSubmission(assignment, token, { artifact_role: "worker_result", status: "success" }),
				),
			).toMatchObject({ ok: true });
		};
		const accepted = await restarted.coordinator.executeAssignment(spec);
		expect(accepted.accepted.assignment_id).toBe(spec.assignmentId);
		expect(restartedToken).toBe(firstToken);
		expect(restarted.runtime.opened[0].child.session_file).toBe(first.runtime.opened[0].child.session_file);
	});

	it("uses durable acceptance as completion and replays it without regenerating a token", async () => {
		let tokenCalls = 0;
		const { coordinator, protocol, runtime } = fixture(() => {
			tokenCalls += 1;
			if (tokenCalls > 1) throw new Error("token must not be regenerated");
			return "token";
		});
		coordinator.initialize();
		coordinator.acquireOwnership();
		protocol.start(
			"prepare",
			protocol.read().owner && {
				owner_id: protocol.read().owner!.owner_id,
				generation: protocol.read().owner!.generation,
			},
		);
		const spec: ExperimentAssignmentSpec = {
			assignmentId: "assignment-worker-prepare-a",
			childId: "worker-prepare-a",
			role: "worker",
			stage: "prepare",
			workUnit: "a",
			kind: "worker_result",
			executionAttempt: 1,
			generation: 1,
			reviewRound: 0,
			inputDigest: digest("input"),
			targetDigest: digest("target"),
			readScope: [],
			writeScope: [],
			prompt: "work",
		};
		runtime.onPrompt = async (assignment, token) => {
			expect(
				protocol.acceptSubmission(
					acceptedSubmission(assignment, token, { artifact_role: "worker_result", status: "success" }),
				),
			).toMatchObject({ ok: true });
			throw new Error("prompt transport failed after submission");
		};
		runtime.releaseError = new Error("dispose failed");
		const first = await coordinator.executeAssignment(spec);
		const replay = await coordinator.executeAssignment(spec);
		expect(replay).toEqual(first);
		expect(tokenCalls).toBe(1);
		expect(runtime.opened).toHaveLength(1);
		expect(runtime.released).toEqual([spec.childId]);
	});

	it("preserves prompt and missing-submission errors when child cleanup fails", async () => {
		for (const promptError of [new Error("prompt failed"), undefined]) {
			const { coordinator, protocol, runtime } = fixture();
			coordinator.initialize();
			coordinator.acquireOwnership();
			const owner = protocol.read().owner!;
			protocol.start("prepare", { owner_id: owner.owner_id, generation: owner.generation });
			runtime.releaseError = new Error("dispose failed");
			runtime.onPrompt = async () => {
				if (promptError) throw promptError;
			};
			const execution = coordinator.executeAssignment({
				assignmentId: `assignment-planner-prepare-${promptError ? "prompt" : "missing"}`,
				childId: `planner-prepare-${promptError ? "prompt" : "missing"}`,
				role: "planner",
				stage: "prepare",
				kind: "plan",
				executionAttempt: 1,
				generation: 1,
				reviewRound: 0,
				inputDigest: digest("input"),
				targetDigest: digest("target"),
				readScope: [],
				writeScope: [],
				prompt: "plan",
			});
			await expect(execution).rejects.toThrow(promptError ? "prompt failed" : "completed without submitting");
		}
	});

	it("fails when a child prompt returns without a durable submission", async () => {
		const { coordinator, protocol } = fixture();
		coordinator.initialize();
		coordinator.acquireOwnership();
		const owner = protocol.read().owner!;
		protocol.start("prepare", { owner_id: owner.owner_id, generation: owner.generation });
		await expect(
			coordinator.executeAssignment({
				assignmentId: "assignment-planner-prepare",
				childId: "planner-prepare",
				role: "planner",
				stage: "prepare",
				kind: "plan",
				executionAttempt: 1,
				generation: 1,
				reviewRound: 0,
				inputDigest: digest("input"),
				targetDigest: digest("target"),
				readScope: [],
				writeScope: [],
				prompt: "plan",
			}),
		).rejects.toThrow("completed without submitting");
	});
});
