import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentSession } from "../../src/core/agent-session.ts";
import type { CreateAgentSessionOptions } from "../../src/core/sdk.ts";
import { ExperimentChildRuntime } from "../../src/core/xlab/experiment/child-runtime.ts";
import {
	type ExperimentAssignment,
	type ExperimentChildIdentity,
	ExperimentProtocolStore,
} from "../../src/core/xlab/experiment/protocol.ts";

function digest(value: string): string {
	return createHash("sha256").update(value).digest("hex");
}

function assignment(
	identity: ExperimentChildIdentity,
	policyDigest: string,
	kind: ExperimentAssignment["kind"],
	readScope: string[] = [],
	writeScope: string[] = [],
): Omit<ExperimentAssignment, "assignment_token_digest" | "status" | "created_at" | "updated_at"> {
	return {
		schema_version: "xlab.experiment_assignment.v1",
		assignment_id: `assignment-${identity.child_id}`,
		child: identity,
		kind,
		input_digest: digest("input"),
		target_digest: digest("target"),
		policy_digest: policyDigest,
		read_scope: readScope,
		write_scope: writeScope,
	};
}

describe("native experiment child runtime", () => {
	const cleanups: Array<() => void> = [];

	afterEach(() => {
		while (cleanups.length > 0) {
			cleanups.pop()?.();
		}
	});

	function fixture() {
		const root = mkdtempSync(join(tmpdir(), "xlab-child-runtime-"));
		cleanups.push(() => rmSync(root, { recursive: true, force: true }));
		const runDir = join(root, "run");
		const cwd = join(root, "project");
		const agentDir = join(root, "agent");
		mkdirSync(cwd, { recursive: true });
		mkdirSync(agentDir, { recursive: true });
		const policyDigest = digest("policy");
		const protocol = new ExperimentProtocolStore(runDir);
		protocol.initialize({ runId: "run-1", ideaDigest: digest("idea"), policyDigest });
		return { root, runDir, cwd, agentDir, policyDigest, protocol };
	}

	it("persists an explicit child session before registration and reopens it exactly", async () => {
		const { runDir, cwd, agentDir, policyDigest, protocol } = fixture();
		const seen: CreateAgentSessionOptions[] = [];
		const fakeSession = { abort: vi.fn(), dispose: vi.fn() } as unknown as AgentSession;
		const runtime = new ExperimentChildRuntime({
			runDir,
			agentDir,
			protocol,
			parentSession: "parent-session",
			createSession: (async (options: CreateAgentSessionOptions) => {
				seen.push(options);
				return {
					session: fakeSession,
					extensionsResult: { extensions: [], errors: [], runtime: undefined },
				} as never;
			}) as never,
		});
		const identity = runtime.createIdentity({
			child_id: "worker-code-integration",
			parent_run_id: "run-1",
			role: "worker",
			stage: "code",
			work_unit: "final_integration_smoke",
			execution_attempt: 1,
			generation: 1,
			review_round: 0,
			cwd,
		});
		expect(protocol.read().children[identity.child_id]).toEqual(identity);
		const record = protocol.createAssignment(assignment(identity, policyDigest, "worker_result"), "token");

		const opened = await runtime.openAssignment(record, "token");
		expect(opened.identity.session_id).toBe("worker-code-integration");
		expect(opened.resumed).toBe(false);
		expect(seen[0].sessionManager?.getSessionFile()).toBe(identity.session_file);
		expect(seen[0].resourceLoader).toBeDefined();
		expect(seen[0].tools).toEqual(["submit_worker_result"]);
	});

	it("isolates role capabilities and accepts only the bound structured submission", async () => {
		const { runDir, cwd, agentDir, policyDigest, protocol } = fixture();
		const readable = join(cwd, "input.txt");
		writeFileSync(readable, "input");
		let captured: CreateAgentSessionOptions | undefined;
		const fakeSession = { abort: vi.fn(), dispose: vi.fn() } as unknown as AgentSession;
		const runtime = new ExperimentChildRuntime({
			runDir,
			agentDir,
			protocol,
			clock: () => "2026-08-08T00:00:00.000Z",
			createSession: (async (options: CreateAgentSessionOptions) => {
				captured = options;
				return {
					session: fakeSession,
					extensionsResult: { extensions: [], errors: [], runtime: undefined },
				} as never;
			}) as never,
		});
		const identity = runtime.createIdentity({
			child_id: "reviewer-code-idea-alignment-r1",
			parent_run_id: "run-1",
			role: "reviewer",
			stage: "code",
			execution_attempt: 1,
			generation: 1,
			review_round: 1,
			cwd,
		});
		const record = protocol.createAssignment(
			{
				...assignment(identity, policyDigest, "review", [cwd], []),
				result_validation: { reviewer_role: "idea_alignment" },
			},
			"token",
		);
		await runtime.openAssignment(record, "token");

		expect(captured?.tools).toEqual(["read", "submit_review"]);
		expect(captured?.tools).not.toEqual(expect.arrayContaining(["write", "edit", "bash", "xlab_finish"]));
		const submit = captured?.customTools?.find((tool) => tool.name === "submit_review");
		expect(submit).toBeDefined();
		const accepted = await submit!.execute(
			"call-1",
			{
				result: {
					artifact_role: "reviewer_report",
					role: "idea_alignment",
					reviewer_kind: "agent",
					verdict: "PASS",
					blocking: true,
					summary: "passed",
					checked_artifacts: ["input.txt"],
					issues: [],
					structured_findings: { checked: true },
				},
				output_paths: [],
			},
			undefined,
			undefined,
			undefined as never,
		);
		expect(accepted.details).toMatchObject({ accepted: true, disposition: "accepted" });
		expect(protocol.read().assignments[record.assignment_id].status).toBe("accepted");
	});

	it("reopens the exact worker session and rejects duplicate live ownership", async () => {
		const { runDir, cwd, agentDir, policyDigest, protocol } = fixture();
		const sessions = [
			{ abort: vi.fn(), dispose: vi.fn() },
			{ abort: vi.fn(), dispose: vi.fn() },
		] as unknown as AgentSession[];
		let index = 0;
		const runtime = new ExperimentChildRuntime({
			runDir,
			agentDir,
			protocol,
			createSession: (async () => ({
				session: sessions[index++],
				extensionsResult: { extensions: [], errors: [], runtime: undefined },
			})) as never,
		});
		const identity = runtime.createIdentity({
			child_id: "worker-code-exact",
			parent_run_id: "run-1",
			role: "worker",
			stage: "code",
			work_unit: "implementation",
			execution_attempt: 1,
			generation: 1,
			review_round: 0,
			cwd,
		});
		const record = protocol.createAssignment(assignment(identity, policyDigest, "worker_result"), "token");

		const first = await runtime.openAssignment(record, "token");
		expect(first.identity.session_id).toBe(identity.session_id);
		await expect(runtime.openAssignment(record, "token")).rejects.toThrow("already open");
		runtime.release(identity.child_id);
		const advanced = runtime.advanceIdentity({
			child_id: identity.child_id,
			execution_attempt: 2,
			generation: 2,
			review_round: 1,
		});
		expect(advanced).toMatchObject({
			session_id: identity.session_id,
			session_file: identity.session_file,
			execution_attempt: 2,
			generation: 2,
			review_round: 1,
		});
		const repair = protocol.createAssignment(
			{
				...assignment(advanced, policyDigest, "worker_result"),
				assignment_id: "assignment-worker-code-exact-repair-2",
			},
			"repair-token",
		);
		const reopened = await runtime.openAssignment(repair, "repair-token");
		expect(reopened.identity.session_file).toBe(identity.session_file);
		expect(index).toBe(2);
	});

	it("marks durable sessions with prior context as resumed", async () => {
		const { runDir, cwd, agentDir, policyDigest, protocol } = fixture();
		const fakeSession = { abort: vi.fn(), dispose: vi.fn() } as unknown as AgentSession;
		const runtime = new ExperimentChildRuntime({
			runDir,
			agentDir,
			protocol,
			createSession: (async () => ({
				session: fakeSession,
				extensionsResult: { extensions: [], errors: [], runtime: undefined },
			})) as never,
		});
		const identity = runtime.createIdentity({
			child_id: "planner-resumed",
			parent_run_id: "run-1",
			role: "planner",
			stage: "prepare",
			execution_attempt: 1,
			generation: 1,
			review_round: 0,
			cwd,
		});
		const manager = (await import("../../src/core/session-manager.ts")).SessionManager.open(identity.session_file);
		manager.appendCustomMessageEntry("xlab.experiment.assignment", "prior assignment context", false);
		const record = protocol.createAssignment(assignment(identity, policyDigest, "plan"), "token");

		const opened = await runtime.openAssignment(record, "token");
		expect(opened.resumed).toBe(true);
	});

	it("enforces worker path scopes and rejects symlink escapes", async () => {
		const { root, runDir, cwd, agentDir, policyDigest, protocol } = fixture();
		const outside = join(root, "outside");
		mkdirSync(outside);
		writeFileSync(join(cwd, "input.txt"), "input");
		writeFileSync(join(outside, "secret.txt"), "secret");
		symlinkSync(join(outside, "secret.txt"), join(cwd, "escape.txt"));
		let captured: CreateAgentSessionOptions | undefined;
		const fakeSession = { abort: vi.fn(), dispose: vi.fn() } as unknown as AgentSession;
		const runtime = new ExperimentChildRuntime({
			runDir,
			agentDir,
			protocol,
			createSession: (async (options: CreateAgentSessionOptions) => {
				captured = options;
				return {
					session: fakeSession,
					extensionsResult: { extensions: [], errors: [], runtime: undefined },
				} as never;
			}) as never,
		});
		const identity = runtime.createIdentity({
			child_id: "worker-code-scoped",
			parent_run_id: "run-1",
			role: "worker",
			stage: "code",
			work_unit: "implementation",
			execution_attempt: 1,
			generation: 1,
			review_round: 0,
			cwd,
		});
		const record = protocol.createAssignment(
			assignment(identity, policyDigest, "worker_result", [cwd], [cwd]),
			"token",
		);
		await runtime.openAssignment(record, "token");
		const read = captured?.customTools?.find((tool) => tool.name === "read");
		const write = captured?.customTools?.find((tool) => tool.name === "write");

		await expect(
			read!.execute("read-escape", { path: "escape.txt" }, undefined, undefined, undefined as never),
		).rejects.toThrow("outside the assignment scope");
		await expect(
			write!.execute(
				"write-escape",
				{ path: "../outside/new.txt", content: "no" },
				undefined,
				undefined,
				undefined as never,
			),
		).rejects.toThrow("outside the assignment scope");
	});

	it("rejects invalid results and late submissions after cancellation", async () => {
		const { runDir, cwd, agentDir, policyDigest, protocol } = fixture();
		let captured: CreateAgentSessionOptions | undefined;
		const fakeSession = { abort: vi.fn(), dispose: vi.fn() } as unknown as AgentSession;
		const runtime = new ExperimentChildRuntime({
			runDir,
			agentDir,
			protocol,
			validateResult: (_assignment, result) =>
				result && typeof result === "object" ? undefined : "Result must be structured.",
			createSession: (async (options: CreateAgentSessionOptions) => {
				captured = options;
				return {
					session: fakeSession,
					extensionsResult: { extensions: [], errors: [], runtime: undefined },
				} as never;
			}) as never,
		});
		const identity = runtime.createIdentity({
			child_id: "reviewer-code-late",
			parent_run_id: "run-1",
			role: "reviewer",
			stage: "code",
			execution_attempt: 1,
			generation: 1,
			review_round: 1,
			cwd,
		});
		const record = protocol.createAssignment(assignment(identity, policyDigest, "review", [cwd]), "token");
		await runtime.openAssignment(record, "token");
		const submit = captured?.customTools?.find((tool) => tool.name === "submit_review");

		const invalid = await submit!.execute("invalid", { result: "bad" }, undefined, undefined, undefined as never);
		expect(invalid.details).toMatchObject({ accepted: false, error: "Result must be structured." });
		protocol.cancel("stop");
		const late = await submit!.execute(
			"late",
			{ result: { verdict: "PASS" } },
			undefined,
			undefined,
			undefined as never,
		);
		expect(late.details).toMatchObject({ accepted: false });
		expect(late.details.error).toContain("cancelled");
	});

	it("reports targeted abort ownership", async () => {
		const { runDir, cwd, agentDir, policyDigest, protocol } = fixture();
		const abort = vi.fn(async () => undefined);
		const fakeSession = { abort, dispose: vi.fn() } as unknown as AgentSession;
		const runtime = new ExperimentChildRuntime({
			runDir,
			agentDir,
			protocol,
			createSession: (async () => ({
				session: fakeSession,
				extensionsResult: { extensions: [], errors: [], runtime: undefined },
			})) as never,
		});
		const identity = runtime.createIdentity({
			child_id: "planner-abort",
			parent_run_id: "run-1",
			role: "planner",
			stage: "prepare",
			execution_attempt: 1,
			generation: 1,
			review_round: 0,
			cwd,
		});
		const record = protocol.createAssignment(assignment(identity, policyDigest, "plan"), "token");
		await runtime.openAssignment(record, "token");

		expect(await runtime.abortChild("missing")).toBe(false);
		expect(await runtime.abortChild(identity.child_id)).toBe(true);
		expect(abort).toHaveBeenCalledOnce();
	});

	it("fans cancellation out to live children and releases without deleting durable identity", async () => {
		const { runDir, cwd, agentDir, policyDigest, protocol } = fixture();
		const abort = vi.fn(async () => undefined);
		const dispose = vi.fn();
		const fakeSession = { abort, dispose } as unknown as AgentSession;
		const runtime = new ExperimentChildRuntime({
			runDir,
			agentDir,
			protocol,
			createSession: (async () => ({
				session: fakeSession,
				extensionsResult: { extensions: [], errors: [], runtime: undefined },
			})) as never,
		});
		const identity = runtime.createIdentity({
			child_id: "planner-code",
			parent_run_id: "run-1",
			role: "planner",
			stage: "code",
			execution_attempt: 1,
			generation: 1,
			review_round: 0,
			cwd,
		});
		const record = protocol.createAssignment(assignment(identity, policyDigest, "plan"), "token");
		await runtime.openAssignment(record, "token");

		await runtime.abortAll();
		expect(abort).toHaveBeenCalledOnce();
		runtime.release(identity.child_id);
		expect(dispose).toHaveBeenCalledOnce();
		expect(protocol.read().children[identity.child_id]).toEqual(identity);
	});
});
