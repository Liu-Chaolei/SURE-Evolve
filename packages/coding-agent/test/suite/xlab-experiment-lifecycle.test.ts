import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { XlabIdeaBinding } from "../../src/core/xlab/experiment/idea-binding.ts";
import {
	createXlabExperimentStagingMaterializer,
	NATIVE_EXPERIMENT_POLICY_DIGEST,
	XlabExperimentLifecycle,
} from "../../src/core/xlab/experiment/lifecycle.ts";
import { ExperimentProtocolStore } from "../../src/core/xlab/experiment/protocol.ts";
import { XlabRunManager } from "../../src/core/xlab/run-manager.ts";
import type { XlabSkillPackage } from "../../src/core/xlab/types.ts";

const clock = () => "2026-08-08T00:00:00.000Z";

function ideaBinding(): XlabIdeaBinding {
	const canonicalBytes = Buffer.from(
		'{"schema_version":"xlab.experiment_idea.v1","components":[{"component":"a"}]}\n',
	);
	return {
		canonical: JSON.parse(canonicalBytes.toString()),
		canonicalBytes,
		metadata: {
			schema_version: "xlab.idea_binding.v1",
			resolution_method: "path",
			reference: "idea.json",
			source_schema_version: "xlab.research_idea.result.v2",
			source_digest: "source",
			canonical_digest: createHash("sha256").update(canonicalBytes).digest("hex"),
		},
		warnings: [],
	};
}

function skill(root: string): XlabSkillPackage {
	return {
		manifest: {
			schemaVersion: "2",
			name: "run_experiment",
			version: "1.0.0",
			visibility: "public",
			prompt: "SKILL.md",
			runtime: { kind: "native", driver: "experiment" },
			workflow: { stages: ["prepare", "code", "science", "finalize"].map((id) => ({ id, skill: id })) },
		},
		manifestPath: join(root, "xlab.skill.json"),
		packageDir: root,
		promptPath: join(root, "SKILL.md"),
		prompt: "run",
		source: "project",
		sourceRoot: root,
		dependencies: [],
	};
}

describe("native experiment lifecycle", () => {
	const cleanups: string[] = [];
	afterEach(() => {
		for (const root of cleanups.splice(0)) rmSync(root, { recursive: true, force: true });
	});

	it("publishes Idea, policy, and protocol in one atomic staging rename", () => {
		const root = mkdtempSync(join(tmpdir(), "xlab-native-publication-"));
		cleanups.push(root);
		const manager = new XlabRunManager(root);
		const binding = ideaBinding();
		const record = manager.createRun(skill(root), "--idea idea.json", {
			ideaBinding: binding,
			materializeStaging: createXlabExperimentStagingMaterializer(binding, clock),
		});
		expect(existsSync(join(record.runDir, "experiment", "policy.json"))).toBe(true);
		const snapshot = new ExperimentProtocolStore(record.runDir, clock).read();
		expect(snapshot).toMatchObject({
			run_id: record.runId,
			idea_digest: binding.metadata.canonical_digest,
			policy_digest: NATIVE_EXPERIMENT_POLICY_DIGEST,
			status: "pending",
		});
	});

	it("does not publish a run when staging materialization fails", () => {
		const root = mkdtempSync(join(tmpdir(), "xlab-native-failure-"));
		cleanups.push(root);
		const manager = new XlabRunManager(root);
		expect(() =>
			manager.createRun(skill(root), "", {
				materializeStaging: () => {
					throw new Error("materialization failed");
				},
			}),
		).toThrow("materialization failed");
		expect(manager.listRuns()).toEqual([]);
	});

	it("reconstructs durable state, verifies immutable digests, and projects pause/cancel", async () => {
		const root = mkdtempSync(join(tmpdir(), "xlab-native-lifecycle-"));
		cleanups.push(root);
		mkdirSync(join(root, "agent"));
		const manager = new XlabRunManager(root);
		const binding = ideaBinding();
		const record = manager.createRun(skill(root), "--idea idea.json", {
			ideaBinding: binding,
			materializeStaging: createXlabExperimentStagingMaterializer(binding, clock),
		});
		const abortAll = vi.fn(async () => undefined);
		const lifecycle = new XlabExperimentLifecycle({
			cwd: root,
			agentDir: join(root, "agent"),
			runId: record.runId,
			ownerId: "owner",
			runManager: manager,
			clock,
			coordinatorOptions: { childRuntime: { abortAll } as never },
		});
		const reconstructed = lifecycle.reconstruct();
		reconstructed.coordinator.pause("wait");
		expect(lifecycle.status().status).toBe("paused");
		expect((await lifecycle.cancel("stop")).status).toBe("cancelled");
		expect(abortAll).toHaveBeenCalledOnce();

		writeFileSync(join(record.runDir, "inputs", "idea.json"), "{}\n");
		expect(() => lifecycle.reconstruct()).toThrow("immutable run inputs");
	});

	it("keeps status read-only and preserves a newer terminal projection", async () => {
		const root = mkdtempSync(join(tmpdir(), "xlab-native-status-"));
		cleanups.push(root);
		mkdirSync(join(root, "agent"));
		const manager = new XlabRunManager(root);
		const binding = ideaBinding();
		const record = manager.createRun(skill(root), "--idea idea.json", {
			ideaBinding: binding,
			materializeStaging: createXlabExperimentStagingMaterializer(binding, clock),
		});
		const lifecycle = new XlabExperimentLifecycle({
			cwd: root,
			agentDir: join(root, "agent"),
			runId: record.runId,
			ownerId: "owner-status",
			runManager: manager,
			clock,
			coordinatorOptions: { childRuntime: { abortAll: vi.fn(async () => undefined) } as never },
		});
		const runPath = join(record.runDir, "run.json");
		const eventsPath = join(record.runDir, "events.jsonl");
		const beforeRun = readFileSync(runPath, "utf-8");
		const beforeEvents = readFileSync(eventsPath, "utf-8");
		expect(lifecycle.status().status).toBe("pending");
		expect(readFileSync(runPath, "utf-8")).toBe(beforeRun);
		expect(readFileSync(eventsPath, "utf-8")).toBe(beforeEvents);

		const cancelled = await lifecycle.cancel("stop");
		const staleFailure = manager.updateRun(record, { status: "failed" }, "late_failure");
		expect(cancelled.status).toBe("cancelled");
		expect(staleFailure.status).toBe("cancelled");
		expect(manager.readRun(record.runId)?.status).toBe("cancelled");
	});

	it("rejects resume while a running owner lease is live", () => {
		const root = mkdtempSync(join(tmpdir(), "xlab-native-live-owner-"));
		cleanups.push(root);
		mkdirSync(join(root, "agent"));
		const manager = new XlabRunManager(root);
		const binding = ideaBinding();
		const record = manager.createRun(skill(root), "--idea idea.json", {
			ideaBinding: binding,
			materializeStaging: createXlabExperimentStagingMaterializer(binding, clock),
		});
		const lifecycle = new XlabExperimentLifecycle({
			cwd: root,
			agentDir: join(root, "agent"),
			runId: record.runId,
			ownerId: "owner-a",
			runManager: manager,
			clock,
		});
		const state = lifecycle.reconstruct();
		state.coordinator.acquireOwnership();
		state.protocol.start("prepare", state.coordinator.acquireOwnership());
		const contender = new XlabExperimentLifecycle({
			cwd: root,
			agentDir: join(root, "agent"),
			runId: record.runId,
			ownerId: "owner-b",
			runManager: manager,
			clock,
		});
		expect(contender.resume([])).rejects.toThrow("live owner");
	});

	it("resumes an interrupted running run after its owner lease expires", async () => {
		const root = mkdtempSync(join(tmpdir(), "xlab-native-expired-owner-"));
		cleanups.push(root);
		mkdirSync(join(root, "agent"));
		const manager = new XlabRunManager(root);
		const binding = ideaBinding();
		const record = manager.createRun(skill(root), "--idea idea.json", {
			ideaBinding: binding,
			materializeStaging: createXlabExperimentStagingMaterializer(binding, clock),
		});
		let now = "2026-08-08T00:00:00.000Z";
		const mutableClock = () => now;
		const first = new XlabExperimentLifecycle({
			cwd: root,
			agentDir: join(root, "agent"),
			runId: record.runId,
			ownerId: "owner-a",
			runManager: manager,
			clock: mutableClock,
			leaseDurationMs: 30,
		});
		const state = first.reconstruct();
		state.protocol.start("prepare", state.coordinator.acquireOwnership());
		now = "2026-08-08T00:00:00.031Z";
		const contender = new XlabExperimentLifecycle({
			cwd: root,
			agentDir: join(root, "agent"),
			runId: record.runId,
			ownerId: "owner-b",
			runManager: manager,
			clock: mutableClock,
			leaseDurationMs: 30,
		});
		const resumed = await contender.resume(
			["prepare", "code", "science", "finalize"].map((stage) => ({ stage, run: async () => undefined })) as never,
		);
		expect(resumed.status).toBe("success");
		expect(contender.reconstruct().snapshot.owner).toMatchObject({ owner_id: "owner-b", generation: 2 });
	});

	it("settles indexing failure as incomplete and resumes publication without rerunning stages", async () => {
		const root = mkdtempSync(join(tmpdir(), "xlab-native-publication-retry-"));
		cleanups.push(root);
		mkdirSync(join(root, "agent"));
		const manager = new XlabRunManager(root);
		const binding = ideaBinding();
		const record = manager.createRun(skill(root), "--idea idea.json", {
			ideaBinding: binding,
			materializeStaging: createXlabExperimentStagingMaterializer(binding, clock),
		});
		const stageRuns = vi.fn(async () => undefined);
		let failIndexing = true;
		const finalizeSuccess = vi.fn(() => {
			if (failIndexing) throw new Error("index unavailable");
			return { artifactRefs: [], manifestPath: "manifest.json" };
		});
		const lifecycle = new XlabExperimentLifecycle({
			cwd: root,
			agentDir: join(root, "agent"),
			runId: record.runId,
			ownerId: "owner-publication",
			runManager: manager,
			clock,
			finalizeSuccess,
		});
		const drivers = ["prepare", "code", "science", "finalize"].map((stage) => ({ stage, run: stageRuns })) as never;

		const incomplete = await lifecycle.start(drivers);
		expect(incomplete).toMatchObject({
			status: "incomplete",
			lastRepair: expect.stringContaining("index unavailable"),
			publicationRecoveryReason: expect.stringContaining("index unavailable"),
		});
		const incompleteSnapshot = lifecycle.reconstruct().snapshot;
		expect(incompleteSnapshot).toMatchObject({
			status: "incomplete",
			publication_recovery_reason: expect.stringContaining("index unavailable"),
		});
		expect(lifecycle.status().publicationRecoveryReason).toContain("index unavailable");
		expect(stageRuns).toHaveBeenCalledTimes(4);

		await expect(lifecycle.resume([])).resolves.toMatchObject({
			status: "incomplete",
			publicationRecoveryReason: expect.stringContaining("index unavailable"),
		});
		expect(finalizeSuccess).toHaveBeenCalledTimes(2);

		failIndexing = false;
		const recovered = await lifecycle.resume([]);
		expect(recovered).toMatchObject({ status: "success", manifestPath: "manifest.json" });
		expect(recovered.lastRepair).toBeUndefined();
		expect(recovered.publicationRecoveryReason).toBeUndefined();
		expect(lifecycle.reconstruct().snapshot).toMatchObject({ status: "success" });
		expect(lifecycle.reconstruct().snapshot.publication_recovery_reason).toBeUndefined();
		expect(stageRuns).toHaveBeenCalledTimes(4);
		expect(finalizeSuccess).toHaveBeenCalledTimes(3);
	});

	it("does not resume an ordinary incomplete terminal run as publication recovery", async () => {
		const root = mkdtempSync(join(tmpdir(), "xlab-native-ordinary-incomplete-"));
		cleanups.push(root);
		mkdirSync(join(root, "agent"));
		const manager = new XlabRunManager(root);
		const binding = ideaBinding();
		const record = manager.createRun(skill(root), "--idea idea.json", {
			ideaBinding: binding,
			materializeStaging: createXlabExperimentStagingMaterializer(binding, clock),
		});
		const protocol = new ExperimentProtocolStore(record.runDir, clock);
		for (const stage of ["prepare", "code", "science", "finalize"] as const) {
			protocol.start(stage);
			protocol.completeStage(stage);
		}
		protocol.finish("incomplete", "ordinary scientific incompleteness");
		manager.updateRun(record, { status: "incomplete" }, "ordinary_incomplete");
		const finalizeSuccess = vi.fn(() => ({ artifactRefs: [], manifestPath: "manifest.json" }));
		const lifecycle = new XlabExperimentLifecycle({
			cwd: root,
			agentDir: join(root, "agent"),
			runId: record.runId,
			ownerId: "owner-ordinary-incomplete",
			runManager: manager,
			clock,
			finalizeSuccess,
		});

		await expect(lifecycle.resume([])).rejects.toThrow("not resumable");
		expect(finalizeSuccess).not.toHaveBeenCalled();
		expect(lifecycle.reconstruct().snapshot.publication_recovery_reason).toBeUndefined();
	});

	it("rejects a modified canonical policy", () => {
		const root = mkdtempSync(join(tmpdir(), "xlab-native-policy-"));
		cleanups.push(root);
		mkdirSync(join(root, "agent"));
		const manager = new XlabRunManager(root);
		const binding = ideaBinding();
		const record = manager.createRun(skill(root), "--idea idea.json", {
			ideaBinding: binding,
			materializeStaging: createXlabExperimentStagingMaterializer(binding, clock),
		});
		writeFileSync(join(record.runDir, "experiment", "policy.json"), JSON.stringify({ changed: true }));
		const lifecycle = new XlabExperimentLifecycle({
			cwd: root,
			agentDir: join(root, "agent"),
			runId: record.runId,
			ownerId: "owner",
		});
		expect(() => lifecycle.reconstruct()).toThrow("policy digest");
	});
});
