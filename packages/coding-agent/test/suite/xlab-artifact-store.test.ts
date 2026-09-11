import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { XlabArtifactStore } from "../../src/core/xlab/artifact-store.ts";
import type { XlabManifestEnvelope, XlabRunRecord } from "../../src/core/xlab/types.ts";

function createRun(cwd: string, runId: string): XlabRunRecord {
	const now = new Date().toISOString();
	return {
		runId,
		skillName: "run_experiment",
		skillVersion: "1.0.0",
		command: "run_experiment",
		status: "running",
		cwd,
		packageDir: cwd,
		runDir: join(cwd, ".xlab", "runs", runId),
		args: "",
		startedAt: now,
		updatedAt: now,
	};
}

function createManifest(
	runId: string,
	path: string,
	lineage: Record<string, unknown>,
	parents: string[],
): XlabManifestEnvelope {
	return {
		schema_version: "2",
		run_id: runId,
		skill_name: "run_experiment",
		skill_version: "1.0.0",
		status: "success",
		created_at: new Date().toISOString(),
		inputs: {},
		outputs: {},
		validation: { passed: true },
		artifacts: [{ type: "experiment_result", schema_version: "1", path, parents, metadata: { lineage } }],
	};
}

describe("XlabArtifactStore occurrence lineage", () => {
	const temporaryDirectories: string[] = [];

	afterEach(() => {
		for (const directory of temporaryDirectories.splice(0)) {
			rmSync(directory, { recursive: true, force: true });
		}
	});

	it("replays an identical occurrence without replacing its timestamp", async () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-artifact-replay-"));
		temporaryDirectories.push(cwd);
		const payload = join(cwd, "result.json");
		writeFileSync(payload, '{"score":1}\n', "utf-8");
		const store = new XlabArtifactStore(cwd);
		const run = createRun(cwd, "run-replay");
		const manifest = createManifest("run-replay", payload, {}, []);
		const [reference] = store.commitManifestArtifacts(run, manifest, (path) => path);
		const original = store.get(reference.artifactId)?.occurrences?.[0];
		await new Promise((resolve) => setTimeout(resolve, 5));

		store.commitManifestArtifacts(run, manifest, (path) => path);

		expect(store.get(reference.artifactId)?.occurrences).toEqual([original]);
	});

	it("rejects invalid declarations before publishing any index rows", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-artifact-preflight-"));
		temporaryDirectories.push(cwd);
		const valid = join(cwd, "valid.json");
		const invalid = join(cwd, "invalid.json");
		writeFileSync(valid, "{}\n", "utf-8");
		writeFileSync(invalid, '{"bad":true}\n', "utf-8");
		const store = new XlabArtifactStore(cwd);
		const run = createRun(cwd, "run-preflight");
		const manifest = createManifest(run.runId, valid, {}, []);
		manifest.artifacts.push({ type: "invalid", schema_version: "1", path: invalid, bytes: 1 });

		expect(() => store.commitManifestArtifacts(run, manifest, (path) => path)).toThrow("byte count");
		expect(store.list()).toEqual([]);
	});

	it("validates required artifacts, content digests, and type conflicts", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-artifact-contract-"));
		temporaryDirectories.push(cwd);
		const payload = join(cwd, "result.json");
		const bytes = Buffer.from('{"score":1}\n');
		writeFileSync(payload, bytes);
		const store = new XlabArtifactStore(cwd);
		const firstRun = createRun(cwd, "run-contract-a");
		const first = createManifest(firstRun.runId, payload, {}, []);
		first.artifacts[0]!.bytes = bytes.length;
		first.artifacts[0]!.digest = `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
		store.commitManifestArtifacts(firstRun, first, (path) => path, [
			{ type: "experiment_result", schemaVersion: "1", required: true },
		]);

		const missing = createManifest("run-contract-b", payload, {}, []);
		expect(() =>
			store.commitManifestArtifacts(createRun(cwd, missing.run_id), missing, (path) => path, [
				{ type: "final_audit", schemaVersion: "1", required: true },
			]),
		).toThrow("missing required artifact");

		const conflict = createManifest("run-contract-c", payload, {}, []);
		conflict.artifacts[0]!.type = "different_type";
		expect(() => store.commitManifestArtifacts(createRun(cwd, conflict.run_id), conflict, (path) => path)).toThrow(
			"conflicts with",
		);
	});

	it("keeps duplicate payload lineage and parent digests occurrence-specific", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-artifact-lineage-"));
		temporaryDirectories.push(cwd);
		const payload = join(cwd, "result.json");
		writeFileSync(payload, '{"score":1}\n', "utf-8");
		const store = new XlabArtifactStore(cwd);
		const firstLineage = {
			child_id: "worker-a",
			session_id: "session-a",
			work_unit: "train",
			execution_attempt: 1,
			generation: 2,
			review_round: 0,
			parent_artifact_digests: [`sha256:${"a".repeat(64)}`],
		};
		const secondLineage = {
			child_id: "reviewer-b",
			session_id: "session-b",
			work_unit: "review",
			execution_attempt: 3,
			generation: 4,
			review_round: 2,
			parent_artifact_digests: [`sha256:${"b".repeat(64)}`, `sha256:${"c".repeat(64)}`],
		};

		const [first] = store.commitManifestArtifacts(
			createRun(cwd, "run-a"),
			createManifest("run-a", payload, firstLineage, firstLineage.parent_artifact_digests),
			(path) => path,
		);
		const [second] = store.commitManifestArtifacts(
			createRun(cwd, "run-b"),
			createManifest("run-b", payload, secondLineage, secondLineage.parent_artifact_digests),
			(path) => path,
		);

		expect(second.artifactId).toBe(first.artifactId);
		const stored = new XlabArtifactStore(cwd).get(first.artifactId);
		expect(stored?.occurrences).toEqual([
			expect.objectContaining({ producer_run_id: "run-a", lineage: firstLineage }),
			expect.objectContaining({ producer_run_id: "run-b", lineage: secondLineage }),
		]);
		expect(stored?.parents).toEqual([]);
		expect(stored?.metadata).toEqual({});
		const objectMetadata = JSON.parse(readFileSync(first.metadataPath, "utf-8")) as Record<string, unknown>;
		expect(objectMetadata.metadata).toEqual({});
		expect(objectMetadata.parents).toEqual([]);
	});
});
