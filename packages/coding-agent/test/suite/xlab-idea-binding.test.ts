import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { XlabArtifactStore } from "../../src/core/xlab/artifact-store.ts";
import { bindXlabExperimentIdea } from "../../src/core/xlab/experiment/idea-binding.ts";
import type { XlabArtifactReference, XlabManifestEnvelope, XlabRunRecord } from "../../src/core/xlab/types.ts";
import { XlabWorkspaceManager } from "../../src/core/xlab/workspace-manager.ts";

function writeJson(path: string, value: unknown): void {
	writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
}

function successfulIdea(): Record<string, unknown> {
	return {
		schema_version: "xlab.research_idea.v2",
		status: "success",
		blockers: [],
		idea_result: {
			title: "Bound idea",
			components: [
				{ component: "Evidence memory", description: "Ground each decision." },
				{ component: "Verifier", explanation: "Check the evidence." },
			],
		},
	};
}

function commitIdea(cwd: string, title: string, suffix: string, type = "research_idea_result"): XlabArtifactReference {
	const payloadPath = join(cwd, `idea-result-${suffix}.json`);
	writeJson(payloadPath, {
		schema_version: "xlab.research_idea.result.v2",
		status: "success",
		blockers: [],
		title,
		components: [{ component: `component-${suffix}`, explanation: "Evidence." }],
	});
	const run: XlabRunRecord = {
		runId: `idea-run-${suffix}`,
		skillName: "research_idea",
		skillVersion: "2.0.0",
		command: "research_idea",
		status: "success",
		cwd,
		packageDir: cwd,
		runDir: join(cwd, ".xlab", "runs", `idea-run-${suffix}`),
		args: "",
		startedAt: new Date().toISOString(),
		updatedAt: new Date().toISOString(),
	};
	const manifest: XlabManifestEnvelope = {
		schema_version: "2",
		run_id: run.runId,
		skill_name: run.skillName,
		skill_version: run.skillVersion,
		status: "success",
		created_at: new Date().toISOString(),
		inputs: {},
		outputs: { title },
		validation: { passed: true },
		artifacts: [
			{
				type,
				schema_version: "2",
				path: payloadPath,
				metadata: { title },
			},
		],
	};
	return new XlabArtifactStore(cwd).commitManifestArtifacts(run, manifest, (path) => path)[0];
}

describe("XLab experiment Idea binding", () => {
	const cleanups: Array<() => void> = [];

	afterEach(() => {
		while (cleanups.length > 0) {
			cleanups.pop()?.();
		}
	});

	it("normalizes a successful Research Idea path into immutable canonical bytes", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-idea-binding-"));
		cleanups.push(() => rmSync(cwd, { recursive: true, force: true }));
		mkdirSync(join(cwd, "ideas"), { recursive: true });
		writeJson(join(cwd, "ideas", "idea.json"), successfulIdea());

		const result = bindXlabExperimentIdea({ cwd, args: "--idea ideas/idea.json" });
		expect(result.ok).toBe(true);
		if (!result.ok) {
			throw new Error(result.error);
		}
		expect(result.binding.metadata.resolution_method).toBe("path");
		expect(result.binding.metadata.source_schema_version).toBe("xlab.research_idea.v2");
		expect(result.binding.canonical.schema_version).toBe("xlab.experiment_idea.v1");
		expect(result.binding.canonical.components).toEqual([
			{ component: "Evidence memory", explanation: "Ground each decision.", index: "1" },
			{ component: "Verifier", explanation: "Check the evidence.", index: "2" },
		]);
		expect(result.binding.metadata.canonical_digest).toMatch(/^[a-f0-9]{64}$/);
		expect(result.binding.canonicalBytes.at(-1)).toBe(10);
	});

	it("fails closed for incomplete, malformed, and ambiguous component inputs", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-idea-binding-invalid-"));
		cleanups.push(() => rmSync(cwd, { recursive: true, force: true }));
		const path = join(cwd, "idea.json");

		writeJson(path, { ...successfulIdea(), status: "incomplete", blockers: ["missing evidence"] });
		expect(bindXlabExperimentIdea({ cwd, args: "--idea idea.json" })).toMatchObject({ ok: false });

		writeFileSync(path, "not json", "utf-8");
		expect(bindXlabExperimentIdea({ cwd, args: "--idea idea.json" })).toMatchObject({ ok: false });

		const duplicate = successfulIdea();
		duplicate.idea_result = {
			components: [{ component: "same" }, { component: "same" }],
		};
		writeJson(path, duplicate);
		const result = bindXlabExperimentIdea({ cwd, args: "--idea idea.json" });
		expect(result).toMatchObject({ ok: false });
		if (!result.ok) {
			expect(result.error).toContain("duplicate component");
		}
	});

	it("binds canonical artifact IDs and rejects legacy artifact types", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-idea-binding-artifact-"));
		cleanups.push(() => rmSync(cwd, { recursive: true, force: true }));
		const canonical = commitIdea(cwd, "Canonical artifact", "canonical");

		const resolved = bindXlabExperimentIdea({ cwd, args: `--idea ${canonical.artifactId}` });
		expect(resolved).toMatchObject({
			ok: true,
			binding: {
				metadata: {
					resolution_method: "artifact-id",
					reference: canonical.artifactId,
					source_artifact_id: canonical.artifactId,
					source_artifact_type: "research_idea_result",
					source_schema_version: "xlab.research_idea.result.v2",
				},
				warnings: [],
			},
		});

		const legacy = commitIdea(cwd, "Legacy artifact", "legacy", "research_idea_idea_result");
		expect(bindXlabExperimentIdea({ cwd, args: `--idea ${legacy.artifactId}` })).toEqual({
			ok: false,
			error: `Artifact ${legacy.artifactId} is not a Research Idea payload.`,
		});
	});

	it("fails closed when a stored Idea artifact is corrupted", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-idea-binding-corrupt-"));
		cleanups.push(() => rmSync(cwd, { recursive: true, force: true }));
		const artifact = commitIdea(cwd, "Corrupt artifact", "corrupt");
		writeFileSync(artifact.payloadPath, "corrupted", "utf-8");

		expect(bindXlabExperimentIdea({ cwd, args: `--idea ${artifact.artifactId}` })).toEqual({
			ok: false,
			error: `Artifact object failed integrity validation: ${artifact.artifactId}`,
		});
	});

	it("resolves producer runs to their result artifact", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-idea-binding-run-"));
		cleanups.push(() => rmSync(cwd, { recursive: true, force: true }));
		const resultArtifact = commitIdea(cwd, "Producer result", "producer");
		const wrapperArtifact = commitIdea(cwd, "Producer wrapper", "wrapper", "research_idea");
		const runDir = join(cwd, ".xlab", "runs", "producer-run");
		mkdirSync(runDir, { recursive: true });
		const run: XlabRunRecord = {
			runId: "producer-run",
			skillName: "research_idea",
			skillVersion: "2.0.0",
			command: "research_idea",
			status: "success",
			cwd,
			packageDir: cwd,
			runDir,
			args: "",
			startedAt: new Date().toISOString(),
			updatedAt: new Date().toISOString(),
			artifactRefs: [wrapperArtifact, resultArtifact],
		};
		writeJson(join(runDir, "run.json"), run);

		const resolved = bindXlabExperimentIdea({ cwd, args: "--idea producer-run" });
		expect(resolved).toMatchObject({
			ok: true,
			binding: {
				canonical: { title: "Producer result" },
				metadata: {
					resolution_method: "producer-run",
					reference: "producer-run",
					source_artifact_id: resultArtifact.artifactId,
					producer: { run_id: "producer-run", skill: "research_idea", version: "2.0.0" },
				},
			},
		});
	});

	it("resolves explicit and current workspace Idea handles", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-idea-binding-workspace-"));
		cleanups.push(() => rmSync(cwd, { recursive: true, force: true }));
		const artifact = commitIdea(cwd, "Workspace idea", "workspace");
		const run: XlabRunRecord = {
			runId: "workspace-run",
			skillName: "research_idea",
			skillVersion: "2.0.0",
			command: "research_idea",
			status: "success",
			cwd,
			packageDir: cwd,
			runDir: join(cwd, ".xlab", "runs", "workspace-run"),
			args: "",
			startedAt: new Date().toISOString(),
			updatedAt: new Date().toISOString(),
		};
		const manifest: XlabManifestEnvelope = {
			schema_version: "2",
			run_id: run.runId,
			skill_name: run.skillName,
			skill_version: run.skillVersion,
			status: "success",
			created_at: run.startedAt,
			inputs: {},
			outputs: { title: "Workspace idea" },
			validation: { passed: true },
			artifacts: [],
		};
		const workspaceManager = new XlabWorkspaceManager(cwd);
		workspaceManager.registerRunArtifacts({
			run,
			manifest,
			artifactRefs: [artifact],
			workspaceSlug: "test-topic",
		});

		const explicit = bindXlabExperimentIdea({
			cwd,
			args: "--idea idea-v1",
			workspaceSlug: "test-topic",
			workspaceManager,
		});
		expect(explicit).toMatchObject({
			ok: true,
			binding: {
				metadata: {
					resolution_method: "workspace-handle",
					reference: "idea-v1",
					workspace: { slug: "test-topic", handle: "idea-v1" },
				},
			},
		});

		const current = workspaceManager.resolveSkillArgs("run_experiment", "", {
			workspaceSlug: "test-topic",
		});
		expect(current.error).toBeUndefined();
		const currentResult = bindXlabExperimentIdea({
			cwd,
			args: current.args,
			workspaceSlug: current.workspaceSlug,
			workspaceManager,
		});
		expect(currentResult).toMatchObject({
			ok: true,
			binding: {
				metadata: {
					resolution_method: "workspace-handle",
					workspace: { slug: "test-topic", handle: "idea-v1" },
				},
			},
		});
	});

	it("resolves an exact indexed title and rejects ambiguous titles", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-idea-binding-title-"));
		cleanups.push(() => rmSync(cwd, { recursive: true, force: true }));
		const artifact = commitIdea(cwd, "Exact indexed title", "one");

		const resolved = bindXlabExperimentIdea({ cwd, args: '--idea "Exact indexed title"' });
		expect(resolved).toMatchObject({
			ok: true,
			binding: {
				metadata: { resolution_method: "title", source_artifact_id: artifact.artifactId },
			},
		});
		expect(bindXlabExperimentIdea({ cwd, args: '--idea "exact indexed title"' })).toMatchObject({ ok: false });

		commitIdea(cwd, "Exact indexed title", "two");
		const ambiguous = bindXlabExperimentIdea({ cwd, args: '--idea "Exact indexed title"' });
		expect(ambiguous).toMatchObject({ ok: false });
		if (!ambiguous.ok) {
			expect(ambiguous.error).toContain("ambiguous");
		}
	});

	it("rejects a missing natural-language reference instead of inventing an Idea", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-idea-binding-lookup-"));
		cleanups.push(() => rmSync(cwd, { recursive: true, force: true }));
		const result = bindXlabExperimentIdea({ cwd, args: '--idea "an idea about graph agents"' });
		expect(result).toMatchObject({ ok: false });
		if (!result.ok) {
			expect(result.error).toContain("existing Research Idea JSON file");
		}
	});
});
