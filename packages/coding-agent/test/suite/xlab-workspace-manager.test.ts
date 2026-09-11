import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { XlabArtifactReference, XlabManifestEnvelope, XlabRunRecord } from "../../src/core/xlab/types.ts";
import { slugifyWorkspaceName, XlabWorkspaceManager } from "../../src/core/xlab/workspace-manager.ts";

function createTempDir(): string {
	return mkdtempSync(join(tmpdir(), "xlab-workspace-test-"));
}

function writeJson(path: string, value: unknown): void {
	writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
}

function baseRun(tempDir: string, runId: string, skillName: string, args = "--topic Test Topic"): XlabRunRecord {
	const runDir = join(tempDir, ".xlab", "runs", runId);
	mkdirSync(join(runDir, "artifacts"), { recursive: true });
	return {
		runId,
		skillName,
		skillVersion: "1.0.0",
		command: skillName,
		status: "success",
		cwd: tempDir,
		packageDir: join(tempDir, "xlab", "skills", skillName),
		runDir,
		args,
		startedAt: "2026-01-01T00:00:00.000Z",
		updatedAt: "2026-01-01T00:00:00.000Z",
		finishedAt: "2026-01-01T00:00:00.000Z",
		manifestPath: join(runDir, "manifest.json"),
		workspaceSlug: "test-topic",
	};
}

function manifest(
	run: XlabRunRecord,
	artifacts: XlabManifestEnvelope["artifacts"],
	inputs: Record<string, unknown> = {},
): XlabManifestEnvelope {
	return {
		schema_version: "2",
		run_id: run.runId,
		skill_name: run.skillName,
		skill_version: run.skillVersion,
		status: run.status,
		created_at: "2026-01-01T00:00:00.000Z",
		inputs,
		outputs: {},
		validation: { passed: true },
		artifacts,
	};
}

function refs(...items: Array<[string, string]>): XlabArtifactReference[] {
	return items.map(([artifactId, type], index) => ({
		artifactId,
		type,
		schemaVersion: "1",
		digest: artifactId.replace("sha256:", ""),
		payloadPath: `/tmp/payload-${index}`,
		metadataPath: `/tmp/artifact-${index}.json`,
	}));
}

function refsWithPaths(...items: Array<[string, string, string]>): XlabArtifactReference[] {
	return items.map(([artifactId, type, payloadPath], index) => ({
		artifactId,
		type,
		schemaVersion: "2",
		digest: artifactId.replace("sha256:", ""),
		payloadPath,
		metadataPath: `/tmp/artifact-path-${index}.json`,
	}));
}

describe("XLab workspace manager", () => {
	const cleanups: Array<() => void> = [];

	afterEach(() => {
		while (cleanups.length > 0) {
			cleanups.pop()?.();
		}
	});

	it("generates readable workspace slugs", () => {
		expect(slugifyWorkspaceName("Scientific Discovery Agents")).toBe("scientific-discovery-agents");
		expect(slugifyWorkspaceName("  Graph/RAG: Survey++  ")).toBe("graph-rag-survey");
		expect(slugifyWorkspaceName("!!!")).toBe("xlab-workspace");
	});

	it("allocates graph, survey, and idea handles with lineage", async () => {
		const tempDir = createTempDir();
		cleanups.push(() => rmSync(tempDir, { recursive: true, force: true }));
		const manager = new XlabWorkspaceManager(tempDir);

		const graphRun = baseRun(tempDir, "graph-run", "knowledge_graph");
		writeJson(graphRun.manifestPath!, {});
		const graph = manager.registerRunArtifacts({
			run: graphRun,
			manifest: manifest(graphRun, [{ type: "graph_db", schema_version: "1", path: "artifacts/graph.db" }], {
				topic: "Test Topic",
			}),
			artifactRefs: refs(["sha256:graphdb", "graph_db"], ["sha256:method", "method_graph"]),
		});
		expect(graph?.handles[0].handle).toBe("graph-v1");
		expect(manager.readCurrent("test-topic").current.graph).toBe("graph-v1");

		const surveyRun = baseRun(tempDir, "survey-run", "literature_survey");
		writeJson(surveyRun.manifestPath!, {});
		const survey = manager.registerRunArtifacts({
			run: surveyRun,
			manifest: manifest(surveyRun, [
				{
					type: "literature_survey_json",
					schema_version: "1",
					path: "artifacts/survey.json",
					parents: ["sha256:graphdb"],
				},
			]),
			artifactRefs: refs(["sha256:survey", "literature_survey_json"]),
		});
		expect(survey?.handles[0].handle).toBe("survey-v1");
		expect(survey?.handles[0].parentHandles).toEqual(["graph-v1"]);
		expect(manager.readCurrent("test-topic").current.survey).toBe("survey-v1");

		const ideaRun = baseRun(tempDir, "idea-run", "research_idea");
		writeJson(ideaRun.manifestPath!, {});
		const idea = manager.registerRunArtifacts({
			run: ideaRun,
			manifest: manifest(ideaRun, [
				{
					type: "research_idea",
					schema_version: "1",
					path: "artifacts/research_idea.json",
					parents: ["sha256:survey"],
				},
			]),
			artifactRefs: refs(["sha256:idea", "research_idea"]),
		});
		expect(idea?.handles[0].handle).toBe("idea-v1");
		expect(idea?.handles[0].parentHandles).toEqual(["survey-v1"]);
		expect(manager.readCurrent("test-topic").current.idea).toBe("idea-v1");
	});

	it("includes citation traces and merges known and unknown survey parents", async () => {
		const tempDir = createTempDir();
		cleanups.push(() => rmSync(tempDir, { recursive: true, force: true }));
		const manager = new XlabWorkspaceManager(tempDir);
		const graphRun = baseRun(tempDir, "graph-run", "knowledge_graph");
		writeJson(graphRun.manifestPath!, {});
		manager.registerRunArtifacts({
			run: graphRun,
			manifest: manifest(graphRun, [{ type: "graph_db", schema_version: "1", path: "artifacts/graph.db" }]),
			artifactRefs: refs(["sha256:graph", "graph_db"]),
		});

		const surveyRun = baseRun(tempDir, "survey-run", "literature_survey");
		writeJson(surveyRun.manifestPath!, {});
		const registered = manager.registerRunArtifacts({
			run: surveyRun,
			manifest: manifest(surveyRun, [
				{
					type: "literature_survey_json",
					schema_version: "1",
					path: "artifacts/survey.json",
					parents: ["sha256:graph", "sha256:unknown"],
				},
				{
					type: "citation_trace",
					schema_version: "1",
					path: "artifacts/citations.json",
					parents: ["sha256:graph"],
				},
			]),
			artifactRefs: refs(["sha256:survey", "literature_survey_json"], ["sha256:citations", "citation_trace"]),
		});

		expect(registered?.handles).toHaveLength(1);
		expect(registered?.handles[0].artifactIds).toEqual(["sha256:survey", "sha256:citations"]);
		expect(registered?.handles[0].parents).toEqual(["sha256:graph", "sha256:unknown"]);
		expect(registered?.handles[0].parentHandles).toEqual(["graph-v1"]);
	});

	it("resolves workspace handles and current defaults to concrete manifest paths", async () => {
		const tempDir = createTempDir();
		cleanups.push(() => rmSync(tempDir, { recursive: true, force: true }));
		const manager = new XlabWorkspaceManager(tempDir);
		const graphRun = baseRun(tempDir, "graph-run", "knowledge_graph");
		writeJson(graphRun.manifestPath!, {});
		manager.registerRunArtifacts({
			run: graphRun,
			manifest: manifest(graphRun, [{ type: "graph_db", schema_version: "1", path: "artifacts/graph.db" }]),
			artifactRefs: refs(["sha256:graph", "graph_db"]),
		});
		const surveyRun = baseRun(tempDir, "survey-run", "literature_survey");
		writeJson(surveyRun.manifestPath!, {});
		manager.registerRunArtifacts({
			run: surveyRun,
			manifest: manifest(surveyRun, [
				{
					type: "literature_survey_json",
					schema_version: "1",
					path: "artifacts/survey.json",
					parents: ["sha256:graph"],
				},
			]),
			artifactRefs: refs(["sha256:survey", "literature_survey_json"]),
		});

		const explicitSurvey = manager.resolveSkillArgs(
			"research_idea",
			"--workspace test-topic --survey survey-v1 --topic Next",
		);
		expect(explicitSurvey.error).toBeUndefined();
		expect(explicitSurvey.args).toContain("--survey .xlab/runs/survey-run/manifest.json");
		expect(explicitSurvey.args).not.toContain("--workspace");

		const defaultSurvey = manager.resolveSkillArgs("research_idea", "--workspace test-topic --topic Next");
		expect(defaultSurvey.error).toBeUndefined();
		expect(defaultSurvey.args).toContain("--survey .xlab/runs/survey-run/manifest.json");
		expect(defaultSurvey.warnings).toContain("Using survey-v1 from workspace test-topic.");

		const defaultGraph = manager.resolveSkillArgs("literature_survey", "--workspace test-topic --topic Next");
		expect(defaultGraph.error).toBeUndefined();
		expect(defaultGraph.args).toContain("--graph .xlab/runs/graph-run/manifest.json");
		expect(defaultGraph.warnings).toContain("Using graph-v1 from workspace test-topic.");
	});

	it("resolves experiment idea handles to their primary payload", async () => {
		const tempDir = createTempDir();
		cleanups.push(() => rmSync(tempDir, { recursive: true, force: true }));
		const manager = new XlabWorkspaceManager(tempDir);
		const ideaRun = baseRun(tempDir, "idea-run", "research_idea");
		const ideaPayload = join(ideaRun.runDir, "artifacts", "idea_result.json");
		writeJson(ideaRun.manifestPath!, {});
		writeJson(ideaPayload, { schema_version: "xlab.research_idea.result.v2" });
		manager.registerRunArtifacts({
			run: ideaRun,
			manifest: manifest(ideaRun, [
				{
					type: "research_idea_result",
					schema_version: "2",
					path: "artifacts/idea_result.json",
				},
			]),
			artifactRefs: refsWithPaths(
				["sha256:idea", "research_idea_result", ideaPayload],
				["sha256:report", "research_idea_report", join(ideaRun.runDir, "artifacts", "idea_report.json")],
			),
		});

		const explicit = manager.resolveSkillArgs(
			"run_experiment",
			"--workspace test-topic --idea idea-v1 --project-root experiment",
		);
		expect(explicit.error).toBeUndefined();
		expect(explicit.args).toContain(`--idea ${ideaPayload}`);
		expect(explicit.args).not.toContain("idea-v1");

		const current = manager.resolveSkillArgs("run_experiment", "--workspace test-topic --project-root experiment");
		expect(current.error).toBeUndefined();
		expect(current.args).toContain(`--idea ${ideaPayload}`);
		expect(current.warnings).toContain("Using idea-v1 from workspace test-topic.");

		const literalPath = manager.resolveSkillArgs(
			"run_experiment",
			"--workspace test-topic --idea ./ideas/custom.json",
		);
		expect(literalPath.args).toContain("--idea ./ideas/custom.json");

		const unresolved = manager.resolveSkillArgs("run_experiment", "--workspace test-topic --idea idea-v9");
		expect(unresolved.error).toContain("Cannot resolve idea-v9");
	});

	it("does not expose legacy Research Idea result artifacts as Idea handles", async () => {
		const tempDir = createTempDir();
		cleanups.push(() => rmSync(tempDir, { recursive: true, force: true }));
		const manager = new XlabWorkspaceManager(tempDir);
		const ideaRun = baseRun(tempDir, "legacy-idea-run", "research_idea");
		const ideaPayload = join(ideaRun.runDir, "artifacts", "idea_result.json");
		writeJson(ideaRun.manifestPath!, {});
		writeJson(ideaPayload, { schema_version: "xlab.research_idea.result.v2" });
		const registered = manager.registerRunArtifacts({
			run: ideaRun,
			manifest: manifest(ideaRun, [
				{
					type: "research_idea_idea_result",
					schema_version: "2",
					path: "artifacts/idea_result.json",
				},
			]),
			artifactRefs: refsWithPaths(["sha256:legacy-idea", "research_idea_idea_result", ideaPayload]),
		});

		expect(registered.handles).toEqual([]);
		const resolved = manager.resolveSkillArgs("run_experiment", "--workspace test-topic --idea idea-v1");
		expect(resolved.error).toContain("Cannot resolve idea-v1");
	});

	it("requires the primary idea artifact to be a successful payload", async () => {
		const tempDir = createTempDir();
		cleanups.push(() => rmSync(tempDir, { recursive: true, force: true }));
		const manager = new XlabWorkspaceManager(tempDir);
		const ideaRun = baseRun(tempDir, "idea-run", "research_idea");
		ideaRun.status = "incomplete";
		writeJson(ideaRun.manifestPath!, {});
		manager.registerRunArtifacts({
			run: ideaRun,
			manifest: manifest(ideaRun, [
				{
					type: "research_idea_result",
					schema_version: "2",
					path: "artifacts/idea_result.json",
				},
			]),
			artifactRefs: refsWithPaths([
				"sha256:report",
				"research_idea_report",
				join(ideaRun.runDir, "artifacts", "idea_report.json"),
			]),
		});

		const resolved = manager.resolveSkillArgs("run_experiment", "--workspace test-topic --idea idea-v1");
		expect(resolved.error).toContain("does not reference a Research Idea payload");
	});

	it("requires a survey path or handle when no current survey is registered", async () => {
		const tempDir = createTempDir();
		cleanups.push(() => rmSync(tempDir, { recursive: true, force: true }));
		const manager = new XlabWorkspaceManager(tempDir);
		manager.ensureWorkspace("test-topic");

		const resolved = manager.resolveSkillArgs("research_idea", "--workspace test-topic --topic Next");
		expect(resolved.error).toContain("No current survey handle");
	});

	it("reports stale survey dependencies when the current graph advances", async () => {
		const tempDir = createTempDir();
		cleanups.push(() => rmSync(tempDir, { recursive: true, force: true }));
		const manager = new XlabWorkspaceManager(tempDir);
		const firstGraphRun = baseRun(tempDir, "graph-run-1", "knowledge_graph");
		writeJson(firstGraphRun.manifestPath!, {});
		manager.registerRunArtifacts({
			run: firstGraphRun,
			manifest: manifest(firstGraphRun, [{ type: "graph_db", schema_version: "1", path: "artifacts/graph.db" }]),
			artifactRefs: refs(["sha256:graph1", "graph_db"]),
		});
		const surveyRun = baseRun(tempDir, "survey-run", "literature_survey");
		writeJson(surveyRun.manifestPath!, {});
		manager.registerRunArtifacts({
			run: surveyRun,
			manifest: manifest(surveyRun, [
				{
					type: "literature_survey_json",
					schema_version: "1",
					path: "artifacts/survey.json",
					parents: ["sha256:graph1"],
				},
			]),
			artifactRefs: refs(["sha256:survey", "literature_survey_json"]),
		});
		const secondGraphRun = baseRun(tempDir, "graph-run-2", "knowledge_graph");
		writeJson(secondGraphRun.manifestPath!, {});
		manager.registerRunArtifacts({
			run: secondGraphRun,
			manifest: manifest(secondGraphRun, [{ type: "graph_db", schema_version: "1", path: "artifacts/graph.db" }]),
			artifactRefs: refs(["sha256:graph2", "graph_db"]),
		});

		expect(manager.staleWarnings("test-topic").join("\n")).toContain("survey-v1 may be stale");
	});
});
