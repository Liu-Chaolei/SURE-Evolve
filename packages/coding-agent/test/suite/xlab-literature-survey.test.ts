import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { XlabHookRunner } from "../../src/core/xlab/hooks.ts";
import { discoverXlabSkillPackages } from "../../src/core/xlab/manifest.ts";
import { validateXlabPackageSelfContained } from "../../src/core/xlab/package-validator.ts";
import type { XlabRunRecord, XlabSkillPackage } from "../../src/core/xlab/types.ts";

const REPOSITORY_ROOT = fileURLToPath(new URL("../../../..", import.meta.url));
const SURVEY_TOPIC = "Retrieval augmented generation for scientific agents";
const SURVEY_ARGS = `"${SURVEY_TOPIC}" --graph .xlab/runs/kg-run`;
const LITERATURE_SURVEY_ENV_OVERRIDES = [
	"LLM_MODEL",
	"LLM_BASE_URL",
	"LLM_API_BASE",
	"LLM_CONTEXT_WINDOW",
	"XLAB_LITERATURE_SURVEY_MODEL",
	"XLAB_LITERATURE_SURVEY_LLM_BASE_URL",
	"OPENAI_API_KEY",
	"LLM_API_KEY",
	"SEMANTIC_SCHOLAR_API_KEY",
	"S2_API_KEY",
] as const;

function copyRepositorySkill(tempDir: string, skillName: string): void {
	const target = join(tempDir, "xlab", "skills", skillName);
	mkdirSync(join(tempDir, "xlab", "skills"), { recursive: true });
	cpSync(join(REPOSITORY_ROOT, "xlab", "skills", skillName), target, { recursive: true });
}

function setupLiteratureSurveyProject(tempDir: string): XlabSkillPackage {
	copyRepositorySkill(tempDir, "paper_collect");
	copyRepositorySkill(tempDir, "knowledge_graph");
	copyRepositorySkill(tempDir, "literature_survey");
	const discovered = discoverXlabSkillPackages(tempDir);
	expect(discovered.diagnostics.filter((diagnostic) => diagnostic.type === "error")).toEqual([]);
	const skillPackage = discovered.packages.find((entry) => entry.manifest.name === "literature_survey");
	expect(skillPackage).toBeDefined();
	return skillPackage!;
}

function createRun(tempDir: string, skillPackage: XlabSkillPackage, runId: string): XlabRunRecord {
	return {
		runId,
		skillName: skillPackage.manifest.name,
		skillVersion: skillPackage.manifest.version,
		command: "write-literature-survey",
		status: "running",
		cwd: tempDir,
		packageDir: skillPackage.packageDir,
		runDir: join(tempDir, ".xlab", "runs", runId),
		args: SURVEY_ARGS,
		startedAt: new Date().toISOString(),
		updatedAt: new Date().toISOString(),
	};
}

function writeGraphFixture(tempDir: string): string {
	const graphDir = join(tempDir, ".xlab", "runs", "kg-run");
	const artifactsDir = join(graphDir, "artifacts");
	mkdirSync(artifactsDir, { recursive: true });
	const methodGraphPath = join(artifactsDir, "method_graph.json");
	writeFileSync(
		methodGraphPath,
		`${JSON.stringify(
			{
				schema_version: "xlab.method_graph.v2",
				nodes: [
					{
						id: "paper:kg-1",
						node_type: "Paper",
						name: "Retrieval-Augmented Scientific Agents",
						paper_id: "kg-1",
						aliases: [],
						provenance: { paper_id: "kg-1" },
						metadata: {
							title: "Retrieval-Augmented Scientific Agents",
							abstract:
								"Retrieval augmented generation grounds scientific agents in indexed paper evidence and tool traces.",
							year: 2024,
							venue: "XLab Conference",
						},
					},
					{
						id: "paper:kg-2",
						node_type: "Paper",
						name: "Evaluating Literature Review Agents",
						paper_id: "kg-2",
						aliases: [],
						provenance: { paper_id: "kg-2" },
						metadata: {
							title: "Evaluating Literature Review Agents",
							abstract:
								"Evaluation of scientific literature agents benefits from citation traceability and reproducible evidence graphs.",
							year: 2025,
							venue: "Agent Evaluation Workshop",
						},
					},
					{
						id: "paper:kg-3",
						node_type: "Paper",
						name: "Knowledge Graph Context for Research Planning",
						paper_id: "kg-3",
						aliases: [],
						provenance: { paper_id: "kg-3" },
						metadata: {
							title: "Knowledge Graph Context for Research Planning",
							abstract:
								"Method graphs connect paper nodes, core concepts, and relations so downstream research workflows can reuse structured context.",
							year: 2023,
							venue: "Graph Systems",
						},
					},
					{
						id: "core:retrieval",
						node_type: "Core",
						name: "retrieval grounding",
						paper_id: "kg-1",
						aliases: ["RAG"],
						provenance: { paper_id: "kg-1" },
					},
				],
				edges: [
					{ source: "core:retrieval", target: "paper:kg-1", relation: "supports" },
					{ source: "paper:kg-2", target: "paper:kg-3", relation: "extends" },
				],
				counts: { nodes: 4, edges: 2, papers: 3 },
			},
			null,
			2,
		)}\n`,
		"utf-8",
	);
	writeFileSync(
		join(graphDir, "manifest.json"),
		`${JSON.stringify(
			{
				schema_version: "2",
				run_id: "kg-run",
				skill_name: "knowledge_graph",
				skill_version: "3.0.0",
				status: "success",
				created_at: "2026-01-01T00:00:00Z",
				inputs: {},
				outputs: {},
				validation: { passed: true },
				artifacts: [{ type: "method_graph", schema_version: "xlab.method_graph.v2", path: methodGraphPath }],
			},
			null,
			2,
		)}\n`,
		"utf-8",
	);
	return graphDir;
}

function writeSuccessfulSurveyFixture(tempDir: string, run: XlabRunRecord): void {
	writeGraphFixture(tempDir);
	const artifactsDir = join(run.runDir, "artifacts");
	const stateDir = join(run.runDir, "state");
	const xcientistStateDir = join(stateDir, "xcientist");
	mkdirSync(artifactsDir, { recursive: true });
	mkdirSync(xcientistStateDir, { recursive: true });
	const references = [
		{
			paper_id: "kg-1",
			title: "Retrieval-Augmented Scientific Agents",
			year: 2024,
			venue: "XLab Conference",
			source: "knowledge_graph",
		},
		{
			paper_id: "kg-2",
			title: "Evaluating Literature Review Agents",
			year: 2025,
			venue: "Agent Evaluation Workshop",
			source: "knowledge_graph",
		},
		{
			paper_id: "kg-3",
			title: "Knowledge Graph Context for Research Planning",
			year: 2023,
			venue: "Graph Systems",
			source: "knowledge_graph",
		},
	];
	const sections = [
		{
			id: "section:grounding",
			title: "Retrieval grounding",
			summary:
				"SurveyAgent synthesis links retrieval grounding, evaluation, and graph context across the fixture papers.",
			paper_ids: ["kg-1", "kg-2"],
		},
	];
	const keyClaims = [
		{
			id: "claim:traceability",
			claim: "Citation-traceable graph context improves reliability of literature review agents.",
			paper_ids: ["kg-2", "kg-3"],
		},
	];
	const researchGaps = [
		{
			id: "gap:planning",
			gap: "Few systems reuse structured method graphs when planning downstream scientific-agent experiments.",
			paper_ids: ["kg-1", "kg-2", "kg-3"],
		},
	];
	const traces = [
		{
			claim_id: "claim:traceability",
			claim: "Citation-traceable graph context improves reliability of literature review agents.",
			paper_ids: ["kg-2", "kg-3"],
			evidence: ["Evaluation and graph-system papers both emphasize reproducible evidence graphs."],
		},
	];
	const surveyAgentResult = {
		schema_version: "xlab.literature_survey.agent.v1",
		mode: "xcientist_survey_agent",
		engine: {
			mode: "integrated_xcientist_survey_agent",
			xcientist: {
				collected_paper_ids: references.map((reference) => reference.paper_id),
				raw_references: references,
				internal_state_dir: xcientistStateDir,
			},
		},
		sections,
		key_claims: keyClaims,
		research_gaps: researchGaps,
		cited_paper_ids: references.map((reference) => reference.paper_id),
		markdown_body:
			"## Retrieval grounding\n" +
			"SurveyAgent synthesis links retrieval grounding, evaluation, and graph context across the fixture papers [paper:kg-1] [paper:kg-2].\n\n" +
			"## Research gaps\n" +
			"Downstream research planning still needs structured graph reuse and traceability [paper:kg-2] [paper:kg-3].\n",
	};
	for (const name of [
		"seed_papers.json",
		"expanded_papers.json",
		"collected_papers.json",
		"keynotes.json",
		"clustering_result.json",
		"analysis_context.json",
		"outline.raw.json",
		"draft.raw.json",
		"references.raw.json",
		"evaluation.json",
		"engine_result.json",
	]) {
		writeFileSync(join(xcientistStateDir, name), `${JSON.stringify({ ok: true }, null, 2)}\n`, "utf-8");
	}
	writeFileSync(join(xcientistStateDir, "draft.raw.md"), "# Draft\n\nFixture SurveyAgent draft.\n", "utf-8");
	writeFileSync(
		join(xcientistStateDir, "refined.raw.md"),
		"# Refined\n\nFixture SurveyAgent refined survey.\n",
		"utf-8",
	);
	writeFileSync(
		join(artifactsDir, "survey.md"),
		[
			`# ${SURVEY_TOPIC}`,
			"",
			"## Retrieval grounding",
			"SurveyAgent synthesis links retrieval grounding, evaluation, and graph context across the fixture papers [paper:kg-1] [paper:kg-2]. The fixture content is intentionally long enough to exercise the packaged audit without invoking an external provider. It preserves the same artifact contracts as the integrated SurveyAgent path and includes explicit paper citations for traceability.",
			"",
			"## Research gaps",
			"Downstream research planning still needs structured graph reuse and traceability [paper:kg-2] [paper:kg-3]. The survey fixture therefore records sections, claims, gaps, references, citation traces, source signatures, and Xcientist provenance so the hook test can focus on pre_finish quality gating rather than live provider behavior.",
			"",
			"## References",
			"1. [kg-1] Retrieval-Augmented Scientific Agents (2024).",
			"2. [kg-2] Evaluating Literature Review Agents (2025).",
			"3. [kg-3] Knowledge Graph Context for Research Planning (2023).",
			"",
		].join("\n"),
		"utf-8",
	);
	writeFileSync(
		join(artifactsDir, "survey.json"),
		`${JSON.stringify(
			{
				schema_version: "xlab.literature_survey.v1",
				run_id: run.runId,
				topic: SURVEY_TOPIC,
				paper_count: references.length,
				sections,
				clusters: [
					{
						id: "cluster:rag",
						name: "Retrieval-grounded agents",
						paper_ids: ["kg-1", "kg-2", "kg-3"],
						summary: "Retrieval-grounded agents combine indexed evidence, evaluation, and graph context.",
					},
				],
				chronology: [
					{ year: 2023, paper_ids: ["kg-3"] },
					{ year: 2024, paper_ids: ["kg-1"] },
					{ year: 2025, paper_ids: ["kg-2"] },
				],
				key_claims: keyClaims,
				research_gaps: researchGaps,
				references,
				citation_trace_path: join(artifactsDir, "citations.json"),
				runtime: { mode: "survey-agent", survey_agent_engine: "integrated_xcientist_survey_agent" },
				source: { source_type: "knowledge_graph" },
				generated_at: "2026-01-01T00:00:00Z",
			},
			null,
			2,
		)}\n`,
		"utf-8",
	);
	writeFileSync(
		join(artifactsDir, "citations.json"),
		`${JSON.stringify({ schema_version: "xlab.citation_trace.v1", run_id: run.runId, references, traces }, null, 2)}\n`,
		"utf-8",
	);
	writeFileSync(
		join(artifactsDir, "survey_report.json"),
		`${JSON.stringify(
			{
				schema_version: "xlab.literature_survey.report.v1",
				generated_at: "2026-01-01T00:00:00Z",
				passed: true,
				blocking_errors: [],
				warnings: [],
				counts: {
					papers: references.length,
					clusters: 1,
					sections: sections.length,
					claims: keyClaims.length,
					gaps: researchGaps.length,
					references: references.length,
					traces: traces.length,
				},
				checks: { survey_agent_mode: true },
			},
			null,
		)}\n`,
		"utf-8",
	);
	writeFileSync(
		join(stateDir, "survey_agent_result.json"),
		`${JSON.stringify(surveyAgentResult, null, 2)}\n`,
		"utf-8",
	);
	writeFileSync(
		join(stateDir, "paper_index.json"),
		`${JSON.stringify(
			{
				schema_version: "xlab.literature_survey.paper_index.v1",
				source_type: "knowledge_graph",
				paper_count: references.length,
				warnings: [],
			},
			null,
			2,
		)}\n`,
		"utf-8",
	);
	writeFileSync(
		join(stateDir, "checkpoint.json"),
		`${JSON.stringify(
			{
				schema_version: "xlab.literature_survey.checkpoint.v1",
				phase: "render_artifacts",
				completed_stages: [
					"ingest_sources",
					"ingest_graph_context",
					"cluster",
					"survey_agent",
					"citation_traces",
					"render_artifacts",
				],
				attempted_stages: [
					"ingest_sources",
					"ingest_graph_context",
					"cluster",
					"survey_agent",
					"citation_traces",
					"render_artifacts",
				],
				blocked_stages: [],
				source_signatures: { graph_method_graph: "fixture-signature" },
				counts: {
					papers: references.length,
					sections: sections.length,
					claims: keyClaims.length,
					gaps: researchGaps.length,
					traces: traces.length,
				},
				warnings: [],
			},
			null,
			2,
		)}\n`,
		"utf-8",
	);
	writeFileSync(
		join(run.runDir, "manifest.json"),
		`${JSON.stringify(
			{
				schema_version: "2",
				run_id: run.runId,
				skill_name: "literature_survey",
				skill_version: "1.0.0",
				status: "success",
				created_at: "2026-01-01T00:00:00Z",
				inputs: {},
				outputs: { topic: SURVEY_TOPIC },
				validation: { passed: true },
				artifacts: [
					{ type: "literature_survey_document", schema_version: "1", path: join(artifactsDir, "survey.md") },
					{ type: "literature_survey_json", schema_version: "1", path: join(artifactsDir, "survey.json") },
					{ type: "citation_trace", schema_version: "1", path: join(artifactsDir, "citations.json") },
					{ type: "survey_report", schema_version: "1", path: join(artifactsDir, "survey_report.json") },
				],
			},
			null,
			2,
		)}\n`,
		"utf-8",
	);
}

function readRunJson(run: XlabRunRecord, relativePath: string): Record<string, unknown> {
	return JSON.parse(readFileSync(join(run.runDir, relativePath), "utf-8")) as Record<string, unknown>;
}

describe("literature_survey XLab skill", () => {
	const temporaryDirectories: string[] = [];

	afterEach(() => {
		for (const directory of temporaryDirectories.splice(0)) {
			rmSync(directory, { recursive: true, force: true });
		}
	});

	it("is discoverable as a self-contained Harness v2 package", () => {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-literature-survey-"));
		temporaryDirectories.push(tempDir);
		const skillPackage = setupLiteratureSurveyProject(tempDir);

		expect(skillPackage.manifest).toEqual(
			expect.objectContaining({
				schemaVersion: "2",
				name: "literature_survey",
				version: "1.0.0",
				visibility: "public",
			}),
		);
		expect(skillPackage.manifest.runtime).toEqual(
			expect.objectContaining({
				kind: "python",
				entrypoint: "scripts/run_survey_phase.py",
				lockfile: "requirements.lock",
				requiredSecrets: ["OPENAI_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"],
				secretAliases: {
					OPENAI_API_KEY: ["LLM_API_KEY"],
					SEMANTIC_SCHOLAR_API_KEY: ["S2_API_KEY"],
				},
			}),
		);
		expect(skillPackage.manifest.permissions?.tools).toEqual(["bash", "read"]);
		expect(skillPackage.manifest.config?.environment?.map((variable) => variable.name)).toEqual(
			expect.arrayContaining(["LLM_BASE_URL", "LLM_MODEL", "LLM_CONTEXT_WINDOW"]),
		);
		expect(skillPackage.manifest.ui?.argumentHint).toBe("<topic> [options]");
		const commandArguments = skillPackage.manifest.ui?.arguments ?? [];
		expect(commandArguments.map((argument) => argument.name)).toEqual([
			"--topic",
			"--graph",
			"--language",
			"--depth",
			"--max-papers",
			"--min-papers",
			"--facet",
			"--full-text",
			"--resume",
			"--input",
			"--papers",
			"--paper-set",
		]);
		expect(commandArguments.find((argument) => argument.name === "--depth")?.choices).toEqual([
			"brief",
			"standard",
			"deep",
		]);
		expect(commandArguments.find((argument) => argument.name === "--facet")?.repeatable).toBe(true);
		const parserSource = readFileSync(
			join(skillPackage.packageDir, "scripts/literature_survey_lib/inputs.py"),
			"utf-8",
		);
		const parserArguments = [...parserSource.matchAll(/parser\.add_argument\("(--[a-z-]+)"/g)].map(
			(match) => match[1],
		);
		expect(new Set(commandArguments.map((argument) => argument.name))).toEqual(new Set(parserArguments));
		const publicPackages = discoverXlabSkillPackages(tempDir).packages.filter(
			(candidate) => candidate.manifest.visibility !== "internal",
		);
		expect(publicPackages.every((candidate) => Boolean(candidate.manifest.ui?.argumentHint))).toBe(true);
		expect(skillPackage.manifest.ui?.phases?.map((phase) => phase.id)).toEqual([
			"init",
			"ingest_sources",
			"ingest_graph_context",
			"cluster",
			"survey_agent",
			"citation_traces",
			"render_artifacts",
			"audit",
		]);
		expect(skillPackage.manifest.artifacts?.map((artifact) => [artifact.type, artifact.schemaVersion])).toEqual([
			["literature_survey_document", "1"],
			["literature_survey_json", "1"],
			["citation_trace", "1"],
			["survey_report", "1"],
		]);
		expect(validateXlabPackageSelfContained(skillPackage.packageDir)).toEqual([]);
	});

	it("initializes, validates, and blocks untraceable citations through hooks", async () => {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-literature-survey-hooks-"));
		temporaryDirectories.push(tempDir);
		const skillPackage = setupLiteratureSurveyProject(tempDir);
		const runner = new XlabHookRunner(skillPackage);
		const run = createRun(tempDir, skillPackage, "survey-run");
		for (const key of LITERATURE_SURVEY_ENV_OVERRIDES) {
			delete process.env[key];
		}
		const context = {
			run,
			skill: skillPackage.manifest,
			cwd: tempDir,
			packageDir: skillPackage.packageDir,
			runDir: run.runDir,
			args: run.args,
		};

		const preStart = await runner.run("pre_start", context);
		expect(preStart.ok).toBe(true);
		expect(JSON.stringify(preStart.state_patch)).toContain("Literature survey initialized");
		expect(JSON.stringify(preStart.state_patch)).toContain("MiniMax-M3");
		expect(JSON.stringify(preStart.state_patch)).toContain("SEMANTIC_SCHOLAR_API_KEY");

		writeSuccessfulSurveyFixture(tempDir, run);
		const manifest = readRunJson(run, "manifest.json");
		expect(manifest.schema_version).toBe("2");
		expect(manifest.status).toBe("success");
		const accepted = await runner.run("pre_finish", {
			...context,
			event: { finish: { status: "success" }, manifest },
		});
		expect(accepted.ok).toBe(true);
		expect(JSON.stringify(accepted.state_patch)).toContain("Literature survey validated");
		expect(accepted.state_patch).toMatchObject({
			artifacts: expect.arrayContaining([
				expect.objectContaining({ type: "literature_survey_document", path: expect.stringContaining("survey.md") }),
				expect.objectContaining({ type: "literature_survey_json", path: expect.stringContaining("survey.json") }),
				expect.objectContaining({ type: "citation_trace", path: expect.stringContaining("citations.json") }),
				expect.objectContaining({ type: "survey_report", path: expect.stringContaining("survey_report.json") }),
			]),
		});

		const citationsPath = join(run.runDir, "artifacts", "citations.json");
		const citations = JSON.parse(readFileSync(citationsPath, "utf-8"));
		citations.traces[0].paper_ids.push("missing-paper");
		writeFileSync(citationsPath, `${JSON.stringify(citations, null, 2)}\n`, "utf-8");

		const rejected = await runner.run("pre_finish", {
			...context,
			event: { finish: { status: "success" }, manifest },
		});
		expect(rejected.ok).toBe(false);
		expect(rejected.repair).toContain("Resolve the audit blockers");
		expect(JSON.stringify(rejected.state_patch)).toContain("missing-paper");
	});

	it("allows only packaged production phases and scoped read-only inspection", async () => {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-literature-survey-policy-"));
		temporaryDirectories.push(tempDir);
		const skillPackage = setupLiteratureSurveyProject(tempDir);
		const runner = new XlabHookRunner(skillPackage);
		const run = createRun(tempDir, skillPackage, "survey-run");
		const context = {
			run,
			skill: skillPackage.manifest,
			cwd: tempDir,
			packageDir: skillPackage.packageDir,
			runDir: run.runDir,
			args: run.args,
		};
		const script = join(skillPackage.packageDir, "scripts", "run_survey_phase.py");
		const productionCommand = `python ${script} synthesize --run-id ${run.runId} --run-dir ${run.runDir}`;
		const allowed = await runner.run("pre_tool_call", {
			...context,
			event: { toolName: "bash", input: { command: productionCommand } },
		});
		expect(allowed.ok).toBe(true);

		mkdirSync(join(run.runDir, "state"), { recursive: true });
		writeFileSync(join(run.runDir, "state", "checkpoint.json"), "{}\n", "utf-8");
		const inspection = await runner.run("pre_tool_call", {
			...context,
			event: {
				toolName: "bash",
				input: { command: `jq . ${join(run.runDir, "state", "checkpoint.json")}` },
			},
		});
		expect(inspection.ok).toBe(true);

		for (const command of [
			`python ${script} smoke --run-id ${run.runId} --run-dir ${run.runDir}`,
			`${productionCommand} && ls ${run.runDir}`,
			`${productionCommand} || true`,
			`${productionCommand} | tee ${join(run.runDir, "logs", "output.log")}`,
			`${productionCommand} > ${join(run.runDir, "logs", "output.log")}`,
			`${productionCommand} $(id)`,
			`curl https://api.openai.com/v1/models`,
			`python /tmp/Xcientist-2/run.py`,
			`find ${run.runDir} -delete`,
			`python -c 'open("${join(run.runDir, "artifacts", "survey.json")}", "w")'`,
		]) {
			const rejected = await runner.run("pre_tool_call", {
				...context,
				event: { toolName: "bash", input: { command } },
			});
			expect(rejected.ok, command).toBe(false);
		}
	});

	it("surfaces resumable checkpoint and partial artifacts on cancellation", async () => {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-literature-survey-cancel-"));
		temporaryDirectories.push(tempDir);
		const skillPackage = setupLiteratureSurveyProject(tempDir);
		const runner = new XlabHookRunner(skillPackage);
		const run = createRun(tempDir, skillPackage, "survey-run");
		writeSuccessfulSurveyFixture(tempDir, run);

		const cancelled = await runner.run("on_cancel", {
			run,
			skill: skillPackage.manifest,
			cwd: tempDir,
			packageDir: skillPackage.packageDir,
			runDir: run.runDir,
			args: run.args,
		});

		expect(cancelled.ok).toBe(true);
		expect(cancelled.state_patch).toMatchObject({
			phase: expect.objectContaining({ status: "incomplete" }),
			checkpoint: expect.objectContaining({ resumable: true }),
			artifacts: expect.arrayContaining([
				expect.objectContaining({ type: "literature_survey_document", status: "partial" }),
				expect.objectContaining({ type: "citation_trace", status: "partial" }),
			]),
		});
		expect(JSON.stringify(cancelled.state_patch)).toContain("render_artifacts");
	});

	it("surfaces legal incomplete synthesis results instead of treating them as crashes", async () => {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-literature-survey-incomplete-"));
		temporaryDirectories.push(tempDir);
		const skillPackage = setupLiteratureSurveyProject(tempDir);
		const runner = new XlabHookRunner(skillPackage);
		const run = createRun(tempDir, skillPackage, "survey-run");
		const context = {
			run,
			skill: skillPackage.manifest,
			cwd: tempDir,
			packageDir: skillPackage.packageDir,
			runDir: run.runDir,
			args: run.args,
		};

		mkdirSync(join(run.runDir, "artifacts"), { recursive: true });
		writeFileSync(
			join(run.runDir, "manifest.json"),
			`${JSON.stringify(
				{
					schema_version: "2",
					run_id: run.runId,
					skill_name: "literature_survey",
					skill_version: "1.0.0",
					status: "incomplete",
					created_at: "2026-01-01T00:00:00Z",
					inputs: {},
					outputs: {},
					validation: { passed: false },
					artifacts: [],
				},
				null,
				2,
			)}\n`,
			"utf-8",
		);
		writeFileSync(
			join(run.runDir, "artifacts", "survey_report.json"),
			`${JSON.stringify(
				{
					schema_version: "xlab.literature_survey.report.v1",
					passed: false,
					blocking_errors: ["Survey Agent generation skipped: OPENAI_API_KEY is missing."],
					warnings: [],
					counts: { papers: 3, sections: 0, claims: 0, gaps: 0 },
					checks: { survey_agent_completed: false },
					blocked_stages: ["survey_agent"],
				},
				null,
				2,
			)}\n`,
			"utf-8",
		);

		const result = await runner.run("post_tool_result", {
			...context,
			event: {
				toolName: "bash",
				input: { command: "python scripts/run_survey_phase.py synthesize" },
				isError: true,
				result: "Command exited with code 2",
			},
		});

		expect(result.ok).toBe(true);
		expect(JSON.stringify(result.state_patch)).toContain("incomplete");
		expect(JSON.stringify(result.state_patch)).toContain("Survey Agent generation skipped");
	});
});
