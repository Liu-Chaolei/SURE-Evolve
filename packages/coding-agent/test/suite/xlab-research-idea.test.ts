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
const TOPIC = "Survey-grounded scientific agents";
const RESEARCH_IDEA_ALGORITHM_ID = "xlab.research_idea.algorithm.v2";
const RESEARCH_IDEA_ALGORITHM_SPEC_VERSION = "xlab.research_idea.algorithm-spec.v2";
const SURVEY_ARGS = `--survey .xlab/runs/survey-run --topic "${TOPIC}"`;
const RESEARCH_IDEA_ENV_OVERRIDES = [
	"LLM_MODEL",
	"LLM_BASE_URL",
	"LLM_API_BASE",
	"OPENAI_API_KEY",
	"LLM_API_KEY",
	"XLAB_RESEARCH_IDEA_FAKE_LLM",
	"XLAB_RESEARCH_IDEA_ENABLE_LLM",
] as const;

function copyRepositorySkill(tempDir: string, skillName: string): void {
	const target = join(tempDir, "xlab", "skills", skillName);
	mkdirSync(join(tempDir, "xlab", "skills"), { recursive: true });
	cpSync(join(REPOSITORY_ROOT, "xlab", "skills", skillName), target, {
		recursive: true,
	});
}

function setupResearchIdeaProject(tempDir: string): XlabSkillPackage {
	copyRepositorySkill(tempDir, "paper_collect");
	copyRepositorySkill(tempDir, "knowledge_graph");
	copyRepositorySkill(tempDir, "literature_survey");
	copyRepositorySkill(tempDir, "research_idea");
	const discovered = discoverXlabSkillPackages(tempDir);
	expect(discovered.diagnostics.filter((diagnostic) => diagnostic.type === "error")).toEqual([]);
	const skillPackage = discovered.packages.find((entry) => entry.manifest.name === "research_idea");
	expect(skillPackage).toBeDefined();
	return skillPackage!;
}

function createRun(tempDir: string, skillPackage: XlabSkillPackage, runId: string, args = SURVEY_ARGS): XlabRunRecord {
	return {
		runId,
		skillName: skillPackage.manifest.name,
		skillVersion: skillPackage.manifest.version,
		command: "generate-research-ideas",
		status: "running",
		cwd: tempDir,
		packageDir: skillPackage.packageDir,
		runDir: join(tempDir, ".xlab", "runs", runId),
		args,
		startedAt: new Date().toISOString(),
		updatedAt: new Date().toISOString(),
	};
}

function writeSurveyFixture(tempDir: string, options?: { passed?: boolean; runtimeMode?: string }): string {
	const passed = options?.passed ?? true;
	const runtimeMode = options?.runtimeMode ?? "survey-agent";
	const runDir = join(tempDir, ".xlab", "runs", "survey-run");
	const artifactsDir = join(runDir, "artifacts");
	mkdirSync(artifactsDir, { recursive: true });
	const references = [
		{
			paper_id: "p1",
			title: "Survey-Grounded Research Ideation",
			year: 2026,
			venue: "XLab",
		},
		{
			paper_id: "p2",
			title: "Memory Guided MCTS for Scientific Agents",
			year: 2025,
			venue: "Agents",
		},
		{
			paper_id: "p3",
			title: "Evidence Traceability in Experimental Design",
			year: 2024,
			venue: "Evaluation",
		},
	];
	const surveyPath = join(artifactsDir, "survey.json");
	const citationsPath = join(artifactsDir, "citations.json");
	const reportPath = join(artifactsDir, "survey_report.json");
	writeJson(surveyPath, {
		schema_version: "xlab.literature_survey.v1",
		run_id: "survey-run",
		topic: TOPIC,
		paper_count: references.length,
		sections: [
			{
				id: "section:grounding",
				title: "Grounded ideation",
				summary: "Research-idea workflows should ground candidate generation in traceable survey evidence.",
				paper_ids: ["p1", "p3"],
			},
		],
		clusters: [
			{
				id: "cluster:agents",
				name: "Scientific agents",
				paper_ids: ["p1", "p2"],
			},
		],
		key_claims: [
			{
				id: "claim:trace",
				claim: "Evidence traceability improves reliability of downstream experiment plans.",
				paper_ids: ["p3"],
			},
		],
		research_gaps: [
			{
				id: "gap:mcts",
				gap: "Few systems connect literature survey memory with MCTS-style idea evolution and executable evaluation plans.",
				paper_ids: ["p1", "p2", "p3"],
			},
		],
		references,
		runtime: { mode: runtimeMode, survey_engine: "integrated_research_survey" },
		generated_at: "2026-01-01T00:00:00Z",
	});
	writeJson(citationsPath, {
		schema_version: "xlab.citation_trace.v1",
		run_id: "survey-run",
		references,
		traces: [
			{
				claim_id: "claim:trace",
				claim: "Evidence traceability improves reliability of downstream experiment plans.",
				paper_ids: ["p3"],
				evidence: ["Fixture evidence."],
			},
		],
	});
	writeJson(reportPath, {
		schema_version: "xlab.literature_survey.report.v1",
		generated_at: "2026-01-01T00:00:00Z",
		passed,
		blocking_errors: passed ? [] : ["Survey Agent generation failed."],
		blockers: [],
		warnings: [],
		counts: {
			papers: references.length,
			sections: 1,
			claims: 1,
			gaps: 1,
			traces: 1,
		},
		checks: { survey_agent_mode: runtimeMode === "survey-agent" },
	});
	writeJson(join(runDir, "manifest.json"), {
		schema_version: "2",
		run_id: "survey-run",
		skill_name: "literature_survey",
		skill_version: "1.0.0",
		status: passed ? "success" : "incomplete",
		created_at: "2026-01-01T00:00:00Z",
		inputs: {},
		outputs: { topic: TOPIC },
		validation: { passed },
		artifacts: [
			{ type: "literature_survey_json", schema_version: "1", path: surveyPath },
			{ type: "citation_trace", schema_version: "1", path: citationsPath },
			{ type: "survey_report", schema_version: "1", path: reportPath },
		],
	});
	return runDir;
}

function writeSuccessfulIdeaFixture(tempDir: string, run: XlabRunRecord): void {
	writeSurveyFixture(tempDir);
	const artifactsDir = join(run.runDir, "artifacts");
	const stateDir = join(run.runDir, "state");
	const researchIdeaStateDir = join(stateDir, "research_idea");
	mkdirSync(artifactsDir, { recursive: true });
	mkdirSync(researchIdeaStateDir, { recursive: true });
	const sourceContext = {
		schema_version: "xlab.research_idea.source_context.v1",
		topic: TOPIC,
		survey_path: join(tempDir, ".xlab", "runs", "survey-run"),
		reference_count: 3,
		evidence_count: 2,
		references: [
			{
				paper_id: "p1",
				title: "Survey-Grounded Research Ideation",
				year: 2026,
				venue: "XLab",
			},
			{
				paper_id: "p2",
				title: "Memory Guided MCTS for Scientific Agents",
				year: 2025,
				venue: "Agents",
			},
			{
				paper_id: "p3",
				title: "Evidence Traceability in Experimental Design",
				year: 2024,
				venue: "Evaluation",
			},
		],
		selected_evidence: [
			{
				paper_ids: ["p1"],
				title: "Grounded ideation",
				summary: "Research-idea workflows should ground candidate generation in traceable survey evidence.",
			},
			{
				paper_ids: ["p3"],
				title: "Traceability",
				summary: "Evidence traceability improves reliability of downstream experiment plans.",
			},
		],
		source_signatures: { survey: "fixture-survey-signature" },
	};
	const algorithmProvenance = {
		algorithm: RESEARCH_IDEA_ALGORITHM_ID,
		runtime_profile: "xlab.research_idea.runtime.v1",
		evidence_profile: "xlab.research_idea.evidence.v1",
		success_profile: "xlab.research_idea.success.v1",
	};
	const requiredModes = [
		"moonshot_inventor",
		"bridge_builder",
		"steady_engineer",
		"ambitious_realist",
		"evidence_first",
	];
	const workflowTrace = [
		{
			workflow: "xlab.research_idea.workflow.v1",
			stage: "knowledge_acquisition",
			status: "success",
		},
		{
			workflow: "xlab.research_idea.workflow.v1",
			stage: "advanced_analysis",
			status: "success",
		},
		{
			workflow: "xlab.research_idea.workflow.v1",
			stage: "idea_generation",
			status: "success",
		},
	];
	const operationTrace = [
		{
			event: "llm_call",
			stage: "advanced_analysis",
			op_name: "advanced_analysis",
			status: "success",
			model: "gpt-5-mini",
		},
		{
			event: "llm_call",
			stage: "idea_generation",
			op_name: "idea_materialization",
			status: "success",
			model: "gpt-5.4",
		},
		{
			event: "llm_call",
			stage: "idea_fusion",
			op_name: "idea_fusion",
			status: "success",
			model: "gpt-5.4",
		},
	];
	const mctsEvolution = {
		total_iterations: requiredModes.length,
		iterations: requiredModes.map((mode, index) => ({
			iteration: index + 1,
			node_id: index + 1,
			depth: 1,
			title: `Candidate ${mode}`,
			operator: "memory_guided_edit",
			defects: ["underspecified_evaluation"],
			score: 0.8,
			visits: 1,
			path: ["root", mode],
			evaluation: { novelty: 4, feasibility: 4, composite: 0.8 },
			idea_taste_mode: mode,
		})),
		pareto_front: Object.fromEntries(requiredModes.map((mode) => [mode, { score: 0.8, title: `Candidate ${mode}` }])),
	};
	const ideaResult = {
		title: "Fused survey-grounded idea",
		abstract: "A provider-generated fused idea grounded in survey evidence.",
		core_contribution: "Memory-guided survey evidence fusion for scientific agents.",
		research_question: "How can scientific agents use survey evidence for idea search?",
		hypothesis: "Survey-grounded MCTS improves idea traceability.",
		method: "Run survey-grounded retrieval, MCTS candidate search, and fused evaluation.",
		experiment_plan: ["Compare provider-generated ideas against non-grounded ideation baselines."],
		data_requirements: ["Completed SurveyAgent literature survey with citation traces."],
		baselines: ["Non-grounded ideation baseline."],
		metrics: ["Traceability, novelty, feasibility, and protocol completeness."],
		risks: ["Provider and evidence coverage limitations."],
		introduction: "Survey evidence motivates a memory-guided scientific-agent idea.",
		components: [
			{
				component: "Evidence memory",
				explanation: "Grounds each idea component.",
			},
		],
		algorithm: [{ name: "MCTS fusion", steps: ["retrieve", "search", "fuse"] }],
		reference_papers: ["Survey-Grounded Research Ideation"],
		mcts_evolution: mctsEvolution,
		idea_source: "fused",
		source_modes: requiredModes,
		fusion_metadata: {
			host_idea_mode: "fusion_agent",
			selected_components: ["Evidence memory"],
		},
		fusion_evolution: {
			source_modes: requiredModes,
			selected_components: ["Evidence memory"],
			ranked_candidates: requiredModes.map((mode) => ({ mode, score: 0.8 })),
		},
		workflow_provenance: {
			algorithm: "xlab-research-idea",
			package_native_workflow: true,
			required_modes: requiredModes,
			completed_modes: requiredModes,
			all_modes_completed: true,
			fusion_required: true,
			fusion_used: true,
			fusion_metadata_present: true,
			mcts_iterations: requiredModes.length,
			pareto_front_present: true,
			search_trace_present: true,
		},
	};
	const researchIdea = {
		schema_version: "xlab.research_idea.v2",
		status: "success",
		blockers: [],
		completed_stages: ["ingest_survey", "organize_context", "idea_generation", "materialize_xlab_artifact", "audit"],
		algorithm_provenance: algorithmProvenance,
		topic: TOPIC,
		research_question: "How can scientific agents use survey evidence for idea search?",
		hypothesis: "Survey-grounded MCTS improves idea traceability.",
		method: "Run evidence retrieval, MCTS, and fusion.",
		expected_contribution: "A traceable research ideation method.",
		experiment_plan: ["Compare against non-grounded ideation baselines."],
		data_requirements: ["completed literature survey"],
		baselines: ["non-grounded ideation"],
		metrics: ["traceability", "novelty", "feasibility"],
		risks: ["retrieval coverage"],
		source_evidence: [{ paper_ids: ["p1"], summary: "Survey evidence" }],
		source_context: sourceContext,
		idea_result: ideaResult,
		workflow_trace: workflowTrace,
		operation_trace: [],
	};
	const ideaTrace = {
		schema_version: "xlab.research_idea.trace.v2",
		status: "success",
		blockers: [],
		completed_stages: researchIdea.completed_stages,
		algorithm_provenance: algorithmProvenance,
		run_id: run.runId,
		topic: TOPIC,
		workflow_trace: workflowTrace,
		operation_trace: operationTrace,
		source_context: sourceContext,
		mcts_evolution: mctsEvolution,
		fusion_evolution: ideaResult.fusion_evolution,
		fusion_metadata: ideaResult.fusion_metadata,
		workflow_namespaces: ["xlab.research_idea.workflow.v1"],
	};
	const researchIdeaWorkflowState = {
		run: { workflow_trace: workflowTrace, operation_trace: operationTrace },
		retrieval: {
			rag_hits: [{ query: "survey evidence", hits: sourceContext.selected_evidence }],
			references: sourceContext.references,
		},
		analysis: {
			entries: [{ tldr: "Provider-generated analysis." }],
			root_idea: { title: "Root idea", method: "Survey-grounded MCTS." },
		},
		ideation: {
			latest_candidate: { title: ideaResult.title },
			mode_candidates: requiredModes.map((mode) => ({
				idea_taste_mode: mode,
			})),
			fusion_result: { fused_entry: { title: ideaResult.title } },
		},
		persistence: { idea_result: ideaResult },
	};
	const resources = Object.fromEntries(
		["survey", "graph", "component_index", "keynotes", "outcome_model", "component_model"].map((role) => [
			role,
			{
				resource_id: `fixture-${role}`,
				logical_uri:
					role === "outcome_model" || role === "component_model"
						? `xlab-cache://models/fixture-${role}`
						: `xlab-resource://fixture/${role}`,
			},
		]),
	);
	const resourcePreflight = {
		schema_version: "xlab.research_idea.resource_preflight.v1",
		implementation: {
			name: "xlab-native-research-idea",
			algorithm_spec: RESEARCH_IDEA_ALGORITHM_ID,
			algorithm_spec_version: RESEARCH_IDEA_ALGORITHM_SPEC_VERSION,
			success_policy: "xlab.research_idea.success.v1",
			resource_profile: "xlab.research_idea.evidence.v1",
			lightweight_success_paths: false,
		},
		passed: true,
		checks: {
			provider_available: true,
			required_python_modules: true,
			graph_db_present: true,
			graph_db_has_nodes: true,
			component_index_dir_present: true,
			component_faiss_present: true,
			component_metadata_present: true,
			component_metadata_nonempty: true,
			outcome_sentence_transformer_present: true,
			component_novelty_model_present: true,
			keynote_cache_present: true,
			survey_resource_paths_resolved: true,
		},
		blocking_errors: [],
		warnings: [],
		resources,
	};
	writeJson(join(artifactsDir, "idea_result.json"), ideaResult);
	writeJson(join(artifactsDir, "research_idea.json"), researchIdea);
	writeJson(join(artifactsDir, "idea_trace.json"), ideaTrace);
	writeJson(join(artifactsDir, "idea_report.json"), {
		schema_version: "xlab.research_idea.report.v2",
		status: "success",
		generated_at: "2026-01-01T00:00:00Z",
		passed: true,
		blocking_errors: [],
		warnings: [],
		counts: {
			references: sourceContext.references.length,
			evidence: researchIdea.source_evidence.length,
			rag_hits: sourceContext.selected_evidence.length,
			components: ideaResult.components.length,
			algorithm_steps: ideaResult.algorithm.length,
			baselines: researchIdea.baselines.length,
			metrics: researchIdea.metrics.length,
			candidates: requiredModes.length,
			mcts_iterations: requiredModes.length,
			research_idea_modes: requiredModes.length,
			workflow_stages: workflowTrace.length,
			operations: operationTrace.length * 2,
			provider_success_ops: 3,
		},
		checks: {
			research_idea_workflow_trace: true,
			provider_required_ops_present: true,
		},
	});
	writeJson(join(stateDir, "resource_preflight.json"), resourcePreflight);
	writeJson(join(researchIdeaStateDir, "artifact.json"), researchIdeaWorkflowState);
	writeJson(join(stateDir, "checkpoint.json"), {
		schema_version: "xlab.research_idea.checkpoint.v2",
		state_layout_id: "xlab.research_idea.state-layout.v1",
		phase: "audit",
		completed_stages: [
			"ingest_survey",
			"organize_context",
			"resource_preflight",
			"knowledge_acquisition",
			"advanced_analysis",
			"idea_generation",
			"materialize_xlab_artifact",
			"audit",
		],
		source_signatures: { survey: "fixture-survey-signature" },
		counts: {
			references: 3,
			evidence: 1,
			rag_hits: 2,
			components: 1,
			mcts_iterations: requiredModes.length,
		},
		warnings: [],
	});
	writeJson(join(run.runDir, "manifest.json"), {
		schema_version: "2",
		run_id: run.runId,
		skill_name: "research_idea",
		skill_version: "1.0.0",
		status: "success",
		created_at: "2026-01-01T00:00:00Z",
		inputs: {},
		outputs: { topic: TOPIC, title: ideaResult.title },
		validation: { passed: true },
		artifacts: [
			{
				type: "research_idea",
				schema_version: "2",
				path: "artifacts/research_idea.json",
			},
			{
				type: "research_idea_result",
				schema_version: "2",
				path: "artifacts/idea_result.json",
			},
			{
				type: "research_idea_trace",
				schema_version: "2",
				path: "artifacts/idea_trace.json",
			},
			{
				type: "research_idea_report",
				schema_version: "2",
				path: "artifacts/idea_report.json",
			},
		],
	});
}

function readRunJson(run: XlabRunRecord, relativePath: string): Record<string, unknown> {
	return JSON.parse(readFileSync(join(run.runDir, relativePath), "utf-8")) as Record<string, unknown>;
}

function writeJson(path: string, value: unknown): void {
	writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
}

describe("research_idea XLab skill", () => {
	const temporaryDirectories: string[] = [];

	afterEach(() => {
		for (const directory of temporaryDirectories.splice(0)) {
			rmSync(directory, { recursive: true, force: true });
		}
	});

	it("is discoverable as a self-contained Harness v2 package", () => {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-research-idea-"));
		temporaryDirectories.push(tempDir);
		const skillPackage = setupResearchIdeaProject(tempDir);

		expect(skillPackage.manifest).toEqual(
			expect.objectContaining({
				schemaVersion: "2",
				name: "research_idea",
				version: "2.0.0",
				visibility: "public",
			}),
		);
		expect(skillPackage.manifest.runtime).toEqual(
			expect.objectContaining({
				kind: "python",
				entrypoint: "scripts/run_idea_phase.py",
				lockfile: "requirements.lock",
				requiredSecrets: ["OPENAI_API_KEY"],
			}),
		);
		expect(skillPackage.manifest.config?.environment?.map((variable) => variable.name)).toContain("OPENAI_BASE_URL");
		expect(
			skillPackage.manifest.config?.environment?.find((variable) => variable.name === "OPENAI_BASE_URL"),
		).toEqual(
			expect.objectContaining({
				secret: false,
				required: false,
				default: "https://api.openai.com/v1",
			}),
		);
		expect(skillPackage.manifest.artifacts?.map((artifact) => [artifact.type, artifact.schemaVersion])).toEqual([
			["research_idea", "2"],
			["research_idea_result", "2"],
			["research_idea_trace", "2"],
			["research_idea_report", "2"],
		]);
		expect(validateXlabPackageSelfContained(skillPackage.packageDir)).toEqual([]);
	});

	it.each([
		["algorithm/prompts/idea_fusion.py"],
		["algorithm/runtime_adapters.py"],
		["resources/evidence.py"],
		["resources/embedding.py"],
	])("fails pre-start when package-native runtime file %s is missing", async (relativePath) => {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-research-idea-preflight-"));
		temporaryDirectories.push(tempDir);
		const skillPackage = setupResearchIdeaProject(tempDir);
		rmSync(join(skillPackage.packageDir, "scripts", "research_idea_lib", relativePath));
		const run = createRun(tempDir, skillPackage, "idea-run");
		const runner = new XlabHookRunner(skillPackage);

		const result = await runner.run("pre_start", {
			run,
			skill: skillPackage.manifest,
			cwd: tempDir,
			packageDir: skillPackage.packageDir,
			runDir: run.runDir,
			args: run.args,
		});

		expect(result.ok).toBe(false);
		expect(result.repair).toContain("package-native runtime and resource files");
		expect(JSON.stringify(result.state_patch)).toContain("package-native resource preflight failed");
		expect(JSON.stringify(result.state_patch)).toContain(relativePath.split("/").at(-1));
		expect(JSON.stringify(result.state_patch)).not.toContain("xcientist");
	});

	it("initializes, validates, and blocks incomplete idea success through hooks", async () => {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-research-idea-hooks-"));
		temporaryDirectories.push(tempDir);
		const skillPackage = setupResearchIdeaProject(tempDir);
		const runner = new XlabHookRunner(skillPackage);
		const run = createRun(tempDir, skillPackage, "idea-run");
		for (const key of RESEARCH_IDEA_ENV_OVERRIDES) {
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
		expect(JSON.stringify(preStart.state_patch)).toContain("Research idea initialized");
		expect(JSON.stringify(preStart.state_patch)).toContain("OPENAI_API_KEY is unavailable");
		expect(JSON.stringify(preStart.state_patch)).toContain(
			`Package-native ${RESEARCH_IDEA_ALGORITHM_ID} structural checks passed`,
		);
		expect(JSON.stringify(preStart.state_patch)).toContain("declared runtime resources");

		writeSuccessfulIdeaFixture(tempDir, run);
		rmSync(join(skillPackage.packageDir, "scripts", "run_idea_phase.py"));
		const manifest = readRunJson(run, "manifest.json");
		expect(manifest.schema_version).toBe("2");
		expect(manifest.status).toBe("success");
		const accepted = await runner.run("pre_finish", {
			...context,
			event: { finish: { status: "success" }, manifest },
		});
		expect(accepted.ok).toBe(true);
		expect(JSON.stringify(accepted.state_patch)).toContain("Research idea validated");
		expect(accepted.state_patch).toMatchObject({
			artifacts: expect.arrayContaining([
				expect.objectContaining({
					type: "research_idea",
					path: expect.stringContaining("research_idea.json"),
				}),
				expect.objectContaining({
					type: "research_idea_result",
					path: expect.stringContaining("idea_result.json"),
				}),
				expect.objectContaining({
					type: "research_idea_trace",
					path: expect.stringContaining("idea_trace.json"),
				}),
				expect.objectContaining({
					type: "research_idea_report",
					path: expect.stringContaining("idea_report.json"),
				}),
			]),
		});

		const reportPath = join(run.runDir, "artifacts", "idea_report.json");
		const report = JSON.parse(readFileSync(reportPath, "utf-8"));
		report.passed = false;
		report.blocking_errors = ["source_evidence is empty"];
		writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf-8");
		manifest.status = "incomplete";
		writeJson(join(run.runDir, "manifest.json"), manifest);

		const rejected = await runner.run("pre_finish", {
			...context,
			event: { finish: { status: "success" }, manifest },
		});
		expect(rejected.ok).toBe(false);
		expect(rejected.repair).toContain("Resolve the audit blockers");
		expect(JSON.stringify(rejected.state_patch)).toContain("source_evidence");
	});

	it("surfaces legal incomplete research-idea results instead of treating them as crashes", async () => {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-research-idea-incomplete-"));
		temporaryDirectories.push(tempDir);
		const skillPackage = setupResearchIdeaProject(tempDir);
		const runner = new XlabHookRunner(skillPackage);
		const run = createRun(tempDir, skillPackage, "idea-run");
		const context = {
			run,
			skill: skillPackage.manifest,
			cwd: tempDir,
			packageDir: skillPackage.packageDir,
			runDir: run.runDir,
			args: run.args,
		};

		mkdirSync(join(run.runDir, "artifacts"), { recursive: true });
		writeJson(join(run.runDir, "manifest.json"), {
			schema_version: "2",
			run_id: run.runId,
			skill_name: "research_idea",
			skill_version: "1.0.0",
			status: "incomplete",
			created_at: "2026-01-01T00:00:00Z",
			inputs: {},
			outputs: {},
			validation: { passed: false },
			artifacts: [],
		});
		writeJson(join(run.runDir, "artifacts", "idea_report.json"), {
			schema_version: "xlab.research_idea.report.v1",
			passed: false,
			blocking_errors: ["Research-idea generation skipped: OPENAI_API_KEY is missing."],
			warnings: [],
			counts: {
				references: 3,
				evidence: 0,
				rag_hits: 0,
				components: 0,
				candidates: 0,
				mcts_iterations: 0,
			},
			checks: { research_idea_completed: false },
			blocked_stages: ["research_idea"],
		});

		const result = await runner.run("post_tool_result", {
			...context,
			event: {
				toolName: "bash",
				input: { command: "python scripts/run_idea_phase.py synthesize" },
				isError: true,
				details: { status: "incomplete" },
			},
		});

		expect(result.ok).toBe(true);
		expect(JSON.stringify(result.state_patch)).toContain("incomplete");
		expect(JSON.stringify(result.state_patch)).toContain("Research-idea generation skipped");
	});

	it("blocks direct MCP, provider, and external-checkout bypasses", async () => {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-research-idea-blocks-"));
		temporaryDirectories.push(tempDir);
		const skillPackage = setupResearchIdeaProject(tempDir);
		const runner = new XlabHookRunner(skillPackage);
		const run = createRun(tempDir, skillPackage, "idea-run");
		const context = {
			run,
			skill: skillPackage.manifest,
			cwd: tempDir,
			packageDir: skillPackage.packageDir,
			runDir: run.runDir,
			args: run.args,
		};

		const mcp = await runner.run("pre_tool_call", {
			...context,
			event: { toolName: "mcp__paperscope__search_papers", input: {} },
		});
		expect(mcp.ok).toBe(false);
		expect(mcp.repair).toContain("run_idea_phase.py");

		const provider = await runner.run("pre_tool_call", {
			...context,
			event: {
				toolName: "bash",
				input: { command: "curl https://api.openai.com/v1/chat/completions" },
			},
		});
		expect(provider.ok).toBe(false);
		expect(provider.repair).toContain("run_idea_phase.py");

		const external = await runner.run("pre_tool_call", {
			...context,
			event: {
				toolName: "bash",
				input: {
					command: "python /tmp/external-checkout/src/agents/idea_agent/run.py",
				},
			},
		});
		expect(external.ok).toBe(false);
		expect(external.repair).toContain("package-local");

		const fakeArtifact = await runner.run("pre_tool_call", {
			...context,
			event: {
				toolName: "bash",
				input: {
					command: "python -c 'print({})' > .xlab/runs/idea-run/artifacts/research_idea.json",
				},
			},
		});
		expect(fakeArtifact.ok).toBe(false);
		expect(fakeArtifact.repair).toContain("run_idea_phase.py");
	});

	it("enforces external_mutations=false for non-Bash tools", async () => {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-research-idea-mutations-"));
		temporaryDirectories.push(tempDir);
		const skillPackage = setupResearchIdeaProject(tempDir);
		const runner = new XlabHookRunner(skillPackage);
		const run = createRun(tempDir, skillPackage, "idea-run");
		const context = {
			run,
			skill: skillPackage.manifest,
			cwd: tempDir,
			packageDir: skillPackage.packageDir,
			runDir: run.runDir,
			args: run.args,
		};
		const gate = (toolName: string, input: Record<string, unknown>) =>
			runner.run("pre_tool_call", { ...context, event: { toolName, input } });

		expect(
			(
				await gate("write", {
					file_path: join(run.runDir, "state", "notes.json"),
				})
			).ok,
		).toBe(true);
		expect(
			(
				await gate("edit", {
					path: join(run.runDir, "artifacts", "draft.json"),
				})
			).ok,
		).toBe(true);
		expect(
			(
				await gate("read", {
					file_path: join(skillPackage.packageDir, "SKILL.md"),
				})
			).ok,
		).toBe(true);
		for (const [toolName, input] of [
			["write", { file_path: join(tempDir, "outside.json") }],
			["edit", { path: join(run.runDir, "..", "other-run", "state.json") }],
			["write", {}],
			["edit", { path: "$HOME/.bashrc" }],
			["read", { file_path: join(tempDir, "outside.json") }],
			["notebook", { notebook_path: join(tempDir, "outside.ipynb") }],
		] as const) {
			expect((await gate(toolName, input)).ok, `${toolName} ${JSON.stringify(input)}`).toBe(false);
		}
	});

	it("allows only whole packaged CLI commands and scoped read-only inspections", async () => {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-research-idea-commands-"));
		temporaryDirectories.push(tempDir);
		const skillPackage = setupResearchIdeaProject(tempDir);
		const runner = new XlabHookRunner(skillPackage);
		const run = createRun(tempDir, skillPackage, "idea-run");
		const script = join(skillPackage.packageDir, "scripts", "run_idea_phase.py");
		const context = {
			run,
			skill: skillPackage.manifest,
			cwd: tempDir,
			packageDir: skillPackage.packageDir,
			runDir: run.runDir,
			args: run.args,
		};
		const bash = (command: string) =>
			runner.run("pre_tool_call", {
				...context,
				event: { toolName: "bash", input: { command } },
			});
		const canonical = `python ${script} synthesize --run-dir ${run.runDir} --run-id ${run.runId} --cwd ${tempDir}`;

		for (const command of [
			canonical,
			`ls ${run.runDir}`,
			`find ${run.runDir} -maxdepth 2 -type f`,
			`rg idea ${join(run.runDir, "artifacts", "request.json")}`,
			`python -m json.tool ${join(run.runDir, "artifacts", "request.json")}`,
		]) {
			expect((await bash(command)).ok, command).toBe(true);
		}
		for (const command of [
			`${canonical}; touch ${join(tempDir, "escaped")}`,
			`${canonical} && rm -rf ${tempDir}`,
			`${canonical} > ${join(tempDir, "escaped")}`,
			`${canonical} $(touch ${join(tempDir, "escaped")})`,
			`echo ${canonical}`,
			`find ${run.runDir} -delete`,
			`find ${run.runDir} -exec touch ${join(tempDir, "escaped")} ;`,
			`ls ${run.runDir}; touch ${join(tempDir, "escaped")}`,
			`rg --pre python idea ${join(run.runDir, "artifacts", "request.json")}`,
			`ls ${tempDir}`,
			`python ${script} synthesize --run-dir ${join(tempDir, "other-run")} --run-id ${run.runId}`,
		]) {
			expect((await bash(command)).ok, command).toBe(false);
		}
	});
});
