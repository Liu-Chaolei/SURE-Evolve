import { spawnSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ENV_AGENT_DIR } from "../../src/config.ts";
import { createAgentSessionServices } from "../../src/core/agent-session-services.ts";
import type { ExtensionFactory, ExtensionUIContext } from "../../src/core/extensions/index.ts";
import { XlabArtifactStore } from "../../src/core/xlab/artifact-store.ts";
import { xlabExtension } from "../../src/core/xlab/index.ts";
import { XLAB_CONTROL_COMMANDS, XLAB_PRODUCT_COMMANDS } from "../../src/core/xlab/product-commands.ts";
import type { XlabManifestEnvelope, XlabRunRecord, XlabRunStatus } from "../../src/core/xlab/types.ts";
import { XlabWorkspaceManager } from "../../src/core/xlab/workspace-manager.ts";
import { createHarness, getUserTexts, type Harness } from "./harness.ts";

const REPOSITORY_ROOT = fileURLToPath(new URL("../../../..", import.meta.url));

function writeJson(path: string, value: unknown): void {
	writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
}

function setupSkillPackage(
	tempDir: string,
	options?: {
		dirName?: string;
		name?: string;
		hook?: string;
		hooks?: Record<string, Array<Record<string, unknown>>>;
		artifacts?: Array<{
			type?: string;
			path?: string;
			required?: boolean;
			description?: string;
		}>;
		root?: ".xlab/skills" | "xlab/skills";
		prompt?: string;
		requiredSecrets?: string[];
		runtime?: Record<string, unknown>;
		ui?: Record<string, unknown>;
		workflow?: { stages: Array<Record<string, unknown>> };
	},
): void {
	const name = options?.name ?? "paper_collect";
	const skillDir = join(tempDir, options?.root ?? ".xlab/skills", options?.dirName ?? name);
	mkdirSync(join(skillDir, "skill"), { recursive: true });
	mkdirSync(join(skillDir, "hooks"), { recursive: true });
	writeFileSync(
		join(skillDir, "skill", "SKILL.md"),
		options?.prompt ?? "Collect evidence and write the final manifest.",
		"utf-8",
	);
	if (options?.hook) {
		writeFileSync(join(skillDir, "hooks", "index.ts"), options.hook, "utf-8");
	}
	writeJson(join(skillDir, "xlab.skill.json"), {
		schema_version: "2",
		name,
		version: "1.0.0",
		visibility: "public",
		description: "Demo XLab paper collection",
		prompt: "skill/SKILL.md",
		runtime: options?.runtime ?? {
			kind: "agent",
			required_secrets: options?.requiredSecrets,
		},
		workflow: options?.workflow,
		ui: options?.ui,
		hooks:
			options?.hooks ??
			(options?.hook
				? {
						pre_finish: [{ module: "hooks/index.ts", handler: "preFinish" }],
					}
				: undefined),
		artifacts: options?.artifacts,
	});
}

async function createXlabHarness(options?: {
	projectTrusted?: boolean;
	input?: ExtensionUIContext["input"];
	uiContext?: Partial<ExtensionUIContext>;
	tools?: AgentTool[];
	initialActiveToolNames?: string[];
}): Promise<Harness> {
	const harness = await createHarness({
		extensionFactories: [xlabExtension],
		settings: {},
		tools: options?.tools,
		initialActiveToolNames: options?.initialActiveToolNames,
	});
	if (options?.projectTrusted !== undefined) {
		harness.settingsManager.setProjectTrusted(options.projectTrusted);
	}
	const baseUiContext = harness.session.extensionRunner.getUIContext();
	const uiContext =
		options?.input || options?.uiContext
			? {
					...baseUiContext,
					...options?.uiContext,
					...(options?.input ? { input: options.input } : {}),
				}
			: undefined;
	await harness.session.bindExtensions({
		uiContext,
		mode: uiContext ? "tui" : undefined,
		onError: (error) => {
			throw new Error(JSON.stringify(error));
		},
		commandContextActions: {
			waitForIdle: () => harness.session.agent.waitForIdle(),
			newSession: async () => ({ cancelled: false }),
			fork: async () => ({ cancelled: false }),
			navigateTree: async () => ({ cancelled: false }),
			switchSession: async () => ({ cancelled: false }),
			reload: async () => {},
		},
	});
	return harness;
}

function writeValidManifest(
	tempDir: string,
	runId: string,
	overrides: Partial<{
		schema_version: string;
		run_id: string;
		skill_name: string;
		skill_version: string;
		status: string;
		created_at: string;
		inputs: unknown;
		outputs: unknown;
		validation: unknown;
		artifacts: unknown;
	}> = {},
): void {
	writeJson(join(tempDir, ".xlab", "runs", runId, "manifest.json"), {
		schema_version: "2",
		run_id: runId,
		skill_name: "paper_collect",
		skill_version: "1.0.0",
		status: "success",
		created_at: new Date().toISOString(),
		inputs: {},
		outputs: {},
		validation: {},
		artifacts: [],
		...overrides,
	});
}

async function waitForCondition(predicate: () => boolean): Promise<void> {
	for (let i = 0; i < 100; i++) {
		if (predicate()) {
			return;
		}
		await new Promise((resolve) => setTimeout(resolve, 10));
	}
}

function getOnlyRunId(tempDir: string): string {
	const entries = readdirSync(join(tempDir, ".xlab", "runs"), {
		withFileTypes: true,
	});
	const runIds = entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name);
	expect(runIds).toHaveLength(1);
	return runIds[0];
}

function readRunState(tempDir: string, runId: string): unknown {
	return JSON.parse(readFileSync(join(tempDir, ".xlab", "runs", runId, "state.json"), "utf-8"));
}

function readWorkflowState(tempDir: string, runId: string): unknown {
	return JSON.parse(readFileSync(join(tempDir, ".xlab", "runs", runId, "workflow.json"), "utf-8"));
}

function fixtureRun(tempDir: string, runId: string, status: XlabRunStatus, skillName = "paper_collect"): XlabRunRecord {
	const timestamp = "2026-08-05T00:00:00.000Z";
	return {
		runId,
		skillName,
		skillVersion: "1.0.0",
		command: skillName === "paper_collect" ? "collect-papers" : skillName,
		status,
		cwd: tempDir,
		packageDir: join(tempDir, ".xlab", "skills", skillName),
		runDir: join(tempDir, ".xlab", "runs", runId),
		args: "",
		startedAt: timestamp,
		updatedAt: timestamp,
	};
}

function fixtureManifest(run: XlabRunRecord, artifacts: XlabManifestEnvelope["artifacts"]): XlabManifestEnvelope {
	return {
		schema_version: "2",
		run_id: run.runId,
		skill_name: run.skillName,
		skill_version: run.skillVersion,
		status: run.status,
		created_at: run.startedAt,
		inputs: {},
		outputs: {},
		validation: { passed: run.status === "success" },
		artifacts,
	};
}

function createAgentDir(): string {
	const agentDir = join(REPOSITORY_ROOT, ".tmp", `xlab-agent-${Date.now()}-${Math.random().toString(36).slice(2)}`);
	mkdirSync(agentDir, { recursive: true });
	return agentDir;
}

function createWriteTool(): AgentTool {
	return {
		name: "write",
		label: "Write",
		description: "Write test tool",
		parameters: Type.Object({ path: Type.String(), content: Type.String() }),
		execute: async () => ({
			content: [{ type: "text", text: "write:ok" }],
			details: {},
		}),
	};
}

function createReadTool(): AgentTool {
	return {
		name: "read",
		label: "Read",
		description: "Read test tool",
		parameters: Type.Object({ path: Type.String() }),
		execute: async (_toolCallId, params) => ({
			content: [
				{
					type: "text",
					text: `read:${String((params as { path: string }).path)}`,
				},
			],
			details: {},
		}),
	};
}

function latestToolResultText(harness: Harness): string {
	const result = harness.session.messages.filter((message) => message.role === "toolResult").at(-1);
	return result && "content" in result ? JSON.stringify(result.content) : "";
}

function latestToolResultIsError(harness: Harness): boolean {
	const result = harness.session.messages.filter((message) => message.role === "toolResult").at(-1);
	return Boolean(result && "isError" in result && result.isError);
}

function copyRepositoryPaperCollectSkill(tempDir: string): void {
	cpSync(join(REPOSITORY_ROOT, "xlab", "skills", "paper_collect"), join(tempDir, "xlab", "skills", "paper_collect"), {
		filter: (source) => !source.split(/[\\/]/).includes("__pycache__"),
		recursive: true,
	});
}

function writePaperCollectFixture(tempDir: string, runId: string): void {
	const runDir = join(tempDir, ".xlab", "runs", runId);
	const artifactsDir = join(runDir, "artifacts");
	const metadataDir = join(artifactsDir, "metadata");
	const logsDir = join(artifactsDir, "logs");
	mkdirSync(metadataDir, { recursive: true });
	mkdirSync(logsDir, { recursive: true });
	const papers = [
		{
			id: "S2-GNN-REVIEW",
			paper_id: "S2-GNN-REVIEW",
			title: "Graph Neural Networks: A Review of Methods and Applications",
			authors: [{ author_id: "A1", name: "Ada Author" }],
			external_ids: { DOI: "10.1000/gnn.review" },
			pdf_candidates: [],
			discovery: [
				{
					provider: "semantic_scholar",
					kind: "semantic_search",
					operation: "semantic_scholar.search_papers",
				},
			],
			dedupe_key: "s2:S2-GNN-REVIEW",
			abstract: "This review organizes graph neural network methods and applications.",
			year: 2021,
			metadata_status: "complete",
			relevance: {
				tier: "core",
				score: 0.96,
				lexical_score: 0.95,
				direct_score: 0.8,
				graph_support: 1,
			},
			download: {
				status: "unavailable",
				path: null,
				url: null,
				sha256: null,
				bytes: null,
				error: null,
			},
		},
		{
			id: "S2-GCN",
			paper_id: "S2-GCN",
			title: "Semi-Supervised Classification with Graph Convolutional Networks",
			authors: [{ author_id: "A2", name: "Bob Author" }],
			external_ids: { DOI: "10.1000/gcn" },
			pdf_candidates: [],
			discovery: [
				{
					provider: "semantic_scholar",
					kind: "reference",
					operation: "semantic_scholar.get_paper_references",
				},
			],
			dedupe_key: "s2:S2-GCN",
			abstract: "We study semi-supervised node classification with graph convolutional networks.",
			year: 2017,
			metadata_status: "complete",
			relevance: {
				tier: "related",
				score: 0.75,
				lexical_score: 0.72,
				direct_score: 0.4,
				graph_support: 1,
			},
			download: {
				status: "unavailable",
				path: null,
				url: null,
				sha256: null,
				bytes: null,
				error: null,
			},
		},
	];
	writeJson(join(artifactsDir, "papers.manifest.json"), {
		schema_version: "xlab.paper_set.v3",
		query: "graph neural networks",
		target_count: 2,
		max_count: 10,
		candidate_count: papers.length,
		collected_count: papers.length,
		seed_count: 1,
		edge_count: 1,
		generated_at: new Date().toISOString(),
		papers,
		search: {
			round: 1,
			followup_count: 0,
			stop_reason: "target_reached",
			novelty_ratio: 1,
		},
		quality: {
			tiers: { core: 1, related: 1, boundary: 0 },
			metadata: { complete: 2, partial: 0, minimal: 0, invalid: 0 },
			abstract_coverage: 1,
			graph_connected_ratio: 1,
		},
		files: {
			papers: "artifacts/metadata/papers.jsonl",
			edges: "artifacts/metadata/edges.jsonl",
		},
	});
	writeFileSync(
		join(metadataDir, "papers.jsonl"),
		`${papers.map((paper) => JSON.stringify(paper)).join("\n")}\n`,
		"utf-8",
	);
	writeFileSync(
		join(metadataDir, "edges.jsonl"),
		`${JSON.stringify({ schema_version: "xlab.paper_edge.v1", source_paper_id: "S2-GNN-REVIEW", target_paper_id: "S2-GCN", relation: "cites", discovered_from: "semantic_scholar_references", seed_id: "S2-GNN-REVIEW" })}\n`,
		"utf-8",
	);
	writeFileSync(
		join(logsDir, "provider_results.jsonl"),
		`${[
			{
				provider: "tavily",
				operation: "tavily.search",
				input: { query: "graph neural networks survey review research papers" },
				is_error: false,
			},
			{
				provider: "semantic_scholar",
				operation: "semantic_scholar.search_papers",
				input: { query: "graph neural networks", limit: 100 },
				is_error: false,
			},
		]
			.map((record) => JSON.stringify(record))
			.join("\n")}\n`,
		"utf-8",
	);
}

describe("XLab extension", () => {
	const cleanups: Array<() => void> = [];

	afterEach(() => {
		while (cleanups.length > 0) {
			cleanups.pop()?.();
		}
		vi.unstubAllEnvs();
	});

	it("starts a skill run from a slash command and activates xlab_finish only for that run", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir);

		expect(harness.session.getActiveToolNames()).not.toContain("xlab_finish");
		harness.setResponses([fauxAssistantMessage("working")]);

		await harness.session.prompt("/xlab collect-papers graph neural networks");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => getUserTexts(harness).length > 0);

		expect(getUserTexts(harness)[0]).toContain("<xlab_invocation");
		expect(getUserTexts(harness)[0]).toContain("graph neural networks");
		expect(harness.session.getActiveToolNames()).toContain("xlab_finish");
		expect(harness.session.getActiveToolNames()).toContain("xlab_update_state");
		expect(harness.session.getActiveToolNames()).toContain("xlab_stage");
		expect(existsSync(join(harness.tempDir, ".xlab", "runs"))).toBe(true);
	});

	it("publishes an XLab dashboard widget when a skill run starts", async () => {
		const widgetUpdates: Array<{
			key: string;
			content: string[] | undefined;
			placement?: string;
		}> = [];
		const statusUpdates: Array<{ key: string; text: string | undefined }> = [];
		const harness = await createXlabHarness({
			uiContext: {
				setStatus: (key, text) => statusUpdates.push({ key, text }),
				setWidget: (key, content, options) => {
					widgetUpdates.push({
						key,
						content: Array.isArray(content) ? content : undefined,
						placement: options?.placement,
					});
				},
			},
		});
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir);
		harness.setResponses([fauxAssistantMessage("working")]);

		await harness.session.prompt("/xlab collect-papers graph neural networks");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => widgetUpdates.some((update) => update.key === "xlab.dashboard" && update.content));

		let dashboard: string[] = [];
		for (let index = widgetUpdates.length - 1; index >= 0; index -= 1) {
			const update = widgetUpdates[index];
			if (update.key === "xlab.dashboard" && update.content) {
				dashboard = update.content;
				break;
			}
		}
		expect(dashboard.join("\n")).toContain("XLab /xlab collect-papers");
		expect(dashboard.join("\n")).toContain("Setup: /xlab check-xlab-setup");
		expect(widgetUpdates.some((update) => update.placement === "aboveEditor")).toBe(true);
		expect(
			statusUpdates.some((update) => update.key === "xlab" && update.text?.includes("XLab xlab collect-papers")),
		).toBe(true);
		expect(harness.session.getActiveToolNames()).toContain("xlab_stage");
	});

	it("initializes missing required secrets through masked dialogs", async () => {
		const firstSecret = "XLAB_TEST_REQUIRED_SECRET_A";
		const secondSecret = "XLAB_TEST_REQUIRED_SECRET_B";
		const firstValue = "first-secret-value";
		const secondValue = "second-secret-value";
		const previousFirst = process.env[firstSecret];
		const previousSecond = process.env[secondSecret];
		delete process.env[firstSecret];
		delete process.env[secondSecret];
		const prompts: Array<{ title: string; secret: boolean | undefined }> = [];
		const input: ExtensionUIContext["input"] = async (title, _placeholder, options) => {
			prompts.push({ title, secret: options?.secret });
			return title.includes(firstSecret) ? firstValue : secondValue;
		};
		const harness = await createXlabHarness({ input });
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			requiredSecrets: [firstSecret, secondSecret],
		});
		harness.setResponses([fauxAssistantMessage("working")]);

		try {
			await harness.session.prompt("/xlab collect-papers topic");
			await harness.session.agent.waitForIdle();
			await waitForCondition(() => getUserTexts(harness).length > 0);

			expect(prompts).toEqual([
				{ title: expect.stringContaining(firstSecret), secret: true },
				{ title: expect.stringContaining(secondSecret), secret: true },
			]);
			expect(process.env[firstSecret]).toBe(firstValue);
			expect(process.env[secondSecret]).toBe(secondValue);
			expect(JSON.stringify(harness.session.messages)).not.toContain(firstValue);
			expect(JSON.stringify(harness.session.messages)).not.toContain(secondValue);
			const runId = getOnlyRunId(harness.tempDir);
			const runRecord = readFileSync(join(harness.tempDir, ".xlab", "runs", runId, "run.json"), "utf-8");
			expect(runRecord).not.toContain(firstValue);
			expect(runRecord).not.toContain(secondValue);
		} finally {
			if (previousFirst === undefined) {
				delete process.env[firstSecret];
			} else {
				process.env[firstSecret] = previousFirst;
			}
			if (previousSecond === undefined) {
				delete process.env[secondSecret];
			} else {
				process.env[secondSecret] = previousSecond;
			}
		}
	});

	it("prompts knowledge_graph for its separate extraction LLM API", async () => {
		const previousUrl = process.env.KNOWLEDGE_GRAPH_LLM_API_URL;
		const previousKey = process.env.KNOWLEDGE_GRAPH_LLM_API_KEY;
		const previousModel = process.env.KNOWLEDGE_GRAPH_LLM_MODEL;
		delete process.env.KNOWLEDGE_GRAPH_LLM_API_URL;
		delete process.env.KNOWLEDGE_GRAPH_LLM_API_KEY;
		delete process.env.KNOWLEDGE_GRAPH_LLM_MODEL;
		const endpointValue = "https://kg-llm.example.test/v1";
		const keyValue = "kg-extraction-secret";
		const modelValue = "graph-extractor-model";
		const prompts: Array<{
			title: string;
			placeholder: string | undefined;
			secret: boolean | undefined;
		}> = [];
		const input: ExtensionUIContext["input"] = async (title, placeholder, options) => {
			prompts.push({ title, placeholder, secret: options?.secret });
			if (title.includes("KNOWLEDGE_GRAPH_LLM_API_URL")) {
				return endpointValue;
			}
			if (title.includes("KNOWLEDGE_GRAPH_LLM_API_KEY")) {
				return keyValue;
			}
			return modelValue;
		};
		const harness = await createXlabHarness({ input });
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			name: "knowledge_graph",
			requiredSecrets: ["KNOWLEDGE_GRAPH_LLM_API_URL", "KNOWLEDGE_GRAPH_LLM_API_KEY", "KNOWLEDGE_GRAPH_LLM_MODEL"],
		});
		harness.setResponses([fauxAssistantMessage("working")]);

		try {
			await harness.session.prompt("/xlab build-knowledge-graph previous-paper-collect-run");
			await harness.session.agent.waitForIdle();
			await waitForCondition(() => getUserTexts(harness).length > 0);

			expect(prompts).toHaveLength(3);
			expect(prompts.map((prompt) => prompt.title).join("\n")).toContain("separate LLM API for paper extraction");
			expect(prompts.map((prompt) => prompt.title).join("\n")).toContain(
				"independent from the Pi Agent chat/coding model",
			);
			expect(prompts[0]).toMatchObject({
				title: expect.stringContaining("paper-extraction LLM endpoint URL"),
				placeholder: expect.stringContaining("/chat/completions"),
				secret: false,
			});
			expect(prompts[1]).toMatchObject({
				title: expect.stringContaining("paper-extraction LLM API key"),
				placeholder: expect.stringContaining("extraction LLM API key"),
				secret: true,
			});
			expect(prompts[2]).toMatchObject({
				title: expect.stringContaining("paper-extraction LLM model id"),
				placeholder: expect.stringContaining("deepseek-chat"),
				secret: false,
			});
			expect(process.env.KNOWLEDGE_GRAPH_LLM_API_URL).toBe(endpointValue);
			expect(process.env.KNOWLEDGE_GRAPH_LLM_API_KEY).toBe(keyValue);
			expect(process.env.KNOWLEDGE_GRAPH_LLM_MODEL).toBe(modelValue);
			expect(JSON.stringify(harness.session.messages)).not.toContain(keyValue);
			const runId = getOnlyRunId(harness.tempDir);
			const runRecord = readFileSync(join(harness.tempDir, ".xlab", "runs", runId, "run.json"), "utf-8");
			expect(runRecord).not.toContain(keyValue);
		} finally {
			if (previousUrl === undefined) {
				delete process.env.KNOWLEDGE_GRAPH_LLM_API_URL;
			} else {
				process.env.KNOWLEDGE_GRAPH_LLM_API_URL = previousUrl;
			}
			if (previousKey === undefined) {
				delete process.env.KNOWLEDGE_GRAPH_LLM_API_KEY;
			} else {
				process.env.KNOWLEDGE_GRAPH_LLM_API_KEY = previousKey;
			}
			if (previousModel === undefined) {
				delete process.env.KNOWLEDGE_GRAPH_LLM_MODEL;
			} else {
				process.env.KNOWLEDGE_GRAPH_LLM_MODEL = previousModel;
			}
		}
	});

	it("does not create a run when required-secret initialization is cancelled", async () => {
		const firstSecret = "XLAB_TEST_ROLLED_BACK_SECRET";
		const secondSecret = "XLAB_TEST_CANCELLED_SECRET";
		const previousFirst = process.env[firstSecret];
		const previousSecond = process.env[secondSecret];
		delete process.env[firstSecret];
		delete process.env[secondSecret];
		const harness = await createXlabHarness({
			input: async (title) => (title.includes(firstSecret) ? "temporary-value" : undefined),
		});
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			requiredSecrets: [firstSecret, secondSecret],
		});

		try {
			await harness.session.prompt("/xlab collect-papers topic");
			await harness.session.agent.waitForIdle();
			expect(existsSync(join(harness.tempDir, ".xlab", "runs"))).toBe(false);
			expect(process.env[firstSecret]).toBeUndefined();
			expect(process.env[secondSecret]).toBeUndefined();
			expect(getUserTexts(harness)).toHaveLength(0);
		} finally {
			if (previousFirst === undefined) {
				delete process.env[firstSecret];
			} else {
				process.env[firstSecret] = previousFirst;
			}
			if (previousSecond === undefined) {
				delete process.env[secondSecret];
			} else {
				process.env[secondSecret] = previousSecond;
			}
		}
	});

	it("reports setup status without creating an XLab run", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);

		await harness.session.prompt("/xlab check-xlab-setup");

		const entriesText = JSON.stringify(harness.sessionManager.getEntries());
		expect(entriesText).toContain("XLab setup status");
		expect(entriesText).toContain("TAVILY_API_KEY");
		expect(entriesText).toContain("SEMANTIC_SCHOLAR_API_KEY");
		expect(existsSync(join(harness.tempDir, ".xlab", "runs"))).toBe(false);
		expect(getUserTexts(harness)).toHaveLength(0);
	});

	it("persists and switches an OpenAI-compatible runtime model through configure-xlab", async () => {
		const agentDir = createAgentDir();
		const previousAgentDir = process.env[ENV_AGENT_DIR];
		const previousOpenai = process.env.OPENAI_API_KEY;
		const previousLlmKey = process.env.LLM_API_KEY;
		const previousOpenaiBaseUrl = process.env.OPENAI_BASE_URL;
		const previousLlmBaseUrl = process.env.LLM_BASE_URL;
		const previousLlmModel = process.env.LLM_MODEL;
		const apiKey = "runtime-secret-value";
		const prompts: Array<{
			title: string;
			placeholder: string | undefined;
			secret: boolean | undefined;
		}> = [];
		const values = ["", "", "https://runtime.example.test/v1/chat/completions", "xlab-runtime-model", apiKey];
		const input: ExtensionUIContext["input"] = async (title, placeholder, options) => {
			prompts.push({ title, placeholder, secret: options?.secret });
			return values.shift();
		};
		vi.stubEnv(ENV_AGENT_DIR, agentDir);
		const harness = await createXlabHarness({ input });
		cleanups.push(harness.cleanup, () => rmSync(agentDir, { recursive: true, force: true }));

		try {
			await harness.session.prompt("/xlab configure-xlab runtime");

			expect(prompts.map((prompt) => prompt.title).join("\n")).toContain("OpenAI-compatible runtime API");
			expect(prompts).toEqual([
				{
					title: expect.stringContaining("provider id"),
					placeholder: "xlab-openai-compatible",
					secret: undefined,
				},
				{
					title: expect.stringContaining("display name"),
					placeholder: "XLab OpenAI-compatible",
					secret: undefined,
				},
				{
					title: expect.stringContaining("base URL"),
					placeholder: "https://api.example.com/v1",
					secret: false,
				},
				{
					title: expect.stringContaining("model id"),
					placeholder: expect.stringContaining("deepseek-chat"),
					secret: false,
				},
				{
					title: expect.stringContaining("API key"),
					placeholder: "Paste the runtime API key",
					secret: true,
				},
			]);

			const modelsJson = JSON.parse(readFileSync(join(agentDir, "models.json"), "utf-8"));
			expect(modelsJson.providers["xlab-openai-compatible"]).toMatchObject({
				name: "XLab OpenAI-compatible",
				baseUrl: "https://runtime.example.test/v1",
				api: "openai-completions",
				models: [{ id: "xlab-runtime-model", name: "xlab-runtime-model" }],
			});
			expect(JSON.stringify(modelsJson)).not.toContain(apiKey);
			expect(harness.authStorage.get("xlab-openai-compatible")).toEqual({
				type: "api_key",
				key: apiKey,
			});
			expect(harness.session.model?.provider).toBe("xlab-openai-compatible");
			expect(harness.session.model?.id).toBe("xlab-runtime-model");
			expect(harness.settingsManager.getDefaultProvider()).toBe("xlab-openai-compatible");
			expect(harness.settingsManager.getDefaultModel()).toBe("xlab-runtime-model");
			expect(process.env.OPENAI_API_KEY).toBe(apiKey);
			expect(process.env.LLM_API_KEY).toBe(apiKey);
			expect(process.env.OPENAI_BASE_URL).toBe("https://runtime.example.test/v1");
			expect(process.env.LLM_BASE_URL).toBe("https://runtime.example.test/v1");
			expect(process.env.LLM_MODEL).toBe("xlab-runtime-model");

			const entriesText = JSON.stringify(harness.sessionManager.getEntries());
			expect(entriesText).toContain("Saved and switched Pi Agent runtime model");
			expect(entriesText).toContain("openai-completions");
			expect(entriesText).not.toContain(apiKey);
			expect(JSON.stringify(harness.session.messages)).not.toContain(apiKey);

			await harness.session.prompt("/xlab check-xlab-setup");
			const statusText = JSON.stringify(harness.sessionManager.getEntries());
			expect(statusText).toContain("Pi Agent runtime model: configured (xlab-openai-compatible/xlab-runtime-model)");
			expect(statusText).not.toContain(apiKey);
			expect(existsSync(join(harness.tempDir, ".xlab", "runs"))).toBe(false);
		} finally {
			if (previousAgentDir === undefined) {
				delete process.env[ENV_AGENT_DIR];
			} else {
				process.env[ENV_AGENT_DIR] = previousAgentDir;
			}
			if (previousOpenai === undefined) {
				delete process.env.OPENAI_API_KEY;
			} else {
				process.env.OPENAI_API_KEY = previousOpenai;
			}
			if (previousLlmKey === undefined) {
				delete process.env.LLM_API_KEY;
			} else {
				process.env.LLM_API_KEY = previousLlmKey;
			}
			if (previousOpenaiBaseUrl === undefined) {
				delete process.env.OPENAI_BASE_URL;
			} else {
				process.env.OPENAI_BASE_URL = previousOpenaiBaseUrl;
			}
			if (previousLlmBaseUrl === undefined) {
				delete process.env.LLM_BASE_URL;
			} else {
				process.env.LLM_BASE_URL = previousLlmBaseUrl;
			}
			if (previousLlmModel === undefined) {
				delete process.env.LLM_MODEL;
			} else {
				process.env.LLM_MODEL = previousLlmModel;
			}
		}
	});

	it("configures scholarly API keys through masked dialogs", async () => {
		const previousTavily = process.env.TAVILY_API_KEY;
		const previousSemanticScholar = process.env.SEMANTIC_SCHOLAR_API_KEY;
		const previousS2 = process.env.S2_API_KEY;
		delete process.env.TAVILY_API_KEY;
		delete process.env.SEMANTIC_SCHOLAR_API_KEY;
		delete process.env.S2_API_KEY;
		const tavilyValue = "tavily-secret-value";
		const semanticScholarValue = "semantic-secret-value";
		const prompts: Array<{ title: string; secret: boolean | undefined }> = [];
		const input: ExtensionUIContext["input"] = async (title, _placeholder, options) => {
			prompts.push({ title, secret: options?.secret });
			if (title.includes("TAVILY_API_KEY")) {
				return tavilyValue;
			}
			if (title.includes("SEMANTIC_SCHOLAR_API_KEY")) {
				return semanticScholarValue;
			}
			return "";
		};
		const harness = await createXlabHarness({ input });
		cleanups.push(harness.cleanup);

		try {
			await harness.session.prompt("/xlab configure-xlab scholarly-services");

			expect(prompts).toEqual([
				{ title: expect.stringContaining("TAVILY_API_KEY"), secret: true },
				{
					title: expect.stringContaining("SEMANTIC_SCHOLAR_API_KEY"),
					secret: true,
				},
				{ title: expect.stringContaining("S2_API_KEY"), secret: true },
			]);
			expect(process.env.TAVILY_API_KEY).toBe(tavilyValue);
			expect(process.env.SEMANTIC_SCHOLAR_API_KEY).toBe(semanticScholarValue);
			expect(process.env.S2_API_KEY).toBeUndefined();
			expect(JSON.stringify(harness.session.messages)).not.toContain(tavilyValue);
			expect(JSON.stringify(harness.session.messages)).not.toContain(semanticScholarValue);
			expect(JSON.stringify(harness.sessionManager.getEntries())).not.toContain(tavilyValue);
			expect(JSON.stringify(harness.sessionManager.getEntries())).not.toContain(semanticScholarValue);
			expect(existsSync(join(harness.tempDir, ".xlab", "runs"))).toBe(false);
		} finally {
			if (previousTavily === undefined) {
				delete process.env.TAVILY_API_KEY;
			} else {
				process.env.TAVILY_API_KEY = previousTavily;
			}
			if (previousSemanticScholar === undefined) {
				delete process.env.SEMANTIC_SCHOLAR_API_KEY;
			} else {
				process.env.SEMANTIC_SCHOLAR_API_KEY = previousSemanticScholar;
			}
			if (previousS2 === undefined) {
				delete process.env.S2_API_KEY;
			} else {
				process.env.S2_API_KEY = previousS2;
			}
		}
	});

	it("rolls back configuration values when setup is cancelled", async () => {
		const previousTavily = process.env.TAVILY_API_KEY;
		const previousSemanticScholar = process.env.SEMANTIC_SCHOLAR_API_KEY;
		delete process.env.TAVILY_API_KEY;
		delete process.env.SEMANTIC_SCHOLAR_API_KEY;
		const temporaryValue = "temporary-tavily-value";
		const harness = await createXlabHarness({
			input: async (title) => (title.includes("TAVILY_API_KEY") ? temporaryValue : undefined),
		});
		cleanups.push(harness.cleanup);

		try {
			await harness.session.prompt("/xlab configure-xlab scholarly-services");

			expect(process.env.TAVILY_API_KEY).toBeUndefined();
			expect(process.env.SEMANTIC_SCHOLAR_API_KEY).toBeUndefined();
			expect(JSON.stringify(harness.sessionManager.getEntries())).not.toContain(temporaryValue);
			expect(existsSync(join(harness.tempDir, ".xlab", "runs"))).toBe(false);
		} finally {
			if (previousTavily === undefined) {
				delete process.env.TAVILY_API_KEY;
			} else {
				process.env.TAVILY_API_KEY = previousTavily;
			}
			if (previousSemanticScholar === undefined) {
				delete process.env.SEMANTIC_SCHOLAR_API_KEY;
			} else {
				process.env.SEMANTIC_SCHOLAR_API_KEY = previousSemanticScholar;
			}
		}
	});

	it("prints export guidance when masked input is unavailable", async () => {
		const previousTavily = process.env.TAVILY_API_KEY;
		delete process.env.TAVILY_API_KEY;
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);

		try {
			await harness.session.prompt("/xlab configure-xlab scholarly-services");

			const entriesText = JSON.stringify(harness.sessionManager.getEntries());
			expect(entriesText).toContain("cannot prompt for integration values in this mode");
			expect(entriesText).toContain("export TAVILY_API_KEY=...");
			expect(existsSync(join(harness.tempDir, ".xlab", "runs"))).toBe(false);
		} finally {
			if (previousTavily === undefined) {
				delete process.env.TAVILY_API_KEY;
			} else {
				process.env.TAVILY_API_KEY = previousTavily;
			}
		}
	});

	it("prints runtime guidance when masked input is unavailable", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);

		await harness.session.prompt("/xlab configure-xlab runtime");

		const entriesText = JSON.stringify(harness.sessionManager.getEntries());
		expect(entriesText).toContain("/xlab configure-xlab runtime cannot prompt for values in this mode");
		expect(entriesText).toContain("OpenAI-compatible runtime provider");
		expect(entriesText).toContain("models.json should define a provider with api");
		expect(entriesText).not.toContain("KNOWLEDGE_GRAPH_LLM_API_KEY");
		expect(existsSync(join(harness.tempDir, ".xlab", "runs"))).toBe(false);
	});

	it("configures knowledge graph extraction LLM values", async () => {
		const previousUrl = process.env.KNOWLEDGE_GRAPH_LLM_API_URL;
		const previousKey = process.env.KNOWLEDGE_GRAPH_LLM_API_KEY;
		const previousModel = process.env.KNOWLEDGE_GRAPH_LLM_MODEL;
		delete process.env.KNOWLEDGE_GRAPH_LLM_API_URL;
		delete process.env.KNOWLEDGE_GRAPH_LLM_API_KEY;
		delete process.env.KNOWLEDGE_GRAPH_LLM_MODEL;
		const keyValue = "kg-init-secret";
		const prompts: Array<{ title: string; secret: boolean | undefined }> = [];
		const input: ExtensionUIContext["input"] = async (title, _placeholder, options) => {
			prompts.push({ title, secret: options?.secret });
			if (title.includes("KNOWLEDGE_GRAPH_LLM_API_URL")) return "https://kg-init.example.test/v1";
			if (title.includes("KNOWLEDGE_GRAPH_LLM_API_KEY")) return keyValue;
			return "kg-init-model";
		};
		const harness = await createXlabHarness({ input });
		cleanups.push(harness.cleanup);

		try {
			await harness.session.prompt("/xlab configure-xlab knowledge-graph");

			expect(prompts).toEqual([
				{
					title: expect.stringContaining("KNOWLEDGE_GRAPH_LLM_API_KEY"),
					secret: true,
				},
				{
					title: expect.stringContaining("KNOWLEDGE_GRAPH_LLM_API_URL"),
					secret: false,
				},
				{
					title: expect.stringContaining("KNOWLEDGE_GRAPH_LLM_MODEL"),
					secret: false,
				},
			]);
			expect(process.env.KNOWLEDGE_GRAPH_LLM_API_URL).toBe("https://kg-init.example.test/v1");
			expect(process.env.KNOWLEDGE_GRAPH_LLM_API_KEY).toBe(keyValue);
			expect(process.env.KNOWLEDGE_GRAPH_LLM_MODEL).toBe("kg-init-model");
			expect(JSON.stringify(harness.sessionManager.getEntries())).not.toContain(keyValue);
			expect(existsSync(join(harness.tempDir, ".xlab", "runs"))).toBe(false);
		} finally {
			if (previousUrl === undefined) {
				delete process.env.KNOWLEDGE_GRAPH_LLM_API_URL;
			} else {
				process.env.KNOWLEDGE_GRAPH_LLM_API_URL = previousUrl;
			}
			if (previousKey === undefined) {
				delete process.env.KNOWLEDGE_GRAPH_LLM_API_KEY;
			} else {
				process.env.KNOWLEDGE_GRAPH_LLM_API_KEY = previousKey;
			}
			if (previousModel === undefined) {
				delete process.env.KNOWLEDGE_GRAPH_LLM_MODEL;
			} else {
				process.env.KNOWLEDGE_GRAPH_LLM_MODEL = previousModel;
			}
		}
	});

	it("updates active run display state through xlab_update_state", async () => {
		const widgetUpdates: Array<string[] | undefined> = [];
		const secretValue = "dashboard-state-secret";
		const previousSecret = process.env.XLAB_TEST_API_KEY;
		process.env.XLAB_TEST_API_KEY = secretValue;
		const harness = await createXlabHarness({
			uiContext: {
				setWidget: (key, content) => {
					if (key === "xlab.dashboard") {
						widgetUpdates.push(Array.isArray(content) ? content : undefined);
					}
				},
			},
		});
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir);

		harness.setResponses([
			fauxAssistantMessage(
				fauxToolCall("xlab_update_state", {
					phase: {
						id: "search",
						label: "Searching papers",
						status: "running",
						progress: 0.4,
					},
					message: "Collected 12 candidates.",
					counters: { candidates: 12, target: 50 },
					next_actions: ["Continue citation expansion"],
				}),
			),
			fauxAssistantMessage("still working"),
		]);

		try {
			await harness.session.prompt("/xlab collect-papers topic");
			await harness.session.agent.waitForIdle();
			await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

			const runId = getOnlyRunId(harness.tempDir);
			expect(readRunState(harness.tempDir, runId)).toMatchObject({
				phase: {
					id: "search",
					label: "Searching papers",
					status: "running",
					progress: 0.4,
				},
				message: "Collected 12 candidates.",
				counters: { candidates: 12, target: 50 },
				next_actions: ["Continue citation expansion"],
			});
			expect(
				harness.sessionManager
					.getEntries()
					.some((entry) => entry.type === "custom" && entry.customType === "xlab.state"),
			).toBe(true);
			const dashboard = widgetUpdates.filter(Boolean).at(-1)?.join("\n") ?? "";
			expect(dashboard).toContain("Phase: Searching papers ▶ · 40%");
			expect(dashboard).toContain("Counters: candidates=12  target=50");
			expect(dashboard).toContain("Next: Continue citation expansion");
			expect(dashboard).not.toContain(secretValue);
		} finally {
			if (previousSecret === undefined) {
				delete process.env.XLAB_TEST_API_KEY;
			} else {
				process.env.XLAB_TEST_API_KEY = previousSecret;
			}
		}
	});

	it("rejects invalid xlab_update_state payloads without writing state", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir);

		harness.setResponses([
			fauxAssistantMessage(
				fauxToolCall("xlab_update_state", {
					counters: { candidates: "twelve" },
				}),
			),
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt("/xlab collect-papers topic");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		const runId = getOnlyRunId(harness.tempDir);
		const toolResult = harness.session.messages.find((message) => message.role === "toolResult");
		expect(toolResult && "isError" in toolResult ? toolResult.isError : false).toBe(true);
		expect(existsSync(join(harness.tempDir, ".xlab", "runs", runId, "state.json"))).toBe(false);
	});

	it("persists display state patches returned by hooks", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			hook: `export function preFinish() { return { ok: false, repair: "need more papers", state_patch: { phase: { id: "validate", label: "Validating collection", status: "blocked" }, counters: { collected: 3, target: 10 }, diagnostics: [{ severity: "warning", message: "Paper count below target.", repair: "Run citation expansion." }] } }; }`,
		});

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				writeValidManifest(harness.tempDir, runId);
				return fauxAssistantMessage(
					fauxToolCall("xlab_finish", {
						status: "success",
						manifest_path: `.xlab/runs/${runId}/manifest.json`,
						summary: "done",
					}),
				);
			},
			fauxAssistantMessage("will repair"),
		]);

		await harness.session.prompt("/xlab collect-papers topic");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		const runId = getOnlyRunId(harness.tempDir);
		expect(readRunState(harness.tempDir, runId)).toMatchObject({
			phase: {
				id: "validate",
				label: "Validating collection",
				status: "blocked",
			},
			counters: { collected: 3, target: 10 },
			diagnostics: [
				{
					severity: "warning",
					message: "Paper count below target.",
					repair: "Run citation expansion.",
				},
			],
		});
	});

	it("applies workflow stage hooks before committing stage transitions", async () => {
		const widgetUpdates: Array<string[] | undefined> = [];
		const harness = await createXlabHarness({
			uiContext: {
				setWidget: (key, content) => {
					if (key === "xlab.dashboard") {
						widgetUpdates.push(Array.isArray(content) ? content : undefined);
					}
				},
			},
		});
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			name: "paper_collect",
			runtime: { kind: "workflow" },
			workflow: {
				stages: [
					{ id: "collect", skill: "paper_collect", max_attempts: 2 },
					{
						id: "validate",
						skill: "paper_collect",
						needs: ["collect"],
						max_attempts: 1,
					},
				],
			},
			hook: `
				export function preStage(ctx) {
					return { ok: true, state_patch: { phase: { id: ctx.event.stage_id, label: "Gate " + ctx.event.stage_id, status: "running" } } };
				}
				export function postStage(ctx) {
					if (ctx.event.action === "complete" && !Array.isArray(ctx.event.artifacts)) {
						return { ok: false, repair: "typed artifacts required", state_patch: { phase: { id: ctx.event.stage_id, label: "Stage blocked", status: "blocked" } } };
					}
					return { ok: true, state_patch: { phase: { id: ctx.event.stage_id, label: "Completed " + ctx.event.stage_id, status: "success" }, counters: { stage_artifacts: ctx.event.artifacts?.length ?? 0 } } };
				}
			`,
			hooks: {
				pre_stage: [{ module: "hooks/index.ts", handler: "preStage" }],
				post_stage: [{ module: "hooks/index.ts", handler: "postStage" }],
			},
		});

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("xlab_stage", { action: "start", stage_id: "collect" })),
			fauxAssistantMessage(fauxToolCall("xlab_stage", { action: "complete", stage_id: "collect" })),
			fauxAssistantMessage(
				fauxToolCall("xlab_stage", {
					action: "complete",
					stage_id: "collect",
					artifacts: [
						{
							type: "dataset",
							schema_version: "1",
							path: "artifacts/dataset.json",
						},
					],
				}),
			),
			fauxAssistantMessage("stage complete"),
		]);

		await harness.session.prompt("/xlab collect-papers dataset benchmark");
		await harness.session.agent.waitForIdle();
		await waitForCondition(
			() => harness.session.messages.filter((message) => message.role === "toolResult").length >= 3,
		);

		const runId = getOnlyRunId(harness.tempDir);
		const toolResults = harness.session.messages.filter((message) => message.role === "toolResult");
		expect(toolResults[1] && "isError" in toolResults[1] ? toolResults[1].isError : false).toBe(true);
		expect(toolResults[1] && "content" in toolResults[1] ? JSON.stringify(toolResults[1].content) : "").toContain(
			"typed artifacts required",
		);
		expect(readRunState(harness.tempDir, runId)).toMatchObject({
			phase: { id: "collect", label: "Completed collect", status: "success" },
			counters: { stage_artifacts: 1 },
		});
		expect(readWorkflowState(harness.tempDir, runId)).toMatchObject({
			stages: [
				expect.objectContaining({
					id: "collect",
					status: "success",
					attempts: 1,
					artifacts: [
						{
							type: "dataset",
							schema_version: "1",
							path: "artifacts/dataset.json",
						},
					],
				}),
				expect.objectContaining({ id: "validate", status: "pending" }),
			],
		});
		const dashboard = widgetUpdates.filter(Boolean).at(-1)?.join("\n") ?? "";
		expect(dashboard).toContain("Workflow: collect ✓  validate ·");
		expect(dashboard).toContain("Counters: stage_artifacts=1");
	});

	it("does not start a skill run before project trust", async () => {
		const harness = await createXlabHarness({ projectTrusted: false });
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir);

		harness.setResponses([fauxAssistantMessage("should not run")]);

		await harness.session.prompt("/xlab collect-papers topic");
		await harness.session.agent.waitForIdle();

		expect(getUserTexts(harness)).toHaveLength(0);
		expect(existsSync(join(harness.tempDir, ".xlab", "runs"))).toBe(false);
		expect(harness.session.getActiveToolNames()).not.toContain("xlab_finish");
	});

	it("rejects an invalid experiment Idea before creating a run", async () => {
		const notifications: Array<{ message: string; level: string }> = [];
		const harness = await createXlabHarness({
			uiContext: {
				notify: (message, level) => notifications.push({ message, level }),
			},
		});
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, { name: "run_experiment" });

		harness.setResponses([fauxAssistantMessage("should not run")]);

		await harness.session.prompt("/xlab run-experiment --idea missing.json");
		await harness.session.agent.waitForIdle();

		expect(notifications).toContainEqual({
			message: "Cannot resolve Idea reference missing.json to an existing Research Idea JSON file.",
			level: "error",
		});
		expect(getUserTexts(harness)).toHaveLength(0);
		expect(existsSync(join(harness.tempDir, ".xlab", "runs"))).toBe(false);
		expect(harness.session.getActiveToolNames()).not.toContain("xlab_finish");
	});

	it("rejects xlab_finish with repair instructions when the manifest is missing", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir);

		harness.setResponses([
			fauxAssistantMessage(
				fauxToolCall("xlab_finish", {
					status: "success",
					manifest_path: ".xlab/runs/missing/manifest.json",
					summary: "done",
				}),
			),
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt("/xlab collect-papers topic");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		const toolResult = harness.session.messages.find((message) => message.role === "toolResult");
		expect(toolResult).toBeDefined();
		expect(toolResult && "isError" in toolResult ? toolResult.isError : false).toBe(true);
		expect(toolResult && "content" in toolResult ? JSON.stringify(toolResult.content) : "").toContain(
			"Create the final manifest",
		);
		expect(harness.session.getActiveToolNames()).toContain("xlab_finish");
	});

	it("finishes a run from a run-relative manifest and restores previous active tools", async () => {
		const widgetUpdates: Array<string[] | undefined> = [];
		const harness = await createXlabHarness({
			uiContext: {
				setWidget: (key, content) => {
					if (key === "xlab.dashboard") {
						widgetUpdates.push(Array.isArray(content) ? content : undefined);
					}
				},
			},
		});
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				writeValidManifest(harness.tempDir, runId);
				return fauxAssistantMessage(
					fauxToolCall("xlab_finish", {
						status: "success",
						manifest_path: "manifest.json",
						summary: "done",
					}),
				);
			},
		]);

		await harness.session.prompt("/xlab collect-papers topic");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		const toolResult = harness.session.messages.find((message) => message.role === "toolResult");
		expect(toolResult && "content" in toolResult ? JSON.stringify(toolResult.content) : "").toContain("finished");
		expect(harness.session.getActiveToolNames()).not.toContain("xlab_finish");
		expect(harness.session.getActiveToolNames()).not.toContain("xlab_update_state");
		expect(widgetUpdates.some((content) => content?.join("\n").includes("finished"))).toBe(true);
		expect(widgetUpdates.at(-1)).toBeUndefined();
	});

	it("removes only xlab_finish when a run ends", async () => {
		const extensionFactories: ExtensionFactory[] = [
			xlabExtension,
			(pi) => {
				pi.registerTool({
					name: "extra_tool",
					label: "Extra Tool",
					description: "Extra tool enabled during a run",
					promptSnippet: "Run extra test behavior",
					parameters: Type.Object({}),
					execute: async () => ({
						content: [{ type: "text", text: "ok" }],
						details: {},
					}),
				});
			},
		];
		const harness = await createHarness({ extensionFactories });
		cleanups.push(harness.cleanup);
		await harness.session.bindExtensions({
			commandContextActions: {
				waitForIdle: () => harness.session.agent.waitForIdle(),
				newSession: async () => ({ cancelled: false }),
				fork: async () => ({ cancelled: false }),
				navigateTree: async () => ({ cancelled: false }),
				switchSession: async () => ({ cancelled: false }),
				reload: async () => {},
			},
		});
		setupSkillPackage(harness.tempDir);
		harness.session.setActiveToolsByName(["read"]);

		harness.setResponses([
			() => {
				harness.session.setActiveToolsByName(["read", "extra_tool", "xlab_finish", "xlab_update_state"]);
				const runId = getOnlyRunId(harness.tempDir);
				writeValidManifest(harness.tempDir, runId);
				return fauxAssistantMessage(
					fauxToolCall("xlab_finish", {
						status: "success",
						manifest_path: `.xlab/runs/${runId}/manifest.json`,
						summary: "done",
					}),
				);
			},
		]);

		await harness.session.prompt("/xlab collect-papers topic");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(harness.session.getActiveToolNames().sort()).toEqual(["extra_tool", "read"]);
	});

	it("discovers repository skill packages from xlab/skills", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			root: "xlab/skills",
			prompt: "Repository skill prompt.",
		});

		harness.setResponses([fauxAssistantMessage("working")]);

		await harness.session.prompt("/xlab collect-papers topic");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => getUserTexts(harness).length > 0);

		expect(getUserTexts(harness)[0]).toContain("Repository skill prompt.");
	});

	it("lets project .xlab skills override repository skills for the same command", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			root: "xlab/skills",
			dirName: "repo",
			name: "paper_collect",
			prompt: "Repository skill prompt.",
		});
		setupSkillPackage(harness.tempDir, {
			root: ".xlab/skills",
			dirName: "project",
			name: "paper_collect",
			prompt: "Project skill prompt.",
		});

		harness.setResponses([fauxAssistantMessage("working")]);

		await harness.session.prompt("/xlab collect-papers topic");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => getUserTexts(harness).length > 0);

		expect(getUserTexts(harness)[0]).toContain("Project skill prompt.");
		expect(getUserTexts(harness)[0]).not.toContain("Repository skill prompt.");
	});

	it("rejects paper_collect finishes that submit a noncanonical manifest", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		copyRepositoryPaperCollectSkill(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				writePaperCollectFixture(harness.tempDir, runId);
				const runDir = join(harness.tempDir, ".xlab", "runs", runId);
				const scriptPath = join(harness.tempDir, "xlab", "skills", "paper_collect", "scripts", "paper_collect.py");
				const audit = spawnSync("python", [scriptPath, "audit", "--run-id", runId, "--run-dir", runDir], {
					encoding: "utf-8",
				});
				expect(audit.status).toBe(0);
				const canonical = JSON.parse(readFileSync(join(runDir, "manifest.json"), "utf-8"));
				writeJson(join(runDir, "alt-manifest.json"), {
					...canonical,
					artifacts: canonical.artifacts.map((artifact: Record<string, unknown>) =>
						artifact.type === "paper_set"
							? {
									...artifact,
									path: `.xlab/runs/${runId}/artifacts/request.json`,
								}
							: artifact,
					),
				});
				return fauxAssistantMessage(
					fauxToolCall("xlab_finish", {
						status: "success",
						manifest_path: `.xlab/runs/${runId}/alt-manifest.json`,
						summary: "done",
					}),
					{ stopReason: "toolUse" },
				);
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(latestToolResultIsError(harness)).toBe(true);
		expect(latestToolResultText(harness)).toContain("canonical manifest");
		expect(harness.session.getActiveToolNames()).toContain("xlab_finish");
	});

	it("rejects paper_collect manifests with noncanonical artifact paths", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		copyRepositoryPaperCollectSkill(harness.tempDir);
		const auditPath = join(
			harness.tempDir,
			"xlab",
			"skills",
			"paper_collect",
			"scripts",
			"paper_collect_lib",
			"audit.py",
		);
		writeFileSync(
			auditPath,
			readFileSync(auditPath, "utf-8").replace(
				'"path": f"{base}/papers.manifest.json"',
				'"path": f"{base}/request.json"',
			),
			"utf-8",
		);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				writePaperCollectFixture(harness.tempDir, runId);
				const runDir = join(harness.tempDir, ".xlab", "runs", runId);
				const scriptPath = join(harness.tempDir, "xlab", "skills", "paper_collect", "scripts", "paper_collect.py");
				const audit = spawnSync("python", [scriptPath, "audit", "--run-id", runId, "--run-dir", runDir], {
					encoding: "utf-8",
				});
				expect(audit.status).toBe(0);
				return fauxAssistantMessage(
					fauxToolCall("xlab_finish", {
						status: "success",
						manifest_path: `.xlab/runs/${runId}/manifest.json`,
						summary: "done",
					}),
					{ stopReason: "toolUse" },
				);
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(latestToolResultIsError(harness)).toBe(true);
		expect(latestToolResultText(harness)).toContain("canonical manifest generated");
		expect(harness.session.getActiveToolNames()).toContain("xlab_finish");
	});

	it("blocks paper_collect bash commands that target another run directory", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const bashRuns: string[] = [];
		const bashTool: AgentTool = {
			name: "bash",
			label: "Bash",
			description: "Bash test tool",
			parameters: Type.Object({ command: Type.String() }),
			execute: async (_toolCallId, params) => {
				bashRuns.push(String((params as { command: string }).command));
				return { content: [{ type: "text", text: "bash:ok" }], details: {} };
			},
		};
		const harness = await createXlabHarness({
			tools: [bashTool],
			initialActiveToolNames: ["bash"],
		});
		cleanups.push(harness.cleanup);
		copyRepositoryPaperCollectSkill(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				const scriptPath = join(harness.tempDir, "xlab", "skills", "paper_collect", "scripts", "paper_collect.py");
				return fauxAssistantMessage(
					fauxToolCall("bash", {
						command: `python ${scriptPath} audit --run-id ${runId} --run-dir ${join(harness.tempDir, "outside-run")}`,
					}),
					{ stopReason: "toolUse" },
				);
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(bashRuns).toHaveLength(0);
		expect(latestToolResultIsError(harness)).toBe(true);
		expect(latestToolResultText(harness)).toContain("for the active XLab run");
	});

	it("blocks paper_collect bash commands with duplicate run directories", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const bashRuns: string[] = [];
		const bashTool: AgentTool = {
			name: "bash",
			label: "Bash",
			description: "Bash test tool",
			parameters: Type.Object({ command: Type.String() }),
			execute: async (_toolCallId, params) => {
				bashRuns.push(String((params as { command: string }).command));
				return { content: [{ type: "text", text: "bash:ok" }], details: {} };
			},
		};
		const harness = await createXlabHarness({
			tools: [bashTool],
			initialActiveToolNames: ["bash"],
		});
		cleanups.push(harness.cleanup);
		copyRepositoryPaperCollectSkill(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				const runDir = join(harness.tempDir, ".xlab", "runs", runId);
				const scriptPath = join(harness.tempDir, "xlab", "skills", "paper_collect", "scripts", "paper_collect.py");
				return fauxAssistantMessage(
					fauxToolCall("bash", {
						command: `python ${scriptPath} audit --run-id ${runId} --run-dir ${runDir} --run-dir ${join(harness.tempDir, "outside-run")}`,
					}),
					{ stopReason: "toolUse" },
				);
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(bashRuns).toHaveLength(0);
		expect(latestToolResultIsError(harness)).toBe(true);
		expect(latestToolResultText(harness)).toContain("for the active XLab run");
	});

	it("blocks paper_collect bash commands with shell control syntax", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const bashRuns: string[] = [];
		const bashTool: AgentTool = {
			name: "bash",
			label: "Bash",
			description: "Bash test tool",
			parameters: Type.Object({ command: Type.String() }),
			execute: async (_toolCallId, params) => {
				bashRuns.push(String((params as { command: string }).command));
				return { content: [{ type: "text", text: "bash:ok" }], details: {} };
			},
		};
		const harness = await createXlabHarness({
			tools: [bashTool],
			initialActiveToolNames: ["bash"],
		});
		cleanups.push(harness.cleanup);
		copyRepositoryPaperCollectSkill(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				const runDir = join(harness.tempDir, ".xlab", "runs", runId);
				const scriptPath = join(harness.tempDir, "xlab", "skills", "paper_collect", "scripts", "paper_collect.py");
				return fauxAssistantMessage(
					fauxToolCall("bash", {
						command: `python ${scriptPath} audit --run-id ${runId} --run-dir ${runDir}; touch ${join(runDir, "artifacts", "forged")}`,
					}),
					{ stopReason: "toolUse" },
				);
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(bashRuns).toHaveLength(0);
		expect(latestToolResultIsError(harness)).toBe(true);
		expect(latestToolResultText(harness)).toContain("without shell control operators");
	});

	it("blocks relative paper_collect bash commands from the repository package path", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const bashRuns: string[] = [];
		const bashTool: AgentTool = {
			name: "bash",
			label: "Bash",
			description: "Bash test tool",
			parameters: Type.Object({ command: Type.String() }),
			execute: async (_toolCallId, params) => {
				bashRuns.push(String((params as { command: string }).command));
				return { content: [{ type: "text", text: "bash:ok" }], details: {} };
			},
		};
		const harness = await createXlabHarness({
			tools: [bashTool],
			initialActiveToolNames: ["bash"],
		});
		cleanups.push(harness.cleanup);
		copyRepositoryPaperCollectSkill(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				const runDir = join(harness.tempDir, ".xlab", "runs", runId);
				return fauxAssistantMessage(
					fauxToolCall("bash", {
						command: `python xlab/skills/paper_collect/scripts/paper_collect.py audit --run-id ${runId} --run-dir ${runDir}`,
					}),
					{ stopReason: "toolUse" },
				);
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(bashRuns).toHaveLength(0);
		expect(latestToolResultIsError(harness)).toBe(true);
		expect(latestToolResultText(harness)).toContain("Use the packaged paper_collect CLI");
	});

	it("blocks paper_collect bash commands from outside the packaged CLI path", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const bashRuns: string[] = [];
		const bashTool: AgentTool = {
			name: "bash",
			label: "Bash",
			description: "Bash test tool",
			parameters: Type.Object({ command: Type.String() }),
			execute: async (_toolCallId, params) => {
				bashRuns.push(String((params as { command: string }).command));
				return { content: [{ type: "text", text: "bash:ok" }], details: {} };
			},
		};
		const harness = await createXlabHarness({
			tools: [bashTool],
			initialActiveToolNames: ["bash"],
		});
		cleanups.push(harness.cleanup);
		copyRepositoryPaperCollectSkill(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				const runDir = join(harness.tempDir, ".xlab", "runs", runId);
				return fauxAssistantMessage(
					fauxToolCall("bash", {
						command: `paper_collect.py audit --run-id ${runId} --run-dir ${runDir}`,
					}),
					{ stopReason: "toolUse" },
				);
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(bashRuns).toHaveLength(0);
		expect(latestToolResultIsError(harness)).toBe(true);
		expect(latestToolResultText(harness)).toContain("Use the packaged paper_collect CLI");
	});

	it("blocks path-qualified paper_collect Python interpreters", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const bashRuns: string[] = [];
		const bashTool: AgentTool = {
			name: "bash",
			label: "Bash",
			description: "Bash test tool",
			parameters: Type.Object({ command: Type.String() }),
			execute: async (_toolCallId, params) => {
				bashRuns.push(String((params as { command: string }).command));
				return { content: [{ type: "text", text: "bash:ok" }], details: {} };
			},
		};
		const harness = await createXlabHarness({
			tools: [bashTool],
			initialActiveToolNames: ["bash"],
		});
		cleanups.push(harness.cleanup);
		copyRepositoryPaperCollectSkill(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				const runDir = join(harness.tempDir, ".xlab", "runs", runId);
				const scriptPath = join(harness.tempDir, "xlab", "skills", "paper_collect", "scripts", "paper_collect.py");
				return fauxAssistantMessage(
					fauxToolCall("bash", {
						command: `./python ${scriptPath} audit --run-id ${runId} --run-dir ${runDir}`,
					}),
					{ stopReason: "toolUse" },
				);
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(bashRuns).toHaveLength(0);
		expect(latestToolResultIsError(harness)).toBe(true);
		expect(latestToolResultText(harness)).toContain("Use the packaged paper_collect CLI");
	});

	it("blocks path-qualified paper_collect inspection tools", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const bashRuns: string[] = [];
		const bashTool: AgentTool = {
			name: "bash",
			label: "Bash",
			description: "Bash test tool",
			parameters: Type.Object({ command: Type.String() }),
			execute: async (_toolCallId, params) => {
				bashRuns.push(String((params as { command: string }).command));
				return { content: [{ type: "text", text: "bash:ok" }], details: {} };
			},
		};
		const harness = await createXlabHarness({
			tools: [bashTool],
			initialActiveToolNames: ["bash"],
		});
		cleanups.push(harness.cleanup);
		copyRepositoryPaperCollectSkill(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				const runDir = join(harness.tempDir, ".xlab", "runs", runId);
				return fauxAssistantMessage(
					fauxToolCall("bash", {
						command: `./grep graph ${join(runDir, "artifacts", "request.json")}`,
					}),
					{ stopReason: "toolUse" },
				);
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(bashRuns).toHaveLength(0);
		expect(latestToolResultIsError(harness)).toBe(true);
		expect(latestToolResultText(harness)).toContain("use read-only inspection scoped to the run/package directory");
	});

	it("blocks rg preprocessors in paper_collect inspection mode", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const bashRuns: string[] = [];
		const bashTool: AgentTool = {
			name: "bash",
			label: "Bash",
			description: "Bash test tool",
			parameters: Type.Object({ command: Type.String() }),
			execute: async (_toolCallId, params) => {
				bashRuns.push(String((params as { command: string }).command));
				return { content: [{ type: "text", text: "bash:ok" }], details: {} };
			},
		};
		const harness = await createXlabHarness({
			tools: [bashTool],
			initialActiveToolNames: ["bash"],
		});
		cleanups.push(harness.cleanup);
		copyRepositoryPaperCollectSkill(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				const runDir = join(harness.tempDir, ".xlab", "runs", runId);
				return fauxAssistantMessage(
					fauxToolCall("bash", {
						command: `rg --pre python graph ${join(runDir, "artifacts", "request.json")}`,
					}),
					{ stopReason: "toolUse" },
				);
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(bashRuns).toHaveLength(0);
		expect(latestToolResultIsError(harness)).toBe(true);
		expect(latestToolResultText(harness)).toContain("use read-only inspection scoped to the run/package directory");
	});

	it("blocks unscoped paper_collect inspection commands", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const bashRuns: string[] = [];
		const bashTool: AgentTool = {
			name: "bash",
			label: "Bash",
			description: "Bash test tool",
			parameters: Type.Object({ command: Type.String() }),
			execute: async (_toolCallId, params) => {
				bashRuns.push(String((params as { command: string }).command));
				return { content: [{ type: "text", text: "bash:ok" }], details: {} };
			},
		};
		const harness = await createXlabHarness({
			tools: [bashTool],
			initialActiveToolNames: ["bash"],
		});
		cleanups.push(harness.cleanup);
		copyRepositoryPaperCollectSkill(harness.tempDir);

		harness.setResponses([
			() =>
				fauxAssistantMessage(fauxToolCall("bash", { command: "rg PRIVATE_KEY $HOME" }), { stopReason: "toolUse" }),
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(bashRuns).toHaveLength(0);
		expect(latestToolResultIsError(harness)).toBe(true);
		expect(latestToolResultText(harness)).toContain("use read-only inspection scoped to the run/package directory");
	});

	it("blocks mutating find commands in paper_collect inspection mode", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const bashRuns: string[] = [];
		const bashTool: AgentTool = {
			name: "bash",
			label: "Bash",
			description: "Bash test tool",
			parameters: Type.Object({ command: Type.String() }),
			execute: async (_toolCallId, params) => {
				bashRuns.push(String((params as { command: string }).command));
				return { content: [{ type: "text", text: "bash:ok" }], details: {} };
			},
		};
		const harness = await createXlabHarness({
			tools: [bashTool],
			initialActiveToolNames: ["bash"],
		});
		cleanups.push(harness.cleanup);
		copyRepositoryPaperCollectSkill(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				const runDir = join(harness.tempDir, ".xlab", "runs", runId);
				return fauxAssistantMessage(fauxToolCall("bash", { command: `find ${runDir} -delete` }), {
					stopReason: "toolUse",
				});
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(bashRuns).toHaveLength(0);
		expect(latestToolResultIsError(harness)).toBe(true);
		expect(latestToolResultText(harness)).toContain("use read-only inspection scoped to the run/package directory");
	});

	it("blocks write tools and read paths outside paper_collect run/package scope", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const writeRuns: string[] = [];
		const writeTool = createWriteTool();
		writeTool.execute = async (_toolCallId, params) => {
			writeRuns.push(String((params as { path: string }).path));
			return { content: [{ type: "text", text: "write:ok" }], details: {} };
		};
		const harness = await createXlabHarness({
			tools: [createReadTool(), writeTool],
			initialActiveToolNames: ["read", "write"],
		});
		cleanups.push(harness.cleanup);
		copyRepositoryPaperCollectSkill(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				return fauxAssistantMessage(
					fauxToolCall("write", {
						path: `.xlab/runs/${runId}/artifacts/papers.manifest.json`,
						content: "{}",
					}),
					{ stopReason: "toolUse" },
				);
			},
			() =>
				fauxAssistantMessage(fauxToolCall("read", { path: join(harness.tempDir, "outside.json") }), {
					stopReason: "toolUse",
				}),
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(
			() => harness.session.messages.filter((message) => message.role === "toolResult").length >= 2,
		);

		const results = harness.session.messages.filter((message) => message.role === "toolResult");
		expect(writeRuns).toHaveLength(0);
		expect(results.every((message) => "isError" in message && message.isError)).toBe(true);
		expect(harness.session.getActiveToolNames()).not.toContain("write");
		expect(JSON.stringify(results[0]?.content)).toContain("Tool write not found");
		expect(JSON.stringify(results[1]?.content)).toContain("inside the active run directory or paper_collect package");
	});

	it("allows read paths inside the active paper_collect run and package scope", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const readRuns: string[] = [];
		const readTool = createReadTool();
		readTool.execute = async (_toolCallId, params) => {
			const path = String((params as { path: string }).path);
			readRuns.push(path);
			return { content: [{ type: "text", text: `read:${path}` }], details: {} };
		};
		const harness = await createXlabHarness({
			tools: [readTool],
			initialActiveToolNames: ["read"],
		});
		cleanups.push(harness.cleanup);
		copyRepositoryPaperCollectSkill(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				return fauxAssistantMessage(
					fauxToolCall("read", {
						path: `.xlab/runs/${runId}/artifacts/request.json`,
					}),
					{ stopReason: "toolUse" },
				);
			},
			() =>
				fauxAssistantMessage(
					fauxToolCall("read", {
						path: join(harness.tempDir, "xlab", "skills", "paper_collect", "SKILL.md"),
					}),
					{ stopReason: "toolUse" },
				),
			fauxAssistantMessage("done"),
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(
			() => harness.session.messages.filter((message) => message.role === "toolResult").length >= 2,
		);

		expect(harness.session.getActiveToolNames()).toContain("read");
		const results = harness.session.messages.filter((message) => message.role === "toolResult");
		expect(readRuns).toHaveLength(2);
		expect(results.every((message) => !("isError" in message) || !message.isError)).toBe(true);
		expect(JSON.stringify(results[0]?.content)).toContain("request.json");
		expect(JSON.stringify(results[1]?.content)).toContain("SKILL.md");
	});

	it("runs the repository paper_collect skill package to a validated finish", async () => {
		vi.stubEnv("TAVILY_API_KEY", "test-tavily-key");
		vi.stubEnv("SEMANTIC_SCHOLAR_API_KEY", "test-semantic-scholar-key");
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		harness.session.setActiveToolsByName(["bash"]);
		copyRepositoryPaperCollectSkill(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				const runDir = join(harness.tempDir, ".xlab", "runs", runId);
				const request: unknown = JSON.parse(readFileSync(join(runDir, "artifacts", "request.json"), "utf-8"));
				expect(request).toMatchObject({
					query: "graph neural networks",
					target_papers: 2,
					max_papers: 10,
				});
				writePaperCollectFixture(harness.tempDir, runId);
				const scriptPath = join(harness.tempDir, "xlab", "skills", "paper_collect", "scripts", "paper_collect.py");
				return fauxAssistantMessage(
					fauxToolCall("bash", {
						command: `python ${scriptPath} audit --run-id ${runId} --run-dir ${runDir}`,
					}),
				);
			},
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				return fauxAssistantMessage(
					fauxToolCall("xlab_update_state", {
						phase: {
							id: "validate",
							label: "Validating generated papers",
							status: "running",
						},
						counters: { collected_papers: 2, target_papers: 2 },
						artifacts: [
							{
								type: "paper_collection",
								name: "Paper collection manifest",
								path: `.xlab/runs/${runId}/artifacts/papers.manifest.json`,
								status: "ready",
							},
						],
					}),
				);
			},
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				return fauxAssistantMessage(
					fauxToolCall("xlab_finish", {
						status: "success",
						manifest_path: `.xlab/runs/${runId}/manifest.json`,
						summary: "Validated the self-contained paper collection workflow.",
					}),
				);
			},
		]);

		await harness.session.prompt('/xlab collect-papers "graph neural networks" --target-papers 2 --max-papers 10');
		await harness.session.agent.waitForIdle();
		await waitForCondition(() =>
			harness.session.messages.some(
				(message) =>
					message.role === "toolResult" &&
					"content" in message &&
					JSON.stringify(message.content).includes("finished with status success"),
			),
		);

		const runId = getOnlyRunId(harness.tempDir);
		const manifestPath = join(harness.tempDir, ".xlab", "runs", runId, "manifest.json");
		if (!existsSync(manifestPath)) {
			throw new Error(
				`paper_collect did not write a manifest: ${JSON.stringify(
					harness.session.messages.filter((message) => message.role === "toolResult"),
				)}`,
			);
		}
		const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
		const paperCollection = JSON.parse(
			readFileSync(join(harness.tempDir, ".xlab", "runs", runId, "artifacts", "papers.manifest.json"), "utf-8"),
		);
		expect(manifest.status).toBe("success");
		expect(manifest.skill_name).toBe("paper_collect");
		expect(manifest.skill_version).toBe("2.2.0");
		expect(manifest.outputs.collected_count).toBe(2);
		expect(manifest.outputs.edge_count).toBe(1);
		expect(paperCollection.papers).toHaveLength(2);
		expect(readRunState(harness.tempDir, runId)).toMatchObject({
			phase: { id: "finish", status: "success" },
		});
	});

	it("rejects xlab_finish when manifest status does not match the finish status", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir);

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				writeValidManifest(harness.tempDir, runId, { status: "failed" });
				return fauxAssistantMessage(
					fauxToolCall("xlab_finish", {
						status: "success",
						manifest_path: `.xlab/runs/${runId}/manifest.json`,
						summary: "done",
					}),
				);
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt("/xlab collect-papers topic");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		const toolResult = harness.session.messages.find((message) => message.role === "toolResult");
		expect(toolResult && "content" in toolResult ? JSON.stringify(toolResult.content) : "").toContain(
			"status must match xlab_finish status",
		);
		expect(harness.session.getActiveToolNames()).toContain("xlab_finish");
	});

	it("rejects xlab_finish when required artifacts are missing", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			artifacts: [{ type: "report", path: "report.md", required: true }],
		});

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				writeValidManifest(harness.tempDir, runId, { artifacts: [] });
				return fauxAssistantMessage(
					fauxToolCall("xlab_finish", {
						status: "success",
						manifest_path: `.xlab/runs/${runId}/manifest.json`,
						summary: "done",
					}),
				);
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt("/xlab collect-papers topic");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		const toolResult = harness.session.messages.find((message) => message.role === "toolResult");
		expect(toolResult && "content" in toolResult ? JSON.stringify(toolResult.content) : "").toContain(
			"missing required artifact",
		);
	});

	it("uses the post-pre-finish manifest for artifacts, lineage, and post-finish hooks", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			name: "knowledge_graph",
			dirName: "knowledge_graph",
			artifacts: [{ type: "graph_db", required: true }],
			hook: `
				import { writeFileSync } from "node:fs";
				export function preFinish(ctx) {
					const event = ctx.event;
					writeFileSync(ctx.runDir + "/artifacts/graph.db", "post-hook graph");
					writeFileSync(event.manifestPath, JSON.stringify({
						...event.manifest,
						inputs: { topic: "rewritten topic" },
						outputs: { source: "pre_finish" },
						validation: { passed: true, source: "pre_finish" },
						artifacts: [{ type: "graph_db", schema_version: "1", path: "artifacts/graph.db" }],
					}));
					return { ok: true };
				}
				export function postFinish(ctx) {
					writeFileSync(ctx.runDir + "/post-finish.json", JSON.stringify(ctx.event));
					return { ok: true };
				}
			`,
			hooks: {
				pre_finish: [{ module: "hooks/index.ts", handler: "preFinish" }],
				post_finish: [{ module: "hooks/index.ts", handler: "postFinish" }],
			},
		});

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				writeValidManifest(harness.tempDir, runId, {
					skill_name: "knowledge_graph",
					artifacts: [],
				});
				return fauxAssistantMessage(
					fauxToolCall("xlab_finish", {
						status: "success",
						manifest_path: `.xlab/runs/${runId}/manifest.json`,
						summary: "done",
					}),
				);
			},
		]);

		await harness.session.prompt("/xlab build-knowledge-graph topic");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => latestToolResultText(harness).includes("finished with status success"));

		const runId = getOnlyRunId(harness.tempDir);
		const stored = new XlabArtifactStore(harness.tempDir).list();
		expect(stored).toEqual([
			expect.objectContaining({
				type: "graph_db",
				validation: { passed: true, source: "pre_finish" },
			}),
		]);
		const workspaceManager = new XlabWorkspaceManager(harness.tempDir);
		const [workspace] = workspaceManager.listWorkspaces();
		expect(workspaceManager.readLineage(workspace.slug).handles).toEqual([
			expect.objectContaining({
				artifactId: stored[0].artifact_id,
				validation: { passed: true, source: "pre_finish" },
			}),
		]);
		const postFinish = JSON.parse(
			readFileSync(join(harness.tempDir, ".xlab", "runs", runId, "post-finish.json"), "utf-8"),
		);
		expect(postFinish.manifest).toMatchObject({
			outputs: { source: "pre_finish" },
			validation: { passed: true, source: "pre_finish" },
			artifacts: [
				{
					type: "graph_db",
					schema_version: "1",
					path: "artifacts/graph.db",
				},
			],
		});
		expect(postFinish.artifactRefs).toEqual([
			expect.objectContaining({ artifactId: stored[0].artifact_id, type: "graph_db" }),
		]);
	});

	it("rejects an invalid manifest rewrite made by pre-finish", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			hook: `
				import { writeFileSync } from "node:fs";
				export function preFinish(ctx) {
					writeFileSync(ctx.event.manifestPath, JSON.stringify({ ...ctx.event.manifest, skill_version: "rewritten-invalid" }));
					return { ok: true };
				}
			`,
		});

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				writeValidManifest(harness.tempDir, runId);
				return fauxAssistantMessage(
					fauxToolCall("xlab_finish", {
						status: "success",
						manifest_path: `.xlab/runs/${runId}/manifest.json`,
						summary: "done",
					}),
				);
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt("/xlab collect-papers topic");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(latestToolResultText(harness)).toContain('skill_version must be \\"1.0.0\\"');
		expect(new XlabArtifactStore(harness.tempDir).list()).toEqual([]);
		expect(harness.session.getActiveToolNames()).toContain("xlab_finish");
	});

	it("rejects invalid present artifacts on incomplete runs without enforcing success requirements", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			artifacts: [{ type: "report", path: "report.md", required: true }],
		});

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				writeValidManifest(harness.tempDir, runId, {
					status: "incomplete",
					artifacts: [{ type: "checkpoint", schema_version: "1", path: "artifacts/missing.json" }],
				});
				return fauxAssistantMessage(
					fauxToolCall("xlab_finish", {
						status: "incomplete",
						manifest_path: `.xlab/runs/${runId}/manifest.json`,
						summary: "partial",
					}),
				);
			},
			fauxAssistantMessage("repairing"),
		]);

		await harness.session.prompt("/xlab collect-papers topic");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		expect(latestToolResultText(harness)).toContain("Manifest artifact path does not exist: artifacts/missing.json");
		expect(latestToolResultText(harness)).not.toContain("missing required artifact");
		expect(harness.session.getActiveToolNames()).toContain("xlab_finish");
	});

	it("runs pre-finish hooks and returns repair without terminating when rejected", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			hook: `export function preFinish() { return { ok: false, repair: "add required evidence" }; }`,
		});

		harness.setResponses([
			() => {
				const runId = getOnlyRunId(harness.tempDir);
				writeValidManifest(harness.tempDir, runId);
				return fauxAssistantMessage(
					fauxToolCall("xlab_finish", {
						status: "success",
						manifest_path: `.xlab/runs/${runId}/manifest.json`,
						summary: "done",
					}),
				);
			},
			fauxAssistantMessage("will repair"),
		]);

		await harness.session.prompt("/xlab collect-papers topic");
		await harness.session.agent.waitForIdle();
		await waitForCondition(() => harness.session.messages.some((message) => message.role === "toolResult"));

		const toolResult = harness.session.messages.find((message) => message.role === "toolResult");
		expect(toolResult && "content" in toolResult ? JSON.stringify(toolResult.content) : "").toContain(
			"add required evidence",
		);
		expect(harness.session.getActiveToolNames()).toContain("xlab_finish");
	});

	it("registers only the XLab root command for product tasks", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir);
		await harness.session.bindExtensions({});

		const commands = harness.session.extensionRunner.getRegisteredCommands();
		const names = commands.map((entry) => entry.invocationName);
		expect(names.filter((name) => name === "xlab")).toHaveLength(1);
		expect(names).not.toEqual(expect.arrayContaining(["paper_collect", "collect", "paper-collect", "init"]));

		const xlab = commands.find((entry) => entry.invocationName === "xlab");
		const root = xlab?.getArgumentCompletions?.("");
		expect(root).toHaveLength(XLAB_PRODUCT_COMMANDS.length + XLAB_CONTROL_COMMANDS.length);
		for (const entry of [...XLAB_PRODUCT_COMMANDS, ...XLAB_CONTROL_COMMANDS]) {
			expect(root).toEqual(
				expect.arrayContaining([
					expect.objectContaining({ value: `${entry.task} `, label: entry.task, description: entry.description }),
				]),
			);
		}
	});

	it("exposes manifest-backed arguments through the XLab product task", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			ui: {
				argumentHint: "<topic> [options]",
				arguments: [
					{
						name: "--depth",
						description: "Set collection depth.",
						valueHint: "<brief|deep>",
						choices: ["brief", "deep"],
						takesValue: true,
					},
					{
						name: "--facet",
						description: "Add a facet.",
						valueHint: "<facet>",
						takesValue: true,
						repeatable: true,
					},
					{
						name: "--resume",
						description: "Resume collection.",
						takesValue: false,
					},
				],
			},
		});
		await harness.session.bindExtensions({});

		const command = harness.session.extensionRunner
			.getRegisteredCommands()
			.find((entry) => entry.invocationName === "xlab");
		expect(command?.getArgumentCompletions?.("collect-papers ")).toEqual(
			expect.arrayContaining([
				expect.objectContaining({ value: "collect-papers --depth ", label: "--depth <brief|deep>" }),
				expect.objectContaining({ value: "collect-papers --facet ", label: "--facet <facet>" }),
				expect.objectContaining({ value: "collect-papers --resume ", label: "--resume" }),
				expect.objectContaining({ value: "collect-papers --workspace ", label: "--workspace <slug>" }),
			]),
		);
		expect(command?.getArgumentCompletions?.("collect-papers --depth ")).toEqual([
			expect.objectContaining({ value: "collect-papers --depth brief ", label: "brief" }),
			expect.objectContaining({ value: "collect-papers --depth deep ", label: "deep" }),
		]);
		const afterFlags = command?.getArgumentCompletions?.("collect-papers --depth deep --facet ranking ");
		expect(afterFlags).toEqual(
			expect.arrayContaining([
				expect.objectContaining({ value: expect.stringContaining("--facet ") }),
				expect.objectContaining({ value: expect.stringContaining("--resume ") }),
			]),
		);
		expect(afterFlags).not.toEqual(
			expect.arrayContaining([expect.objectContaining({ label: expect.stringContaining("--depth") })]),
		);
	});

	it("deduplicates choices and completes unfinished quoted values", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			ui: {
				arguments: [
					{
						name: "--workspace",
						description: "Select a workspace.",
						valueHint: "<slug>",
						choices: ["graph-lab"],
						dynamicSource: "workspace",
						takesValue: true,
					},
				],
			},
		});
		const workspaces = new XlabWorkspaceManager(harness.tempDir);
		workspaces.ensureWorkspace("graph-lab", { displayName: "Graph Lab" });
		workspaces.ensureWorkspace("survey-lab", { displayName: "Survey Lab" });
		await harness.session.bindExtensions({});

		const xlab = harness.session.extensionRunner
			.getRegisteredCommands()
			.find((entry) => entry.invocationName === "xlab");
		const workspaceFlags = await xlab?.getArgumentCompletions?.("collect-papers ");
		expect(workspaceFlags?.filter((item) => item.label === "--workspace <slug>")).toHaveLength(1);
		expect(await xlab?.getArgumentCompletions?.('collect-papers --workspace "gra')).toEqual([
			expect.objectContaining({
				value: 'collect-papers --workspace "graph-lab" ',
				label: "graph-lab",
				description: "Graph Lab",
			}),
		]);
	});

	it("adds common workspace completion to product tasks without package arguments", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir);
		await harness.session.bindExtensions({});

		const command = harness.session.extensionRunner
			.getRegisteredCommands()
			.find((entry) => entry.invocationName === "xlab");
		expect(command?.getArgumentCompletions?.("collect-papers ")).toEqual([
			expect.objectContaining({
				label: "--workspace <slug>",
				value: "collect-papers --workspace ",
			}),
		]);
	});

	it("completes assignment arguments and XLab management operations", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			ui: {
				argumentHint: "<name> [key=value]",
				arguments: [
					{
						name: "language",
						description: "Select language.",
						valueHint: "<en|zh>",
						choices: ["en", "zh"],
						takesValue: true,
					},
				],
			},
		});
		await harness.session.bindExtensions({});

		const xlab = harness.session.extensionRunner
			.getRegisteredCommands()
			.find((entry) => entry.invocationName === "xlab");
		expect(xlab?.getArgumentCompletions?.("collect-papers lang")).toEqual([
			expect.objectContaining({
				value: "collect-papers language=",
				label: "language=<en|zh>",
			}),
		]);
		expect(xlab?.getArgumentCompletions?.("collect-papers language=")).toEqual([
			expect.objectContaining({ value: "collect-papers language=en ", label: "en" }),
			expect.objectContaining({ value: "collect-papers language=zh ", label: "zh" }),
		]);
		expect(xlab?.getArgumentCompletions?.("configure-xlab ")).toEqual(
			expect.arrayContaining([
				expect.objectContaining({ value: "configure-xlab runtime ", label: "runtime" }),
				expect.objectContaining({ value: "configure-xlab knowledge-graph ", label: "knowledge-graph" }),
				expect.objectContaining({ value: "configure-xlab scholarly-services ", label: "scholarly-services" }),
				expect.objectContaining({ value: "configure-xlab all ", label: "all" }),
			]),
		);
	});

	it("completes workspace, run, and artifact management values", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		const workspaces = new XlabWorkspaceManager(harness.tempDir);
		workspaces.ensureWorkspace("graph-lab", { displayName: "Graph Lab" });
		workspaces.ensureWorkspace("survey-lab", { displayName: "Survey Lab" });

		for (const [runId, status] of [
			["run-incomplete", "incomplete"],
			["run-success", "success"],
			["run-failed", "failed"],
			["run-cancelled", "cancelled"],
		] as const) {
			const run = fixtureRun(harness.tempDir, runId, status);
			mkdirSync(run.runDir, { recursive: true });
			writeJson(join(run.runDir, "run.json"), run);
		}

		const payloadPath = join(harness.tempDir, "artifact-fixture.txt");
		writeFileSync(payloadPath, "artifact completion fixture", "utf-8");
		const artifactRun = fixtureRun(harness.tempDir, "artifact-run", "success");
		const [artifact] = new XlabArtifactStore(harness.tempDir).commitManifestArtifacts(
			artifactRun,
			fixtureManifest(artifactRun, [{ type: "paper_set", schema_version: "1", path: payloadPath }]),
			(path) => path,
		);

		const xlab = harness.session.extensionRunner
			.getRegisteredCommands()
			.find((entry) => entry.invocationName === "xlab");
		expect(xlab?.getArgumentCompletions?.("show-workspace graph")).toEqual([
			expect.objectContaining({ value: "show-workspace graph-lab ", description: "Graph Lab" }),
		]);
		expect(xlab?.getArgumentCompletions?.("show-run run-")).toEqual(
			expect.arrayContaining([
				expect.objectContaining({ value: "show-run run-incomplete " }),
				expect.objectContaining({ value: "show-run run-success " }),
			]),
		);
		const resumable = xlab?.getArgumentCompletions?.("resume-run run-");
		expect(resumable).toEqual([expect.objectContaining({ value: "resume-run run-incomplete " })]);
		expect(resumable).not.toEqual(
			expect.arrayContaining([
				expect.objectContaining({ value: expect.stringContaining("run-success") }),
				expect.objectContaining({ value: expect.stringContaining("run-failed") }),
				expect.objectContaining({ value: expect.stringContaining("run-cancelled") }),
			]),
		);
		expect(xlab?.getArgumentCompletions?.(`show-artifact ${artifact?.artifactId.slice(0, 16)}`)).toEqual([
			expect.objectContaining({
				value: `show-artifact ${artifact?.artifactId} `,
				description: "paper_set — paper_collect",
			}),
		]);
	});

	it("completes graph and survey handles from an explicit workspace", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			dirName: "literature-survey",
			name: "literature_survey",
			ui: {
				arguments: [
					{
						name: "--graph",
						description: "Select a knowledge graph.",
						valueHint: "<graph>",
						dynamicSource: "graph",
						takesValue: true,
					},
				],
			},
		});
		setupSkillPackage(harness.tempDir, {
			dirName: "research-idea",
			name: "research_idea",
			ui: {
				arguments: [
					{
						name: "--survey",
						description: "Select a literature survey.",
						valueHint: "<survey>",
						dynamicSource: "survey",
						takesValue: true,
					},
				],
			},
		});
		await harness.session.bindExtensions({});

		const workspaces = new XlabWorkspaceManager(harness.tempDir);
		workspaces.ensureWorkspace("graph-lab", { displayName: "Graph Lab" });
		const graphRun = fixtureRun(harness.tempDir, "graph-run", "success", "knowledge_graph");
		const graphRef = {
			artifactId: "sha256:graph-fixture",
			type: "graph_db",
			schemaVersion: "1",
			digest: "graph-fixture",
			payloadPath: "/unused/graph.db",
			metadataPath: "/unused/graph.json",
		};
		workspaces.registerRunArtifacts({
			run: graphRun,
			workspaceSlug: "graph-lab",
			manifest: fixtureManifest(graphRun, [{ type: "graph_db", schema_version: "1", path: "artifacts/graph.db" }]),
			artifactRefs: [graphRef],
		});
		const surveyRun = fixtureRun(harness.tempDir, "survey-run", "success", "literature_survey");
		workspaces.registerRunArtifacts({
			run: surveyRun,
			workspaceSlug: "graph-lab",
			manifest: fixtureManifest(surveyRun, [
				{
					type: "literature_survey_json",
					schema_version: "1",
					path: "artifacts/survey.json",
					parents: [graphRef.artifactId],
				},
			]),
			artifactRefs: [
				{
					artifactId: "sha256:survey-fixture",
					type: "literature_survey_json",
					schemaVersion: "1",
					digest: "survey-fixture",
					payloadPath: "/unused/survey.json",
					metadataPath: "/unused/survey-metadata.json",
				},
			],
		});

		const xlab = harness.session.extensionRunner
			.getRegisteredCommands()
			.find((entry) => entry.invocationName === "xlab");
		expect(xlab?.getArgumentCompletions?.("write-literature-survey --workspace graph-lab --graph ")).toEqual([
			expect.objectContaining({
				value: "write-literature-survey --workspace graph-lab --graph graph-v1 ",
				label: "graph-v1",
			}),
		]);
		expect(xlab?.getArgumentCompletions?.("generate-research-ideas --workspace graph-lab --survey ")).toEqual([
			expect.objectContaining({
				value: "generate-research-ideas --workspace graph-lab --survey survey-v1 ",
				label: "survey-v1",
			}),
		]);
	});

	it("accepts declared secret aliases without prompting or leaking values", async () => {
		const canonical = "OPENAI_API_KEY";
		const alias = "LLM_API_KEY";
		const aliasValue = "alias-secret-value";
		const previousCanonical = process.env[canonical];
		const previousAlias = process.env[alias];
		delete process.env[canonical];
		process.env[alias] = aliasValue;
		const prompts: string[] = [];
		const harness = await createXlabHarness({
			input: async (title) => {
				prompts.push(title);
				return "should-not-be-requested";
			},
		});
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			runtime: {
				kind: "agent",
				required_secrets: [canonical],
				secret_aliases: { [canonical]: [alias] },
			},
		});
		harness.setResponses([fauxAssistantMessage("working")]);

		try {
			await harness.session.prompt("/xlab collect-papers topic");
			await harness.session.agent.waitForIdle();
			await waitForCondition(() => getUserTexts(harness).length > 0);

			expect(prompts).toEqual([]);
			expect(process.env[canonical]).toBeUndefined();
			expect(JSON.stringify(harness.session.messages)).not.toContain(aliasValue);
			const runId = getOnlyRunId(harness.tempDir);
			expect(readFileSync(join(harness.tempDir, ".xlab", "runs", runId, "run.json"), "utf-8")).not.toContain(
				aliasValue,
			);
		} finally {
			if (previousCanonical === undefined) {
				delete process.env[canonical];
			} else {
				process.env[canonical] = previousCanonical;
			}
			if (previousAlias === undefined) {
				delete process.env[alias];
			} else {
				process.env[alias] = previousAlias;
			}
		}
	});

	it("does not route unknown internal package names as product tasks", async () => {
		const harness = await createXlabHarness();
		cleanups.push(harness.cleanup);
		setupSkillPackage(harness.tempDir, {
			dirName: "unknown",
			name: "unknown_xlab",
		});

		harness.setResponses([fauxAssistantMessage("should not run")]);
		await harness.session.prompt("/xlab unknown_xlab topic");
		await harness.session.agent.waitForIdle();

		expect(getUserTexts(harness)).toHaveLength(0);
		expect(existsSync(join(harness.tempDir, ".xlab", "runs"))).toBe(false);
	});

	it("does not inject default XLab extension when extensions are disabled", async () => {
		const services = await createAgentSessionServices({
			cwd: "/tmp",
			resourceLoaderOptions: {
				noExtensions: true,
				noSkills: true,
				noPromptTemplates: true,
				noThemes: true,
				noContextFiles: true,
			},
		});

		expect(services.resourceLoader.getExtensions().extensions).toHaveLength(0);
	});
});
