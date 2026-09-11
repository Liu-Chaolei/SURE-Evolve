import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { relative } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { getAgentDir, getAuthPath, getModelsPath } from "../../config.ts";
import {
	defineTool,
	type ExtensionAPI,
	type ExtensionCommandContext,
	type ExtensionContext,
	type ExtensionFactory,
} from "../extensions/index.ts";
import { OPENAI_COMPATIBLE_API, upsertOpenAICompatibleModelConfig } from "../models-config-storage.ts";
import { XlabArtifactStore } from "./artifact-store.ts";
import {
	getXlabArgumentCompletions,
	getXlabArgumentValue,
	getXlabPositionalCompletions,
	parseXlabArgumentTokens,
} from "./command-completions.ts";
import { formatXlabDashboard } from "./dashboard.ts";
import { prepareXlabEnvironment } from "./environment.ts";
import { createExperimentStageDrivers, type ExperimentDriverPublication } from "./experiment/drivers.ts";
import { bindXlabExperimentIdea, type XlabIdeaBinding } from "./experiment/idea-binding.ts";
import {
	createXlabExperimentStagingMaterializer,
	NATIVE_EXPERIMENT_POLICY_DIGEST,
	XlabExperimentLifecycle,
} from "./experiment/lifecycle.ts";
import { currentSymbolicMemoryDigest, ExperimentMaterializer } from "./experiment/materializer.ts";
import { XlabHookRunner } from "./hooks.ts";
import { discoverXlabSkillPackages, type XlabDiscoveryDiagnostic } from "./manifest.ts";
import { validateXlabRuntimePrerequisites } from "./package-validator.ts";
import {
	resolveXlabProductPackage,
	validateXlabProductCommands,
	XLAB_CONTROL_COMMANDS,
	XLAB_PRODUCT_COMMANDS,
	xlabProductCommandForTask,
} from "./product-commands.ts";
import { XlabRunManager } from "./run-manager.ts";
import {
	collectXlabSecretUsage,
	formatXlabSecretStatus,
	getXlabInitVariables,
	initializeRequiredXlabSecrets,
	promptForXlabEnvironmentVariables,
	promptVariableForName,
	type XlabInitGroup,
	type XlabPromptVariable,
} from "./secrets.ts";
import { formatXlabDisplayStatus, normalizeXlabDisplayStatePatch } from "./state.ts";
import type {
	XlabArtifactReference,
	XlabArtifactRequirement,
	XlabDisplayState,
	XlabEnvironmentRecord,
	XlabFinishDetails,
	XlabFinishParams,
	XlabManifestArtifact,
	XlabManifestEnvelope,
	XlabRunRecord,
	XlabSkillPackage,
	XlabToolContent,
	XlabUpdateStateDetails,
} from "./types.ts";
import { transitionWorkflowStage } from "./workflow.ts";
import { parseXlabWorkspaceSelection, XlabWorkspaceManager } from "./workspace-manager.ts";

const FINISH_TOOL_NAME = "xlab_finish";
const UPDATE_STATE_TOOL_NAME = "xlab_update_state";
const STAGE_TOOL_NAME = "xlab_stage";
const STATUS_KEY = "xlab";
const DASHBOARD_WIDGET_KEY = "xlab.dashboard";

const finishToolParams = Type.Object({
	status: StringEnum(["success", "incomplete", "failed"] as const, {
		description: "Final XLab run status.",
	}),
	manifest_path: Type.String({
		description: "Path to the final artifact manifest, relative to the project cwd or absolute under the project.",
	}),
	summary: Type.String({
		description: "Concise final summary of the XLab run.",
	}),
	artifacts: Type.Optional(Type.Unsafe<unknown>({ description: "Structured final artifact summary." })),
	error_summary: Type.Optional(
		Type.String({
			description: "Failure or incompleteness summary, when relevant.",
		}),
	),
	next_actions: Type.Optional(Type.Array(Type.String(), { description: "Concrete follow-up actions." })),
});

const displayPhaseParams = Type.Object({
	id: Type.Optional(Type.String({ description: "Skill-defined phase id." })),
	label: Type.Optional(Type.String({ description: "Human-readable phase label." })),
	status: Type.Optional(
		StringEnum(["pending", "running", "success", "failed", "incomplete", "skipped", "blocked"] as const, {
			description: "Skill-defined phase status.",
		}),
	),
	progress: Type.Optional(Type.Number({ description: "Optional phase progress value." })),
});

const displayDiagnosticParams = Type.Object({
	severity: Type.Optional(StringEnum(["info", "warning", "error"] as const)),
	code: Type.Optional(Type.String({ description: "Skill-defined diagnostic code." })),
	message: Type.String({ description: "Human-readable diagnostic message." }),
	repair: Type.Optional(Type.String({ description: "Suggested repair action." })),
	data: Type.Optional(
		Type.Unsafe<unknown>({
			description: "Optional skill-defined diagnostic payload.",
		}),
	),
});

const displayArtifactParams = Type.Object({
	type: Type.Optional(Type.String({ description: "Skill-defined artifact type." })),
	name: Type.Optional(Type.String({ description: "Human-readable artifact name." })),
	path: Type.Optional(
		Type.String({
			description: "Artifact path relative to the project or run workspace.",
		}),
	),
	status: Type.Optional(StringEnum(["draft", "ready", "failed", "incomplete"] as const)),
	summary: Type.Optional(Type.String({ description: "Artifact summary." })),
	metadata: Type.Optional(Type.Record(Type.String(), Type.Unsafe<unknown>({}))),
});

const displayCheckpointParams = Type.Object({
	id: Type.Optional(Type.String({ description: "Skill-defined checkpoint id." })),
	label: Type.Optional(Type.String({ description: "Human-readable checkpoint label." })),
	resumable: Type.Optional(Type.Boolean({ description: "Whether the checkpoint can be resumed." })),
	resume_hint: Type.Optional(Type.String({ description: "Human-readable resume hint." })),
	data: Type.Optional(
		Type.Unsafe<unknown>({
			description: "Optional skill-defined checkpoint payload.",
		}),
	),
});

const updateStateToolParams = Type.Object({
	phase: Type.Optional(displayPhaseParams),
	message: Type.Optional(Type.String({ description: "Current run message for TUI/WebUI display." })),
	progress: Type.Optional(Type.Number({ description: "Optional overall progress value." })),
	counters: Type.Optional(Type.Record(Type.String(), Type.Number())),
	diagnostics: Type.Optional(Type.Array(displayDiagnosticParams)),
	artifacts: Type.Optional(Type.Array(displayArtifactParams)),
	checkpoint: Type.Optional(displayCheckpointParams),
	next_actions: Type.Optional(Type.Array(Type.String())),
});

const stageToolParams = Type.Object({
	action: StringEnum(["start", "complete", "fail", "block", "skip"] as const, {
		description: "Workflow stage transition.",
	}),
	stage_id: Type.String({
		description: "Workflow stage id declared in xlab.skill.json.",
	}),
	checkpoint: Type.Optional(Type.Unsafe<unknown>({ description: "Resumable stage checkpoint." })),
	artifacts: Type.Optional(
		Type.Unsafe<unknown>({
			description: "Stage artifact references or summary.",
		}),
	),
	error: Type.Optional(Type.String({ description: "Structured failure or blocker summary." })),
});

interface AgentActiveRun {
	kind: "agent";
	record: XlabRunRecord;
	skillPackage: XlabSkillPackage;
	hooks: XlabHookRunner;
	previousActiveTools: string[];
}

interface NativeActiveRun {
	kind: "native";
	record: XlabRunRecord;
	skillPackage: XlabSkillPackage;
	lifecycle: XlabExperimentLifecycle;
	execution: Promise<void>;
	previousActiveTools: string[];
}

type ActiveRun = AgentActiveRun | NativeActiveRun;

function isNativeExperiment(skillPackage: XlabSkillPackage): boolean {
	const runtime = skillPackage.manifest.runtime;
	return runtime?.kind === "native" && runtime.driver === "experiment";
}

function nativeExperimentRuntime(
	record: XlabRunRecord,
	ownerId: string,
	skillPackage: XlabSkillPackage,
	binding?: XlabIdeaBinding,
): {
	lifecycle: XlabExperimentLifecycle;
	drivers: ReturnType<typeof createExperimentStageDrivers>;
} {
	const lifecycle = new XlabExperimentLifecycle({
		cwd: record.cwd,
		agentDir: getAgentDir(),
		runId: record.runId,
		ownerId,
		finalizeSuccess: (latest) => {
			const manifestPath = `${latest.runDir}/manifest.json`;
			if (!existsSync(manifestPath)) throw new Error("Native experiment manifest.json is missing.");
			const manifest = JSON.parse(readFileSync(manifestPath, "utf-8")) as XlabManifestEnvelope;
			const artifactRefs = new XlabArtifactStore(latest.cwd).commitManifestArtifacts(
				latest,
				manifest,
				(path) => new XlabRunManager(latest.cwd).resolveRunPath(latest, path),
				skillPackage.manifest.artifacts ?? [],
			);
			return { artifactRefs, manifestPath };
		},
	});
	const reconstructed = lifecycle.reconstruct();
	const idea = binding?.canonical ?? JSON.parse(readFileSync(`${record.runDir}/inputs/idea.json`, "utf-8"));
	const materializer = new ExperimentMaterializer({
		runRoot: record.runDir,
		sharedRoot: record.cwd,
		protocol: reconstructed.protocol,
		skillName: record.skillName,
		skillVersion: record.skillVersion,
	});
	const publication: ExperimentDriverPublication = {
		publishPlan: ({ assignmentId }) => {
			materializer.materializePlan(
				assignmentId,
				(idea.components as Array<{ component: string }>).map((item) => item.component),
			);
		},
		publishWorker: ({ accepted }) => {
			materializer.materializeWorker(accepted.accepted.assignment_id);
		},
		publishReview: ({ stage, workUnit, reviewRound, assignmentIds, validationContext }) => {
			materializer.materializeReviewMatrix({
				stage,
				workUnit: workUnit.id,
				reviewRound,
				assignmentIds,
				validationContext,
			});
		},
		publishFinal: ({ componentDefinitions, componentResults, summary, provenance }) => {
			const symbolicMemoryPath = ".xlab/symbolic_memory/symbolic_memory.json";
			materializer.materializeFinal({
				componentDefinitions: componentDefinitions.map((item) => ({
					component: item.component,
					method_context: item.explanation,
				})),
				componentResults,
				summary,
				provenance,
				symbolicMemoryPath,
				expectedSymbolicMemoryDigest: currentSymbolicMemoryDigest(`${record.cwd}/${symbolicMemoryPath}`),
			});
		},
	};
	return {
		lifecycle,
		drivers: createExperimentStageDrivers({
			idea,
			ideaDigest: reconstructed.snapshot.idea_digest,
			policyDigest: NATIVE_EXPERIMENT_POLICY_DIGEST,
			readScope: ["inputs", "project"],
			projectWriteScope: ["project"],
			maxReviewRounds: 2,
			publication,
		}),
	};
}

async function initializeRequiredSecrets(
	ctx: ExtensionContext,
	skillPackage: XlabSkillPackage,
): Promise<{ ok: boolean; cancelled?: boolean; message?: string }> {
	return initializeRequiredXlabSecrets(ctx, skillPackage);
}

function formatDiagnostics(diagnostics: XlabDiscoveryDiagnostic[]): string {
	return diagnostics.map((diagnostic) => `${diagnostic.type}: ${diagnostic.message}`).join("\n");
}

function mergePromptVariables(variables: XlabPromptVariable[]): XlabPromptVariable[] {
	const byName = new Map<string, XlabPromptVariable>();
	for (const variable of variables) {
		const existing = byName.get(variable.name);
		byName.set(variable.name, {
			...existing,
			...variable,
			required: existing?.required === true || variable.required === true,
			secret: existing?.secret ?? variable.secret,
		});
	}
	return [...byName.values()];
}

function xlabInitVariablesFor(subcommand: XlabInitGroup | "all", packages: XlabSkillPackage[]): XlabPromptVariable[] {
	const groupVariables =
		subcommand === "all"
			? [...getXlabInitVariables("knowledge_graph"), ...getXlabInitVariables("scholarly")]
			: getXlabInitVariables(subcommand);
	const discoveredRequired =
		subcommand === "all"
			? collectXlabSecretUsage(packages)
					.filter((entry) => entry.requiredBy.length > 0)
					.map((entry) => promptVariableForName(entry.name, true))
			: [];
	return mergePromptVariables([...groupVariables, ...discoveredRequired]);
}

function formatXlabInitUsage(): string {
	return [
		"Usage: /xlab configure-xlab <runtime|knowledge-graph|scholarly-services|all>",
		"/xlab configure-xlab runtime saves an OpenAI-compatible agent runtime model and switches to it immediately.",
		"/xlab configure-xlab knowledge-graph configures the separate paper-extraction model used by /xlab build-knowledge-graph.",
		"Do not pass API-key values as slash-command arguments; configuration uses masked prompts when UI is available.",
	].join("\n");
}

function formatNoUiRuntimeInitInstructions(): string {
	return [
		"/xlab configure-xlab runtime cannot prompt for values in this mode.",
		"Run /xlab configure-xlab runtime in interactive mode, or manually configure an OpenAI-compatible runtime provider:",
		`- models: ${getModelsPath()}`,
		`- auth: ${getAuthPath()}`,
		'models.json should define a provider with api: "openai-completions", baseUrl, and models: [{ id: "..." }].',
		"auth.json should store the provider API key as an api_key credential. Do not paste secret values into slash-command arguments.",
	].join("\n");
}

function formatNoUiInitInstructions(subcommand: string, variables: XlabPromptVariable[]): string {
	const missing = variables.filter((variable) => !process.env[variable.name]?.trim());
	const names = missing.length > 0 ? missing : variables;
	const includesKnowledgeGraphLlm = names.some((variable) => variable.name.startsWith("KNOWLEDGE_GRAPH_LLM_"));
	const publicGroup =
		subcommand === "knowledge_graph"
			? "knowledge-graph"
			: subcommand === "scholarly"
				? "scholarly-services"
				: subcommand;
	return [
		`/xlab configure-xlab ${publicGroup} cannot prompt for integration values in this mode.`,
		includesKnowledgeGraphLlm
			? "The KNOWLEDGE_GRAPH_LLM_* values configure a separate paper-extraction model for /xlab build-knowledge-graph, independent from the agent runtime model."
			: undefined,
		"Set the needed environment variables in your shell or secrets manager, then rerun the XLab command. Examples:",
		...names.map((variable) => `export ${variable.name}=...`),
		"Do not paste secret values into slash-command arguments.",
	]
		.filter((line): line is string => line !== undefined)
		.join("\n");
}

function publishXlabInitReport(
	pi: ExtensionAPI,
	ctx: ExtensionCommandContext,
	report: string,
	severity: "info" | "warning" | "error" = "info",
	details?: Record<string, unknown>,
): void {
	pi.appendEntry("xlab.init", {
		report,
		severity,
		...details,
	});
	ctx.ui.notify(report, severity);
}

interface XlabRuntimeModelSetup {
	providerId: string;
	providerName: string;
	baseUrl: string;
	modelId: string;
	apiKey: string;
}

const DEFAULT_XLAB_RUNTIME_PROVIDER_ID = "xlab-openai-compatible";
const DEFAULT_XLAB_RUNTIME_PROVIDER_NAME = "XLab OpenAI-compatible";

function normalizeProviderId(value: string): string {
	return value
		.trim()
		.toLowerCase()
		.replace(/[^a-z0-9_.-]+/g, "-")
		.replace(/^-+|-+$/g, "");
}

function promptValueMessage(title: string, description: string, defaultValue?: string): string {
	return [title, description, defaultValue ? `Default: ${defaultValue}` : undefined]
		.filter((line): line is string => line !== undefined)
		.join("\n");
}

async function promptRequiredXlabRuntimeValue(
	ctx: ExtensionCommandContext,
	title: string,
	placeholder: string,
	options?: {
		secret?: boolean;
		defaultValue?: string;
		normalize?: (value: string) => string;
	},
): Promise<{ ok: true; value: string } | { ok: false; cancelled: true; message: string }> {
	while (true) {
		const value = await ctx.ui.input(title, placeholder, {
			secret: options?.secret,
		});
		if (value === undefined) {
			return {
				ok: false,
				cancelled: true,
				message: "XLab runtime model initialization cancelled.",
			};
		}
		const trimmed = value.trim();
		const normalized = trimmed
			? options?.normalize
				? options.normalize(trimmed)
				: trimmed
			: (options?.defaultValue ?? "");
		if (normalized) {
			return { ok: true, value: normalized };
		}
		ctx.ui.notify("Value cannot be empty. Enter a value or cancel the dialog.", "warning");
	}
}

function normalizeOpenAIBaseUrl(value: string): string {
	return value.replace(/\/+$/, "").replace(/\/chat\/completions$/i, "");
}

function snapshotEnvironment(names: string[]): Map<string, string | undefined> {
	return new Map(names.map((name) => [name, process.env[name]]));
}

function restoreEnvironment(snapshot: Map<string, string | undefined>): void {
	for (const [name, value] of snapshot.entries()) {
		if (value === undefined) {
			delete process.env[name];
		} else {
			process.env[name] = value;
		}
	}
}

function applyRuntimeOpenAIEnvironment(setup: XlabRuntimeModelSetup): Map<string, string | undefined> {
	const aliases = ["OPENAI_API_KEY", "LLM_API_KEY", "OPENAI_BASE_URL", "LLM_BASE_URL", "LLM_MODEL"];
	const previous = snapshotEnvironment(aliases);
	process.env.OPENAI_API_KEY = setup.apiKey;
	process.env.LLM_API_KEY = setup.apiKey;
	process.env.OPENAI_BASE_URL = setup.baseUrl;
	process.env.LLM_BASE_URL = setup.baseUrl;
	process.env.LLM_MODEL = setup.modelId;
	return previous;
}

function registerRuntimeProviderForCurrentSession(pi: ExtensionAPI, setup: XlabRuntimeModelSetup): void {
	pi.registerProvider(setup.providerId, {
		name: setup.providerName,
		baseUrl: setup.baseUrl,
		apiKey: "$OPENAI_API_KEY",
		api: OPENAI_COMPATIBLE_API,
		models: [
			{
				id: setup.modelId,
				name: setup.modelId,
				reasoning: false,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 128000,
				maxTokens: 16384,
			},
		],
	});
}

async function promptForXlabRuntimeModelSetup(
	ctx: ExtensionCommandContext,
): Promise<{ ok: true; setup: XlabRuntimeModelSetup } | { ok: false; cancelled: true; message: string }> {
	const providerId = await promptRequiredXlabRuntimeValue(
		ctx,
		promptValueMessage(
			"Initialize XLab Pi Agent runtime model",
			"Enter a provider id for the OpenAI-compatible runtime API. This is saved under Pi config and used in the footer/model selector.",
			DEFAULT_XLAB_RUNTIME_PROVIDER_ID,
		),
		DEFAULT_XLAB_RUNTIME_PROVIDER_ID,
		{
			defaultValue: DEFAULT_XLAB_RUNTIME_PROVIDER_ID,
			normalize: normalizeProviderId,
		},
	);
	if (!providerId.ok) return providerId;

	const providerName = await promptRequiredXlabRuntimeValue(
		ctx,
		promptValueMessage(
			"Initialize XLab Pi Agent runtime model",
			"Enter a display name for this OpenAI-compatible provider.",
			DEFAULT_XLAB_RUNTIME_PROVIDER_NAME,
		),
		DEFAULT_XLAB_RUNTIME_PROVIDER_NAME,
		{ defaultValue: DEFAULT_XLAB_RUNTIME_PROVIDER_NAME },
	);
	if (!providerName.ok) return providerName;

	const baseUrl = await promptRequiredXlabRuntimeValue(
		ctx,
		promptValueMessage(
			"Initialize XLab Pi Agent runtime model",
			"Enter the OpenAI-compatible base URL for the Pi Agent chat/coding model.",
		),
		"https://api.example.com/v1",
		{ secret: false, normalize: normalizeOpenAIBaseUrl },
	);
	if (!baseUrl.ok) return baseUrl;

	const modelId = await promptRequiredXlabRuntimeValue(
		ctx,
		promptValueMessage(
			"Initialize XLab Pi Agent runtime model",
			"Enter the provider model id for Pi Agent to use now.",
		),
		"gpt-4.1, qwen-plus, deepseek-chat, ...",
		{ secret: false },
	);
	if (!modelId.ok) return modelId;

	const apiKey = await promptRequiredXlabRuntimeValue(
		ctx,
		promptValueMessage(
			"Initialize XLab Pi Agent runtime model",
			"Enter the API key for this OpenAI-compatible runtime provider. It will be stored in Pi auth storage, not in models.json.",
		),
		"Paste the runtime API key",
		{ secret: true },
	);
	if (!apiKey.ok) return apiKey;

	return {
		ok: true,
		setup: {
			providerId: providerId.value,
			providerName: providerName.value,
			baseUrl: baseUrl.value,
			modelId: modelId.value,
			apiKey: apiKey.value,
		},
	};
}

async function runXlabRuntimeModelInit(pi: ExtensionAPI, ctx: ExtensionCommandContext): Promise<boolean> {
	if (!ctx.hasUI) {
		publishXlabInitReport(pi, ctx, formatNoUiRuntimeInitInstructions(), "warning", { subcommand: "llm" });
		return false;
	}

	const setupResult = await promptForXlabRuntimeModelSetup(ctx);
	if (!setupResult.ok) {
		publishXlabInitReport(pi, ctx, setupResult.message, "warning", {
			subcommand: "llm",
			cancelled: true,
		});
		return false;
	}

	const { setup } = setupResult;
	const previousSelectedCredential = ctx.modelRegistry.authStorage.get(setup.providerId);
	let runtimeEnvRollback: Map<string, string | undefined> | undefined;

	try {
		const modelsPath = upsertOpenAICompatibleModelConfig({
			providerId: setup.providerId,
			providerName: setup.providerName,
			baseUrl: setup.baseUrl,
			modelId: setup.modelId,
		});
		ctx.modelRegistry.authStorage.set(setup.providerId, {
			type: "api_key",
			key: setup.apiKey,
		});
		runtimeEnvRollback = applyRuntimeOpenAIEnvironment(setup);
		ctx.modelRegistry.refresh();
		registerRuntimeProviderForCurrentSession(pi, setup);
		const model = ctx.modelRegistry.find(setup.providerId, setup.modelId);
		if (!model) {
			throw new Error(
				`Configured model ${setup.providerId}/${setup.modelId} was not found after refreshing models.json.`,
			);
		}
		const switched = await pi.setModel(model);
		if (!switched) {
			throw new Error(`No API key for ${setup.providerId}/${setup.modelId}.`);
		}

		publishXlabInitReport(
			pi,
			ctx,
			[
				`Saved and switched Pi Agent runtime model to ${setup.providerId}/${setup.modelId}.`,
				`API format: ${OPENAI_COMPATIBLE_API}.`,
				`Model config: ${modelsPath}.`,
				`Runtime API key: stored in ${getAuthPath()}.`,
				"OpenAI-compatible environment aliases for XLab Python skills are available in this process.",
				"/xlab configure-xlab knowledge-graph remains separate for paper-extraction model setup.",
			].join("\n"),
			"info",
			{
				subcommand: "llm",
				providerId: setup.providerId,
				modelId: setup.modelId,
				api: OPENAI_COMPATIBLE_API,
			},
		);
		return true;
	} catch (error) {
		if (runtimeEnvRollback) {
			restoreEnvironment(runtimeEnvRollback);
		}
		if (previousSelectedCredential) {
			ctx.modelRegistry.authStorage.set(setup.providerId, previousSelectedCredential);
		} else {
			ctx.modelRegistry.authStorage.remove(setup.providerId);
		}
		ctx.modelRegistry.refresh();
		publishXlabInitReport(
			pi,
			ctx,
			`Failed to initialize the Pi Agent runtime model: ${error instanceof Error ? error.message : String(error)}`,
			"error",
			{ subcommand: "llm" },
		);
		return false;
	}
}

async function runXlabInitCommand(pi: ExtensionAPI, ctx: ExtensionCommandContext, args: string): Promise<void> {
	const tokens = args.trim() ? args.trim().split(/\s+/) : [];
	const subcommand = (tokens[0] ?? "status").toLowerCase();
	const discovered = discoverXlabSkillPackages(ctx.cwd);
	for (const diagnostic of discovered.diagnostics) {
		pi.appendEntry("xlab.diagnostic", diagnostic);
	}

	if (tokens.length > 1 || subcommand === "help") {
		publishXlabInitReport(pi, ctx, formatXlabInitUsage(), tokens.length > 1 ? "warning" : "info", {
			subcommand: "usage",
		});
		return;
	}

	if (subcommand === "" || subcommand === "status") {
		publishXlabInitReport(
			pi,
			ctx,
			formatXlabSecretStatus(discovered.packages, ctx.modelRegistry, ctx.model),
			"info",
			{
				subcommand: "status",
			},
		);
		return;
	}

	if (subcommand !== "llm" && subcommand !== "knowledge_graph" && subcommand !== "scholarly" && subcommand !== "all") {
		publishXlabInitReport(pi, ctx, formatXlabInitUsage(), "warning", {
			subcommand: "usage",
		});
		return;
	}

	if (!ctx.isProjectTrusted()) {
		publishXlabInitReport(pi, ctx, "Trust this project before initializing API keys for XLab skills.", "error", {
			subcommand,
		});
		return;
	}

	if (subcommand === "llm") {
		await runXlabRuntimeModelInit(pi, ctx);
		return;
	}

	if (subcommand === "all") {
		const ok = await runXlabRuntimeModelInit(pi, ctx);
		if (!ok) {
			return;
		}
	}

	const integrationSubcommands: XlabInitGroup[] =
		subcommand === "all" ? ["knowledge_graph", "scholarly"] : [subcommand];
	const initialized: string[] = [];
	const skipped: string[] = [];
	for (const integrationSubcommand of integrationSubcommands) {
		const variables = xlabInitVariablesFor(integrationSubcommand, discovered.packages);
		if (!ctx.hasUI) {
			publishXlabInitReport(pi, ctx, formatNoUiInitInstructions(integrationSubcommand, variables), "warning", {
				subcommand: integrationSubcommand,
			});
			return;
		}

		const result = await promptForXlabEnvironmentVariables(
			ctx,
			`Initialize XLab ${integrationSubcommand}`,
			variables,
			{
				skipAlreadySet: true,
				allowSkipOptional: true,
			},
		);
		if (!result.ok) {
			publishXlabInitReport(
				pi,
				ctx,
				result.message ?? "XLab initialization cancelled.",
				result.cancelled ? "warning" : "error",
				{
					subcommand: integrationSubcommand,
					cancelled: result.cancelled ?? false,
					initialized,
					skipped: [...skipped, ...result.skipped],
				},
			);
			return;
		}
		initialized.push(...result.initialized);
		skipped.push(...result.skipped);
	}

	const summary =
		initialized.length === 0 && skipped.length === 0
			? "No missing XLab integration environment variables for the selected group."
			: `Initialized ${initialized.length} integration environment variable${
					initialized.length === 1 ? "" : "s"
				} for this process${skipped.length > 0 ? `; skipped ${skipped.join(", ")}` : ""}.`;
	publishXlabInitReport(
		pi,
		ctx,
		`${summary}\n\n${formatXlabSecretStatus(discovered.packages, ctx.modelRegistry)}`,
		"info",
		{
			subcommand,
			initialized,
			skipped,
		},
	);
}

function buildInvocationPrompt(skillPackage: XlabSkillPackage, run: XlabRunRecord): string {
	const relativePackage = relative(run.cwd, skillPackage.packageDir) || ".";
	const workspace = run.workspaceSlug ? `Workspace: ${run.workspaceSlug}` : "Workspace: (none)";
	const dependencies =
		skillPackage.dependencies.length > 0
			? skillPackage.dependencies
					.map(
						(dependency) =>
							`- ${dependency.manifest.name}@${dependency.manifest.version}: ${
								relative(run.cwd, dependency.packageDir) || "."
							}`,
					)
					.join("\n")
			: "(none)";
	const workflow = skillPackage.manifest.workflow
		? skillPackage.manifest.workflow.stages
				.map(
					(stage) =>
						`- ${stage.id}: ${stage.skill}; needs=${stage.needs?.join(",") || "none"}; max_attempts=${
							stage.maxAttempts ?? 1
						}`,
				)
				.join("\n")
		: "(none)";
	return [
		`<xlab_invocation run_id="${run.runId}" skill="${skillPackage.manifest.name}" version="${skillPackage.manifest.version}" command="/xlab ${run.command}" invoked_command="/${run.invokedCommand ?? `xlab ${run.command}`}">`,
		`Package directory: ${relativePackage}`,
		`Run directory: ${relative(run.cwd, run.runDir)}`,
		workspace,
		`User arguments: ${run.args || "(none)"}`,
		"Resolved skill dependencies:",
		dependencies,
		`Prepared environment: ${
			run.environment
				? `${run.environment.kind} at ${relative(run.cwd, run.environment.path)}; use ${run.environment.executable}`
				: "none required"
		}`,
		"Workflow stages:",
		workflow,
		"",
		"Run this XLab skill as a scientific task. Produce durable artifacts in the run directory or project workspace.",
		"Use the skill instructions below. References are relative to the package directory.",
		"",
		skillPackage.prompt,
		"",
		"Completion protocol:",
		`- Finish by calling ${FINISH_TOOL_NAME} as a standalone final tool call.`,
		"- Do not emit a normal final answer instead of the finish tool.",
		"- If the finish tool returns repair instructions, fix the artifacts and call it again.",
		"- For declared workflows, use xlab_stage at every stage boundary.",
		"- The final manifest must use schema_version 2 and include skill_version plus typed artifact entries.",
		"</xlab_invocation>",
	].join("\n");
}

function buildResumePrompt(skillPackage: XlabSkillPackage, run: XlabRunRecord, runManager: XlabRunManager): string {
	const state = runManager.readState(run);
	const workflow = runManager.readWorkflow(run);
	return [
		`<xlab_resume run_id="${run.runId}" skill="${run.skillName}" version="${run.skillVersion}">`,
		`Run directory: ${relative(run.cwd, run.runDir)}`,
		`Original arguments: ${run.args || "(none)"}`,
		`Display checkpoint: ${JSON.stringify(state?.checkpoint ?? null)}`,
		`Workflow state: ${JSON.stringify(workflow ?? null)}`,
		"",
		"Resume from durable artifacts and the last valid checkpoint. Do not repeat successful workflow stages.",
		"Use the skill instructions below, then finish with xlab_finish.",
		"",
		skillPackage.prompt,
		"</xlab_resume>",
	].join("\n");
}

function toolText(text: string): XlabToolContent[] {
	return [{ type: "text", text }];
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function isStringRecord(value: unknown): value is Record<string, unknown> {
	return isRecord(value) && !Array.isArray(value);
}

function isManifestArtifact(value: unknown): value is {
	type: string;
	schema_version: string;
	path: string;
	parents?: string[];
	metadata?: Record<string, unknown>;
} {
	if (!isStringRecord(value)) {
		return false;
	}
	return (
		typeof value.type === "string" &&
		value.type.trim() !== "" &&
		typeof value.schema_version === "string" &&
		value.schema_version.trim() !== "" &&
		typeof value.path === "string" &&
		value.path.trim() !== "" &&
		(value.parents === undefined ||
			(Array.isArray(value.parents) && value.parents.every((parent) => typeof parent === "string"))) &&
		(value.metadata === undefined || isStringRecord(value.metadata))
	);
}

function setRunStatusText(ctx: ExtensionContext, run: XlabRunRecord, state?: XlabDisplayState): void {
	ctx.ui.setStatus(STATUS_KEY, formatXlabDisplayStatus(run.invokedCommand ?? run.command, run.runId, state));
}

function setRunDashboard(ctx: ExtensionContext, active: ActiveRun, state?: XlabDisplayState): void {
	const runManager = new XlabRunManager(active.record.cwd);
	ctx.ui.setWidget(
		DASHBOARD_WIDGET_KEY,
		formatXlabDashboard({
			run: active.record,
			state: state ?? runManager.readState(active.record),
			workflow: runManager.readWorkflow(active.record),
			skillPackage: active.skillPackage,
		}),
		{ placement: "aboveEditor" },
	);
}

function clearRunDashboard(ctx: ExtensionContext): void {
	ctx.ui.setWidget(DASHBOARD_WIDGET_KEY, undefined);
}

function applyStatePatch(
	pi: ExtensionAPI,
	ctx: ExtensionContext,
	active: ActiveRun,
	patchValue: unknown,
	eventType: string,
): { ok: boolean; state?: XlabDisplayState; repair?: string } {
	if (patchValue === undefined) {
		return { ok: true };
	}
	const normalized = normalizeXlabDisplayStatePatch(patchValue);
	if (!normalized.ok || !normalized.state) {
		const repair = normalized.message ?? "Invalid XLab state patch.";
		pi.appendEntry("xlab.diagnostic", {
			type: "warning",
			message: repair,
			path: active.skillPackage.manifestPath,
		});
		return { ok: false, repair };
	}
	const runManager = new XlabRunManager(active.record.cwd);
	const state = runManager.updateState(active.record, normalized.state, eventType);
	pi.appendEntry("xlab.state", {
		runId: active.record.runId,
		command: active.record.command,
		skillName: active.record.skillName,
		state,
	});
	setRunStatusText(ctx, active.record, state);
	setRunDashboard(ctx, active, state);
	return { ok: true, state };
}

function activeToolNames(active: ActiveRun): string[] {
	const allowed = active.skillPackage.manifest.permissions?.tools;
	const previous = allowed
		? active.previousActiveTools.filter((toolName) => allowed.includes(toolName))
		: active.previousActiveTools;
	return [...new Set([...previous, FINISH_TOOL_NAME, UPDATE_STATE_TOOL_NAME, STAGE_TOOL_NAME])];
}

function validateRequiredArtifact(
	requirement: XlabArtifactRequirement,
	manifest: XlabManifestEnvelope,
	runManager: XlabRunManager,
	run: XlabRunRecord,
): string | undefined {
	if (!requirement.required) {
		return undefined;
	}

	const artifacts = manifest.artifacts ?? [];
	const candidate = artifacts.find((artifact) => {
		if (requirement.type && artifact.type !== requirement.type) {
			return false;
		}
		if (requirement.schemaVersion && artifact.schema_version !== requirement.schemaVersion) {
			return false;
		}
		if (requirement.path && artifact.path !== requirement.path) {
			return false;
		}
		if (!requirement.type && !requirement.path) {
			return artifact.path !== undefined || artifact.type !== undefined;
		}
		return true;
	});
	if (!candidate) {
		const description = requirement.description ? ` (${requirement.description})` : "";
		return `Final manifest is missing required artifact${description}.`;
	}
	if (requirement.path) {
		const artifactPath = runManager.resolveRunPath(run, requirement.path);
		if (!artifactPath || !existsSync(artifactPath)) {
			return `Required artifact path does not exist: ${requirement.path}.`;
		}
	}
	if (candidate.path) {
		const artifactPath = runManager.resolveRunPath(run, candidate.path);
		if (!artifactPath || !existsSync(artifactPath)) {
			return `Manifest artifact path does not exist: ${candidate.path}.`;
		}
	}
	return undefined;
}

function validateStageArtifacts(
	value: unknown,
	skillPackage: XlabSkillPackage,
	runManager: XlabRunManager,
	run: XlabRunRecord,
): { artifacts?: XlabManifestArtifact[]; repair?: string } {
	if (!Array.isArray(value) || !value.every(isManifestArtifact)) {
		return {
			repair: `Stage completion for ${skillPackage.manifest.name} requires a typed artifacts array.`,
		};
	}
	const manifest: XlabManifestEnvelope = {
		schema_version: "2",
		run_id: run.runId,
		skill_name: skillPackage.manifest.name,
		skill_version: skillPackage.manifest.version,
		status: "success",
		created_at: new Date().toISOString(),
		inputs: {},
		outputs: {},
		validation: { stage_gate: true },
		artifacts: value,
	};
	for (const requirement of skillPackage.manifest.artifacts ?? []) {
		const error = validateRequiredArtifact(requirement, manifest, runManager, run);
		if (error) {
			return { repair: `${skillPackage.manifest.name}: ${error}` };
		}
	}
	return { artifacts: value };
}

function formatWorkspaceList(manager: XlabWorkspaceManager): string {
	const activeSlug = manager.readIndex().activeWorkspaceSlug;
	const workspaces = manager.listWorkspaces().sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
	if (workspaces.length === 0) {
		return "No XLab workspaces found.";
	}
	return workspaces
		.map((workspace) => {
			const current = manager.readCurrent(workspace.slug).current;
			const marker = workspace.slug === activeSlug ? "*" : " ";
			const handles =
				[current.graph, current.survey, current.idea].filter(Boolean).join(" ") || "no current artifacts";
			return `${marker} ${workspace.slug} — ${handles} — updated ${workspace.updatedAt}`;
		})
		.join("\n");
}

function formatWorkspaceArtifactSummary(artifactId: string, artifactIds: string[]): string {
	const extra = artifactIds.length > 1 ? ` (+${artifactIds.length - 1})` : "";
	return `${artifactId}${extra}`;
}

function formatWorkspaceValidationSummary(validation?: Record<string, unknown>): string {
	if (!validation) {
		return "";
	}
	const summaryKeys = ["passed", "success", "status", "blockers", "warnings"];
	const parts = summaryKeys.flatMap((key) => {
		if (!Object.hasOwn(validation, key)) {
			return [];
		}
		const value = validation[key];
		if (Array.isArray(value)) {
			return [`${key}=${value.length}`];
		}
		if (isRecord(value)) {
			return [`${key}={...}`];
		}
		return [`${key}=${String(value)}`];
	});
	return parts.length > 0 ? ` validation=${parts.slice(0, 3).join(",")}` : "";
}

function splitFirstArgument(value: string): { token?: string; rest: string } {
	const trimmed = value.trimStart();
	if (!trimmed) {
		return { rest: "" };
	}
	const match = trimmed.match(/^(\S+)(?:\s+([\s\S]*))?$/);
	return { token: match?.[1], rest: match?.[2] ?? "" };
}

function gpuAutocompleteItems(): Array<{ value: string; label: string; description: string }> | undefined {
	try {
		const output = execFileSync(
			"nvidia-smi",
			["--query-gpu=index,name,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
			{ encoding: "utf-8", timeout: 2_000, windowsHide: true },
		);
		const visible = process.env.CUDA_VISIBLE_DEVICES?.split(",")
			.map((value) => value.trim())
			.filter(Boolean);
		const visibleIndexes = visible && visible.length > 0 ? new Set(visible) : undefined;
		return output.split("\n").flatMap((line) => {
			const [index, name, freeMemory, utilization] = line.split(",").map((value) => value.trim());
			if (!index || !name || !freeMemory || !utilization || (visibleIndexes && !visibleIndexes.has(index))) {
				return [];
			}
			return [
				{
					value: index,
					label: index,
					description: `${name} — ${freeMemory} MiB free, ${utilization}% utilization`,
				},
			];
		});
	} catch {
		return undefined;
	}
}

function formatXlabProductHelp(packages: XlabSkillPackage[], task?: string): string {
	if (task) {
		const product = xlabProductCommandForTask(task);
		const skillPackage = product
			? packages.find((candidate) => candidate.manifest.name === product.skillName)
			: undefined;
		if (!product) {
			return `Unknown XLab task: ${task}. Run /xlab show-help to list tasks.`;
		}
		if (!skillPackage) {
			return `XLab task /xlab ${task} is unavailable because ${product.skillName} is not installed.`;
		}
		return [
			`/xlab ${task} ${skillPackage.manifest.ui?.argumentHint ?? ""}`.trim(),
			skillPackage.manifest.description,
			`Package: ${skillPackage.manifest.name}@${skillPackage.manifest.version}`,
		]
			.filter((line): line is string => Boolean(line))
			.join("\n");
	}
	const tasks = XLAB_PRODUCT_COMMANDS.map((product) => {
		const skillPackage = packages.find((candidate) => candidate.manifest.name === product.skillName);
		return `- /xlab ${product.task}${
			skillPackage?.manifest.description
				? ` — ${skillPackage.manifest.description}`
				: ` — unavailable (${product.skillName})`
		}`;
	});
	return [
		"XLab research commands:",
		...tasks,
		"XLab management commands:",
		...XLAB_CONTROL_COMMANDS.map((control) => `- /xlab ${control.task} — ${control.description}`),
		"Use /xlab show-help <task> for task arguments.",
	].join("\n");
}

function formatWorkspaceDetails(manager: XlabWorkspaceManager, slug: string): string | undefined {
	const workspace = manager.readWorkspace(slug);
	if (!workspace) {
		return undefined;
	}
	const activeSlug = manager.readIndex().activeWorkspaceSlug;
	const current = manager.readCurrent(workspace.slug).current;
	const lineage = manager.readLineage(workspace.slug);
	const warnings = manager.staleWarnings(workspace.slug);
	const handles = lineage.handles.length
		? lineage.handles
				.map((handle) => {
					const parents = handle.parentHandles.length > 0 ? ` parents=${handle.parentHandles.join(",")}` : "";
					const artifact = formatWorkspaceArtifactSummary(handle.artifactId, handle.artifactIds);
					const validation = formatWorkspaceValidationSummary(handle.validation);
					return `- ${handle.handle} ${handle.skillName}@${handle.skillVersion} ${handle.status} run=${handle.runId} artifact=${artifact}${parents}${validation}`;
				})
				.join("\n")
		: "- no registered artifacts";
	return [
		`${workspace.slug}${workspace.slug === activeSlug ? " (active)" : ""}`,
		`Name: ${workspace.displayName}`,
		workspace.topic ? `Topic: ${workspace.topic}` : undefined,
		`Current graph: ${current.graph ?? "(none)"}`,
		`Current survey: ${current.survey ?? "(none)"}`,
		`Current idea: ${current.idea ?? "(none)"}`,
		"Lineage:",
		handles,
		warnings.length > 0 ? `Warnings:\n${warnings.map((warning) => `- ${warning}`).join("\n")}` : undefined,
	]
		.filter((line): line is string => line !== undefined)
		.join("\n");
}

function validateManifestEnvelope(
	value: unknown,
	run: XlabRunRecord,
	finish: XlabFinishParams,
	runManager: XlabRunManager,
	skillPackage: XlabSkillPackage,
): string | undefined {
	if (!isRecord(value)) {
		return "Final manifest must be a JSON object.";
	}
	const required = [
		"schema_version",
		"run_id",
		"skill_name",
		"skill_version",
		"status",
		"created_at",
		"inputs",
		"outputs",
		"validation",
	];
	for (const key of required) {
		if (!(key in value)) {
			return `Final manifest is missing required field "${key}".`;
		}
	}
	if (value.run_id !== run.runId) {
		return `Final manifest run_id must be "${run.runId}".`;
	}
	if (value.skill_name !== run.skillName) {
		return `Final manifest skill_name must be "${run.skillName}".`;
	}
	if (value.skill_version !== run.skillVersion) {
		return `Final manifest skill_version must be "${run.skillVersion}".`;
	}
	if (value.schema_version !== "2") {
		return 'Final manifest field "schema_version" must be "2".';
	}
	if (value.status !== finish.status) {
		return `Final manifest status must match xlab_finish status "${finish.status}".`;
	}
	if (typeof value.created_at !== "string" || Number.isNaN(Date.parse(value.created_at))) {
		return 'Final manifest field "created_at" must be an ISO date string.';
	}
	if (!isStringRecord(value.inputs)) {
		return 'Final manifest field "inputs" must be a JSON object.';
	}
	if (!isStringRecord(value.outputs)) {
		return 'Final manifest field "outputs" must be a JSON object.';
	}
	if (!isStringRecord(value.validation)) {
		return 'Final manifest field "validation" must be a JSON object.';
	}
	if (!Array.isArray(value.artifacts) || !value.artifacts.every(isManifestArtifact)) {
		return 'Final manifest field "artifacts" must contain typed entries with type, schema_version, and path.';
	}
	const manifest = value as unknown as XlabManifestEnvelope;
	for (const artifact of manifest.artifacts) {
		const artifactPath = runManager.resolveRunPath(run, artifact.path);
		if (!artifactPath || !existsSync(artifactPath)) {
			return `Manifest artifact path does not exist: ${artifact.path}.`;
		}
	}
	if (finish.status === "success") {
		for (const requirement of skillPackage.manifest.artifacts ?? []) {
			const artifactError = validateRequiredArtifact(requirement, manifest, runManager, run);
			if (artifactError) {
				return artifactError;
			}
		}
	}
	return undefined;
}

async function startRun(
	pi: ExtensionAPI,
	ctx: ExtensionCommandContext,
	skillPackage: XlabSkillPackage,
	args: string,
	setActiveRun: (run: ActiveRun | undefined, expected?: ActiveRun) => boolean,
	ownerId: string,
	invokedCommand?: string,
): Promise<void> {
	if (!ctx.isIdle()) {
		ctx.ui.notify("Cannot start an XLab run while the agent is busy.", "warning");
		return;
	}
	if (!ctx.isProjectTrusted()) {
		ctx.ui.notify("Cannot start an XLab run until this project is trusted.", "error");
		return;
	}
	const nativeExperiment = isNativeExperiment(skillPackage);
	if (!nativeExperiment) {
		const initialization = await initializeRequiredSecrets(ctx, skillPackage);
		if (!initialization.ok) {
			ctx.ui.notify(
				initialization.message ?? "XLab secret initialization failed.",
				initialization.cancelled ? "warning" : "error",
			);
			return;
		}
	}

	const workspaceManager = new XlabWorkspaceManager(ctx.cwd);
	const workspaceSelection = parseXlabWorkspaceSelection(args);
	const activeWorkspace = workspaceManager.activeWorkspace();
	const workspaceSlug = workspaceSelection?.explicit
		? workspaceSelection.slug
		: (activeWorkspace?.slug ?? workspaceSelection?.slug);
	const argsWithoutWorkspace = workspaceSelection?.explicit ? workspaceSelection.args : args;
	if (workspaceSlug) {
		workspaceManager.ensureWorkspace(workspaceSlug);
	}
	const resolvedArgs = workspaceManager.resolveSkillArgs(skillPackage.manifest.name, argsWithoutWorkspace, {
		workspaceSlug,
	});
	if (resolvedArgs.error) {
		ctx.ui.notify(resolvedArgs.error, "error");
		return;
	}
	for (const warning of resolvedArgs.warnings) {
		ctx.ui.notify(warning, "warning");
	}

	const runManager = new XlabRunManager(ctx.cwd);
	const ideaBindingResult =
		skillPackage.manifest.name === "run_experiment"
			? bindXlabExperimentIdea({
					cwd: ctx.cwd,
					args: resolvedArgs.args,
					workspaceSlug: resolvedArgs.workspaceSlug,
					workspaceManager,
					artifactStore: new XlabArtifactStore(ctx.cwd),
					runManager,
				})
			: undefined;
	if (ideaBindingResult && !ideaBindingResult.ok) {
		ctx.ui.notify(ideaBindingResult.error, "error");
		return;
	}
	if (ideaBindingResult?.ok) {
		for (const warning of ideaBindingResult.binding.warnings) {
			ctx.ui.notify(warning, "warning");
		}
	}
	const binding = ideaBindingResult?.ok ? ideaBindingResult.binding : undefined;
	const record = runManager.createRun(skillPackage, resolvedArgs.args, {
		workspaceSlug: resolvedArgs.workspaceSlug,
		invokedCommand,
		ideaBinding: binding,
		...(nativeExperiment && binding ? { materializeStaging: createXlabExperimentStagingMaterializer(binding) } : {}),
	});
	if (nativeExperiment) {
		const runtime = nativeExperimentRuntime(record, ownerId, skillPackage, binding);
		const active: NativeActiveRun = {
			kind: "native",
			record,
			skillPackage,
			lifecycle: runtime.lifecycle,
			execution: Promise.resolve(),
			previousActiveTools: pi.getActiveTools(),
		};
		setActiveRun(active);
		setRunStatusText(ctx, active.record);
		setRunDashboard(ctx, active);
		active.execution = settleNativeExecution(pi, ctx, active, setActiveRun, runtime.lifecycle.start(runtime.drivers));
		return;
	}
	const active: AgentActiveRun = {
		kind: "agent",
		record,
		skillPackage,
		hooks: new XlabHookRunner(skillPackage),
		previousActiveTools: pi.getActiveTools(),
	};
	setActiveRun(active);
	setRunStatusText(ctx, active.record);
	setRunDashboard(ctx, active);

	const runtimeIssues = validateXlabRuntimePrerequisites(skillPackage);
	if (runtimeIssues.length > 0) {
		const repair = runtimeIssues.map((issue) => issue.message).join(" ");
		active.record = runManager.updateRun(
			active.record,
			{
				status: "failed",
				finishedAt: new Date().toISOString(),
				lastRepair: repair,
			},
			"runtime_prerequisite_failure",
			runtimeIssues,
		);
		ctx.ui.notify(repair, "error");
		pi.appendEntry("xlab.run", active.record);
		clearActiveRun(pi, ctx, active, setActiveRun);
		return;
	}
	try {
		const environment = prepareXlabEnvironment(active.record.cwd, skillPackage);
		if (environment) {
			active.record = runManager.updateRun(active.record, { environment }, "environment_prepared", environment);
		}
	} catch (error) {
		const repair = `Failed to prepare the locked runtime environment: ${
			error instanceof Error ? error.message : String(error)
		}`;
		active.record = runManager.updateRun(
			active.record,
			{
				status: "failed",
				finishedAt: new Date().toISOString(),
				lastRepair: repair,
			},
			"environment_failure",
			{ repair },
		);
		ctx.ui.notify(repair, "error");
		pi.appendEntry("xlab.run", active.record);
		clearActiveRun(pi, ctx, active, setActiveRun);
		return;
	}

	const preStart = await active.hooks.run("pre_start", {
		run: active.record,
		skill: skillPackage.manifest,
		cwd: ctx.cwd,
		packageDir: skillPackage.packageDir,
		runDir: active.record.runDir,
		args: active.record.args,
	});
	applyStatePatch(pi, ctx, active, preStart.state_patch, "pre_start_state");
	if (!preStart.ok) {
		active.record = runManager.updateRun(
			active.record,
			{
				status: "failed",
				finishedAt: new Date().toISOString(),
				lastRepair: preStart.repair ?? preStart.message,
			},
			"pre_start_repair",
			preStart,
		);
		ctx.ui.notify(preStart.repair ?? preStart.message ?? "XLab pre-start gate rejected the run.", "error");
		pi.appendEntry("xlab.run", active.record);
		clearActiveRun(pi, ctx, active, setActiveRun);
		return;
	}

	active.record = runManager.setStatus(active.record, "running", "started");
	pi.setActiveTools(activeToolNames(active));
	const state = runManager.readState(active.record);
	setRunStatusText(ctx, active.record, state);
	setRunDashboard(ctx, active, state);
	pi.appendEntry("xlab.run", active.record);
	try {
		await pi.sendUserMessage(buildInvocationPrompt(skillPackage, active.record));
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		active.record = runManager.updateRun(
			active.record,
			{
				status: "failed",
				finishedAt: new Date().toISOString(),
				lastRepair: `XLab run failed to start or continue: ${message}`,
			},
			"agent_error",
			{ message },
		);
		const onError = await active.hooks.run("on_error", {
			run: active.record,
			skill: skillPackage.manifest,
			cwd: ctx.cwd,
			packageDir: skillPackage.packageDir,
			runDir: active.record.runDir,
			args: active.record.args,
			event: { reason: "agent_error", message },
		});
		applyStatePatch(pi, ctx, active, onError.state_patch, "on_error_state");
		ctx.ui.setStatus(STATUS_KEY, undefined);
		pi.appendEntry("xlab.run", active.record);
		clearActiveRun(pi, ctx, active, setActiveRun);
		throw error;
	}
}

async function settleNativeExecution(
	pi: ExtensionAPI,
	ctx: ExtensionContext,
	active: NativeActiveRun,
	setActiveRun: (run: ActiveRun | undefined, expected?: ActiveRun) => boolean,
	execution: Promise<XlabRunRecord>,
): Promise<void> {
	try {
		active.record = await execution;
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		try {
			active.record =
				active.lifecycle.status().status === "cancelled"
					? active.lifecycle.status()
					: active.lifecycle.settleFailure(message);
		} catch {
			active.record = active.lifecycle.status();
		}
		if (active.record.status !== "cancelled") {
			ctx.ui.notify(`Native experiment failed: ${message}`, "error");
		}
	}
	pi.appendEntry("xlab.run", active.record);
	setRunStatusText(ctx, active.record);
	clearActiveRun(pi, ctx, active, setActiveRun);
}

function clearActiveRun(
	pi: ExtensionAPI,
	ctx: ExtensionContext,
	active: ActiveRun | undefined,
	setActiveRun: (run: ActiveRun | undefined, expected?: ActiveRun) => boolean,
) {
	if (!active || !setActiveRun(undefined, active)) {
		return;
	}
	if (active.kind === "agent") {
		pi.setActiveTools(
			pi
				.getActiveTools()
				.filter(
					(toolName) =>
						toolName !== FINISH_TOOL_NAME && toolName !== UPDATE_STATE_TOOL_NAME && toolName !== STAGE_TOOL_NAME,
				),
		);
	}
	clearRunDashboard(ctx);
}

export function createXlabExtension(): ExtensionFactory {
	return (pi: ExtensionAPI) => {
		let activeRun: ActiveRun | undefined;
		const runtimeOwnerId = `extension-${randomUUID()}`;
		let autocompleteCwd = process.cwd();

		const setActiveRun = (run: ActiveRun | undefined, expected?: ActiveRun): boolean => {
			if (expected !== undefined && activeRun !== expected) return false;
			activeRun = run;
			return true;
		};

		pi.registerTool(
			defineTool({
				name: UPDATE_STATE_TOOL_NAME,
				label: "XLab State",
				description:
					"Update the active XLab run display state for TUI/WebUI status, diagnostics, artifacts, and checkpoints.",
				promptSnippet: "Update the active XLab run display state",
				promptGuidelines: [
					`Use ${UPDATE_STATE_TOOL_NAME} during active XLab skill runs to report phase, progress, counters, diagnostics, artifacts, checkpoints, or next actions.`,
					"Use skill-defined phase ids and counter names; Harness only validates the common state shape.",
				],
				activeByDefault: false,
				executionMode: "sequential",
				parameters: updateStateToolParams,
				async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
					const active = activeRun;
					if (!active || active.kind === "native") {
						return {
							content: toolText("No active XLab run. Start an XLab slash command before updating state."),
							details: { accepted: false } satisfies XlabUpdateStateDetails,
							isError: true,
						};
					}
					const result = applyStatePatch(pi, ctx, active, params, "state_update");
					if (!result.ok) {
						return {
							content: toolText(result.repair ?? "Invalid XLab state update."),
							details: {
								runId: active.record.runId,
								accepted: false,
								repair: result.repair,
							} satisfies XlabUpdateStateDetails,
							isError: true,
						};
					}
					return {
						content: toolText(`Updated XLab run ${active.record.runId} state.`),
						details: {
							runId: active.record.runId,
							accepted: true,
							state: result.state,
						} satisfies XlabUpdateStateDetails,
					};
				},
			}),
		);

		pi.registerTool(
			defineTool({
				name: STAGE_TOOL_NAME,
				label: "XLab Stage",
				description: "Advance a declared XLab workflow stage with durable attempt and checkpoint state.",
				promptSnippet: "Advance an XLab workflow stage",
				promptGuidelines: [
					`Call ${STAGE_TOOL_NAME} with action=start before executing a declared stage.`,
					"Mark the stage complete only after its artifacts pass the stage contract.",
				],
				activeByDefault: false,
				executionMode: "sequential",
				parameters: stageToolParams,
				async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
					const active = activeRun;
					if (!active || active.kind === "native") {
						return {
							content: toolText("No active XLab run. Start or resume a workflow before updating a stage."),
							details: { accepted: false },
							isError: true,
						};
					}
					const action = params.action;
					const runManager = new XlabRunManager(active.record.cwd);
					const workflow = runManager.readWorkflow(active.record);
					if (!workflow) {
						return {
							content: toolText(`XLab run ${active.record.runId} has no workflow declaration.`),
							details: {
								accepted: false,
								repair: "Use xlab_stage only for declared workflows.",
							},
							isError: true,
						};
					}
					const preview = transitionWorkflowStage(workflow, params.stage_id, action, {
						checkpoint: params.checkpoint,
						artifacts: params.artifacts,
						error: params.error,
					});
					if (!preview.ok) {
						return {
							content: toolText(preview.repair ?? "Workflow stage transition failed."),
							details: { accepted: false, repair: preview.repair },
							isError: true,
						};
					}
					let environment: XlabEnvironmentRecord | undefined;
					const stage = active.skillPackage.manifest.workflow?.stages.find(
						(candidate) => candidate.id === params.stage_id,
					);
					const dependency = stage
						? active.skillPackage.dependencies.find((candidate) => candidate.manifest.name === stage.skill)
						: undefined;
					if ((action === "start" || action === "complete") && dependency) {
						const initialization = await initializeRequiredSecrets(ctx, dependency);
						if (!initialization.ok) {
							const repair = initialization.message ?? "XLab stage secret initialization failed.";
							return {
								content: toolText(repair),
								details: { accepted: false, repair },
								isError: true,
							};
						}
						const issues = validateXlabRuntimePrerequisites(dependency);
						if (issues.length > 0) {
							const repair = issues.map((issue) => issue.message).join(" ");
							return {
								content: toolText(repair),
								details: { accepted: false, repair },
								isError: true,
							};
						}
						try {
							environment = prepareXlabEnvironment(active.record.cwd, dependency);
						} catch (error) {
							const repair = `Failed to prepare stage environment: ${
								error instanceof Error ? error.message : String(error)
							}`;
							return {
								content: toolText(repair),
								details: { accepted: false, repair },
								isError: true,
							};
						}
						if (action === "start") {
							const dependencyGate = await new XlabHookRunner(dependency).run("pre_start", {
								run: environment ? { ...active.record, environment } : active.record,
								skill: dependency.manifest,
								cwd: ctx.cwd,
								packageDir: dependency.packageDir,
								runDir: active.record.runDir,
								args: active.record.args,
								event: { stage: params.stage_id },
							});
							applyStatePatch(pi, ctx, active, dependencyGate.state_patch, `${params.stage_id}_pre_start_state`);
							if (!dependencyGate.ok) {
								const repair =
									dependencyGate.repair ??
									dependencyGate.message ??
									`${dependency.manifest.name} pre-start hook rejected the stage.`;
								return {
									content: toolText(repair),
									details: { accepted: false, repair },
									isError: true,
								};
							}
						}
					}
					let stageArtifacts: XlabManifestArtifact[] | undefined;
					let stageArtifactRefs: XlabArtifactReference[] | undefined;
					if (action === "complete" && dependency) {
						const validation = validateStageArtifacts(params.artifacts, dependency, runManager, active.record);
						if (!validation.artifacts) {
							return {
								content: toolText(validation.repair ?? "Stage artifacts failed validation."),
								details: { accepted: false, repair: validation.repair },
								isError: true,
							};
						}
						stageArtifacts = validation.artifacts;
						const dependencyGate = await new XlabHookRunner(dependency).run("pre_finish", {
							run: environment ? { ...active.record, environment } : active.record,
							skill: dependency.manifest,
							cwd: ctx.cwd,
							packageDir: dependency.packageDir,
							runDir: active.record.runDir,
							args: active.record.args,
							event: {
								stage: params.stage_id,
								finish: { status: "success" },
								manifest: { artifacts: stageArtifacts },
							},
						});
						applyStatePatch(pi, ctx, active, dependencyGate.state_patch, `${params.stage_id}_pre_finish_state`);
						if (!dependencyGate.ok) {
							const repair =
								dependencyGate.repair ??
								dependencyGate.message ??
								`${dependency.manifest.name} artifact hook rejected the stage.`;
							return {
								content: toolText(repair),
								details: { accepted: false, repair },
								isError: true,
							};
						}
						const stageManifest: XlabManifestEnvelope = {
							schema_version: "2",
							run_id: active.record.runId,
							skill_name: dependency.manifest.name,
							skill_version: dependency.manifest.version,
							status: "success",
							created_at: new Date().toISOString(),
							inputs: {
								parent_skill: active.record.skillName,
								stage: params.stage_id,
							},
							outputs: {},
							validation: { stage_gate: true },
							artifacts: stageArtifacts,
						};
						const producerRun: XlabRunRecord = {
							...active.record,
							skillName: dependency.manifest.name,
							skillVersion: dependency.manifest.version,
						};
						try {
							stageArtifactRefs = new XlabArtifactStore(active.record.cwd).commitManifestArtifacts(
								producerRun,
								stageManifest,
								(path) => runManager.resolveRunPath(active.record, path),
								dependency.manifest.artifacts ?? [],
							);
						} catch (error) {
							const repair = `Stage artifact commit failed: ${
								error instanceof Error ? error.message : String(error)
							}`;
							return {
								content: toolText(repair),
								details: { accepted: false, repair },
								isError: true,
							};
						}
					}
					const point = action === "start" ? "pre_stage" : "post_stage";
					const gate = await active.hooks.run(point, {
						run: active.record,
						skill: active.skillPackage.manifest,
						cwd: ctx.cwd,
						packageDir: active.skillPackage.packageDir,
						runDir: active.record.runDir,
						args: active.record.args,
						event: params,
					});
					applyStatePatch(pi, ctx, active, gate.state_patch, `${point}_state`);
					if (!gate.ok) {
						const repair = gate.repair ?? gate.message ?? `XLab ${point} hook rejected the transition.`;
						return {
							content: toolText(repair),
							details: { accepted: false, repair },
							isError: true,
						};
					}
					const transition = runManager.transitionStage(active.record, params.stage_id, action, {
						checkpoint: params.checkpoint,
						artifacts: stageArtifactRefs ?? stageArtifacts ?? params.artifacts,
						error: params.error,
					});
					if (!transition.ok) {
						return {
							content: toolText(transition.repair ?? "Workflow stage transition failed."),
							details: { accepted: false, repair: transition.repair },
							isError: true,
						};
					}
					setRunDashboard(ctx, active, runManager.readState(active.record));
					return {
						content: toolText(
							`Workflow stage ${params.stage_id} marked ${action}.${
								environment ? ` Use locked interpreter ${environment.executable}.` : ""
							}`,
						),
						details: {
							accepted: true,
							workflow: transition.state,
							environment,
						},
					};
				},
			}),
		);

		pi.registerTool(
			defineTool({
				name: FINISH_TOOL_NAME,
				label: "XLab Finish",
				description: "Finish the active XLab scientific run after validating its artifact manifest and gates.",
				promptSnippet: "Finish an active XLab skill run with artifact validation",
				promptGuidelines: [
					`Use ${FINISH_TOOL_NAME} as the final standalone action for active XLab skill runs.`,
					"If it returns repair instructions, update artifacts and call it again.",
				],
				activeByDefault: false,
				executionMode: "sequential",
				parameters: finishToolParams,
				async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
					const active = activeRun;
					if (!active || active.kind === "native") {
						return {
							content: toolText("No active XLab run. Start an XLab slash command before calling xlab_finish."),
							details: { accepted: false } satisfies XlabFinishDetails,
						};
					}

					const finish = params as XlabFinishParams;
					const runManager = new XlabRunManager(active.record.cwd);
					const manifestPath = runManager.resolveRunPath(active.record, finish.manifest_path);
					if (!manifestPath || !existsSync(manifestPath)) {
						const repair = `Create the final manifest at ${finish.manifest_path} inside the project or run workspace, then call ${FINISH_TOOL_NAME} again.`;
						active.record = runManager.updateRun(active.record, { lastRepair: repair }, "finish_repair", {
							repair,
						});
						return {
							content: toolText(repair),
							details: {
								runId: active.record.runId,
								accepted: false,
								repair,
							} satisfies XlabFinishDetails,
						};
					}

					let manifest: unknown;
					try {
						manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
					} catch (error) {
						const repair = `Final manifest must be valid JSON: ${error instanceof Error ? error.message : String(error)}`;
						active.record = runManager.updateRun(active.record, { lastRepair: repair }, "finish_repair", {
							repair,
						});
						return {
							content: toolText(repair),
							details: {
								runId: active.record.runId,
								accepted: false,
								repair,
							} satisfies XlabFinishDetails,
						};
					}

					if (finish.status !== "failed") {
						const gate = await active.hooks.run("pre_finish", {
							run: active.record,
							skill: active.skillPackage.manifest,
							cwd: ctx.cwd,
							packageDir: active.skillPackage.packageDir,
							runDir: active.record.runDir,
							args: active.record.args,
							event: { finish, manifest, manifestPath },
						});
						applyStatePatch(pi, ctx, active, gate.state_patch, "pre_finish_state");
						if (!gate.ok) {
							const repair =
								gate.repair ?? gate.message ?? "Repair the XLab artifacts and call xlab_finish again.";
							active.record = runManager.updateRun(active.record, { lastRepair: repair }, "finish_repair", gate);
							return {
								content: toolText(repair),
								details: {
									runId: active.record.runId,
									accepted: false,
									repair,
									diagnostics: gate.diagnostics,
								} satisfies XlabFinishDetails,
							};
						}
					}

					try {
						manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
					} catch (error) {
						const repair = `Final manifest must be valid JSON: ${error instanceof Error ? error.message : String(error)}`;
						active.record = runManager.updateRun(active.record, { lastRepair: repair }, "finish_repair", {
							repair,
						});
						return {
							content: toolText(repair),
							details: {
								runId: active.record.runId,
								accepted: false,
								repair,
							} satisfies XlabFinishDetails,
						};
					}

					const envelopeError = validateManifestEnvelope(
						manifest,
						active.record,
						finish,
						runManager,
						active.skillPackage,
					);
					if (envelopeError) {
						const repair = `${envelopeError} Repair the manifest and call ${FINISH_TOOL_NAME} again.`;
						active.record = runManager.updateRun(active.record, { lastRepair: repair }, "finish_repair", {
							repair,
						});
						return {
							content: toolText(repair),
							details: {
								runId: active.record.runId,
								accepted: false,
								repair,
							} satisfies XlabFinishDetails,
						};
					}
					const manifestEnvelope = manifest as XlabManifestEnvelope;
					const workflow = runManager.readWorkflow(active.record);
					if (finish.status === "success" && workflow) {
						const unfinished = workflow.stages.filter((stage) => !stage.optional && stage.status !== "success");
						if (unfinished.length > 0) {
							const repair = `Complete required workflow stages before a successful finish: ${unfinished
								.map((stage) => `${stage.id}=${stage.status}`)
								.join(", ")}.`;
							active.record = runManager.updateRun(active.record, { lastRepair: repair }, "finish_repair", {
								repair,
							});
							return {
								content: toolText(repair),
								details: {
									runId: active.record.runId,
									accepted: false,
									repair,
								} satisfies XlabFinishDetails,
								isError: true,
							};
						}
					}

					let artifactRefs: XlabArtifactReference[];
					try {
						const artifactStore = new XlabArtifactStore(active.record.cwd);
						artifactRefs = artifactStore.commitManifestArtifacts(
							active.record,
							manifestEnvelope,
							(path) => runManager.resolveRunPath(active.record, path),
							active.skillPackage.manifest.artifacts ?? [],
						);
					} catch (error) {
						const repair = `Artifact store commit failed: ${
							error instanceof Error ? error.message : String(error)
						}. Repair the artifact payload and call ${FINISH_TOOL_NAME} again.`;
						active.record = runManager.updateRun(
							active.record,
							{ lastRepair: repair },
							"artifact_commit_failure",
							{
								repair,
							},
						);
						return {
							content: toolText(repair),
							details: {
								runId: active.record.runId,
								accepted: false,
								repair,
							} satisfies XlabFinishDetails,
							isError: true,
						};
					}

					let workspaceHandle: string | undefined;
					try {
						const workspaceRegistration = new XlabWorkspaceManager(active.record.cwd).registerRunArtifacts({
							run: {
								...active.record,
								status: finish.status,
								manifestPath,
								summary: finish.summary,
								artifactRefs,
							},
							manifest: manifestEnvelope,
							artifactRefs,
						});
						workspaceHandle = workspaceRegistration?.handles[0]?.handle;
						for (const warning of workspaceRegistration?.warnings ?? []) {
							ctx.ui.notify(warning, "warning");
						}
					} catch (error) {
						pi.appendEntry("xlab.diagnostic", {
							type: "warning",
							message: `XLab workspace registration failed: ${error instanceof Error ? error.message : String(error)}`,
							path: active.skillPackage.manifestPath,
						});
					}

					active.record = runManager.updateRun(
						active.record,
						{
							status: finish.status,
							manifestPath,
							summary: finish.summary,
							errorSummary: finish.error_summary,
							artifacts: finish.artifacts,
							artifactRefs,
							workspaceHandle,
							finishedAt: new Date().toISOString(),
						},
						"finished",
						{ finish, manifestPath, workspaceHandle },
					);
					pi.appendEntry("xlab.run", active.record);
					const postFinish = await active.hooks.run("post_finish", {
						run: active.record,
						skill: active.skillPackage.manifest,
						cwd: ctx.cwd,
						packageDir: active.skillPackage.packageDir,
						runDir: active.record.runDir,
						args: active.record.args,
						event: { finish, manifest, manifestPath, artifactRefs },
					});
					applyStatePatch(pi, ctx, active, postFinish.state_patch, "post_finish_state");
					if (!postFinish.ok) {
						pi.appendEntry("xlab.diagnostic", {
							type: "warning",
							message: postFinish.message ?? postFinish.repair ?? "XLab post-finish hook failed.",
							path: active.skillPackage.manifestPath,
						});
					}
					const state = runManager.updateState(
						active.record,
						{
							phase: {
								id: "finish",
								label: `${active.record.skillName} finished`,
								status: finish.status,
								progress: 1,
							},
							message: finish.summary,
						},
						"terminal_state",
					);
					pi.appendEntry("xlab.state", {
						runId: active.record.runId,
						command: active.record.command,
						skillName: active.record.skillName,
						state,
					});
					setRunStatusText(ctx, active.record, state);
					setRunDashboard(ctx, active, state);
					ctx.ui.setStatus(STATUS_KEY, undefined);
					clearActiveRun(pi, ctx, active, setActiveRun);
					return {
						content: toolText(`XLab run ${active.record.runId} finished with status ${finish.status}.`),
						details: {
							runId: active.record.runId,
							status: finish.status,
							accepted: true,
						} satisfies XlabFinishDetails,
						terminate: true,
					};
				},
			}),
		);

		pi.on("session_start", async (_event, ctx) => {
			autocompleteCwd = ctx.cwd;
		});

		pi.on("tool_call", async (event, ctx) => {
			const active = activeRun;
			if (
				!active ||
				active.kind === "native" ||
				event.toolName === FINISH_TOOL_NAME ||
				event.toolName === UPDATE_STATE_TOOL_NAME ||
				event.toolName === STAGE_TOOL_NAME
			) {
				return undefined;
			}
			const runManager = new XlabRunManager(active.record.cwd);
			active.record = runManager.updateRun(active.record, {}, "tool_call", {
				toolName: event.toolName,
				toolCallId: event.toolCallId,
				input: event.input,
			});
			const gate = await active.hooks.run("pre_tool_call", {
				run: active.record,
				skill: active.skillPackage.manifest,
				cwd: ctx.cwd,
				packageDir: active.skillPackage.packageDir,
				runDir: active.record.runDir,
				args: active.record.args,
				event,
			});
			applyStatePatch(pi, ctx, active, gate.state_patch, "pre_tool_call_state");
			if (!gate.ok) {
				const repair = gate.repair ?? gate.message ?? "Repair the tool call and retry.";
				active.record = runManager.updateRun(active.record, { lastRepair: repair }, "tool_call_repair", gate);
				return { block: true, reason: repair };
			}
			return undefined;
		});

		pi.on("tool_result", async (event, ctx) => {
			const active = activeRun;
			if (
				event.toolName === FINISH_TOOL_NAME ||
				event.toolName === UPDATE_STATE_TOOL_NAME ||
				event.toolName === STAGE_TOOL_NAME
			) {
				if (isRecord(event.details) && event.details.accepted === false) {
					return { isError: true };
				}
				return undefined;
			}
			if (!active || active.kind === "native") {
				return undefined;
			}
			const runManager = new XlabRunManager(active.record.cwd);
			active.record = runManager.updateRun(active.record, {}, "tool_result", {
				toolName: event.toolName,
				toolCallId: event.toolCallId,
				isError: event.isError,
			});
			const gate = await active.hooks.run("post_tool_result", {
				run: active.record,
				skill: active.skillPackage.manifest,
				cwd: ctx.cwd,
				packageDir: active.skillPackage.packageDir,
				runDir: active.record.runDir,
				args: active.record.args,
				event,
			});
			applyStatePatch(pi, ctx, active, gate.state_patch, "post_tool_result_state");
			if (!gate.ok) {
				const repair = gate.repair ?? gate.message ?? "Repair the tool result issue and retry.";
				active.record = runManager.updateRun(active.record, { lastRepair: repair }, "tool_result_repair", gate);
				return {
					content: toolText(repair),
					details: { repair, diagnostics: gate.diagnostics },
					isError: true,
				};
			}
			return undefined;
		});

		pi.on("agent_end", async (_event, ctx) => {
			const active = activeRun;
			if (!active || active.kind === "native") {
				return;
			}
			if (active.record.status !== "running") {
				return;
			}
			const runManager = new XlabRunManager(active.record.cwd);
			active.record = runManager.updateRun(
				active.record,
				{
					lastRepair: `The XLab run is still active. Call ${FINISH_TOOL_NAME} after producing the required manifest.`,
				},
				"finish_missing",
			);
			ctx.ui.notify(`XLab run ${active.record.runId} is still waiting for ${FINISH_TOOL_NAME}.`, "warning");
		});

		pi.on("session_shutdown", async (_event, ctx) => {
			const active = activeRun;
			if (!active) {
				return;
			}
			if (active.kind === "native") {
				active.record = await active.lifecycle.cancel("session_shutdown");
				await active.execution;
				pi.appendEntry("xlab.run", active.record);
				ctx.ui.setStatus(STATUS_KEY, undefined);
				clearActiveRun(pi, ctx, active, setActiveRun);
				return;
			}
			const runManager = new XlabRunManager(active.record.cwd);
			active.record = runManager.setStatus(active.record, "incomplete", "session_shutdown");
			pi.appendEntry("xlab.run", active.record);
			const onError = await active.hooks.run("on_error", {
				run: active.record,
				skill: active.skillPackage.manifest,
				cwd: ctx.cwd,
				packageDir: active.skillPackage.packageDir,
				runDir: active.record.runDir,
				args: active.record.args,
				event: { reason: "session_shutdown" },
			});
			applyStatePatch(pi, ctx, active, onError.state_patch, "on_error_state");
			ctx.ui.setStatus(STATUS_KEY, undefined);
			clearActiveRun(pi, ctx, active, setActiveRun);
		});

		pi.registerCommand("xlab", {
			description: "Run research workflows and manage XLab",
			argumentHint: "<verb-noun-command> [arguments]",
			getArgumentCompletions: (prefix) => {
				const taskMatch = prefix.match(/^(\S+)(?:\s+([\s\S]*))?$/);
				const product = taskMatch ? xlabProductCommandForTask(taskMatch[1] ?? "") : undefined;
				if (product && taskMatch) {
					const discovered = discoverXlabSkillPackages(autocompleteCwd);
					const skillPackage = resolveXlabProductPackage(discovered.packages, product.task);
					if (!skillPackage) return null;
					const commandArguments = [...(skillPackage.manifest.ui?.arguments ?? [])];
					if (!commandArguments.some((argument) => argument.name === "--workspace")) {
						commandArguments.push({
							name: "--workspace",
							description: "Select an XLab workspace; defaults to the active workspace.",
							syntax: "flag" as const,
							valueHint: "<slug>",
							dynamicSource: "workspace" as const,
							takesValue: true,
						});
					}
					const argumentPrefix = taskMatch[2] ?? "";
					const completions = getXlabArgumentCompletions(commandArguments, argumentPrefix, (argument, context) => {
						const manager = new XlabWorkspaceManager(autocompleteCwd);
						if (argument.dynamicSource === "workspace") {
							return manager.listWorkspaces().map((workspace) => ({
								value: workspace.slug,
								label: workspace.slug,
								description: workspace.displayName,
							}));
						}
						const family =
							argument.dynamicSource === "graph" ||
							argument.dynamicSource === "survey" ||
							argument.dynamicSource === "idea"
								? argument.dynamicSource
								: undefined;
						const selectedWorkspace = getXlabArgumentValue(context.argumentPrefix, "--workspace");
						const workspace = selectedWorkspace
							? manager.readWorkspace(selectedWorkspace)
							: manager.activeWorkspace();
						if (family && workspace) {
							return manager
								.readLineage(workspace.slug)
								.handles.filter((handle) => handle.family === family)
								.map((handle) => ({
									value: handle.handle,
									label: handle.handle,
									description: `${handle.skillName} — ${handle.status}`,
								}));
						}
						if (argument.dynamicSource === "run") {
							return new XlabRunManager(autocompleteCwd).listRuns().map((run) => ({
								value: run.runId,
								label: run.runId,
								description: `${run.skillName} — ${run.status}`,
							}));
						}
						if (argument.dynamicSource === "artifact") {
							return new XlabArtifactStore(autocompleteCwd).list().map((artifact) => ({
								value: artifact.artifact_id,
								label: artifact.artifact_id,
								description: `${artifact.type} — ${artifact.producer.skill}`,
							}));
						}
						if (argument.dynamicSource === "gpu") {
							return gpuAutocompleteItems();
						}
						// File and directory values intentionally fall through to the TUI's
						// project-aware file completer when no manifest choices are available.
						return undefined;
					});
					return (
						completions?.map((item) => ({
							...item,
							value: `${product.task} ${item.value}`,
						})) ?? null
					);
				}
				return getXlabPositionalCompletions(prefix, (completed, current) => {
					const match = (value: string) => value.toLowerCase().startsWith(current.toLowerCase());
					if (completed.length === 0) {
						return [...XLAB_PRODUCT_COMMANDS, ...XLAB_CONTROL_COMMANDS]
							.filter((entry) => match(entry.task))
							.map((entry) => ({
								value: entry.task,
								label: entry.task,
								description: entry.description,
							}));
					}
					if (completed.length === 1 && completed[0] === "show-help") {
						return XLAB_PRODUCT_COMMANDS.filter((entry) => match(entry.task)).map((entry) => ({
							value: entry.task,
							label: entry.task,
							description: entry.description,
						}));
					}
					if (completed.length === 1 && completed[0] === "configure-xlab") {
						return [
							{
								value: "runtime",
								description: "Configure the agent runtime model and credentials.",
							},
							{
								value: "knowledge-graph",
								description: "Configure the paper-extraction model used for knowledge graphs.",
							},
							{
								value: "scholarly-services",
								description: "Configure scholarly search and metadata services.",
							},
							{
								value: "all",
								description: "Configure every XLab runtime and research integration.",
							},
						]
							.filter((entry) => match(entry.value))
							.map((entry) => ({ ...entry, label: entry.value }));
					}
					if (
						completed.length === 1 &&
						(completed[0] === "show-workspace" || completed[0] === "select-workspace")
					) {
						return new XlabWorkspaceManager(autocompleteCwd)
							.listWorkspaces()
							.filter((workspace) => match(workspace.slug))
							.map((workspace) => ({
								value: workspace.slug,
								label: workspace.slug,
								description: workspace.displayName,
							}));
					}
					if (completed.length === 1 && ["show-run", "resume-run", "cancel-run"].includes(completed[0] ?? "")) {
						const resume = completed[0] === "resume-run";
						return new XlabRunManager(autocompleteCwd)
							.listRuns()
							.filter(
								(run) =>
									match(run.runId) && (!resume || !["success", "failed", "cancelled"].includes(run.status)),
							)
							.map((run) => ({
								value: run.runId,
								label: run.runId,
								description: `${run.skillName} — ${run.status}`,
							}));
					}
					if (completed.length === 1 && completed[0] === "show-artifact") {
						return new XlabArtifactStore(autocompleteCwd)
							.list()
							.filter((artifact) => match(artifact.artifact_id))
							.map((artifact) => ({
								value: artifact.artifact_id,
								label: artifact.artifact_id,
								description: `${artifact.type} — ${artifact.producer.skill}`,
							}));
					}
					return [];
				});
			},
			handler: async (args, ctx) => {
				const first = splitFirstArgument(args);
				const subcommand = first.token ?? "show-help";
				const remainder = first.rest;
				const remainderTokens = parseXlabArgumentTokens(remainder);
				const requireArity = (minimum: number, maximum: number, usage: string): boolean => {
					if (!remainderTokens || remainderTokens.length < minimum || remainderTokens.length > maximum) {
						ctx.ui.notify(`Usage: ${usage}`, "warning");
						return false;
					}
					return true;
				};
				const discovered = discoverXlabSkillPackages(ctx.cwd);
				const product = xlabProductCommandForTask(subcommand);
				if (product) {
					if (activeRun) {
						ctx.ui.notify(`XLab run ${activeRun.record.runId} is already active.`, "warning");
						return;
					}
					const skillPackage = resolveXlabProductPackage(discovered.packages, subcommand);
					if (!skillPackage) {
						ctx.ui.notify(
							`XLab task ${subcommand} is unavailable because ${product.skillName} is not installed.`,
							"error",
						);
						return;
					}
					await startRun(pi, ctx, skillPackage, remainder, setActiveRun, runtimeOwnerId, `xlab ${subcommand}`);
					return;
				}
				if (subcommand === "show-help") {
					if (!requireArity(0, 1, "/xlab show-help [task]")) return;
					ctx.ui.notify(formatXlabProductHelp(discovered.packages, remainderTokens?.[0]));
					return;
				}
				if (subcommand === "check-xlab-setup") {
					if (!requireArity(0, 0, "/xlab check-xlab-setup")) return;
					const catalogErrors = validateXlabProductCommands();
					const missingTasks = XLAB_PRODUCT_COMMANDS.filter(
						(entry) => !discovered.packages.some((candidate) => candidate.manifest.name === entry.skillName),
					).map((entry) => `${entry.task} (${entry.skillName})`);
					const report = [
						formatXlabSecretStatus(discovered.packages, ctx.modelRegistry, ctx.model),
						catalogErrors.length > 0
							? `Product catalog errors:\n${catalogErrors.map((error) => `- ${error}`).join("\n")}`
							: "Product catalog: valid",
						missingTasks.length > 0
							? `Unavailable tasks: ${missingTasks.join(", ")}`
							: "Product tasks: all available",
						discovered.diagnostics.length > 0
							? `Discovery diagnostics:\n${formatDiagnostics(discovered.diagnostics)}`
							: "Package discovery: healthy",
					].join("\n");
					publishXlabInitReport(pi, ctx, report, "info", {
						subcommand: "check-xlab-setup",
					});
					return;
				}
				if (subcommand === "configure-xlab") {
					const usage = "/xlab configure-xlab <runtime|knowledge-graph|scholarly-services|all>";
					if (!requireArity(1, 1, usage)) return;
					const setup = remainderTokens?.[0];
					const setupMap: Record<string, string> = {
						runtime: "llm",
						"knowledge-graph": "knowledge_graph",
						"scholarly-services": "scholarly",
						all: "all",
					};
					if (!setup || !setupMap[setup]) {
						ctx.ui.notify(`Usage: ${usage}`, "warning");
						return;
					}
					await runXlabInitCommand(pi, ctx, setupMap[setup]);
					return;
				}
				const identifier = remainderTokens?.[0];
				const diagnostics =
					discovered.diagnostics.length > 0 ? `\n${formatDiagnostics(discovered.diagnostics)}` : "";
				const workspaceManager = new XlabWorkspaceManager(ctx.cwd);
				if (subcommand === "list-workspaces") {
					if (!requireArity(0, 0, "/xlab list-workspaces")) return;
					ctx.ui.notify(formatWorkspaceList(workspaceManager));
					return;
				}
				if (subcommand === "show-workspace") {
					if (!requireArity(1, 1, "/xlab show-workspace <slug>")) return;
					const target = identifier;
					if (!target) return;
					const details = formatWorkspaceDetails(workspaceManager, target);
					ctx.ui.notify(details ?? `XLab workspace not found: ${target}`);
					return;
				}
				if (subcommand === "select-workspace") {
					if (!requireArity(1, 1, "/xlab select-workspace <slug>")) return;
					if (!identifier) return;
					const workspace = workspaceManager.setActiveWorkspace(identifier);
					ctx.ui.notify(
						workspace ? `Selected XLab workspace ${workspace.slug}.` : `XLab workspace not found: ${identifier}`,
					);
					return;
				}
				if (subcommand === "list-runs") {
					if (!requireArity(0, 0, "/xlab list-runs")) return;
					const runs = new XlabRunManager(ctx.cwd).listRuns().slice(0, 20);
					ctx.ui.notify(
						runs.length > 0
							? runs
									.map(
										(run) =>
											`${run.runId} ${run.skillName}@${run.skillVersion} ${run.status} ${run.updatedAt}`,
									)
									.join("\n")
							: "No XLab runs found.",
					);
					return;
				}
				if (subcommand === "show-run") {
					if (!requireArity(1, 1, "/xlab show-run <run-id>")) return;
					if (!identifier) return;
					const manager = new XlabRunManager(ctx.cwd);
					const record = manager.readRun(identifier);
					let displayRecord = record;
					if (record && record.skillName === "run_experiment") {
						try {
							displayRecord = new XlabExperimentLifecycle({
								cwd: record.cwd,
								agentDir: getAgentDir(),
								runId: record.runId,
								ownerId: `extension-${record.runId}`,
							}).status();
						} catch {
							// Preserve the durable run record when native reconstruction diagnostics are unavailable.
						}
					}
					ctx.ui.notify(
						displayRecord
							? JSON.stringify(
									{
										run: displayRecord,
										state: manager.readState(displayRecord),
										workflow: manager.readWorkflow(displayRecord),
									},
									null,
									2,
								)
							: `XLab run not found: ${identifier}`,
					);
					return;
				}
				if (subcommand === "show-artifact") {
					if (!requireArity(1, 1, "/xlab show-artifact <artifact-id>")) return;
					if (!identifier) return;
					const metadata = new XlabArtifactStore(ctx.cwd).get(identifier);
					ctx.ui.notify(metadata ? JSON.stringify(metadata, null, 2) : `Artifact not found: ${identifier}`);
					return;
				}
				if (subcommand === "cancel-run") {
					if (!requireArity(1, 1, "/xlab cancel-run <run-id>")) return;
					if (!identifier) return;
					const runManager = new XlabRunManager(ctx.cwd);
					const record = runManager.readRun(identifier);
					if (!record) {
						ctx.ui.notify(`XLab run not found: ${identifier}`, "error");
						return;
					}
					const active = activeRun?.record.runId === identifier ? activeRun : undefined;
					if (record.skillName === "run_experiment") {
						const native =
							active?.kind === "native"
								? active.lifecycle
								: new XlabExperimentLifecycle({
										cwd: record.cwd,
										agentDir: getAgentDir(),
										runId: record.runId,
										ownerId: `extension-${record.runId}`,
									});
						const cancelled = await native.cancel("user_cancel");
						if (active?.kind === "native") {
							active.record = cancelled;
							await active.execution;
							clearActiveRun(pi, ctx, active, setActiveRun);
						}
					} else if (active?.kind === "agent") {
						const cancelled = await active.hooks.run("on_cancel", {
							run: active.record,
							skill: active.skillPackage.manifest,
							cwd: ctx.cwd,
							packageDir: active.skillPackage.packageDir,
							runDir: active.record.runDir,
							args: active.record.args,
							event: { reason: "user_cancel" },
						});
						applyStatePatch(pi, ctx, active, cancelled.state_patch, "on_cancel_state");
						active.record = runManager.setStatus(active.record, "cancelled", "user_cancel");
						pi.appendEntry("xlab.run", active.record);
						clearActiveRun(pi, ctx, active, setActiveRun);
					} else {
						runManager.setStatus(record, "cancelled", "user_cancel");
					}
					ctx.ui.setStatus(STATUS_KEY, undefined);
					ctx.ui.notify(`Cancelled XLab run ${identifier}.`);
					return;
				}
				if (subcommand === "resume-run") {
					if (!requireArity(1, 1, "/xlab resume-run <run-id>")) return;
					if (!identifier) return;
					if (activeRun) {
						ctx.ui.notify(`XLab run ${activeRun.record.runId} is already active.`, "warning");
						return;
					}
					if (!ctx.isIdle() || !ctx.isProjectTrusted()) {
						ctx.ui.notify("XLab resume requires an idle agent and a trusted project.", "error");
						return;
					}
					const runManager = new XlabRunManager(ctx.cwd);
					const record = runManager.readRun(identifier);
					if (!record) {
						ctx.ui.notify(`XLab run not found: ${identifier}`, "error");
						return;
					}
					if (record.status === "success" || record.status === "failed" || record.status === "cancelled") {
						ctx.ui.notify(`Terminal XLab run cannot be resumed: ${record.status}`, "error");
						return;
					}
					const skillPackage = discovered.packages.find(
						(candidate) =>
							candidate.manifest.name === record.skillName && candidate.manifest.version === record.skillVersion,
					);
					if (!skillPackage) {
						ctx.ui.notify(
							`Cannot resume without ${record.skillName}@${record.skillVersion} in the resolved dependency closure.`,
							"error",
						);
						return;
					}
					if (isNativeExperiment(skillPackage)) {
						try {
							const runtime = nativeExperimentRuntime(record, runtimeOwnerId, skillPackage);
							const active: NativeActiveRun = {
								kind: "native",
								record,
								skillPackage,
								lifecycle: runtime.lifecycle,
								execution: Promise.resolve(),
								previousActiveTools: pi.getActiveTools(),
							};
							setActiveRun(active);
							setRunStatusText(ctx, active.record);
							setRunDashboard(ctx, active);
							active.execution = settleNativeExecution(
								pi,
								ctx,
								active,
								setActiveRun,
								runtime.lifecycle.resume(runtime.drivers),
							);
						} catch (error) {
							ctx.ui.notify(
								`Native experiment resume failed: ${error instanceof Error ? error.message : String(error)}`,
								"error",
							);
						}
						return;
					}
					const initialization = await initializeRequiredSecrets(ctx, skillPackage);
					if (!initialization.ok) {
						ctx.ui.notify(
							initialization.message ?? "XLab secret initialization failed.",
							initialization.cancelled ? "warning" : "error",
						);
						return;
					}
					const runtimeIssues = validateXlabRuntimePrerequisites(skillPackage);
					if (runtimeIssues.length > 0) {
						ctx.ui.notify(runtimeIssues.map((issue) => issue.message).join(" "), "error");
						return;
					}
					let environment: XlabEnvironmentRecord | undefined;
					try {
						environment = prepareXlabEnvironment(record.cwd, skillPackage);
					} catch (error) {
						ctx.ui.notify(
							`Failed to restore the locked runtime environment: ${
								error instanceof Error ? error.message : String(error)
							}`,
							"error",
						);
						return;
					}
					let resumedRecord = runManager.resumeRun(record);
					if (environment) {
						resumedRecord = runManager.updateRun(
							resumedRecord,
							{ environment },
							"environment_prepared",
							environment,
						);
					}
					const active: AgentActiveRun = {
						kind: "agent",
						record: resumedRecord,
						skillPackage,
						hooks: new XlabHookRunner(skillPackage),
						previousActiveTools: pi.getActiveTools(),
					};
					setActiveRun(active);
					const gate = await active.hooks.run("on_resume", {
						run: active.record,
						skill: skillPackage.manifest,
						cwd: ctx.cwd,
						packageDir: skillPackage.packageDir,
						runDir: active.record.runDir,
						args: active.record.args,
						event: {
							state: runManager.readState(active.record),
							workflow: runManager.readWorkflow(active.record),
						},
					});
					applyStatePatch(pi, ctx, active, gate.state_patch, "on_resume_state");
					if (!gate.ok) {
						active.record = runManager.setStatus(active.record, "incomplete", "resume_rejected");
						clearActiveRun(pi, ctx, active, setActiveRun);
						ctx.ui.notify(gate.repair ?? gate.message ?? "XLab resume hook rejected the run.", "error");
						return;
					}
					pi.setActiveTools(activeToolNames(active));
					const state = runManager.readState(active.record);
					setRunStatusText(ctx, active.record, state);
					setRunDashboard(ctx, active, state);
					pi.appendEntry("xlab.run", active.record);
					await pi.sendUserMessage(buildResumePrompt(skillPackage, active.record, runManager));
					return;
				}
				ctx.ui.notify(`Unknown XLab command: ${subcommand}. Run /xlab show-help.${diagnostics}`, "warning");
			},
		});
	};
}

export const xlabExtension = createXlabExtension();
