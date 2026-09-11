import type { ExtensionContext } from "../extensions/index.ts";
import { xlabProductCommandForSkill } from "./product-commands.ts";
import type { XlabSkillPackage } from "./types.ts";

export type XlabInitGroup = "knowledge_graph" | "scholarly";

export interface XlabEnvironmentVariableDefinition {
	name: string;
	group?: XlabInitGroup;
	description: string;
	secret?: boolean;
	required?: boolean;
	promptLabel?: string;
	promptDescription?: string;
	placeholder?: string;
}

export interface XlabSecretUsage {
	name: string;
	requiredBy: string[];
	optionalBy: string[];
}

export interface XlabPromptVariable {
	name: string;
	description?: string;
	secret?: boolean;
	required?: boolean;
	promptLabel?: string;
	promptDescription?: string;
	placeholder?: string;
	satisfiedBy?: string[];
}

export interface XlabPromptResult {
	ok: boolean;
	cancelled?: boolean;
	initialized: string[];
	skipped: string[];
	message?: string;
}

export const XLAB_COMMON_ENVIRONMENT_VARIABLES: XlabEnvironmentVariableDefinition[] = [
	{
		name: "KNOWLEDGE_GRAPH_LLM_API_KEY",
		group: "knowledge_graph",
		description:
			"Extraction LLM API key used only by /xlab build-knowledge-graph; independent from the agent runtime model.",
		secret: true,
		required: true,
		promptLabel: "paper-extraction LLM API key",
		promptDescription:
			"This key is sent only to the separate LLM endpoint used by /xlab build-knowledge-graph Step 2 extraction. It is not the Pi Agent runtime API key and is kept in this process only.",
		placeholder: "Paste the extraction LLM API key",
	},
	{
		name: "KNOWLEDGE_GRAPH_LLM_API_URL",
		group: "knowledge_graph",
		description:
			"OpenAI-compatible base URL or full chat-completions URL for the separate /xlab build-knowledge-graph extraction LLM.",
		secret: false,
		required: true,
		promptLabel: "paper-extraction LLM endpoint URL",
		promptDescription:
			"Use an OpenAI-compatible base URL or full /chat/completions endpoint. This configures the LLM used to extract paper graph entities, independent from the Pi Agent chat/coding model.",
		placeholder: "https://api.example.com/v1 or https://api.example.com/v1/chat/completions",
	},
	{
		name: "KNOWLEDGE_GRAPH_LLM_MODEL",
		group: "knowledge_graph",
		description: "Provider model identifier for the separate /xlab build-knowledge-graph paper-extraction LLM.",
		secret: false,
		required: true,
		promptLabel: "paper-extraction LLM model id",
		promptDescription:
			"Enter the provider model name used for PaperGraph-style extraction, such as gpt-4.1, qwen-plus, or deepseek-chat. This is separate from the Pi Agent runtime model.",
		placeholder: "gpt-4.1, qwen-plus, deepseek-chat, ...",
	},
	{
		name: "LLM_BASE_URL",
		description:
			"Optional OpenAI-compatible base URL for Python LLM agents; literature_survey defaults to https://api.minimaxi.com/v1.",
		secret: false,
		required: false,
	},
	{
		name: "LLM_MODEL",
		description: "Optional model name for Python LLM agents; literature_survey defaults to MiniMax-M3.",
		secret: false,
		required: false,
	},
	{
		name: "LLM_CONTEXT_WINDOW",
		description: "Optional context-window hint for Python LLM agents; literature_survey defaults to 512000.",
		secret: false,
		required: false,
	},
	{
		name: "WEB_SEARCH_MCP_URL",
		group: "scholarly",
		description: "Optional HTTP MCP endpoint for web search; defaults to the local web-search service.",
		secret: false,
		required: false,
		placeholder: "http://127.0.0.1:17890/mcp",
	},
	{
		name: "SEMANTIC_SCHOLAR_API_KEY",
		group: "scholarly",
		description: "Semantic Scholar API key required by scholarly collection/survey flows.",
		secret: true,
		required: true,
	},
	{
		name: "S2_API_KEY",
		group: "scholarly",
		description: "Optional Semantic Scholar/S2 alias used by code paths that accept S2 naming.",
		secret: true,
		required: false,
	},
];

function isSet(name: string): boolean {
	return Boolean(process.env[name]?.trim());
}

function unique(values: string[]): string[] {
	return [...new Set(values)];
}

const COMMON_SECRET_ALIASES: Record<string, string[]> = {
	OPENAI_API_KEY: ["LLM_API_KEY"],
	SEMANTIC_SCHOLAR_API_KEY: ["S2_API_KEY"],
};

export function getXlabRequiredSecretNames(skillPackage: XlabSkillPackage): string[] {
	return unique(skillPackage.manifest.runtime.requiredSecrets ?? []);
}

export function getXlabSecretAliases(skillPackage: XlabSkillPackage, name: string): string[] {
	return unique([
		...(COMMON_SECRET_ALIASES[name] ?? []),
		...(skillPackage.manifest.runtime.secretAliases?.[name] ?? []),
	]).filter((alias) => alias !== name);
}

export function getXlabSecretRequirementNames(skillPackage: XlabSkillPackage, name: string): string[] {
	return unique([name, ...getXlabSecretAliases(skillPackage, name)]);
}

export function isXlabSecretRequirementSatisfied(skillPackage: XlabSkillPackage, name: string): boolean {
	return getXlabSecretRequirementNames(skillPackage, name).some(isSet);
}

export function getMissingXlabRequiredSecretNames(skillPackage: XlabSkillPackage): string[] {
	return getXlabRequiredSecretNames(skillPackage).filter(
		(secret) => !isXlabSecretRequirementSatisfied(skillPackage, secret),
	);
}

export function collectXlabSecretUsage(packages: XlabSkillPackage[]): XlabSecretUsage[] {
	const usage = new Map<string, { requiredBy: Set<string>; optionalBy: Set<string> }>();
	const entryFor = (name: string): { requiredBy: Set<string>; optionalBy: Set<string> } => {
		const existing = usage.get(name);
		if (existing) {
			return existing;
		}
		const created = {
			requiredBy: new Set<string>(),
			optionalBy: new Set<string>(),
		};
		usage.set(name, created);
		return created;
	};
	for (const skillPackage of packages) {
		const skillName = `${skillPackage.manifest.name}@${skillPackage.manifest.version}`;
		for (const secret of skillPackage.manifest.runtime.requiredSecrets ?? []) {
			entryFor(secret).requiredBy.add(skillName);
		}
		for (const secret of skillPackage.manifest.runtime.optionalSecrets ?? []) {
			entryFor(secret).optionalBy.add(skillName);
		}
	}
	return [...usage.entries()]
		.map(([name, value]) => ({
			name,
			requiredBy: unique([...value.requiredBy]),
			optionalBy: unique([...value.optionalBy]),
		}))
		.sort((left, right) => left.name.localeCompare(right.name));
}

function definitionFor(name: string): XlabEnvironmentVariableDefinition | undefined {
	return XLAB_COMMON_ENVIRONMENT_VARIABLES.find((definition) => definition.name === name);
}

const KNOWLEDGE_GRAPH_LLM_VARIABLES = new Set([
	"KNOWLEDGE_GRAPH_LLM_API_URL",
	"KNOWLEDGE_GRAPH_LLM_API_KEY",
	"KNOWLEDGE_GRAPH_LLM_MODEL",
]);

function isKnowledgeGraphLlmVariable(name: string): boolean {
	return KNOWLEDGE_GRAPH_LLM_VARIABLES.has(name);
}

function placeholderFor(variable: XlabPromptVariable): string {
	if (variable.placeholder) {
		return variable.placeholder;
	}
	return variable.secret === false ? "Enter value" : "Paste the secret value";
}

function promptTitleFor(title: string, variable: XlabPromptVariable, index: number, total: number): string {
	const optional = variable.required === false;
	const label = variable.promptLabel ?? variable.name;
	const step = total > 1 ? `${index + 1}/${total} ` : "";
	const description = variable.promptDescription ?? variable.description;
	const lines = [title];
	if (isKnowledgeGraphLlmVariable(variable.name)) {
		lines.push(
			"Knowledge graph uses a separate LLM API for paper extraction.",
			"This is independent from the Pi Agent chat/coding model.",
		);
	}
	lines.push(`Enter ${step}${label} (${variable.name})${optional ? " (optional; leave blank to skip)" : ""}.`);
	if (description) {
		lines.push(description);
	}
	lines.push("Values entered here are kept only in this Claude Code process.");
	return lines.join("\n");
}

function requiredSecretsMessage(skillPackage: XlabSkillPackage, missingSecrets: string[]): string {
	const command = xlabProductCommandForSkill(skillPackage.manifest.name)?.task ?? skillPackage.manifest.name;
	if (missingSecrets.some(isKnowledgeGraphLlmVariable)) {
		return [
			`/xlab ${command} needs a separate LLM API for paper graph extraction.`,
			"This is independent from the Pi Agent chat/coding model.",
			"Set the required environment variables in your shell or secrets manager, then rerun the command:",
			...missingSecrets.map((secret) => `export ${secret}=...`),
			"Do not paste secret values into slash-command arguments.",
		].join("\n");
	}
	return `Set the required environment variable${missingSecrets.length === 1 ? "" : "s"} before running /xlab ${
		command
	}: ${missingSecrets.join(", ")}.`;
}

export function promptVariableForName(name: string, required = true): XlabPromptVariable {
	const definition = definitionFor(name);
	return {
		name,
		description: definition?.description,
		secret: definition?.secret ?? true,
		required,
		promptLabel: definition?.promptLabel,
		promptDescription: definition?.promptDescription,
		placeholder: definition?.placeholder,
	};
}

export function getXlabInitVariables(group: XlabInitGroup): XlabPromptVariable[] {
	return XLAB_COMMON_ENVIRONMENT_VARIABLES.filter((definition) => definition.group === group).map((definition) => ({
		name: definition.name,
		description: definition.description,
		secret: definition.secret ?? true,
		required: definition.required ?? false,
		promptLabel: definition.promptLabel,
		promptDescription: definition.promptDescription,
		placeholder: definition.placeholder,
	}));
}

export async function promptForXlabEnvironmentVariables(
	ctx: ExtensionContext,
	title: string,
	variables: XlabPromptVariable[],
	options?: { skipAlreadySet?: boolean; allowSkipOptional?: boolean },
): Promise<XlabPromptResult> {
	const targets = unique(variables.map((variable) => variable.name)).flatMap((name) => {
		const variable = variables.find((candidate) => candidate.name === name);
		if (!variable) {
			return [];
		}
		const requirementNames = [name, ...(variable.satisfiedBy ?? [])];
		return options?.skipAlreadySet !== false && requirementNames.some(isSet) ? [] : [variable];
	});
	if (targets.length === 0) {
		return { ok: true, initialized: [], skipped: [] };
	}
	if (!ctx.hasUI) {
		const targetNames = targets.map((variable) => variable.name);
		const message = targetNames.some(isKnowledgeGraphLlmVariable)
			? [
					"Set the separate /xlab build-knowledge-graph paper-extraction LLM API before running this command.",
					"This is independent from the Pi Agent chat/coding model.",
					...targetNames.map((name) => `export ${name}=...`),
				].join("\n")
			: `Set environment variables before running this command: ${targetNames.join(", ")}.`;
		return {
			ok: false,
			initialized: [],
			skipped: [],
			message,
		};
	}

	const initialized: string[] = [];
	const skipped: string[] = [];
	const previousValues = new Map(targets.map((variable) => [variable.name, process.env[variable.name]]));
	const rollback = (): void => {
		for (const name of initialized) {
			const previousValue = previousValues.get(name);
			if (previousValue === undefined) {
				delete process.env[name];
			} else {
				process.env[name] = previousValue;
			}
		}
	};

	for (const [index, variable] of targets.entries()) {
		while (true) {
			const optional = variable.required === false;
			const value = await ctx.ui.input(
				promptTitleFor(title, variable, index, targets.length),
				placeholderFor(variable),
				{ secret: variable.secret !== false },
			);
			if (value === undefined) {
				rollback();
				return {
					ok: false,
					cancelled: true,
					initialized: [],
					skipped,
					message: `XLab initialization cancelled while requesting ${variable.name}.`,
				};
			}
			const normalized = value.trim();
			if (!normalized) {
				if (optional && options?.allowSkipOptional !== false) {
					skipped.push(variable.name);
					break;
				}
				ctx.ui.notify(`${variable.name} cannot be empty. Enter a value or cancel the dialog.`, "warning");
				continue;
			}
			try {
				process.env[variable.name] = normalized;
			} catch (error) {
				rollback();
				return {
					ok: false,
					initialized: [],
					skipped,
					message: `Cannot initialize ${variable.name}: ${error instanceof Error ? error.message : String(error)}`,
				};
			}
			initialized.push(variable.name);
			break;
		}
	}
	return { ok: true, initialized, skipped };
}

export async function initializeRequiredXlabSecrets(
	ctx: ExtensionContext,
	skillPackage: XlabSkillPackage,
): Promise<{ ok: boolean; cancelled?: boolean; message?: string }> {
	const missingSecrets = getMissingXlabRequiredSecretNames(skillPackage);
	if (missingSecrets.length === 0) {
		return { ok: true };
	}
	if (!ctx.hasUI) {
		return {
			ok: false,
			message: requiredSecretsMessage(skillPackage, missingSecrets),
		};
	}

	const result = await promptForXlabEnvironmentVariables(
		ctx,
		`Initialize ${skillPackage.manifest.name}`,
		missingSecrets.map((secret) => ({
			...promptVariableForName(secret, true),
			satisfiedBy: getXlabSecretAliases(skillPackage, secret),
		})),
		{ skipAlreadySet: false, allowSkipOptional: false },
	);
	if (!result.ok) {
		return { ok: false, cancelled: result.cancelled, message: result.message };
	}
	if (result.initialized.length > 0) {
		ctx.ui.notify(
			`Initialized ${result.initialized.length} required environment variable${
				result.initialized.length === 1 ? "" : "s"
			} for this process.`,
		);
	}
	return { ok: true };
}

export function formatXlabSecretStatus(
	packages: XlabSkillPackage[],
	modelRegistry?: ExtensionContext["modelRegistry"],
	currentModel?: ExtensionContext["model"],
): string {
	const usage = collectXlabSecretUsage(packages);
	const usageByName = new Map(usage.map((entry) => [entry.name, entry]));
	const names = unique([
		...XLAB_COMMON_ENVIRONMENT_VARIABLES.map((definition) => definition.name),
		...usage.map((entry) => entry.name),
	]);
	const lines = names.map((name) => {
		const definition = definitionFor(name);
		const entry = usageByName.get(name);
		const requiredBy = entry?.requiredBy.length ? ` required by ${entry.requiredBy.join(", ")}` : "";
		const optionalBy = entry?.optionalBy.length ? ` optional for ${entry.optionalBy.join(", ")}` : "";
		const description = definition?.description ? ` — ${definition.description}` : "";
		const requiredEntry = usage.find((usageEntry) => usageEntry.requiredBy.length > 0 && usageEntry.name === name);
		const requiredPackages = requiredEntry
			? packages.filter((skillPackage) =>
					requiredEntry.requiredBy.includes(`${skillPackage.manifest.name}@${skillPackage.manifest.version}`),
				)
			: [];
		const set =
			requiredPackages.length > 0
				? requiredPackages.some((skillPackage) => isXlabSecretRequirementSatisfied(skillPackage, name))
				: isSet(name);
		const aliasText = requiredPackages
			.flatMap((skillPackage) => getXlabSecretAliases(skillPackage, name))
			.filter((alias) => isSet(alias));
		const status = aliasText.length > 0 ? `set via ${unique(aliasText).join("/")}` : set ? "set" : "missing";
		return `- ${name}: ${status}${requiredBy}${optionalBy}${description}`;
	});
	const availableModels = modelRegistry?.getAvailable() ?? [];
	const currentRuntimeModel = currentModel
		? availableModels.find(
				(candidate) =>
					candidate.provider === currentModel.provider &&
					candidate.id === currentModel.id &&
					candidate.api === "openai-completions",
			)
		: undefined;
	const namedXlabRuntimeModel = availableModels.find(
		(candidate) => candidate.api === "openai-completions" && candidate.provider.toLowerCase().includes("xlab"),
	);
	const model = currentRuntimeModel ?? namedXlabRuntimeModel;
	const runtimeStatus = model
		? `- Pi Agent runtime model: configured (${model.provider}/${model.id})`
		: "- Pi Agent runtime model: not configured by /xlab configure-xlab runtime";
	return [
		"XLab setup status:",
		runtimeStatus,
		"",
		"Integration environment values:",
		...lines,
		"",
		"/xlab configure-xlab runtime saves an OpenAI-compatible agent runtime model and applies it immediately.",
		"Integration values entered with /xlab configure-xlab knowledge-graph or scholarly-services are kept only in this process.",
		"The KNOWLEDGE_GRAPH_LLM_* values configure a separate paper-extraction model for /xlab build-knowledge-graph; they are independent from the agent runtime model.",
		"Use /xlab configure-xlab <runtime|knowledge-graph|scholarly-services|all> to initialize missing values with masked prompts.",
	].join("\n");
}
