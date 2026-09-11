import { randomUUID } from "node:crypto";
import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { getModelsPath } from "../config.ts";
import { stripJsonComments } from "../utils/json.ts";
import { normalizePath } from "../utils/paths.ts";

export const OPENAI_COMPATIBLE_API = "openai-completions";

export interface OpenAICompatibleModelConfig {
	providerId: string;
	providerName: string;
	baseUrl: string;
	modelId: string;
	modelName?: string;
	contextWindow?: number;
	maxTokens?: number;
}

export interface UpsertModelsConfigOptions {
	modelsPath?: string;
}

type JsonObject = Record<string, unknown>;

const MODELS_FILE_WRITE_OPTIONS = { encoding: "utf-8", mode: 0o600 } as const;

function isJsonObject(value: unknown): value is JsonObject {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readModelsConfig(modelsPath: string): JsonObject {
	if (!existsSync(modelsPath)) {
		return { providers: {} };
	}

	const content = readFileSync(modelsPath, "utf-8");
	if (!content.trim()) {
		return { providers: {} };
	}

	let parsed: unknown;
	try {
		parsed = JSON.parse(stripJsonComments(content));
	} catch (error) {
		throw new Error(`Failed to parse models.json: ${error instanceof Error ? error.message : String(error)}`);
	}

	if (!isJsonObject(parsed)) {
		throw new Error("Invalid models.json: root value must be an object.");
	}

	const providers = parsed.providers;
	if (providers !== undefined && !isJsonObject(providers)) {
		throw new Error("Invalid models.json: providers must be an object.");
	}

	return { ...parsed, providers: providers ? { ...providers } : {} };
}

function getExistingModels(providerId: string, providerConfig: JsonObject): JsonObject[] {
	const models = providerConfig.models;
	if (models === undefined) {
		return [];
	}
	if (!Array.isArray(models)) {
		throw new Error(`Invalid models.json: providers.${providerId}.models must be an array.`);
	}
	return models.map((model, index) => {
		if (!isJsonObject(model)) {
			throw new Error(`Invalid models.json: providers.${providerId}.models[${index}] must be an object.`);
		}
		if (typeof model.id !== "string" || model.id.trim() === "") {
			throw new Error(
				`Invalid models.json: providers.${providerId}.models[${index}].id must be a non-empty string.`,
			);
		}
		return { ...model };
	});
}

function assertPositiveInteger(name: string, value: number | undefined): void {
	if (value !== undefined && (!Number.isInteger(value) || value <= 0)) {
		throw new Error(`${name} must be a positive integer.`);
	}
}

function writeRestrictiveJson(path: string, value: unknown): void {
	mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
	const tmpPath = `${path}.${process.pid}.${randomUUID()}.tmp`;
	try {
		writeFileSync(tmpPath, `${JSON.stringify(value, null, 2)}\n`, MODELS_FILE_WRITE_OPTIONS);
		chmodSync(tmpPath, 0o600);
		renameSync(tmpPath, path);
		chmodSync(path, 0o600);
	} catch (error) {
		rmSync(tmpPath, { force: true });
		throw error;
	}
}

export function upsertOpenAICompatibleModelConfig(
	config: OpenAICompatibleModelConfig,
	options: UpsertModelsConfigOptions = {},
): string {
	const providerId = config.providerId.trim();
	const providerName = config.providerName.trim();
	const baseUrl = config.baseUrl.trim();
	const modelId = config.modelId.trim();
	if (!providerId) {
		throw new Error("Provider id cannot be empty.");
	}
	if (!providerName) {
		throw new Error("Provider name cannot be empty.");
	}
	if (!baseUrl) {
		throw new Error("Base URL cannot be empty.");
	}
	if (!modelId) {
		throw new Error("Model id cannot be empty.");
	}
	assertPositiveInteger("contextWindow", config.contextWindow);
	assertPositiveInteger("maxTokens", config.maxTokens);

	const modelsPath = normalizePath(options.modelsPath ?? getModelsPath());
	const root = readModelsConfig(modelsPath);
	const providers = root.providers as JsonObject;
	const existingProvider = providers[providerId];
	if (existingProvider !== undefined && !isJsonObject(existingProvider)) {
		throw new Error(`Invalid models.json: providers.${providerId} must be an object.`);
	}

	const providerConfig: JsonObject = existingProvider ? { ...existingProvider } : {};
	const nextModel: JsonObject = {
		id: modelId,
		name: config.modelName?.trim() || modelId,
	};
	if (config.contextWindow !== undefined) {
		nextModel.contextWindow = config.contextWindow;
	}
	if (config.maxTokens !== undefined) {
		nextModel.maxTokens = config.maxTokens;
	}

	const existingModels = getExistingModels(providerId, providerConfig).filter((model) => model.id !== modelId);
	providers[providerId] = {
		...providerConfig,
		name: providerName,
		baseUrl,
		api: OPENAI_COMPATIBLE_API,
		models: [...existingModels, nextModel],
	};

	writeRestrictiveJson(modelsPath, { ...root, providers });
	return modelsPath;
}
