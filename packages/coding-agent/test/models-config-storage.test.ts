import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { upsertOpenAICompatibleModelConfig } from "../src/core/models-config-storage.ts";

describe("models config storage", () => {
	let tempDir: string;
	let modelsPath: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-test-models-config-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		modelsPath = join(tempDir, "nested", "models.json");
	});

	afterEach(() => {
		if (existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});

	test("creates a missing models.json with an OpenAI-compatible provider", () => {
		const writtenPath = upsertOpenAICompatibleModelConfig(
			{
				providerId: "xlab-openai-compatible",
				providerName: "XLab OpenAI-compatible",
				baseUrl: "https://api.example.com/v1",
				modelId: "demo-model",
			},
			{ modelsPath },
		);

		expect(writtenPath).toBe(modelsPath);
		const config = JSON.parse(readFileSync(modelsPath, "utf-8"));
		expect(config).toEqual({
			providers: {
				"xlab-openai-compatible": {
					name: "XLab OpenAI-compatible",
					baseUrl: "https://api.example.com/v1",
					api: "openai-completions",
					models: [{ id: "demo-model", name: "demo-model" }],
				},
			},
		});
	});

	test("preserves unrelated providers and upserts the selected model", () => {
		mkdirSync(join(tempDir, "nested"), { recursive: true });
		writeFileSync(
			modelsPath,
			JSON.stringify({
				providers: {
					other: {
						name: "Other",
						baseUrl: "https://other.example/v1",
						api: "openai-completions",
						models: [{ id: "other-model" }],
					},
					"xlab-openai-compatible": {
						name: "Old",
						baseUrl: "https://old.example/v1",
						api: "openai-completions",
						models: [{ id: "old-model" }],
					},
				},
			}),
		);

		upsertOpenAICompatibleModelConfig(
			{
				providerId: "xlab-openai-compatible",
				providerName: "New",
				baseUrl: "https://new.example/v1",
				modelId: "new-model",
			},
			{ modelsPath },
		);

		const config = JSON.parse(readFileSync(modelsPath, "utf-8"));
		expect(config.providers.other.models).toEqual([{ id: "other-model" }]);
		expect(config.providers["xlab-openai-compatible"]).toMatchObject({
			name: "New",
			baseUrl: "https://new.example/v1",
			api: "openai-completions",
			models: [{ id: "old-model" }, { id: "new-model", name: "new-model" }],
		});
	});

	test("refuses malformed JSON without overwriting it", () => {
		mkdirSync(join(tempDir, "nested"), { recursive: true });
		writeFileSync(modelsPath, "{ not json", "utf-8");

		expect(() =>
			upsertOpenAICompatibleModelConfig(
				{
					providerId: "xlab-openai-compatible",
					providerName: "XLab",
					baseUrl: "https://api.example.com/v1",
					modelId: "demo-model",
				},
				{ modelsPath },
			),
		).toThrow("Failed to parse models.json");
		expect(readFileSync(modelsPath, "utf-8")).toBe("{ not json");
	});
});
