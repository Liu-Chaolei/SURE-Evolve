import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { XlabArtifactStore } from "../src/core/xlab/artifact-store.ts";
import { formatXlabDashboard } from "../src/core/xlab/dashboard.ts";
import { prepareXlabEnvironment } from "../src/core/xlab/environment.ts";
import { validateXlabPackageSelfContained } from "../src/core/xlab/package-validator.ts";
import { XlabRunManager } from "../src/core/xlab/run-manager.ts";
import { collectXlabSecretUsage, formatXlabSecretStatus } from "../src/core/xlab/secrets.ts";
import type {
	XlabManifestEnvelope,
	XlabRunRecord,
	XlabSkillManifest,
	XlabSkillPackage,
} from "../src/core/xlab/types.ts";
import { createWorkflowState, transitionWorkflowStage, validateWorkflowStages } from "../src/core/xlab/workflow.ts";

function createRun(cwd: string, runId: string): XlabRunRecord {
	return {
		runId,
		skillName: "test_skill",
		skillVersion: "1.0.0",
		command: "test_skill",
		status: "running",
		cwd,
		packageDir: cwd,
		runDir: join(cwd, ".xlab", "runs", runId),
		args: "",
		startedAt: new Date().toISOString(),
		updatedAt: new Date().toISOString(),
	};
}

function createManifest(runId: string, artifacts: XlabManifestEnvelope["artifacts"]): XlabManifestEnvelope {
	return {
		schema_version: "2",
		run_id: runId,
		skill_name: "test_skill",
		skill_version: "1.0.0",
		status: "success",
		created_at: new Date().toISOString(),
		inputs: {},
		outputs: {},
		validation: { passed: true },
		artifacts,
	};
}

describe("XLab workflow and artifact runtime", () => {
	const temporaryDirectories: string[] = [];

	afterEach(() => {
		for (const directory of temporaryDirectories.splice(0)) {
			rmSync(directory, { recursive: true, force: true });
		}
	});

	it("enforces workflow prerequisites, terminal stages, and attempt limits", () => {
		const stages = [
			{ id: "collect", skill: "collect", maxAttempts: 1 },
			{ id: "review", skill: "review", needs: ["collect"], maxAttempts: 2 },
		];
		expect(validateWorkflowStages(stages)).toBeUndefined();
		const manifest = {
			schemaVersion: "2",
			name: "workflow",
			version: "1.0.0",
			visibility: "public",
			prompt: "SKILL.md",
			runtime: { kind: "workflow" },
			workflow: { stages },
		} satisfies XlabSkillManifest;
		const initial = createWorkflowState("run-1", manifest);
		expect(initial).toBeDefined();
		if (!initial) {
			throw new Error("workflow state was not created");
		}
		expect(transitionWorkflowStage(initial, "review", "start").ok).toBe(false);
		const collecting = transitionWorkflowStage(initial, "collect", "start");
		expect(collecting.ok).toBe(true);
		const collected = transitionWorkflowStage(collecting.state, "collect", "complete");
		expect(collected.ok).toBe(true);
		expect(transitionWorkflowStage(collected.state, "collect", "start").ok).toBe(false);
		const reviewing = transitionWorkflowStage(collected.state, "review", "start");
		expect(reviewing.ok).toBe(true);
		expect(reviewing.stage?.attempts).toBe(1);
		const blocked = transitionWorkflowStage(reviewing.state, "review", "block", { checkpoint: { page: 3 } });
		expect(blocked.ok).toBe(true);
		const retry = transitionWorkflowStage(blocked.state, "review", "start");
		expect(retry.ok).toBe(true);
		expect(retry.stage?.attempts).toBe(2);
		const failed = transitionWorkflowStage(retry.state, "review", "fail");
		expect(transitionWorkflowStage(failed.state, "review", "start").ok).toBe(false);
	});

	it("rejects cyclic workflow declarations", () => {
		expect(
			validateWorkflowStages([
				{ id: "left", skill: "left", needs: ["right"] },
				{ id: "right", skill: "right", needs: ["left"] },
			]),
		).toContain("acyclic");
	});

	it("accepts exactly pinned Python requirements with extras", () => {
		const packageDirectory = mkdtempSync(join(tmpdir(), "xlab-python-lock-"));
		temporaryDirectories.push(packageDirectory);
		writeFileSync(
			join(packageDirectory, "requirements.lock"),
			"mineru[all]==3.4.4\nmineru[core,vllm]==3.4.4\n",
			"utf-8",
		);
		expect(validateXlabPackageSelfContained(packageDirectory)).toEqual([]);
		writeFileSync(join(packageDirectory, "requirements.lock"), "mineru[all]>=3.4.4\n", "utf-8");
		expect(validateXlabPackageSelfContained(packageDirectory)).toEqual([
			expect.objectContaining({ message: expect.stringContaining("Unlocked Python requirement") }),
		]);
	});

	it("aggregates secret usage and formats masked setup status", () => {
		const packages: XlabSkillPackage[] = [
			{
				manifest: {
					schemaVersion: "2",
					name: "collect",
					version: "1.0.0",
					visibility: "public",
					prompt: "SKILL.md",
					runtime: { kind: "agent", requiredSecrets: ["TAVILY_API_KEY"], optionalSecrets: ["S2_API_KEY"] },
				},
				manifestPath: join("skill", "xlab.skill.json"),
				packageDir: "skill",
				promptPath: join("skill", "SKILL.md"),
				prompt: "Collect.",
				source: "repository",
				sourceRoot: ".",
				dependencies: [],
			},
			{
				manifest: {
					schemaVersion: "2",
					name: "survey",
					version: "1.0.0",
					visibility: "public",
					prompt: "SKILL.md",
					runtime: { kind: "agent", requiredSecrets: ["OPENAI_API_KEY", "TAVILY_API_KEY"] },
				},
				manifestPath: join("survey", "xlab.skill.json"),
				packageDir: "survey",
				promptPath: join("survey", "SKILL.md"),
				prompt: "Survey.",
				source: "repository",
				sourceRoot: ".",
				dependencies: [],
			},
		];
		const previousTavily = process.env.TAVILY_API_KEY;
		const previousOpenai = process.env.OPENAI_API_KEY;
		const previousS2 = process.env.S2_API_KEY;
		process.env.TAVILY_API_KEY = "secret-value";
		delete process.env.OPENAI_API_KEY;
		delete process.env.S2_API_KEY;
		try {
			expect(collectXlabSecretUsage(packages)).toContainEqual({
				name: "TAVILY_API_KEY",
				requiredBy: ["collect@1.0.0", "survey@1.0.0"],
				optionalBy: [],
			});
			const status = formatXlabSecretStatus(packages);
			expect(status).toContain("TAVILY_API_KEY: set required by collect@1.0.0, survey@1.0.0");
			expect(status).toContain("OPENAI_API_KEY: missing required by survey@1.0.0");
			expect(status).not.toContain("secret-value");
		} finally {
			if (previousTavily === undefined) {
				delete process.env.TAVILY_API_KEY;
			} else {
				process.env.TAVILY_API_KEY = previousTavily;
			}
			if (previousOpenai === undefined) {
				delete process.env.OPENAI_API_KEY;
			} else {
				process.env.OPENAI_API_KEY = previousOpenai;
			}
			if (previousS2 === undefined) {
				delete process.env.S2_API_KEY;
			} else {
				process.env.S2_API_KEY = previousS2;
			}
		}
	});

	it("treats declared secret aliases as set while keeping values masked", () => {
		const skillPackage: XlabSkillPackage = {
			manifest: {
				schemaVersion: "2",
				name: "survey",
				version: "1.0.0",
				visibility: "public",
				prompt: "SKILL.md",
				runtime: {
					kind: "agent",
					requiredSecrets: ["OPENAI_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"],
					secretAliases: {
						OPENAI_API_KEY: ["LLM_API_KEY"],
						SEMANTIC_SCHOLAR_API_KEY: ["S2_API_KEY"],
					},
				},
				config: {
					environment: [{ name: "LLM_MODEL", required: false, default: "test-model" }],
				},
			},
			manifestPath: join("survey", "xlab.skill.json"),
			packageDir: "survey",
			promptPath: join("survey", "SKILL.md"),
			prompt: "Survey.",
			source: "repository",
			sourceRoot: ".",
			dependencies: [],
		};
		const previousOpenai = process.env.OPENAI_API_KEY;
		const previousLlm = process.env.LLM_API_KEY;
		const previousSemantic = process.env.SEMANTIC_SCHOLAR_API_KEY;
		const previousS2 = process.env.S2_API_KEY;
		delete process.env.OPENAI_API_KEY;
		delete process.env.SEMANTIC_SCHOLAR_API_KEY;
		process.env.LLM_API_KEY = "llm-alias-secret";
		process.env.S2_API_KEY = "s2-alias-secret";
		try {
			const status = formatXlabSecretStatus([skillPackage]);
			expect(status).toContain("OPENAI_API_KEY: set via LLM_API_KEY required by survey@1.0.0");
			expect(status).toContain("SEMANTIC_SCHOLAR_API_KEY: set via S2_API_KEY required by survey@1.0.0");
			expect(status).not.toContain("llm-alias-secret");
			expect(status).not.toContain("s2-alias-secret");
			expect(collectXlabSecretUsage([skillPackage]).map((usage) => usage.name)).not.toContain("LLM_MODEL");
		} finally {
			for (const [name, value] of [
				["OPENAI_API_KEY", previousOpenai],
				["LLM_API_KEY", previousLlm],
				["SEMANTIC_SCHOLAR_API_KEY", previousSemantic],
				["S2_API_KEY", previousS2],
			] as const) {
				if (value === undefined) {
					delete process.env[name];
				} else {
					process.env[name] = value;
				}
			}
		}
	});

	it("formats a compact XLab dashboard from run, workflow, and display state", () => {
		const run = createRun("/tmp/xlab-dashboard", "run-dashboard");
		const previousSecret = process.env.XLAB_TEST_API_KEY;
		process.env.XLAB_TEST_API_KEY = "dashboard-secret-value";
		try {
			const lines = formatXlabDashboard({
				run,
				workflow: {
					schemaVersion: "1",
					runId: run.runId,
					updatedAt: "2026-01-01T00:00:00.000Z",
					stages: [
						{
							id: "collect",
							skill: "collect",
							needs: [],
							optional: false,
							maxAttempts: 1,
							attempts: 1,
							status: "success",
						},
						{
							id: "validate",
							skill: "validate",
							needs: ["collect"],
							optional: false,
							maxAttempts: 1,
							attempts: 1,
							status: "running",
						},
					],
				},
				state: {
					phase: { id: "validate", label: "Validating", status: "running", progress: 0.5 },
					message: "Checking dashboard-secret-value outputs.",
					counters: { target: 10, collected: 7, rejected: 1 },
					diagnostics: [
						{ severity: "warning", message: "Need more evidence.", repair: "Expand citation search." },
					],
					artifacts: [
						{
							type: "paper_set",
							name: "papers.manifest.json",
							path: "artifacts/papers.manifest.json",
							status: "ready",
						},
					],
					checkpoint: {
						id: "page-2",
						label: "Page 2",
						resumable: true,
						resume_hint: "Use /xlab resume-run run-dashboard.",
					},
					next_actions: ["Collect more papers", "Review blockers"],
				},
				skillPackage: {
					manifest: {
						schemaVersion: "2",
						name: "survey",
						version: "1.0.0",
						visibility: "public",
						prompt: "SKILL.md",
						runtime: { kind: "agent" },
						ui: { primaryCounters: ["collected", "target"], artifactTypes: ["paper_set"] },
					},
					manifestPath: "survey/xlab.skill.json",
					packageDir: "survey",
					promptPath: "survey/SKILL.md",
					prompt: "Survey.",
					source: "repository",
					sourceRoot: ".",
					dependencies: [],
				},
			});

			expect(lines.length).toBeLessThanOrEqual(10);
			expect(lines.join("\n")).toContain("XLab /test_skill");
			expect(lines.join("\n")).toContain("Workflow: collect ✓  validate ▶");
			expect(lines.join("\n")).toContain("Counters: collected=7  target=10");
			expect(lines.join("\n")).toContain("Blocker: warning: Need more evidence. Repair: Expand citation search.");
			expect(lines.join("\n")).toContain("Artifacts: ready=1");
			expect(lines.join("\n")).toContain("Checkpoint: Page 2");
			expect(lines.join("\n")).not.toContain("dashboard-secret-value");
		} finally {
			if (previousSecret === undefined) {
				delete process.env.XLAB_TEST_API_KEY;
			} else {
				process.env.XLAB_TEST_API_KEY = previousSecret;
			}
		}
	});

	it("uses the declared Python directly for empty lockfiles", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-empty-lock-env-"));
		temporaryDirectories.push(cwd);
		const packageDirectory = join(cwd, "skill");
		const binDirectory = join(cwd, "bin");
		mkdirSync(packageDirectory, { recursive: true });
		mkdirSync(binDirectory, { recursive: true });
		const python = join(binDirectory, "python");
		writeFileSync(python, "#!/bin/sh\necho Python 3.13.0\n", "utf-8");
		chmodSync(python, 0o755);
		writeFileSync(join(packageDirectory, "requirements.lock"), "# Standard-library runtime.\n", "utf-8");
		const skillPackage: XlabSkillPackage = {
			manifest: {
				schemaVersion: "2",
				name: "empty_lock_skill",
				version: "1.0.0",
				visibility: "public",
				prompt: "SKILL.md",
				runtime: { kind: "python", lockfile: "requirements.lock", executables: [python] },
			},
			manifestPath: join(packageDirectory, "xlab.skill.json"),
			packageDir: packageDirectory,
			promptPath: join(packageDirectory, "SKILL.md"),
			prompt: "Run the empty lockfile skill.",
			source: "project",
			sourceRoot: cwd,
			dependencies: [],
		};

		const environment = prepareXlabEnvironment(cwd, skillPackage);

		expect(environment).toMatchObject({ kind: "python-system", executable: python, lockfile: "requirements.lock" });
		expect(environment?.path).toContain(join(".xlab", "environments", "empty_lock_skill"));
		expect(existsSync(join(environment?.path ?? "", "bin", "python"))).toBe(false);
		expect(JSON.parse(readFileSync(join(environment?.path ?? "", "environment.json"), "utf-8"))).toMatchObject({
			kind: "python-system",
			executable: python,
		});
	});

	it("resolves canonical run-relative and legacy project-relative paths without allowing traversal", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-run-paths-"));
		temporaryDirectories.push(cwd);
		const run = createRun(cwd, "run-paths");
		const manager = new XlabRunManager(cwd);

		expect(manager.resolveRunPath(run, "artifacts/result.json")).toBe(join(run.runDir, "artifacts", "result.json"));
		expect(manager.resolveRunPath(run, ".xlab/runs/run-paths/artifacts/result.json")).toBe(
			join(run.runDir, "artifacts", "result.json"),
		);
		expect(manager.resolveRunPath(run, "../../outside.json")).toBeUndefined();
	});

	it("deduplicates identical payloads while separating file and directory hashes", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-artifacts-"));
		temporaryDirectories.push(cwd);
		const payload = join(cwd, "payload.txt");
		const emptyDirectory = join(cwd, "empty");
		writeFileSync(payload, "", "utf-8");
		mkdirSync(emptyDirectory);
		const store = new XlabArtifactStore(cwd);
		const runOne = createRun(cwd, "run-1");
		const runTwo = createRun(cwd, "run-2");
		const runThree = createRun(cwd, "run-3");
		const fileManifest = createManifest(runOne.runId, [{ type: "fixture", schema_version: "1", path: payload }]);
		const directoryManifest = createManifest(runThree.runId, [
			{ type: "fixture_directory", schema_version: "1", path: emptyDirectory },
		]);
		const [first] = store.commitManifestArtifacts(runOne, fileManifest, (path) => path);
		const secondFileManifest = createManifest(runTwo.runId, [
			{ type: "fixture", schema_version: "1", path: payload, parents: [first.artifactId] },
		]);
		const [second] = store.commitManifestArtifacts(runTwo, secondFileManifest, (path) => path);
		const [directory] = store.commitManifestArtifacts(runThree, directoryManifest, (path) => path);
		expect(second.artifactId).toBe(first.artifactId);
		expect(directory.artifactId).not.toBe(first.artifactId);
		const reopened = new XlabArtifactStore(cwd).get(first.artifactId);
		expect(reopened?.digest).toBe(first.digest);
		expect(reopened?.payloadPath).toBe(first.payloadPath);
		expect(reopened?.metadataPath).toBe(first.metadataPath);
		expect(existsSync(reopened?.payloadPath ?? "")).toBe(true);
		expect(readFileSync(reopened?.payloadPath ?? "", "utf-8")).toBe("");
	});
});
