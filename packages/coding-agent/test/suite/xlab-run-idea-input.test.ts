import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { XlabIdeaBinding } from "../../src/core/xlab/experiment/idea-binding.ts";
import { XlabRunManager } from "../../src/core/xlab/run-manager.ts";
import type { XlabSkillPackage } from "../../src/core/xlab/types.ts";

function skillPackage(cwd: string): XlabSkillPackage {
	return {
		manifest: {
			schemaVersion: "2",
			name: "run_experiment",
			version: "1.0.0",
			visibility: "public",
			prompt: "SKILL.md",
			runtime: { kind: "workflow" },
			workflow: { stages: [{ id: "prepare", skill: "prepare" }] },
		},
		manifestPath: join(cwd, "xlab.skill.json"),
		packageDir: cwd,
		promptPath: join(cwd, "SKILL.md"),
		prompt: "Run experiment.",
		source: "project",
		sourceRoot: cwd,
		dependencies: [],
	};
}

function ideaBinding(): XlabIdeaBinding {
	const canonical = {
		schema_version: "xlab.experiment_idea.v1",
		status: "success",
		blockers: [],
		components: [{ component: "Verifier", explanation: "Check evidence.", index: "1" }],
	};
	return {
		canonical,
		canonicalBytes: Buffer.from(`${JSON.stringify(canonical, null, 2)}\n`, "utf-8"),
		metadata: {
			schema_version: "xlab.idea_binding.v1",
			resolution_method: "path",
			reference: "ideas/source.json",
			source_schema_version: "xlab.research_idea.v2",
			source_digest: "a".repeat(64),
			canonical_digest: "b".repeat(64),
		},
		warnings: [],
	};
}

describe("XLab run Idea inputs", () => {
	const cleanups: Array<() => void> = [];

	afterEach(() => {
		while (cleanups.length > 0) {
			cleanups.pop()?.();
		}
	});

	it("publishes immutable Idea bytes and binding metadata with the run", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-run-idea-input-"));
		cleanups.push(() => rmSync(cwd, { recursive: true, force: true }));
		const binding = ideaBinding();
		const manager = new XlabRunManager(cwd);

		const run = manager.createRun(skillPackage(cwd), "--idea ideas/source.json --seed 7", { ideaBinding: binding });

		expect(readFileSync(join(run.runDir, "inputs", "idea.json"))).toEqual(binding.canonicalBytes);
		expect(JSON.parse(readFileSync(join(run.runDir, "inputs", "idea-binding.json"), "utf-8"))).toEqual(
			binding.metadata,
		);
		expect(run.args).toBe(`--idea .xlab/runs/${run.runId}/inputs/idea.json --seed 7`);
		expect(manager.readRun(run.runId)?.args).toBe(run.args);
		expect(existsSync(join(run.runDir, "workflow.json"))).toBe(true);
		expect(readdirSync(manager.runsRoot).some((name) => name.startsWith(".staging-"))).toBe(false);
	});

	it("removes staging state when an Idea input cannot be written", () => {
		const cwd = mkdtempSync(join(tmpdir(), "xlab-run-idea-failure-"));
		cleanups.push(() => rmSync(cwd, { recursive: true, force: true }));
		const binding = ideaBinding();
		(binding as { canonicalBytes: unknown }).canonicalBytes = {};
		const manager = new XlabRunManager(cwd);

		expect(() => manager.createRun(skillPackage(cwd), "--idea source.json", { ideaBinding: binding })).toThrow();
		expect(manager.listRuns()).toEqual([]);
		expect(readdirSync(manager.runsRoot).some((name) => name.startsWith(".staging-"))).toBe(false);
	});
});
