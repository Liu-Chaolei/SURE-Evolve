import { cpSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { discoverXlabSkillPackages } from "../../src/core/xlab/manifest.ts";
import { validateXlabPackageSelfContained } from "../../src/core/xlab/package-validator.ts";

const REPOSITORY_ROOT = fileURLToPath(new URL("../../../..", import.meta.url));

function setupRunExperimentProject(tempDir: string) {
	const skillsDir = join(tempDir, "xlab", "skills");
	cpSync(join(REPOSITORY_ROOT, "xlab", "skills", "run_experiment"), join(skillsDir, "run_experiment"), {
		recursive: true,
	});
	const discovered = discoverXlabSkillPackages(tempDir);
	expect(discovered.diagnostics.filter((diagnostic) => diagnostic.type === "error")).toEqual([]);
	const skillPackage = discovered.packages.find((entry) => entry.manifest.name === "run_experiment");
	expect(skillPackage).toBeDefined();
	return skillPackage!;
}

describe("run_experiment XLab skill", () => {
	const temporaryDirectories: string[] = [];

	afterEach(() => {
		for (const directory of temporaryDirectories.splice(0)) {
			rmSync(directory, { recursive: true, force: true });
		}
	});

	it("declares the native durable experiment runtime", () => {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-run-experiment-"));
		temporaryDirectories.push(tempDir);
		const skillPackage = setupRunExperimentProject(tempDir);

		expect(skillPackage.manifest).toMatchObject({
			schemaVersion: "2",
			name: "run_experiment",
			version: "2.0.0",
			visibility: "public",
			runtime: { kind: "native", driver: "experiment" },
		});
		expect(skillPackage.manifest.runtime.entrypoint).toBeUndefined();
		expect(skillPackage.manifest.runtime.lockfile).toBeUndefined();
		expect(skillPackage.manifest.hooks ?? []).toEqual([]);
		expect(skillPackage.manifest.workflow?.stages.map((stage) => stage.id)).toEqual([
			"prepare",
			"code",
			"science",
			"finalize",
		]);
		expect(
			skillPackage.manifest.artifacts?.map((artifact) => [artifact.type, artifact.schemaVersion, artifact.required]),
		).toEqual([
			["ablation_results", "1", true],
			["final_audit", "1", true],
			["symbolic_memory_receipt", "1", true],
		]);
		expect(validateXlabPackageSelfContained(skillPackage.packageDir)).toEqual([]);
	});

	it("exposes only the canonical Idea selector", () => {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-run-experiment-args-"));
		temporaryDirectories.push(tempDir);
		const skillPackage = setupRunExperimentProject(tempDir);

		expect(skillPackage.manifest.ui?.arguments?.map((argument) => argument.name)).toEqual(["--idea"]);
		expect(skillPackage.manifest.ui?.argumentHint).toBe("--idea <idea reference>");
	});
});
