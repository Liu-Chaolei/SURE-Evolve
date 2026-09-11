import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { discoverXlabSkillPackages } from "../../src/core/xlab/manifest.ts";

describe("native XLab runtime manifest", () => {
	const cleanups: string[] = [];
	afterEach(() => {
		for (const root of cleanups.splice(0)) rmSync(root, { recursive: true, force: true });
	});

	it("parses a native runtime driver", () => {
		const root = mkdtempSync(join(tmpdir(), "xlab-native-manifest-"));
		cleanups.push(root);
		const packageDir = join(root, ".xlab", "skills", "native_experiment");
		mkdirSync(packageDir, { recursive: true });
		writeFileSync(join(packageDir, "SKILL.md"), "Run natively.\n");
		writeFileSync(
			join(packageDir, "xlab.skill.json"),
			JSON.stringify({
				schema_version: "2",
				name: "native_experiment",
				version: "1.0.0",
				visibility: "public",
				prompt: "SKILL.md",
				runtime: { kind: "native", driver: "experiment" },
			}),
		);
		const result = discoverXlabSkillPackages(root);
		expect(result.diagnostics).toEqual([]);
		expect(result.packages[0]?.manifest.runtime).toEqual({
			kind: "native",
			driver: "experiment",
			entrypoint: undefined,
			lockfile: undefined,
			containerfile: undefined,
			executables: undefined,
			requiredSecrets: undefined,
			optionalSecrets: undefined,
			secretAliases: undefined,
		});
	});
});
