import { existsSync, lstatSync, readdirSync, readFileSync, readlinkSync, statSync } from "node:fs";
import { delimiter, join, relative, resolve } from "node:path";
import { isXlabSecretRequirementSatisfied } from "./secrets.ts";
import type { XlabSkillPackage } from "./types.ts";

const TEXT_EXTENSIONS = new Set([
	".js",
	".json",
	".lock",
	".md",
	".mjs",
	".py",
	".sh",
	".toml",
	".ts",
	".txt",
	".yaml",
	".yml",
]);
const SKIP_DIRECTORIES = new Set([".git", ".venv", "__pycache__", "node_modules", "dist", "build"]);
const MAX_TEXT_BYTES = 1024 * 1024;

export interface XlabPackageValidationIssue {
	message: string;
	path?: string;
}

function isInside(base: string, candidate: string): boolean {
	const path = relative(base, candidate);
	return path === "" || (!path.startsWith("..") && !path.startsWith("/") && path !== "..");
}

function executableExists(name: string): boolean {
	if (name.includes("/") || name.includes("\\")) {
		return existsSync(name);
	}
	return (process.env.PATH ?? "")
		.split(delimiter)
		.some((directory) => directory !== "" && existsSync(join(directory, name)));
}

export function validateXlabRuntimePrerequisites(skillPackage: XlabSkillPackage): XlabPackageValidationIssue[] {
	const issues: XlabPackageValidationIssue[] = [];
	for (const executable of skillPackage.manifest.runtime.executables ?? []) {
		if (!executableExists(executable)) {
			issues.push({ message: `Required executable is not available on PATH: ${executable}` });
		}
	}
	for (const secret of skillPackage.manifest.runtime.requiredSecrets ?? []) {
		if (!isXlabSecretRequirementSatisfied(skillPackage, secret)) {
			issues.push({ message: `Required secret environment variable is missing: ${secret}` });
		}
	}
	return issues;
}

export function validateXlabPackageSelfContained(packageDir: string): XlabPackageValidationIssue[] {
	const issues: XlabPackageValidationIssue[] = [];
	const walk = (directory: string): void => {
		for (const name of readdirSync(directory)) {
			if (SKIP_DIRECTORIES.has(name)) {
				continue;
			}
			const path = join(directory, name);
			const lstat = lstatSync(path);
			if (lstat.isSymbolicLink()) {
				const target = resolve(directory, readlinkSync(path));
				if (!isInside(packageDir, target)) {
					issues.push({ message: "Symbolic link escapes the skill package.", path });
				}
				continue;
			}
			if (lstat.isDirectory()) {
				walk(path);
				continue;
			}
			if (!lstat.isFile() || statSync(path).size > MAX_TEXT_BYTES) {
				continue;
			}
			const extension = name.includes(".") ? `.${name.split(".").pop()?.toLowerCase()}` : "";
			if (!TEXT_EXTENSIONS.has(extension)) {
				continue;
			}
			const text = readFileSync(path, "utf-8");
			const lines = text.split(/\r?\n/);
			if (name === "requirements.lock") {
				for (const [index, line] of lines.entries()) {
					const requirement = line.trim();
					if (
						requirement !== "" &&
						!requirement.startsWith("#") &&
						!/^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*\])?==[^\s;]+(?:\s*;\s*.+)?$/.test(
							requirement,
						)
					) {
						issues.push({
							message: `Unlocked Python requirement at line ${index + 1}; use an exact == version.`,
							path,
						});
					}
				}
			}
			for (const [index, line] of lines.entries()) {
				if (/\/(?:aistor|hpc_stor\d*)\//.test(line)) {
					issues.push({
						message: `Host-specific absolute path at line ${index + 1}.`,
						path,
					});
				}
				if (
					/(?:file:\/\/|\s-e\s+|path\s*=\s*["']|dependencies\s*=).*\.{2}\//.test(line) ||
					/(?:Xcientist|XAgora|XForge|PaperGraph)\/.+\.(?:py|js|sh)/.test(line)
				) {
					issues.push({
						message: `External local code dependency at line ${index + 1}.`,
						path,
					});
				}
			}
		}
	};
	walk(packageDir);
	return issues;
}
