import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { XlabEnvironmentRecord, XlabSkillPackage } from "./types.ts";

function commandOutput(executable: string, args: string[], timeout = 30_000): string {
	const result = spawnSync(executable, args, {
		encoding: "utf-8",
		maxBuffer: 10 * 1024 * 1024,
		timeout,
	});
	if (result.error) {
		throw result.error;
	}
	if (result.status !== 0) {
		throw new Error(result.stderr.trim() || `${executable} ${args.join(" ")} exited with status ${result.status}`);
	}
	return `${result.stdout}${result.stderr}`.trim();
}

function resolvePythonExecutable(requested: string): string {
	const version = commandOutput(requested, [
		"-c",
		"import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
	]);
	const [major, minor] = version.split(".").map(Number);
	if (major > 3 || (major === 3 && minor >= 10)) {
		return requested;
	}
	const condaPython = commandOutput("conda", [
		"run",
		"--no-capture-output",
		"-n",
		"xlab",
		"python",
		"-c",
		"import sys; print(sys.executable)",
	]);
	return condaPython.split(/\r?\n/).at(-1)?.trim() || requested;
}

function readLockedRequirements(lockfilePath: string): string[] {
	return readFileSync(lockfilePath, "utf-8")
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter((line) => line !== "" && !line.startsWith("#"));
}

function writeJson(path: string, value: unknown): void {
	writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
}

export function prepareXlabEnvironment(cwd: string, skillPackage: XlabSkillPackage): XlabEnvironmentRecord | undefined {
	const runtime = skillPackage.manifest.runtime;
	if (runtime.kind !== "python" || !runtime.lockfile) {
		return undefined;
	}
	const python = runtime.executables?.[0] ?? "python";
	const resolvedPython = resolvePythonExecutable(python);
	const lockfilePath = join(skillPackage.packageDir, runtime.lockfile);
	const lockfile = readFileSync(lockfilePath, "utf-8");
	const requirements = readLockedRequirements(lockfilePath);
	const runtimeVersion = commandOutput(resolvedPython, ["--version"]);
	const digest = createHash("sha256")
		.update(skillPackage.manifest.name)
		.update("\0")
		.update(skillPackage.manifest.version)
		.update("\0")
		.update(runtimeVersion)
		.update("\0")
		.update(lockfile)
		.digest("hex");
	const environmentDir = join(cwd, ".xlab", "environments", skillPackage.manifest.name, digest.slice(0, 16));
	const executable = join(environmentDir, "bin", "python");
	const metadataPath = join(environmentDir, "environment.json");
	const lockDigest = createHash("sha256").update(lockfile).digest("hex");
	if (requirements.length === 0) {
		mkdirSync(environmentDir, { recursive: true });
		if (existsSync(metadataPath)) {
			return JSON.parse(readFileSync(metadataPath, "utf-8")) as XlabEnvironmentRecord;
		}
		const record: XlabEnvironmentRecord = {
			kind: "python-system",
			environmentId: digest,
			path: environmentDir,
			executable: resolvedPython,
			lockfile: runtime.lockfile,
			lockDigest,
			runtimeVersion,
		};
		writeJson(metadataPath, record);
		return record;
	}
	if (existsSync(executable) && existsSync(metadataPath)) {
		return JSON.parse(readFileSync(metadataPath, "utf-8")) as XlabEnvironmentRecord;
	}

	mkdirSync(environmentDir, { recursive: true });
	commandOutput(resolvedPython, ["-m", "venv", environmentDir]);
	commandOutput(
		executable,
		["-m", "pip", "install", "--disable-pip-version-check", "--requirement", lockfilePath],
		600_000,
	);
	const record: XlabEnvironmentRecord = {
		kind: "python-venv",
		environmentId: digest,
		path: environmentDir,
		executable,
		lockfile: runtime.lockfile,
		lockDigest,
		runtimeVersion,
	};
	writeJson(metadataPath, record);
	return record;
}
