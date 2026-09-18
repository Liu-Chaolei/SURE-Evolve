import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, symlinkSync } from "node:fs";
import { join } from "node:path";
import type { AsrEvolutionConfig } from "./asr-execution.ts";
import { atomicWriteJson, sha256 } from "./protocol.ts";

export function freezeAsrDeployment(
	cwd: string,
	stagingDir: string,
	runDir: string,
	config: AsrEvolutionConfig,
): AsrEvolutionConfig {
	const source = join(stagingDir, "controller_source");
	const deployed = join(runDir, "controller_source");
	mkdirSync(source, { recursive: true });
	for (const name of ["package.json", "tsconfig.json", "tsconfig.base.json"])
		cpSync(join(cwd, name), join(source, name));
	const filter = (path: string) => !path.split("/").includes("__pycache__");
	for (const entry of readdirSync(join(cwd, "packages"), { withFileTypes: true })) {
		if (!entry.isDirectory() || !existsSync(join(cwd, "packages", entry.name, "src"))) continue;
		const target = join(source, "packages", entry.name);
		mkdirSync(target, { recursive: true });
		cpSync(join(cwd, "packages", entry.name, "src"), join(target, "src"), { recursive: true });
		cpSync(join(cwd, "packages", entry.name, "package.json"), join(target, "package.json"));
		const modules = join(cwd, "packages", entry.name, "node_modules");
		if (existsSync(modules)) symlinkSync(modules, join(target, "node_modules"), "dir");
	}
	symlinkSync(join(cwd, "node_modules"), join(source, "node_modules"), "dir");
	for (const skill of ["research_idea", "run_experiment", "asr_evolution"]) {
		cpSync(join(cwd, "xlab/skills", skill), join(source, "xlab/skills", skill), { recursive: true, filter });
	}
	cpSync(join(config.icefall, "icefall"), join(source, "model/icefall"), {
		recursive: true,
		dereference: true,
		filter,
	});
	cpSync(config.recipe, join(source, "model/recipe"), { recursive: true, dereference: true, filter });
	cpSync(config.sure_pythonpath, join(source, "sure_eval_src"), { recursive: true, filter });
	const hashes: Record<string, string> = {};
	const visit = (directory: string) => {
		for (const entry of readdirSync(directory, { withFileTypes: true })) {
			const path = join(directory, entry.name);
			if (entry.isSymbolicLink()) continue; // Dependency installations are bound, never copied or modified.
			if (entry.isDirectory()) visit(path);
			else hashes[path.slice(source.length + 1)] = sha256(readFileSync(path));
		}
	};
	visit(source);
	atomicWriteJson(join(source, "source_manifest.json"), hashes);
	return {
		...config,
		deployment_root: deployed,
		runtime: join(deployed, "xlab/skills/run_experiment/scripts"),
		research_script: join(deployed, "xlab/skills/research_idea/scripts/run_idea_phase.py"),
		icefall: join(deployed, "model"),
		recipe: join(deployed, "model/recipe"),
		sure_pythonpath: join(deployed, "sure_eval_src"),
	};
}
