import { join, resolve } from "node:path";
import { XlabRunManager } from "../run-manager.ts";
import { cancelAsrEvolution, createAsrEvolution, runAsrEvolution } from "./asr-evolution.ts";
import { type AsrEvolutionConfig, AsrSlurmExecutor, readJson, validateAsrConfig } from "./asr-execution.ts";
import { createAsrExperiment, runAsrExperiment } from "./asr-experiment.ts";
import { atomicWriteJson } from "./protocol.ts";

const [operation, cwdValue, input, workspace] = process.argv.slice(2);
const cwd = resolve(cwdValue ?? process.cwd());
const manager = new XlabRunManager(cwd);

async function main(): Promise<void> {
	if (operation === "create") {
		const record = createAsrEvolution(cwd, resolve(input), workspace);
		console.log(JSON.stringify({ run_id: record.runId, run_dir: record.runDir }));
		return;
	}
	if (operation === "check") {
		validateAsrConfig(readJson<AsrEvolutionConfig>(resolve(input)));
		console.log("ASR profile and input checks passed; no model or API requests executed.");
		return;
	}
	if (operation === "baseline") {
		const config = readJson<AsrEvolutionConfig>(resolve(input));
		validateAsrConfig(config);
		const record = createAsrExperiment(cwd, config, {
			task_description: "Run the fixed native Zipformer baseline; no scientific changes.",
		});
		console.log(JSON.stringify({ run_id: record.runId, run_dir: record.runDir }));
		await runAsrExperiment(record);
		return;
	}
	const record = manager.readRun(input);
	if (!record) throw new Error(`Unknown native ASR run ${input}.`);
	if (operation === "cancel") {
		if (record.skillName === "asr_evolution") await cancelAsrEvolution(record);
		else await new AsrSlurmExecutor(readJson(join(record.runDir, "inputs/asr.json")), record.runDir).cancel();
		return;
	}
	if (operation !== "run") throw new Error("Expected create, check, baseline, run, or cancel.");
	atomicWriteJson(join(record.runDir, "controller.json"), {
		pid: process.pid,
		started_at: new Date().toISOString(),
		argv: process.argv,
	});
	if (record.skillName === "asr_evolution") await runAsrEvolution(record);
	else await runAsrExperiment(record);
}

main().catch((error: unknown) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exitCode = 1;
});
