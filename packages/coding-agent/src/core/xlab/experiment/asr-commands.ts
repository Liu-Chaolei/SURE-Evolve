import { spawn } from "node:child_process";
import { closeSync, existsSync, openSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionCommandContext } from "../../extensions/index.ts";
import { getXlabArgumentValue, parseXlabArgumentTokens } from "../command-completions.ts";
import { XlabRunManager } from "../run-manager.ts";
import type { XlabRunRecord } from "../types.ts";
import { cancelAsrEvolution, createAsrEvolution } from "./asr-evolution.ts";
import { type AsrEvolutionConfig, AsrSlurmExecutor, readJson, validateAsrConfig } from "./asr-execution.ts";
import { createAsrExperiment } from "./asr-experiment.ts";
import { XlabExperimentLifecycle } from "./lifecycle.ts";
import { atomicWriteJson } from "./protocol.ts";

export function launchNativeAsr(record: XlabRunRecord): number {
	const controllerPath = join(record.runDir, "controller.json");
	if (existsSync(controllerPath)) {
		const previous = readJson<{ pid: number }>(controllerPath);
		const processPath = `/proc/${previous.pid}/cmdline`;
		if (existsSync(processPath) && readFileSync(processPath, "utf-8").includes(record.runId))
			throw new Error(`Run ${record.runId} already has a live controller.`);
	}
	const config = readJson<AsrEvolutionConfig>(
		join(record.runDir, record.skillName === "asr_evolution" ? "inputs/config.json" : "inputs/asr.json"),
	);
	const cli = config.deployment_root
		? join(config.deployment_root, "packages/coding-agent/src/core/xlab/experiment/asr-cli.ts")
		: join(dirname(fileURLToPath(import.meta.url)), "asr-cli.ts");
	const log = openSync(join(record.runDir, "logs/controller.log"), "a");
	try {
		const child = spawn(process.execPath, ["--import", "tsx", cli, "run", record.cwd, record.runId], {
			cwd: record.cwd,
			detached: true,
			env: {
				...process.env,
				...(config.deployment_root ? { TSX_TSCONFIG_PATH: join(config.deployment_root, "tsconfig.json") } : {}),
			},
			stdio: ["ignore", log, log],
		});
		if (!child.pid) throw new Error("Native ASR controller did not start.");
		child.on("error", (error) =>
			atomicWriteJson(join(record.runDir, "controller_error.json"), { message: error.message }),
		);
		atomicWriteJson(controllerPath, { pid: child.pid, run_id: record.runId, started_at: new Date().toISOString() });
		child.unref();
		return child.pid;
	} finally {
		closeSync(log);
	}
}

export async function handleNativeAsrCommand(
	command: string,
	args: string,
	ctx: ExtensionCommandContext,
): Promise<boolean> {
	const tokens = parseXlabArgumentTokens(args);
	const manager = new XlabRunManager(ctx.cwd);
	const existing =
		tokens?.length === 1 && ["resume-run", "cancel-run"].includes(command) ? manager.readRun(tokens[0]) : undefined;
	const baseline = getXlabArgumentValue(args, "--baseline-profile");
	const profile = getXlabArgumentValue(args, "--profile");
	const isNew = command === "run-evolution" || (command === "run-experiment" && Boolean(baseline || profile));
	const isExisting =
		existing && (existing.skillName === "asr_evolution" || existsSync(join(existing.runDir, "inputs/asr.json")));
	if (!isNew && !isExisting) return false;
	if (!ctx.isProjectTrusted() || !ctx.isIdle())
		throw new Error("Native ASR commands require an idle agent and trusted project.");
	if (existing && command === "cancel-run") {
		if (existing.skillName === "asr_evolution") await cancelAsrEvolution(existing);
		else {
			await new AsrSlurmExecutor(readJson(join(existing.runDir, "inputs/asr.json")), existing.runDir).cancel();
			await new XlabExperimentLifecycle({
				cwd: existing.cwd,
				agentDir: join(existing.runDir, "private-agent"),
				runId: existing.runId,
				ownerId: `cancel-${process.pid}`,
			}).cancel("user_cancel");
		}
		const controller = join(existing.runDir, "controller.json");
		if (existsSync(controller)) {
			const { pid } = readJson<{ pid: number }>(controller);
			const path = `/proc/${pid}/cmdline`;
			if (existsSync(path) && readFileSync(path, "utf-8").includes(existing.runId)) process.kill(pid, "SIGTERM");
		}
		ctx.ui.notify(`Cancelled native ASR run ${existing.runId}.`);
		return true;
	}
	let record = existing;
	if (record && ["success", "failed", "cancelled"].includes(record.status))
		throw new Error(`Cannot resume terminal run ${record.status}.`);
	if (!record && command === "run-evolution") {
		const path = getXlabArgumentValue(args, "--config");
		if (!path) throw new Error("Usage: /xlab run-evolution --config <profile.json> [--workspace <slug>]");
		record = createAsrEvolution(ctx.cwd, resolve(ctx.cwd, path), getXlabArgumentValue(args, "--workspace"));
	} else if (!record) {
		const idea = getXlabArgumentValue(args, "--idea");
		if (baseline && idea) throw new Error("--baseline-profile and --idea are mutually exclusive.");
		if (!baseline && !idea) throw new Error("--profile requires --idea.");
		const config = readJson<AsrEvolutionConfig>(resolve(ctx.cwd, baseline ?? profile!));
		validateAsrConfig(config);
		record = createAsrExperiment(
			ctx.cwd,
			config,
			{ task_description: "Execute the frozen ASR protocol and canonical Idea." },
			idea,
		);
	}
	const pid = launchNativeAsr(record);
	ctx.ui.notify(
		`Started native ASR run ${record.runId} (controller ${pid}). Logs: ${record.runDir}/logs/controller.log`,
	);
	return true;
}
