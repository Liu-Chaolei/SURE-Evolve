import { spawn } from "node:child_process";
import { closeSync, existsSync, openSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import lockfile from "proper-lockfile";
import { XlabArtifactStore } from "../artifact-store.ts";
import { discoverXlabSkillPackages } from "../manifest.ts";
import { XlabRunManager } from "../run-manager.ts";
import type { XlabManifestEnvelope, XlabRunRecord } from "../types.ts";
import { XlabWorkspaceManager } from "../workspace-manager.ts";
import { freezeAsrDeployment } from "./asr-deployment.ts";
import {
	type AsrConditionResult,
	type AsrEvolutionConfig,
	AsrSlurmExecutor,
	readJson,
	validateAsrConfig,
} from "./asr-execution.ts";
import { createAsrExperiment, runAsrExperiment, xiAgentOptions, xiEnvironment } from "./asr-experiment.ts";
import { XlabExperimentLifecycle } from "./lifecycle.ts";
import { atomicWriteJson, canonicalDigest, sha256 } from "./protocol.ts";

interface EvolutionRound {
	index: number;
	idea_run_id?: string;
	idea_path?: string;
	experiment_run_id?: string;
	results?: AsrConditionResult[];
	feedback?: Record<string, unknown>;
}

export interface AsrEvolutionState {
	schema_version: "xlab.asr_evolution_state.v1";
	config_digest: string;
	status: "pending" | "running" | "paused" | "success" | "cancelled";
	baseline_run_id?: string;
	baseline?: AsrConditionResult;
	best?: AsrConditionResult;
	rounds: EvolutionRound[];
	selection?: AsrConditionResult[];
	test?: AsrConditionResult[];
	error?: string;
}

export function createAsrEvolution(cwd: string, configPath: string, workspace?: string): XlabRunRecord {
	const config = readJson<AsrEvolutionConfig>(configPath);
	if (workspace) config.workspace = workspace;
	validateAsrConfig(config);
	xiAgentOptions(config); // Validate registration only; no API request.
	const skills = discoverXlabSkillPackages(cwd).packages;
	const skill = skills.find((item) => item.manifest.name === "asr_evolution");
	if (!skill) throw new Error("asr_evolution skill is missing.");
	new XlabWorkspaceManager(cwd).ensureWorkspace(config.workspace);
	return new XlabRunManager(cwd).createRun(skill, `--config ${JSON.stringify(configPath)}`, {
		workspaceSlug: config.workspace,
		invokedCommand: "run-evolution",
		materializeStaging({ stagingDir, runDir }) {
			const frozen = freezeAsrDeployment(cwd, stagingDir, runDir, config);
			atomicWriteJson(join(stagingDir, "inputs/config.json"), frozen);
			atomicWriteJson(
				join(stagingDir, "inputs/resources.json"),
				Object.fromEntries(
					[config.survey, config.resource_manifest, ...Object.values(config.refs)].map((path) => [
						path,
						sha256(readFileSync(path)),
					]),
				),
			);
			atomicWriteJson(join(stagingDir, "evolution.json"), {
				schema_version: "xlab.asr_evolution_state.v1",
				config_digest: canonicalDigest(frozen),
				status: "pending",
				rounds: [],
			});
		},
	});
}

function contextFor(config: AsrEvolutionConfig, state: AsrEvolutionState, index: number): Record<string, unknown> {
	const implementation: Record<string, string> = {};
	if (state.best) {
		for (const file of ["method.py", "recipe/model.py", "recipe/zipformer.py"])
			implementation[file] = readFileSync(join(state.best.project, file), "utf-8");
	}
	return {
		task_description:
			"Improve English ASR WER on full TEDLIUM3 starting from icefall Zipformer. Propose one evidence-backed research method per round. Architecture, training methods and decoding are open to change within the fixed data, evaluation and compute protocol. Every canonical component must have a genuine independently disabled full-run condition. There is no fixed component or ablation count cap.",
		round_index: index,
		total_rounds: 6,
		task_card: { dataset: "TEDLIUM_release-3", metric: "wer", direction: "lower" },
		base_model_profile: { model: "icefall Zipformer", recipe: config.recipe, data: config.data },
		execution_contract: {
			training: config.training,
			hardware: config.slurm,
			scratch_only: true,
			no_extra_data: true,
			no_smoke: true,
			evaluation: {
				pipeline: config.pipeline_id,
				search: "regular",
				selection: "after_round_6_only",
				test: "frozen_final_model_only",
			},
		},
		current_best: state.best ?? null,
		implementation,
		prior_rounds: state.rounds
			.filter((round) => round.feedback)
			.map((round) => ({ index: round.index, results: round.results, feedback: round.feedback })),
	};
}

function shellArgument(value: string): string {
	return `'${value.replace(/'/g, `'"'"'`)}'`;
}

async function ideaPhase(
	config: AsrEvolutionConfig,
	cwd: string,
	run: XlabRunRecord,
	phase: string,
	argumentsText?: string,
): Promise<void> {
	const cli = config.research_script ?? join(cwd, "xlab/skills/research_idea/scripts/run_idea_phase.py");
	const argumentsPath = join(run.runDir, "research_arguments.txt");
	if (argumentsText) writeFileSync(argumentsPath, argumentsText, { mode: 0o600 });
	const argv = [
		cli,
		phase,
		"--cwd",
		cwd,
		"--run-dir",
		run.runDir,
		"--run-id",
		run.runId,
		...(argumentsText ? ["--arguments-file", argumentsPath] : []),
	];
	const log = openSync(join(run.runDir, "logs", `${phase}.log`), "a");
	try {
		await new Promise<void>((done, reject) => {
			const child = spawn(config.python, argv, {
				cwd,
				env: xiEnvironment(config.env_file),
				stdio: ["ignore", log, log],
			});
			const onSignal = () => child.kill("SIGTERM");
			process.once("SIGTERM", onSignal);
			child.once("error", (error) => {
				process.off("SIGTERM", onSignal);
				reject(error);
			});
			child.once("exit", (code) => {
				process.off("SIGTERM", onSignal);
				code === 0
					? done()
					: reject(new Error(`Research idea ${phase} failed (${code}); see ${run.runDir}/logs/${phase}.log`));
			});
		});
	} finally {
		closeSync(log);
	}
}

export async function runAsrEvolution(record: XlabRunRecord): Promise<void> {
	const config = readJson<AsrEvolutionConfig>(join(record.runDir, "inputs/config.json"));
	validateAsrConfig(config);
	const resources = readJson<Record<string, string>>(join(record.runDir, "inputs/resources.json"));
	for (const [path, digest] of Object.entries(resources))
		if (sha256(readFileSync(path)) !== digest) throw new Error(`Frozen resource changed: ${path}`);
	const statePath = join(record.runDir, "evolution.json");
	const release = await lockfile.lock(statePath, { realpath: false, stale: 180000, update: 30000 });
	const manager = new XlabRunManager(record.cwd);
	let state = readJson<AsrEvolutionState>(statePath);
	const save = () => {
		atomicWriteJson(statePath, state);
	};
	const active = () => {
		if (existsSync(join(record.runDir, "cancel_requested.json")))
			throw new Error("Evolution cancellation requested.");
	};
	const stage = (name: string, action: "start" | "complete") => {
		const workflow = manager.readWorkflow(record);
		const current = workflow?.stages.find((entry) => entry.id === name);
		if ((action === "start" && current?.status === "running") || current?.status === "success") return;
		const transition = manager.transitionStage(record, name, action);
		if (!transition.ok) throw new Error(transition.repair ?? `Cannot ${action} ${name}`);
	};
	try {
		if (state.config_digest !== canonicalDigest(config))
			throw new Error("Evolution configuration changed after creation.");
		if (state.status === "success") return;
		active();
		state = { ...state, status: "running" };
		delete state.error;
		save();
		manager.updateRun(record, { status: "running" }, "evolution_started");
		if (!state.baseline) {
			stage("baseline", "start");
			if (!state.baseline_run_id) {
				const baseline = createAsrExperiment(record.cwd, config, contextFor(config, state, 0));
				state.baseline_run_id = baseline.runId;
				save();
			}
			const baseline = manager.readRun(state.baseline_run_id)!;
			state.baseline = (await runAsrExperiment(baseline))[0];
			state.best = state.baseline;
			save();
			stage("baseline", "complete");
		}
		stage("baseline", "complete");
		for (let index = 1; index <= config.rounds; index++) {
			active();
			let round = state.rounds.find((entry) => entry.index === index);
			if (round?.feedback) {
				stage(`round-${index}`, "complete");
				continue;
			}
			stage(`round-${index}`, "start");
			if (!round) {
				round = { index };
				state.rounds.push(round);
				save();
			}
			if (!round.idea_path) {
				const skill = discoverXlabSkillPackages(record.cwd).packages.find(
					(item) => item.manifest.name === "research_idea",
				)!;
				if (!round.idea_run_id) {
					round.idea_run_id = manager.createRun(skill, "", { workspaceSlug: config.workspace }).runId;
					save();
				}
				const ideaRun = manager.readRun(round.idea_run_id)!;
				const context = contextFor(config, state, index);
				const args = [
					"--survey",
					config.survey,
					"--topic",
					String(context.task_description),
					"--discussion",
					JSON.stringify(context),
				];
				const previous = state.rounds.find((entry) => entry.index === index - 1);
				if (previous?.idea_path && previous.feedback) {
					const artifact = readJson<{ idea_result: unknown }>(previous.idea_path);
					args.push(
						"--mature-idea",
						JSON.stringify(artifact.idea_result),
						"--experiment-feedback",
						JSON.stringify(previous.feedback),
					);
				}
				const requestExists = existsSync(join(ideaRun.runDir, "state/request.json"));
				if (!requestExists) await ideaPhase(config, record.cwd, ideaRun, "init", args.map(shellArgument).join(" "));
				await ideaPhase(config, record.cwd, ideaRun, requestExists ? "resume" : "synthesize");
				const manifestPath = join(ideaRun.runDir, "manifest.json");
				const manifest = readJson<XlabManifestEnvelope>(manifestPath);
				if (manifest.status !== "success") throw new Error("Research Idea audit did not succeed.");
				const refs = new XlabArtifactStore(record.cwd).commitManifestArtifacts(
					ideaRun,
					manifest,
					(path) => manager.resolveRunPath(ideaRun, path),
					[],
				);
				const completedIdea = manager.updateRun(
					ideaRun,
					{ status: "success", manifestPath, artifactRefs: refs },
					"research_completed",
				);
				new XlabWorkspaceManager(record.cwd).registerRunArtifacts({
					run: completedIdea,
					manifest,
					artifactRefs: refs,
				});
				round.idea_path = join(ideaRun.runDir, "artifacts/research_idea.json");
				save();
			}
			if (!round.experiment_run_id) {
				round.experiment_run_id = createAsrExperiment(
					record.cwd,
					config,
					contextFor(config, state, index),
					round.idea_path,
				).runId;
				save();
			}
			const experiment = manager.readRun(round.experiment_run_id)!;
			round.results = await runAsrExperiment(experiment);
			const reference = round.results.find((result) => result.disabled_components.length === 0)!;
			const ablation = readJson<{ components: Record<string, Record<string, unknown>>; summary: unknown }>(
				join(experiment.runDir, "agent_reports/ablation/final/ablation_results.json"),
			);
			for (const result of round.results) {
				if (result.score < state.best!.score) state.best = result;
			}
			const components = Object.fromEntries(
				Object.entries(ablation.components).map(([name, conclusion]) => {
					const condition = round!.results!.find((result) => result.disabled_components.includes(name))!;
					const benefit = reference.score - condition.score;
					return [
						name,
						{
							...conclusion,
							observed_removal_benefit: benefit,
							reference_wer: reference.score,
							disabled_wer: condition.score,
							condition_id: condition.condition_id,
						},
					];
				}),
			);
			round.feedback = {
				components,
				summary: ablation.summary,
				results: round.results,
				current_best: state.best,
				interpretation:
					"Removal benefit = reference WER - disabled WER; one seed, no claim of statistical significance.",
			};
			atomicWriteJson(join(experiment.runDir, "artifacts/evolution_feedback.json"), round.feedback);
			save();
			stage(`round-${index}`, "complete");
		}
		active();
		stage("evaluation", "start");
		const all = [state.baseline!, ...state.rounds.flatMap((round) => round.results ?? [])];
		const evaluator = new AsrSlurmExecutor(config, record.runDir);
		const evaluate = (result: AsrConditionResult, split: "selection" | "test") =>
			evaluator.runCondition({
				id: `${split}-${canonicalDigest(result).slice(0, 16)}`,
				project: result.project,
				sourceDigest: result.source_digest,
				components: [...result.enabled_components, ...result.disabled_components],
				disabled: result.disabled_components,
				inheritedDisabled: result.inherited_disabled_components,
				split,
				checkpointDir: result.checkpoint_dir,
			});
		if (!state.selection) {
			const results = await Promise.allSettled(all.map((result) => evaluate(result, "selection")));
			if (results.some((result) => result.status === "rejected"))
				throw new Error("Selection evaluation incomplete; resume reuses accepted jobs.");
			state.selection = results.map((result) => (result as PromiseFulfilledResult<AsrConditionResult>).value);
			save();
		}
		let bestIndex = all.findIndex((result) => result.checkpoint === state.best!.checkpoint);
		for (let index = 0; index < state.selection.length; index++)
			if (state.selection[index].score < state.selection[bestIndex].score) bestIndex = index;
		const winner = all[bestIndex];
		atomicWriteJson(join(record.runDir, "artifacts/frozen_final_model.json"), {
			winner,
			selection: state.selection[bestIndex],
		});
		if (!state.test) {
			state.test = await Promise.all([
				evaluate(state.baseline!, "test"),
				...(winner.checkpoint === state.baseline!.checkpoint ? [] : [evaluate(winner, "test")]),
			]);
			save();
		}
		const summary = {
			baseline: state.baseline,
			winner,
			selection: state.selection,
			test: state.test,
			rounds: state.rounds,
			config_digest: state.config_digest,
		};
		const path = join(record.runDir, "artifacts/evolution_summary.json");
		atomicWriteJson(path, summary);
		const manifest: XlabManifestEnvelope = {
			schema_version: "2",
			run_id: record.runId,
			skill_name: "asr_evolution",
			skill_version: "1.0.0",
			status: "success",
			created_at: record.startedAt,
			inputs: { config_digest: state.config_digest },
			outputs: { winner: winner.checkpoint },
			validation: { valid: true, rounds: 6 },
			artifacts: [
				{
					type: "evolution_summary",
					schema_version: "1",
					path: "artifacts/evolution_summary.json",
					digest: `sha256:${sha256(readFileSync(path))}`,
					metadata: { rounds: 6 },
				},
			],
		};
		atomicWriteJson(join(record.runDir, "manifest.json"), manifest);
		const refs = new XlabArtifactStore(record.cwd).commitManifestArtifacts(
			record,
			manifest,
			(pathValue) => manager.resolveRunPath(record, pathValue),
			[],
		);
		stage("evaluation", "complete");
		state.status = "success";
		save();
		manager.updateRun(
			record,
			{
				status: "success",
				artifactRefs: refs,
				manifestPath: join(record.runDir, "manifest.json"),
				finishedAt: new Date().toISOString(),
			},
			"evolution_completed",
		);
	} catch (error) {
		state.status = existsSync(join(record.runDir, "cancel_requested.json")) ? "cancelled" : "paused";
		state.error = error instanceof Error ? error.message : String(error);
		save();
		manager.updateRun(record, { status: state.status, lastRepair: state.error }, "evolution_interrupted");
		throw error;
	} finally {
		await release();
	}
}

export async function cancelAsrEvolution(record: XlabRunRecord): Promise<void> {
	atomicWriteJson(join(record.runDir, "cancel_requested.json"), { requested_at: new Date().toISOString() });
	const config = readJson<AsrEvolutionConfig>(join(record.runDir, "inputs/config.json"));
	const state = readJson<AsrEvolutionState>(join(record.runDir, "evolution.json"));
	const manager = new XlabRunManager(record.cwd);
	const children = [state.baseline_run_id, ...state.rounds.map((round) => round.experiment_run_id)].filter(
		(id): id is string => Boolean(id),
	);
	for (const id of children) {
		const child = manager.readRun(id)!;
		await new AsrSlurmExecutor(config, child.runDir).cancel();
		const lifecycle = new XlabExperimentLifecycle({
			cwd: record.cwd,
			agentDir: join(child.runDir, "private-agent"),
			runId: id,
			ownerId: `cancel-${process.pid}`,
		});
		const snapshot = lifecycle.reconstruct().snapshot;
		if (snapshot.terminal_generation === undefined) await lifecycle.cancel("user_cancel");
	}
	await new AsrSlurmExecutor(config, record.runDir).cancel();
	atomicWriteJson(join(record.runDir, "evolution.json"), { ...state, status: "cancelled" });
	manager.updateRun(record, { status: "cancelled", finishedAt: new Date().toISOString() }, "evolution_cancelled");
}
