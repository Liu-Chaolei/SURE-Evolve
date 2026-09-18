import { cpSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parseEnv } from "node:util";
import { AuthStorage } from "../../auth-storage.ts";
import { ModelRegistry } from "../../model-registry.ts";
import { XlabArtifactStore } from "../artifact-store.ts";
import { discoverXlabSkillPackages } from "../manifest.ts";
import { XlabRunManager } from "../run-manager.ts";
import type { XlabManifestEnvelope, XlabRunRecord } from "../types.ts";
import {
	type AsrConditionResult,
	type AsrEvolutionConfig,
	AsrSlurmExecutor,
	executionTools,
	freezeProject,
	readJson,
	runLocalPython,
} from "./asr-execution.ts";
import { asrPlans } from "./asr-plans.ts";
import { createExperimentStageDrivers } from "./drivers.ts";
import { bindXlabExperimentIdea, type XlabIdeaBinding } from "./idea-binding.ts";
import { createXlabExperimentStagingMaterializer, XlabExperimentLifecycle } from "./lifecycle.ts";
import { currentSymbolicMemoryDigest, ExperimentMaterializer } from "./materializer.ts";
import {
	atomicWriteJson,
	canonicalDigest,
	canonicalJson,
	canonicalJsonBytes,
	type ExperimentAssignment,
	type ExperimentExecutionProfile,
	sha256,
	validateExperimentResult,
} from "./protocol.ts";

export function xiEnvironment(envFile: string): NodeJS.ProcessEnv {
	const loaded = parseEnv(readFileSync(envFile, "utf-8"));
	if (!loaded.XI_API_KEY || !loaded.XI_BASE_URL)
		throw new Error("XI_API_KEY and XI_BASE_URL are required in the private env file.");
	const base = loaded.XI_BASE_URL.replace(/\/+$/, "");
	const url = new URL(base);
	const endpoint = url.pathname === "/" || url.pathname === "" ? `${base}/v1` : base;
	const env: NodeJS.ProcessEnv = Object.fromEntries(
		Object.entries(process.env).filter(([key]) => !/^(ZAI_|XI_|OPENAI_|LLM_|XLAB_RESEARCH_IDEA_)/.test(key)),
	);
	Object.assign(env, {
		OPENAI_API_KEY: loaded.XI_API_KEY,
		OPENAI_BASE_URL: endpoint,
		XLAB_RESEARCH_IDEA_STREAM: "1",
		PYTHONDONTWRITEBYTECODE: "1",
	});
	for (const role of ["AGENT", "GENERATION", "EVALUATION", "FUSION"])
		env[`XLAB_RESEARCH_IDEA_${role}_MODEL`] = "gpt-6-astra";
	return env;
}

export function xiAgentOptions(config: AsrEvolutionConfig) {
	const env = xiEnvironment(config.env_file);
	const authStorage = AuthStorage.inMemory();
	authStorage.setRuntimeApiKey("xi", env.OPENAI_API_KEY!);
	const modelRegistry = ModelRegistry.inMemory(authStorage);
	modelRegistry.registerProvider("xi", {
		apiKey: env.OPENAI_API_KEY,
		baseUrl: env.OPENAI_BASE_URL,
		api: "openai-completions",
		models: [
			{
				id: "gpt-6-astra",
				name: "GPT-6 Astra",
				reasoning: true,
				input: ["text"],
				contextWindow: 128000,
				maxTokens: 16384,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				compat: { supportsStore: false, supportsDeveloperRole: false },
			},
		],
	});
	const model = modelRegistry.find("xi", "gpt-6-astra");
	if (!model) throw new Error("XI model registration failed.");
	return { authStorage, modelRegistry, model };
}

export function createAsrExperiment(
	cwd: string,
	config: AsrEvolutionConfig,
	context: Record<string, unknown>,
	ideaPath?: string,
): XlabRunRecord {
	const manager = new XlabRunManager(cwd);
	const skill = discoverXlabSkillPackages(cwd).packages.find((item) => item.manifest.name === "run_experiment");
	if (!skill) throw new Error("Native run_experiment skill is missing.");
	let binding: XlabIdeaBinding;
	const profile: ExperimentExecutionProfile = { mode: "direct_formal", kind: ideaPath ? "research" : "baseline" };
	if (ideaPath) {
		const result = bindXlabExperimentIdea({ cwd, args: `--idea ${JSON.stringify(ideaPath)}` });
		if (!result.ok) throw new Error(result.error);
		binding = result.binding;
	} else {
		const canonical = {
			schema_version: "xlab.experiment_baseline.v1",
			title: "TEDLIUM-only Zipformer reference under the fixed benchmark protocol",
			components: [],
			baseline_configuration: {
				train_args: ["--enable-musan", "0"],
				decode_args: [],
				decoding_method: "greedy_search",
				rationale:
					"MUSAN is an additional noise corpus excluded by the TEDLIUM-only data contract. Disabling it is an explicit reference configuration, not a proposed scientific component. Native model architecture is retained with the declared NPU compatibility adapter.",
			},
			training: config.training,
			recipe: config.recipe,
		};
		const bytes = canonicalJsonBytes(canonical);
		binding = {
			canonical,
			canonicalBytes: bytes,
			warnings: [],
			metadata: {
				schema_version: "xlab.idea_binding.v1",
				resolution_method: "path",
				reference: config.recipe,
				source_schema_version: canonical.schema_version,
				source_digest: sha256(bytes),
				canonical_digest: sha256(bytes),
			},
		};
	}
	return manager.createRun(
		skill,
		ideaPath ? `--idea ${JSON.stringify(ideaPath)}` : "--baseline-profile inputs/asr.json",
		{
			workspaceSlug: config.workspace,
			invokedCommand: "run-experiment",
			ideaBinding: binding,
			materializeStaging(staging) {
				createXlabExperimentStagingMaterializer(binding, undefined, profile)(staging);
				atomicWriteJson(join(staging.stagingDir, "inputs/asr.json"), config);
				atomicWriteJson(join(staging.stagingDir, "inputs/context.json"), context);
				atomicWriteJson(join(staging.stagingDir, "inputs/execution-profile.json"), profile);
				mkdirSync(join(staging.stagingDir, "project/receipts"), { recursive: true });
				mkdirSync(join(staging.stagingDir, "agent_reports"), { recursive: true });
			},
		},
	);
}

export async function runAsrExperiment(record: XlabRunRecord): Promise<AsrConditionResult[]> {
	const config = readJson<AsrEvolutionConfig>(join(record.runDir, "inputs/asr.json"));
	const profile = readJson<ExperimentExecutionProfile>(join(record.runDir, "inputs/execution-profile.json"));
	const idea = readJson<Record<string, unknown>>(join(record.runDir, "inputs/idea.json"));
	const components = (idea.components as Array<{ component: string }>).map((item) => item.component);
	const project = join(record.runDir, "project");
	const prepareRequest = join(record.runDir, "inputs/prepare-request.json");
	atomicWriteJson(prepareRequest, { config, project });
	await runLocalPython(config, "prepare", prepareRequest);
	const context = readJson<{ current_best?: AsrConditionResult }>(join(record.runDir, "inputs/context.json"));
	const inheritedDisabled = [
		...new Set([
			...(context.current_best?.inherited_disabled_components ?? []),
			...(context.current_best?.disabled_components ?? []),
		]),
	].filter((name) => !components.includes(name));
	if (profile.kind === "research" && context.current_best && !existsSync(join(project, "parent.json"))) {
		for (const name of ["recipe", "library", "method.py"])
			cpSync(join(context.current_best.project, name), join(project, name), { recursive: true });
		atomicWriteJson(join(project, "parent.json"), context.current_best);
	}
	const plans = asrPlans(config, record.runDir, components);
	for (const [stage, plan] of Object.entries(plans))
		atomicWriteJson(join(record.runDir, "inputs", `${stage}-plan.json`), plan);
	const executor = new AsrSlurmExecutor(config, record.runDir);
	const receiptPath = (assignment: ExperimentAssignment) =>
		join(project, "receipts", `${assignment.assignment_id}.json`);
	const execute = async (assignment: ExperimentAssignment, signal?: AbortSignal) => {
		let evidence: unknown;
		if (assignment.child.stage === "prepare") {
			evidence = readJson(join(project, "preparation.json"));
		} else if (assignment.child.stage === "code") {
			const request = join(record.runDir, "inputs", `static-${assignment.assignment_id}.json`);
			const receipt = join(record.runDir, "inputs", `static-${assignment.assignment_id}-result.json`);
			atomicWriteJson(request, { project, components, receipt });
			await runLocalPython(config, "static", request);
			evidence = readJson(receipt);
		} else if (assignment.child.stage === "science") {
			const unit = plans.science.work_units.find((item) => item.id === assignment.child.work_unit);
			if (!unit) throw new Error("Unknown scientific condition.");
			const frozen = freezeProject(project, record.runDir);
			evidence = await executor.runCondition({
				id: unit.id,
				project: frozen.project,
				sourceDigest: frozen.digest,
				components,
				disabled: unit.disabled_components ?? [],
				inheritedDisabled,
				signal,
			});
		} else throw new Error("No execution tool in finalize.");
		const result = {
			artifact_role: "worker_result",
			status: "success",
			work_unit: assignment.child.work_unit,
			evidence,
		};
		atomicWriteJson(receiptPath(assignment), result);
		return result;
	};
	const lifecycle = new XlabExperimentLifecycle({
		cwd: record.cwd,
		agentDir: join(record.runDir, "private-agent"),
		runId: record.runId,
		ownerId: `asr-${process.pid}`,
		coordinatorOptions: {
			maxAssignmentPromptAttempts: 3,
			childRuntimeOptions: {
				agentOptions: xiAgentOptions(config),
				assignmentTools: (assignment) => executionTools(execute, assignment),
				boundPlan: (assignment) => canonicalJson(plans[assignment.child.stage as keyof typeof plans]),
				allowEdits: (assignment) => assignment.child.stage === "code",
				systemPrompt: (assignment) =>
					[
						`You are the native XLab ${assignment.child.role} for ${assignment.child.stage}/${assignment.child.work_unit ?? "finalize"}.`,
						`For a reviewer submission the exact JSON key is "role" and its exact value is ${JSON.stringify(assignment.result_validation?.reviewer_role ?? "not_applicable")}. Never create an exact_assigned_role key. Call submit_review({result: report, output_paths: []}).`,
						`Read ${join(record.runDir, "inputs/context.json")} and ${join(record.runDir, "inputs/idea.json")}.`,
						`The real prepared code is in ${project}. Plan templates are in ${record.runDir}/inputs/*-plan.json. Accepted worker evidence is in ${project}/receipts; reviewer reports are under ${record.runDir}/agent_reports.`,
						"Planner: review the bound plan and submit it with submit_plan({bound_plan:true,output_paths:[]}); this preserves every condition without truncating a large plan to fit a model response.",
						"Use only the granted tools. No smoke, test training, API probes, extra data or pretrained weights. The native recipe's own in-training OOM check is part of formal training.",
						"Worker: call experiment_execute to obtain real evidence and submit its returned object unchanged with submit_worker_result, output_paths=[]. Never invent metrics or results. Code workers first implement the actual idea in recipe/*.py and method.py, using meaningful component_enabled(name) hooks. Baseline workers preserve the prepared recipe.",
						"For an explicit baseline input, baseline_configuration defines the reference, including --enable-musan 0 for TEDLIUM-only training. Compare against this declared benchmark configuration, not unoverridden upstream CLI defaults. The baseline has no proposed scientific components and its static code phase does not implement a new method.",
						"Code phase is static only; static_integration checks syntax and toggle bindings, not runtime success. Runtime integration is evaluated in science after every full condition finishes.",
						"Reviewer: return artifact_role=reviewer_report, your exact assigned role, reviewer_kind=agent, verdict=PASS or FAIL, blocking=true, nonempty summary, checked_artifacts, issues (code,message,required_fix,evidence), and nonempty structured_findings. PASS has no issues. Read actual files before judging.",
						"For statistical_interpretation, structured_findings.component_results must contain every canonical component ({} for baseline). Each record includes condition_id, enabled_components, disabled_components, result=positive/negative/neutral/inconclusive, metric=wer, value as a string, confidence in [0,1], analysis, follow_up_required=false only when its full run completed. Positive means removal helped: reference WER minus disabled WER > 0. Do not claim statistical significance from one seed.",
						"Final reviewer: return artifact_role=final_review, verdict=PASS only with complete evidence, blocking=true, checked_artifacts, issues=[], summary={feasible:boolean,confidence:number,key_findings:[string]}. Do not supply or override component_results.",
						"Every child must finish with its assigned submit_* tool. Do not alter parent-owned protocol, plans, receipts or final manifests.",
					].join("\n"),
				validateResult(assignment, result) {
					const error = validateExperimentResult(assignment, result);
					if (error) return error;
					if (
						assignment.kind === "worker_result" &&
						(!existsSync(receiptPath(assignment)) ||
							canonicalDigest(readJson(receiptPath(assignment))) !== canonicalDigest(result))
					)
						return "Worker result must exactly match the executor's real evidence receipt.";
					if (
						assignment.kind === "plan" &&
						canonicalDigest(result) !== canonicalDigest(plans[assignment.child.stage as keyof typeof plans])
					)
						return "Use the exact frozen executor-bound plan template.";
					if (assignment.result_validation?.reviewer_role === "statistical_interpretation" && components.length) {
						const report = result as {
							structured_findings: {
								component_results: Record<
									string,
									{ result: string; value: string; metric: string; follow_up_required: boolean }
								>;
							};
						};
						const reference = readJson<AsrConditionResult>(
							join(project, "results/science/all-components/result.json"),
						);
						for (const unit of plans.science.work_units.filter((item) => item.disabled_component)) {
							const condition = readJson<AsrConditionResult>(
								join(project, "results/science", unit.id, "result.json"),
							);
							const benefit = reference.score - condition.score;
							const conclusion = report.structured_findings.component_results[unit.disabled_component!];
							if (
								conclusion.follow_up_required !== false ||
								conclusion.metric !== "wer" ||
								conclusion.result !== (benefit > 0 ? "positive" : benefit < 0 ? "negative" : "neutral") ||
								Math.abs(Number(conclusion.value) - benefit) > 1e-12 ||
								!Number.isFinite(Number(conclusion.value))
							) {
								return `Component ${unit.disabled_component} must report the actual removal benefit ${benefit}, with its matching sign. One-seed evidence does not establish significance.`;
							}
						}
					}
					return undefined;
				},
			},
		},
		finalizeSuccess(latest) {
			const manifestPath = join(latest.runDir, "manifest.json");
			const manifest = readJson<XlabManifestEnvelope>(manifestPath);
			const manager = new XlabRunManager(latest.cwd);
			return {
				manifestPath,
				artifactRefs: new XlabArtifactStore(latest.cwd).commitManifestArtifacts(
					latest,
					manifest,
					(path) => manager.resolveRunPath(latest, path),
					[],
				),
			};
		},
	});
	const reconstructed = lifecycle.reconstruct();
	const materializer = new ExperimentMaterializer({
		runRoot: record.runDir,
		sharedRoot: record.cwd,
		protocol: reconstructed.protocol,
	});
	const drivers = createExperimentStageDrivers({
		idea,
		ideaDigest: reconstructed.snapshot.idea_digest,
		policyDigest: reconstructed.snapshot.policy_digest,
		executionProfile: profile,
		planTemplates: plans,
		maxReviewRounds: 3,
		readScope: [join(record.runDir, "inputs"), project, join(record.runDir, "agent_reports")],
		projectWriteScope: [join(project, "recipe"), join(project, "method.py")],
		publication: {
			publishPlan: ({ assignmentId }) => {
				materializer.materializePlan(assignmentId, components);
			},
			publishWorker: ({ accepted }) => {
				materializer.materializeWorker(accepted.accepted.assignment_id);
			},
			publishReview: ({ stage, workUnit, reviewRound, assignmentIds, validationContext }) => {
				materializer.materializeReviewMatrix({
					stage,
					workUnit: workUnit.id,
					reviewRound,
					assignmentIds,
					validationContext,
				});
			},
			publishFinal: ({ componentDefinitions, componentResults, summary, provenance }) => {
				const memoryPath = `.xlab/workspaces/${config.workspace}/symbolic_memory.json`;
				materializer.materializeFinal({
					componentDefinitions: componentDefinitions.map((item) => ({
						component: item.component,
						method_context: item.explanation,
					})),
					componentResults,
					summary,
					provenance,
					symbolicMemoryPath: memoryPath,
					expectedSymbolicMemoryDigest: currentSymbolicMemoryDigest(join(record.cwd, memoryPath)),
				});
			},
		},
	});
	try {
		const snapshot = reconstructed.snapshot;
		if (snapshot.status === "success") {
			// Reuse already audited work below.
		} else if (snapshot.status === "paused") await lifecycle.resume(drivers);
		else await lifecycle.start(drivers);
	} catch (error) {
		const state = lifecycle.reconstruct();
		if (state.snapshot.terminal_generation === undefined)
			state.coordinator.pause(error instanceof Error ? error.message : String(error));
		throw error;
	}
	if (lifecycle.status().status !== "success") throw new Error("Experiment has not completed native finalization.");
	return plans.science.work_units.map((unit) =>
		readJson<AsrConditionResult>(join(project, "results/science", unit.id, "result.json")),
	);
}
