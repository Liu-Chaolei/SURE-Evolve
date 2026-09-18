import { join } from "node:path";
import type { AsrEvolutionConfig } from "./asr-execution.ts";
import { canonicalDigest, type ExperimentPlan, type ExperimentWorkUnit, PREPARE_WORK_UNITS } from "./protocol.ts";

export function asrPlans(
	config: AsrEvolutionConfig,
	runDir: string,
	components: string[],
): Record<"prepare" | "code" | "science", ExperimentPlan> {
	const artifactIds = {
		repos: ["prepare.discovery", "prepare.repos"],
		dataset: ["prepare.dataset"],
		model: ["prepare.model"],
		env: ["prepare.env"],
		synthesis: ["prepare.idea", "prepare.target_inventory"],
	};
	const prepare: ExperimentPlan = {
		stage: "prepare",
		work_units: PREPARE_WORK_UNITS.map((id, index) => ({
			id,
			goal: `Bind and verify the real local ${id} resources for the frozen ASR experiment`,
			needs: index ? [PREPARE_WORK_UNITS[index - 1]] : [],
			input_paths: {
				context: "inputs/context.json",
				profile: "inputs/asr.json",
				source: "project/preparation.json",
			},
			repos_policy: "reference_or_copy",
			project_must_be_self_contained: true,
			research_required: false,
			acquisition_required: false,
			existing_local_hints: [config.data, config.recipe, config.resource_manifest],
			artifact_ids: artifactIds[id],
			done_condition: `Use experiment_execute managed artifact tools; accepted evidence must be in the artifact ledger with ${artifactIds[id].join(", ")}.`,
		})),
	};
	const codeUnit = (id: string, needs: string[]): ExperimentWorkUnit => ({
		id,
		needs,
		goal:
			id === "implementation"
				? "Implement every canonical scientific component with real disable hooks"
				: "Validate syntax and component bindings without executing a model",
		input_paths: { idea: "inputs/idea.json", context: "inputs/context.json", profile: "inputs/asr.json" },
		repos_policy: "reference_or_copy",
		project_must_be_self_contained: true,
		write_scope: "project",
		component_scope: components,
		code_artifacts: [
			{
				path: "method.py",
				artifact_type: "python",
				symbols: ["configuration", "component_enabled"],
				responsibility: "Resolve method configuration and component toggles",
				dependencies: [],
				config_keys: ["XLAB_DISABLED_COMPONENTS"],
				entrypoint_role: "scientific_configuration",
			},
			{
				path: "recipe/train.py",
				artifact_type: "python",
				symbols: ["main"],
				responsibility: "Full Zipformer training",
				dependencies: ["method.py"],
				config_keys: [],
				entrypoint_role: "training",
			},
		],
		interface_contract: {
			configuration: "method.configuration(disabled_components) returns train_args, decode_args, decoding_method",
			toggles:
				"method.component_enabled(name) reads XLAB_DISABLED_COMPONENTS; every component must have a meaningful disabled implementation",
		},
		implementation_requirements: {
			components,
			baseline: components.length === 0,
			no_smoke: true,
			fixed_training: config.training,
		},
		experiment_bindings: { trainer: "recipe/train.py", decoder: "recipe/decode.py", scorer: config.pipeline_id },
		component_disable_hooks: components.map((component) => ({
			component,
			interface: `component_enabled(${JSON.stringify(component)})`,
			environment: "XLAB_DISABLED_COMPONENTS",
		})),
		verify_command: "experiment_execute",
		artifact_ids: [`code.${id}.handoff`],
		evidence: ["syntax", "component_contract"],
		done_condition: `Use experiment_execute managed artifact tools and preserve artifact ledger proof for code.${id}.handoff; runtime dataset and metric evidence is deferred to the full science conditions.`,
	});
	const code: ExperimentPlan = {
		stage: "code",
		work_units: components.length
			? [codeUnit("implementation", []), codeUnit("final_static_integration", ["implementation"])]
			: [codeUnit("final_static_integration", [])],
	};
	const conditions = [
		{ id: "all-components", disabled: [] as string[] },
		...components.map((name, index) => ({
			id: `without-${index + 1}-${canonicalDigest(name).slice(0, 8)}`,
			disabled: [name],
		})),
	];
	const science: ExperimentPlan = {
		stage: "science",
		work_units: conditions.map(({ id, disabled }) => ({
			id,
			needs: [],
			kind: disabled.length ? "component_disabled" : "all_components_reference",
			goal: `Full ASR experiment ${id}`,
			full_run: true,
			run_level: "full",
			enabled_components: components.filter((name) => !disabled.includes(name)),
			disabled_components: disabled,
			...(disabled.length ? { disabled_component: disabled[0], reference_condition_id: "all-components" } : {}),
			output_dir: `results/science/${id}`,
			command: `python -P ${join(config.runtime, "asr_runtime/runner.py")} execute ${join(runDir, "project/results/science", id, "request.json")} # disabled_components=${JSON.stringify(disabled)}`,
			setup_rationale:
				"One full reference and one full disabled condition per actual canonical component; all train from scratch with the same frozen budget.",
			runtime_probe_summary: "No preliminary probe: runtime evidence comes from this formal full run.",
			source_basis: ["inputs/idea.json", "inputs/context.json", "project/preparation.json"],
			training_protocol: config.training,
			evaluation_protocol: { pipeline_id: config.pipeline_id, split: "regular", seed: 42 },
			train_dataset_binding: config.data,
			evaluation_dataset_bindings: [config.refs.regular],
			metric_bindings: [{ metric: "wer", scorer: config.pipeline_id }],
			component_set_description: { enabled: components.filter((name) => !disabled.includes(name)), disabled },
			result_interpretation_rule:
				"disabled benefit = all-components WER minus disabled WER; one seed supports observations, not statistical significance",
			raw_evidence: ["train.log", "decode_regular.log", "sure_regular/score.json", "result.json"],
			artifact_ids: [`science.${id}.evidence`],
			pass_condition: `Require finite SURE metric, complete 30-epoch checkpoint, log and evidence; managed artifact tools must publish science.${id}.evidence in the artifact ledger.`,
		})),
	};
	return { prepare, code, science };
}
