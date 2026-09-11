export type CommandKind = "product" | "control";
export type CommandGroup = "setup" | "synthesis" | "experiments" | "collaboration" | "workspaces" | "runs";

export interface CommandArgument {
  name: string;
  description: string;
  valueHint?: string;
  choices?: readonly string[];
  dynamicSource?: string;
  takesValue: boolean;
  repeatable?: boolean;
}

export interface XLabCommand {
  task: string;
  anchor: string;
  description: string;
  kind: CommandKind;
  group: CommandGroup;
  usage: string;
  example: string;
  result: string;
  sourcePath: string;
  skillName?: string;
  runtimeKind?: string;
  argumentHint?: string;
  arguments: readonly CommandArgument[];
}

const arg = (name: string, description: string, valueHint?: string, extra: Partial<CommandArgument> = {}): CommandArgument => ({
  name,
  description,
  ...(valueHint ? { valueHint } : {}),
  takesValue: valueHint !== undefined,
  ...extra,
});

const product = (
  task: string,
  description: string,
  group: CommandGroup,
  skillName: string,
  runtimeKind: string,
  argumentHint: string,
  args: readonly CommandArgument[],
  example: string,
  result: string,
): XLabCommand => ({
  task,
  anchor: `command-${task}`,
  description,
  kind: "product",
  group,
  usage: `/xlab ${task}${argumentHint ? ` ${argumentHint}` : ""} [--workspace <slug>]`,
  example,
  result,
  sourcePath: `xlab/skills/${skillName}/xlab.skill.json`,
  skillName,
  runtimeKind,
  argumentHint,
  arguments: args,
});

const control = (
  task: string,
  description: string,
  group: CommandGroup,
  usage: string,
  example: string,
  result: string,
): XLabCommand => ({
  task,
  anchor: `command-${task}`,
  description,
  kind: "control",
  group,
  usage,
  example,
  result,
  sourcePath: "README.md",
  arguments: [],
});

export const commands: readonly XLabCommand[] = [
  control("show-help", "Show all XLab commands or detailed help for one task.", "setup", "/xlab show-help [task]", "/xlab show-help", "Prints the command catalog or task-specific usage."),
  control("check-xlab-setup", "Check XLab configuration, packages, and required integrations.", "setup", "/xlab check-xlab-setup", "/xlab check-xlab-setup", "Reports setup checks and actionable configuration gaps."),
  control("configure-xlab", "Configure XLab runtime and research integrations.", "setup", "/xlab configure-xlab <runtime|knowledge-graph|scholarly-services|all>", "/xlab configure-xlab runtime", "Guides configuration for the selected integration area."),

  product("collect-papers", "Find, download, and organize research papers.", "synthesis", "paper_collect", "python", '"<topic>" [options]', [
    arg("--target-papers", "Set the target number of papers to collect.", "<count>"),
    arg("--max-papers", "Set the hard maximum paper count.", "<count>"),
    arg("--facet", "Add a topic facet; repeat for up to eight facets.", "<facet>", { repeatable: true }),
    arg("--download-workers", "Set concurrent PDF download workers.", "<count>"),
    arg("--download-per-host", "Set concurrent downloads allowed per host.", "<count>"),
  ], '/xlab collect-papers "graph neural networks after 2022" --target-papers 50', "Creates a persisted paper-set artifact with source records."),
  product("build-knowledge-graph", "Build a traceable method knowledge graph from papers.", "synthesis", "knowledge_graph", "python", '"<paper-collect-run>" [options]', [
    arg("--mineru-backend", "Select the MinerU execution backend.", "<backend>", { choices: ["auto", "pipeline", "vlm-engine", "hybrid-engine"] }),
    arg("--mineru-method", "Select automatic, text, or OCR parsing.", "<method>", { choices: ["auto", "txt", "ocr"] }),
    arg("--mineru-language", "Set the MinerU document language.", "<language>"),
    arg("--gpu", "Select a GPU index; repeat to select multiple devices.", "<index>", { dynamicSource: "gpu", repeatable: true }),
    arg("--mineru-workers", "Set MinerU worker count.", "<count>"),
    arg("--mineru-timeout", "Set the MinerU timeout in seconds.", "<seconds>"),
    arg("--min-gpu-free-gb", "Require minimum free GPU memory in GiB.", "<gb>"),
    arg("--max-gpu-utilization", "Set maximum allowed GPU utilization percent.", "<percent>"),
    arg("--structure-workers", "Set structure extraction workers.", "<count>"),
    arg("--llm-workers", "Set concurrent graph extraction LLM workers.", "<count>"),
    arg("--llm-timeout", "Set each graph extraction LLM timeout.", "<seconds>"),
    arg("--llm-retries", "Set graph extraction LLM retry count.", "<count>"),
    arg("--llm-validation-retries", "Set validation repair retry count.", "<count>"),
    arg("--llm-rpm", "Set the LLM request-per-minute limit.", "<count>"),
    arg("--llm-max-input-chars", "Bound characters sent for each paper extraction.", "<count>"),
    arg("--llm-max-output-tokens", "Bound graph extraction output tokens.", "<count>"),
    arg("--min-papers", "Set the minimum paper count audit gate.", "<count>"),
    arg("--min-nodes", "Set the minimum graph node audit gate.", "<count>"),
    arg("--min-edges", "Set the minimum graph edge audit gate.", "<count>"),
    arg("--min-mineru-coverage", "Set minimum MinerU coverage ratio.", "<ratio>"),
    arg("--min-structure-success", "Set minimum structure success ratio.", "<ratio>"),
    arg("--min-extraction-success", "Set minimum extraction success ratio.", "<ratio>"),
    arg("--min-core-coverage", "Set minimum core-method coverage ratio.", "<ratio>"),
    arg("--min-evidence-coverage", "Set minimum evidence coverage ratio.", "<ratio>"),
    arg("--desired-baseline-coverage", "Set desired baseline coverage ratio.", "<ratio>"),
    arg("--desired-dataset-coverage", "Set desired dataset coverage ratio.", "<ratio>"),
    arg("--max-ambiguous-alias-ratio", "Set maximum ambiguous alias ratio.", "<ratio>"),
  ], '/xlab build-knowledge-graph "run_7f2a"', "Creates a versioned graph artifact with evidence links and audit results."),
  product("write-literature-survey", "Write a cited literature survey from papers and graph evidence.", "synthesis", "literature_survey", "python", "<topic> [options]", [
    arg("--topic", "Set the survey topic explicitly; a quoted positional topic is preferred.", "<topic>"),
    arg("--graph", "Override the active workspace knowledge graph with a graph handle or path.", "<graph>", { dynamicSource: "graph" }),
    arg("--language", "Set the language recorded for the generated survey.", "<language>"),
    arg("--depth", "Set survey depth metadata and planning detail.", "<brief|standard|deep>", { choices: ["brief", "standard", "deep"] }),
    arg("--max-papers", "Set the maximum number of real papers to materialize.", "<count>"),
    arg("--min-papers", "Set the minimum real-paper count required by the audit.", "<count>"),
    arg("--facet", "Add a survey facet; repeat the flag to add more facets.", "<facet>", { repeatable: true }),
    arg("--full-text", "Request full-text synthesis when the runtime supports it."),
    arg("--resume", "Resume from compatible run-local survey state."),
    arg("--input", "Use a local paper manifest compatibility input instead of the active graph.", "<path>", { dynamicSource: "file" }),
    arg("--papers", "Compatibility alias for --input.", "<path>", { dynamicSource: "file" }),
    arg("--paper-set", "Compatibility alias for --input.", "<path>", { dynamicSource: "file" }),
  ], '/xlab write-literature-survey "robustness under distribution shift" --depth deep', "Creates a cited survey and citation-trace artifacts for acceptance review."),
  product("generate-research-ideas", "Generate evidence-grounded, experiment-ready research ideas.", "synthesis", "research_idea", "python", "[--survey <handle-or-path>] [options]", [
    arg("--survey", "Use a survey workspace handle or artifact path; defaults to the current workspace survey.", "<survey>", { dynamicSource: "survey" }),
    arg("--topic", "Override the topic inferred from the survey.", "<topic>"),
    arg("--mature-idea", "Anchor contract-mode refinement to an existing mature idea.", "<idea>"),
    arg("--refinement-scope", "Provide free-text guidance for the desired refinement; this is not an exact component-name allowlist.", "<scope>"),
    arg("--discussion", "Add scientific discussion context.", "<text>"),
    arg("--experiment-feedback", "Add experiment feedback and trigger replanning; --mature-idea is required.", "<text>"),
  ], "/xlab generate-research-ideas --survey survey-v1", "Creates evidence-linked research-idea artifacts and experiment plans."),
  product("check-idea-novelty", "Check an idea for prior work, overlap, and uncertainty.", "synthesis", "novelty_check", "python", "<idea and evidence context> [options]", [
    arg("--idea", "Provide a research idea artifact path or handle.", "<idea>", { dynamicSource: "file" }),
    arg("--papers", "Provide a paper-set evidence artifact path or handle.", "<papers>", { dynamicSource: "file" }),
    arg("--graph", "Provide a knowledge-graph evidence artifact path or handle.", "<graph>", { dynamicSource: "file" }),
  ], "/xlab check-idea-novelty --idea idea-v1 --graph graph-v1", "Creates an overlap-risk report with prior-work evidence and uncertainty."),
  product("run-research-workflow", "Run the complete papers-to-novelty research workflow.", "synthesis", "research_workflow", "workflow", "<research topic and workflow constraints>", [], '/xlab run-research-workflow "robust learning under distribution shift"', "Runs the staged papers-to-novelty workflow and persists each accepted artifact."),

  product("build-research-dataset", "Build and validate a provenance-preserving research dataset.", "experiments", "data_workflow", "workflow", "<dataset goal, sources, processing, and validation criteria>", [], '/xlab build-research-dataset "Create a labeled robustness benchmark from approved sources"', "Creates a validated dataset artifact with provenance records."),
  product("run-experiment", "Prepare, execute, and review a research experiment.", "experiments", "run_experiment", "native", "--idea <idea reference>", [
    arg("--idea", "Select an existing successful Research Idea by path, artifact ID, producer run, workspace handle, current workspace Idea, or exact indexed title.", "<idea-reference>", { dynamicSource: "idea" }),
  ], "/xlab run-experiment --idea idea-v1", "Creates an experiment run with plans, execution state, metrics, and review artifacts."),
  product("evaluate-model", "Evaluate a model on a frozen dataset and metric contract.", "experiments", "model_eval", "workflow", "<model, dataset, metrics, and optional baseline>", [], '/xlab evaluate-model "model=checkpoint.pt dataset=benchmark-v1 metrics=accuracy,f1"', "Creates an evaluation artifact against the declared dataset and metric contract."),
  product("create-model-mcp-server", "Package and validate a model as an MCP server.", "experiments", "model_mcp", "workflow", "<runner, schema, example, and registration request> [--approve]", [
    arg("--approve", "Explicitly approve optional MCP registration after validation."),
  ], '/xlab create-model-mcp-server "runner=model.py schema=schema.json example=example.json"', "Creates and validates an MCP server package; registration requires explicit approval."),
  product("reproduce-paper", "Reproduce a paper with official code and explicit metrics.", "experiments", "paper_reproduction", "workflow", "<paper, official code revision, experiment, metrics, and resource budget>", [], '/xlab reproduce-paper "paper=doi:10.x/code revision=abc123 metric=accuracy budget=1gpu-8h"', "Creates a reproduction run with pinned inputs, metrics, and comparison results."),

  product("conduct-research-discussion", "Conduct a structured, evidence-linked research discussion.", "collaboration", "research_discussion", "python", "<topic, goal, roles, round limit, and source artifacts>", [], '/xlab conduct-research-discussion "topic=robustness goal=identify gaps rounds=3"', "Creates a structured discussion artifact linked to its source evidence."),
  product("build-scholar-profile", "Build an evidence-backed profile of a researcher.", "collaboration", "scholar_profile", "python", "<scholar name> [key=value options]", [
    arg("scholar_name", "Set the scholar name explicitly.", "<name>"),
    arg("native_name", "Set the scholar's native-language name.", "<name>"),
    arg("seed_url", "Provide an optional seed profile or homepage URL.", "<url>"),
    arg("language", "Select the output language.", "<en|zh>", { choices: ["en", "zh"] }),
  ], '/xlab build-scholar-profile "Geoffrey Hinton" language=en', "Creates a profile whose claims link back to collected evidence."),
  product("manage-research-communications", "Manage local research threads, messages, and synchronization.", "collaboration", "science_gateway", "service-control", "<gateway operation and communication context>", [], '/xlab manage-research-communications "list local threads"', "Updates or reports local research communication state."),

  control("list-workspaces", "List available XLab research workspaces.", "workspaces", "/xlab list-workspaces", "/xlab list-workspaces", "Lists workspace slugs and active state."),
  control("show-workspace", "Show details for an XLab workspace.", "workspaces", "/xlab show-workspace <slug>", "/xlab show-workspace <slug>", "Shows workspace handles, lineage, and stale-dependency warnings."),
  control("select-workspace", "Select the active XLab workspace.", "workspaces", "/xlab select-workspace <slug>", "/xlab select-workspace <slug>", "Sets the active workspace used by subsequent commands."),

  control("list-runs", "List recent XLab runs.", "runs", "/xlab list-runs", "/xlab list-runs", "Lists recent run identifiers and states."),
  control("show-run", "Show the state and workflow of an XLab run.", "runs", "/xlab show-run <run-id>", "/xlab show-run <run-id>", "Shows phases, counters, blockers, artifacts, checkpoints, and next actions."),
  control("resume-run", "Resume an incomplete XLab run.", "runs", "/xlab resume-run <run-id>", "/xlab resume-run <run-id>", "Continues an eligible run from persisted state."),
  control("cancel-run", "Cancel an XLab run.", "runs", "/xlab cancel-run <run-id>", "/xlab cancel-run <run-id>", "Records the run as cancelled."),
  control("show-artifact", "Show metadata and lineage for an XLab artifact.", "runs", "/xlab show-artifact <artifact-id>", "/xlab show-artifact <artifact-id>", "Shows artifact metadata, validation, producer, and parent relationships."),
] as const;

export const commandGroups: readonly { id: CommandGroup; label: string; description: string }[] = [
  { id: "setup", label: "Setup and help", description: "Inspect and configure the local XLab installation." },
  { id: "synthesis", label: "Discovery and synthesis", description: "Move from collected papers through evidence, surveys, ideas, and overlap analysis." },
  { id: "experiments", label: "Data, experiments, and models", description: "Build datasets, execute experiments, evaluate models, and package validated runners." },
  { id: "collaboration", label: "Collaboration and profiles", description: "Create evidence-linked discussions, profiles, and local communication state." },
  { id: "workspaces", label: "Workspaces", description: "Select and inspect versioned research context." },
  { id: "runs", label: "Runs and artifacts", description: "Inspect, resume, cancel, and trace persisted work." },
] as const;

export const commandText = (command: XLabCommand): string => command.example;
