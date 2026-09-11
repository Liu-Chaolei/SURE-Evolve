import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { basename, isAbsolute, join, relative, resolve } from "node:path";
import type {
  XlabHookContext,
  XlabHookResult,
} from "@earendil-works/pi-coding-agent";

const RESEARCH_IDEA_TYPE = "research_idea";
const IDEA_RESULT_TYPE = "research_idea_result";
const TRACE_TYPE = "research_idea_trace";
const REPORT_TYPE = "research_idea_report";
const RESEARCH_IDEA_ALGORITHM_ID = "xlab.research_idea.algorithm.v2";

const NATIVE_RUNTIME_FILES = [
  "scripts/run_idea_phase.py",
  "scripts/research_idea_lib/common.py",
  "scripts/research_idea_lib/inputs.py",
  "scripts/research_idea_lib/config.py",
  "scripts/research_idea_lib/checkpoints.py",
  "scripts/research_idea_lib/survey_repository.py",
  "scripts/research_idea_lib/artifact_mapping.py",
  "scripts/research_idea_lib/pipeline.py",
  "scripts/research_idea_lib/audit.py",
  "scripts/research_idea_lib/manifest.py",
  "scripts/research_idea_lib/llm.py",
  "scripts/research_idea_lib/research_idea_artifacts.py",
  "scripts/research_idea_lib/research_idea_spec.py",
  "scripts/research_idea_lib/algorithm/__init__.py",
  "scripts/research_idea_lib/algorithm/contracts.py",
  "scripts/research_idea_lib/algorithm/defects.py",
  "scripts/research_idea_lib/algorithm/evaluation.py",
  "scripts/research_idea_lib/algorithm/fusion.py",
  "scripts/research_idea_lib/algorithm/memory.py",
  "scripts/research_idea_lib/algorithm/operators.py",
  "scripts/research_idea_lib/algorithm/provider_adapter.py",
  "scripts/research_idea_lib/algorithm/runtime_adapters.py",
  "scripts/research_idea_lib/algorithm/search.py",
  "scripts/research_idea_lib/algorithm/skills.py",
  "scripts/research_idea_lib/algorithm/spec.py",
  "scripts/research_idea_lib/algorithm/tastes.py",
  "scripts/research_idea_lib/algorithm/workflow.py",
  "scripts/research_idea_lib/providers/__init__.py",
  "scripts/research_idea_lib/providers/contracts.py",
  "scripts/research_idea_lib/providers/fake.py",
  "scripts/research_idea_lib/providers/openai_compatible.py",
  "scripts/research_idea_lib/resources/__init__.py",
  "scripts/research_idea_lib/resources/embedding.py",
  "scripts/research_idea_lib/resources/evidence.py",
  "scripts/research_idea_lib/resources/ids.py",
  "scripts/research_idea_lib/resources/manifest.py",
  "scripts/research_idea_lib/resources/model_cache.py",
  "scripts/research_idea_lib/resources/resolver.py",
] as const;

const NATIVE_PROMPT_FILES = [
  "advanced_analysis.py",
  "algorithm_alignment.py",
  "algorithm_structuring.py",
  "component_extraction.py",
  "component_novelty_evaluation.py",
  "experiment_findings_extraction.py",
  "idea_fusion.py",
  "idea_introduction.py",
  "idea_result_alignment.py",
  "keynote_ops.py",
  "mcts_evaluation.py",
  "mechanism_commit_query.py",
  "prompt_modes.py",
  "rag_query.py",
  "re_analysis_replan.py",
  "reference_grounding.py",
  "root_domain_classification.py",
  "skill_instantiation.py",
  "theory_transfer_query.py",
  "topic_background.py",
] as const;

const NATIVE_OPERATOR_RESOURCES = [
  ["alternative-path-contrast", "contrast_patterns.md"],
  ["feedback-closed-loop", "mechanism_cards.md"],
  ["hierarchical-decomposition", "hierarchy_patterns.md"],
  ["mechanism-commit-innovation", "mechanism_commit_patterns.md"],
  ["multi-scale-coordinator", "coordination_patterns.md"],
  ["speculative-execution-with-repair", "speculation_patterns.md"],
  ["surgical-modularity", "modularity_patterns.md"],
  ["theory-transfer-injection", "transfer_patterns.md"],
] as const;

interface ManifestArtifact {
  type?: string;
  path?: string;
}

interface FinishEvent {
  finish?: {
    status?: string;
  };
  manifest?: {
    status?: string;
    artifacts?: ManifestArtifact[];
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numeric(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function failure(
  repair: string,
  message: string,
  counters?: Record<string, number>,
): XlabHookResult {
  return {
    ok: false,
    repair,
    state_patch: {
      phase: {
        id: "validate",
        label: "Validating research idea",
        status: "blocked",
      },
      message,
      counters,
      diagnostics: [{ severity: "error", message, repair }],
    },
  };
}

function paths(ctx: XlabHookContext): {
  script: string;
  artifacts: string;
  state: string;
  checkpoint: string;
  logs: string;
  ideaResult: string;
  researchIdea: string;
  trace: string;
  report: string;
  manifest: string;
} {
  const artifacts = join(ctx.runDir, "artifacts");
  const state = join(ctx.runDir, "state");
  return {
    script: join(ctx.packageDir, "scripts", "run_idea_phase.py"),
    artifacts,
    state,
    checkpoint: join(state, "checkpoint.json"),
    logs: join(ctx.runDir, "logs"),
    ideaResult: join(artifacts, "idea_result.json"),
    researchIdea: join(artifacts, "research_idea.json"),
    trace: join(artifacts, "idea_trace.json"),
    report: join(artifacts, "idea_report.json"),
    manifest: join(ctx.runDir, "manifest.json"),
  };
}

function readJson(path: string): Record<string, unknown> {
  const value: unknown = JSON.parse(readFileSync(path, "utf-8"));
  if (!isRecord(value)) {
    throw new Error(`${path} must contain a JSON object.`);
  }
  return value;
}

function eventValue(ctx: XlabHookContext): Record<string, unknown> {
  return isRecord(ctx.event) ? ctx.event : {};
}

function finishEvent(ctx: XlabHookContext): FinishEvent {
  return eventValue(ctx) as FinishEvent;
}

function runRelativePath(ctx: XlabHookContext, path: string): string {
  return relative(ctx.cwd, join(ctx.runDir, path)).replace(/\\/g, "/");
}

function resolveProjectOrRunPath(ctx: XlabHookContext, path: string): string {
  if (isAbsolute(path)) {
    return resolve(path);
  }
  const runPath = resolve(ctx.runDir, path);
  if (existsSync(runPath)) {
    return runPath;
  }
  return resolve(ctx.cwd, path);
}

function artifactPathFromManifest(
  ctx: XlabHookContext,
  artifacts: ManifestArtifact[],
  type: string,
  fallback: string,
): string {
  const artifact = artifacts.find((entry) => entry.type === type);
  if (artifact?.path) {
    return resolveProjectOrRunPath(ctx, artifact.path);
  }
  return join(ctx.runDir, fallback);
}

function isWithin(path: string, root: string): boolean {
  const relativePath = relative(resolve(root), resolve(path));
  return (
    relativePath === "" ||
    (!relativePath.startsWith("..") && !isAbsolute(relativePath))
  );
}

interface ParsedCommand {
  executable: string;
  args: string[];
}

interface PackagedCliCommand extends ParsedCommand {
  phase: "init" | "synthesize" | "resume" | "audit";
}

const READ_ONLY_INSPECTION_TOOLS = new Set([
  "jq",
  "ls",
  "find",
  "rg",
  "grep",
  "test",
  "[",
]);
const FIND_MUTATING_PREDICATES = [
  "-delete",
  "-exec",
  "-execdir",
  "-ok",
  "-okdir",
  "-fprint",
  "-fprint0",
  "-fprintf",
  "-fls",
] as const;

function splitCommand(command: string): string[] | undefined {
  const tokens: string[] = [];
  let current = "";
  let quote: '"' | "'" | undefined;
  let escaping = false;
  for (const character of command.trim()) {
    if (escaping) {
      current += character;
      escaping = false;
      continue;
    }
    if (character === "\\" && quote !== "'") {
      escaping = true;
      continue;
    }
    if (quote) {
      if (character === quote) {
        quote = undefined;
      } else {
        current += character;
      }
      continue;
    }
    if (character === '"' || character === "'") {
      quote = character;
      continue;
    }
    if (/\s/.test(character)) {
      if (current) {
        tokens.push(current);
        current = "";
      }
      continue;
    }
    current += character;
  }
  if (escaping || quote) {
    return undefined;
  }
  if (current) {
    tokens.push(current);
  }
  return tokens;
}

function hasShellControl(command: string): boolean {
  let quote: '"' | "'" | undefined;
  let escaping = false;
  for (const character of command) {
    if (escaping) {
      escaping = false;
      continue;
    }
    if (character === "\\" && quote !== "'") {
      escaping = true;
      continue;
    }
    if (quote) {
      if (character === quote) {
        quote = undefined;
      }
      continue;
    }
    if (character === '"' || character === "'") {
      quote = character;
      continue;
    }
    if (";&|<>`$\n\r".includes(character)) {
      return true;
    }
  }
  return escaping || quote !== undefined;
}

function parseSimpleCommand(command: string): ParsedCommand | undefined {
  if (hasShellControl(command)) {
    return undefined;
  }
  const tokens = splitCommand(command);
  if (!tokens || tokens.length === 0) {
    return undefined;
  }
  return { executable: tokens[0], args: tokens.slice(1) };
}

function isPythonExecutable(value: string): boolean {
  return /^(?:python|python3(?:\.\d+)?)$/.test(basename(value));
}

function isAllowedPythonExecutable(
  executable: string,
  ctx: XlabHookContext,
): boolean {
  if (!isPythonExecutable(executable)) {
    return false;
  }
  if (basename(executable) === executable) {
    return true;
  }
  const configured = ctx.run.environment?.executable;
  return Boolean(
    configured &&
    isAbsolute(executable) &&
    resolve(executable) === resolve(configured),
  );
}

function isPackagedScriptPath(value: string, ctx: XlabHookContext): boolean {
  const candidates = isAbsolute(value)
    ? [resolve(value)]
    : [resolve(ctx.cwd, value), resolve(ctx.packageDir, value)];
  return candidates.some(
    (candidate) => candidate === resolve(paths(ctx).script),
  );
}

function optionValues(
  args: string[],
  allowed: ReadonlySet<string>,
): Map<string, string> | undefined {
  const options = new Map<string, string>();
  for (let index = 0; index < args.length; index += 1) {
    const token = args[index];
    const separator = token.indexOf("=");
    const name = separator >= 0 ? token.slice(0, separator) : token;
    if (!name.startsWith("--") || !allowed.has(name) || options.has(name)) {
      return undefined;
    }
    const value = separator >= 0 ? token.slice(separator + 1) : args[index + 1];
    if (!value || (separator < 0 && value.startsWith("--"))) {
      return undefined;
    }
    options.set(name, value);
    if (separator < 0) {
      index += 1;
    }
  }
  return options;
}

function resolveCommandPath(value: string, ctx: XlabHookContext): string {
  return isAbsolute(value) ? resolve(value) : resolve(ctx.cwd, value);
}

function parsePackagedCliCommand(
  command: string,
  ctx: XlabHookContext,
): PackagedCliCommand | undefined {
  const parsed = parseSimpleCommand(command);
  if (!parsed) {
    return undefined;
  }
  let script: string;
  let args: string[];
  if (isAllowedPythonExecutable(parsed.executable, ctx)) {
    [script, ...args] = parsed.args;
  } else {
    script = parsed.executable;
    args = parsed.args;
  }
  if (!script || !isPackagedScriptPath(script, ctx)) {
    return undefined;
  }
  const [phase, ...phaseArgs] = args;
  if (
    phase !== "init" &&
    phase !== "synthesize" &&
    phase !== "resume" &&
    phase !== "audit"
  ) {
    return undefined;
  }
  const allowedOptions = new Set(["--run-dir", "--run-id", "--cwd"]);
  if (phase === "init" || phase === "synthesize") {
    allowedOptions.add("--arguments");
  }
  if (phase !== "init") {
    allowedOptions.add("--status");
  }
  const options = optionValues(phaseArgs, allowedOptions);
  if (
    !options ||
    options.get("--run-id") !== ctx.run.runId ||
    !options.has("--run-dir") ||
    resolveCommandPath(options.get("--run-dir") ?? "", ctx) !==
      resolve(ctx.runDir)
  ) {
    return undefined;
  }
  const cwd = options.get("--cwd");
  if (cwd && resolveCommandPath(cwd, ctx) !== resolve(ctx.cwd)) {
    return undefined;
  }
  const status = options.get("--status");
  if (status && status !== "success" && status !== "incomplete") {
    return undefined;
  }
  return { ...parsed, phase };
}

function isAllowedPackagedCli(command: string, ctx: XlabHookContext): boolean {
  return parsePackagedCliCommand(command, ctx) !== undefined;
}

function hasUnsafePathSyntax(value: string): boolean {
  return (
    !value ||
    value.includes("$") ||
    value.startsWith("~") ||
    /[*?[\]{}]/.test(value)
  );
}

function isScopedInspectionPath(value: string, ctx: XlabHookContext): boolean {
  if (hasUnsafePathSyntax(value)) {
    return false;
  }
  const resolved = resolveCommandPath(value, ctx);
  return [ctx.runDir, ctx.packageDir].some((root) => isWithin(resolved, root));
}

function nonOptionOperands(args: string[]): string[] {
  const operands: string[] = [];
  let afterOptions = false;
  for (const value of args) {
    if (!afterOptions && value === "--") {
      afterOptions = true;
      continue;
    }
    if (!afterOptions && value.startsWith("-") && value !== "-") {
      continue;
    }
    if (value !== "]") {
      operands.push(value);
    }
  }
  return operands;
}

function leadingFindPaths(args: string[]): string[] {
  const roots: string[] = [];
  for (const value of args) {
    if (value === "--") {
      continue;
    }
    if (
      value.startsWith("-") ||
      value === "!" ||
      value === "(" ||
      value === ")"
    ) {
      break;
    }
    roots.push(value);
  }
  return roots;
}

function allPathsAreScoped(values: string[], ctx: XlabHookContext): boolean {
  return (
    values.length > 0 &&
    values.every((value) => isScopedInspectionPath(value, ctx))
  );
}

function isRunOrPackageInspection(
  command: string,
  ctx: XlabHookContext,
): boolean {
  const parsed = parseSimpleCommand(command);
  if (!parsed) {
    return false;
  }
  const executable = basename(parsed.executable);
  const pythonJsonTool =
    isAllowedPythonExecutable(parsed.executable, ctx) &&
    parsed.args[0] === "-m" &&
    parsed.args[1] === "json.tool";
  if (
    (!pythonJsonTool &&
      (!READ_ONLY_INSPECTION_TOOLS.has(executable) ||
        executable !== parsed.executable)) ||
    parsed.args.some((value) => value.includes("$") || value.startsWith("~"))
  ) {
    return false;
  }
  if (pythonJsonTool) {
    return allPathsAreScoped(nonOptionOperands(parsed.args.slice(2)), ctx);
  }
  if (executable === "find") {
    if (
      parsed.args.some((value) =>
        FIND_MUTATING_PREDICATES.some(
          (predicate) =>
            value === predicate || value.startsWith(`${predicate}=`),
        ),
      )
    ) {
      return false;
    }
    return allPathsAreScoped(leadingFindPaths(parsed.args), ctx);
  }
  if (executable === "rg" || executable === "grep") {
    if (
      parsed.args.some(
        (value) =>
          value === "--pre" ||
          value.startsWith("--pre=") ||
          value === "--pre-glob" ||
          value.startsWith("--pre-glob="),
      )
    ) {
      return false;
    }
    const operands = nonOptionOperands(parsed.args);
    const patternFromFile =
      parsed.args.includes("-f") || parsed.args.includes("--file");
    return allPathsAreScoped(
      patternFromFile ? operands : operands.slice(1),
      ctx,
    );
  }
  if (executable === "jq") {
    const operands = nonOptionOperands(parsed.args);
    const filterFromFile =
      parsed.args.includes("-f") || parsed.args.includes("--from-file");
    return allPathsAreScoped(
      filterFromFile ? operands : operands.slice(1),
      ctx,
    );
  }
  return allPathsAreScoped(nonOptionOperands(parsed.args), ctx);
}

function mutationTargets(input: Record<string, unknown>): string[] {
  return [input.file_path, input.path, input.notebook_path].filter(
    (value): value is string =>
      typeof value === "string" && value.trim() !== "",
  );
}

function mutationsAreRunScoped(
  input: Record<string, unknown>,
  ctx: XlabHookContext,
): boolean {
  const targets = mutationTargets(input);
  return (
    targets.length > 0 &&
    targets.every(
      (target) =>
        !hasUnsafePathSyntax(target) &&
        isWithin(resolveCommandPath(target, ctx), ctx.runDir),
    )
  );
}

function legalIncomplete(
  ctx: XlabHookContext,
  event: Record<string, unknown>,
): boolean {
  if (
    event.isError !== true ||
    !existsSync(paths(ctx).report) ||
    !existsSync(paths(ctx).manifest)
  ) {
    return false;
  }
  try {
    const report = readJson(paths(ctx).report);
    const manifest = readJson(paths(ctx).manifest);
    return report.passed !== true && manifest.status === "incomplete";
  } catch {
    return false;
  }
}

function diagnosticsFromReport(
  report: Record<string, unknown>,
  repair: string,
): Array<{ severity: "warning" | "error"; message: string; repair?: string }> {
  const warnings = Array.isArray(report.warnings)
    ? report.warnings.filter(
        (value): value is string => typeof value === "string",
      )
    : [];
  const blockers = Array.isArray(report.blocking_errors)
    ? report.blocking_errors.filter(
        (value): value is string => typeof value === "string",
      )
    : [];
  return [
    ...warnings.map((message) => ({ severity: "warning" as const, message })),
    ...blockers.map((message) => ({
      severity: "error" as const,
      message,
      repair,
    })),
  ];
}

function countersFromReport(
  report: Record<string, unknown>,
): Record<string, number> {
  const counts = isRecord(report.counts) ? report.counts : {};
  const warnings = Array.isArray(report.warnings) ? report.warnings.length : 0;
  return {
    references: numeric(counts.references) ?? 0,
    evidence: numeric(counts.evidence) ?? 0,
    rag_hits: numeric(counts.rag_hits) ?? 0,
    components: numeric(counts.components) ?? 0,
    candidates: numeric(counts.candidates) ?? 0,
    mcts_iterations: numeric(counts.mcts_iterations) ?? 0,
    warnings,
  };
}

function cliPhase(
  command: string,
  ctx: XlabHookContext,
): "init" | "synthesize" | "resume" | "audit" | "other" {
  return parsePackagedCliCommand(command, ctx)?.phase ?? "other";
}

function runPython(ctx: XlabHookContext, args: string[], timeout: number) {
  const python = ctx.run.environment?.executable ?? "python";
  return spawnSync(python, [paths(ctx).script, ...args], {
    cwd: ctx.packageDir,
    encoding: "utf-8",
    maxBuffer: 10 * 1024 * 1024,
    timeout,
  });
}

export function preStart(ctx: XlabHookContext): XlabHookResult {
  const promptRoot = join(
    ctx.packageDir,
    "scripts",
    "research_idea_lib",
    "algorithm",
    "prompts",
  );
  const operatorRoot = join(
    ctx.packageDir,
    "scripts",
    "research_idea_lib",
    "algorithm",
    "resources",
  );
  const required = [
    ...NATIVE_RUNTIME_FILES.map((path) => join(ctx.packageDir, path)),
    join(promptRoot, "__init__.py"),
    ...NATIVE_PROMPT_FILES.map((path) => join(promptRoot, path)),
    join(operatorRoot, "DEFAULT_SKILL_TEMPLATES.json"),
    ...NATIVE_OPERATOR_RESOURCES.flatMap(([operator, reference]) => [
      join(operatorRoot, "edit_operator_skills", operator, "SKILL.md"),
      join(
        operatorRoot,
        "edit_operator_skills",
        operator,
        "references",
        reference,
      ),
    ]),
    join(ctx.packageDir, "schemas", "request.schema.json"),
    join(ctx.packageDir, "schemas", "idea.schema.json"),
    join(ctx.packageDir, "schemas", "idea_result.schema.json"),
    join(ctx.packageDir, "schemas", "idea_trace.schema.json"),
    join(ctx.packageDir, "schemas", "idea_report.schema.json"),
  ];
  const missing = required.filter((path) => !existsSync(path));
  if (missing.length > 0) {
    return failure(
      "Restore the missing package-native runtime and resource files inside the research_idea skill package.",
      `Research idea package-native resource preflight failed: ${missing.join(", ")}`,
    );
  }
  for (const directory of [
    paths(ctx).artifacts,
    paths(ctx).state,
    paths(ctx).logs,
  ]) {
    mkdirSync(directory, { recursive: true });
  }
  const initialization = runPython(
    ctx,
    [
      "init",
      "--arguments",
      ctx.args,
      "--run-id",
      ctx.run.runId,
      "--run-dir",
      ctx.runDir,
      "--cwd",
      ctx.cwd,
    ],
    30_000,
  );
  if (initialization.error || initialization.status !== 0) {
    const detail =
      initialization.error?.message ||
      initialization.stderr.trim() ||
      initialization.stdout.trim() ||
      `exit ${initialization.status}`;
    return failure(
      'Use /xlab generate-research-ideas --survey <survey.json | literature_survey run dir | manifest.json> [--topic "..."] [--mature-idea "..."] [--refinement-scope "..."] [--discussion "..."] [--experiment-feedback "..."].',
      `Invalid research idea arguments: ${detail}`,
    );
  }
  let initialized: Record<string, unknown>;
  try {
    const value: unknown = JSON.parse(initialization.stdout);
    if (!isRecord(value)) {
      throw new Error("initialization output must be a JSON object");
    }
    initialized = value;
  } catch (error) {
    return failure(
      "Run run_idea_phase.py init manually and inspect its JSON output.",
      `Research idea initialization returned invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  const warnings = Array.isArray(initialized.warnings)
    ? initialized.warnings.filter(
        (value): value is string => typeof value === "string",
      )
    : [];
  const api = isRecord(initialized.api) ? initialized.api : {};
  const mcts = isRecord(initialized.mcts) ? initialized.mcts : {};
  const modes = Array.isArray(mcts.idea_taste_modes)
    ? mcts.idea_taste_modes.length
    : 0;
  const apiDiagnostic = {
    severity: "info",
    message: `XLab research-idea workflow API: ${String(api.generation_model ?? "unknown")} via ${String(api.openai_base_url ?? "default")}; algorithm ${String(api.algorithm_spec ?? RESEARCH_IDEA_ALGORITHM_ID)}.`,
  } as const;
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "init",
        label: "Research idea initialized",
        status: "running",
      },
      message: `Initialized research idea generation for ${String(initialized.topic || "survey topic")}.`,
      counters: {
        references: 0,
        rag_hits: 0,
        components: 0,
        candidates: 0,
        mcts_iterations: 0,
        warnings: warnings.length,
      },
      diagnostics: [
        apiDiagnostic,
        {
          severity: "info",
          message: `Package-native ${String(api.algorithm_spec ?? RESEARCH_IDEA_ALGORITHM_ID)} structural checks passed for ${modes} idea taste modes and declared runtime resources; this does not assert Xcientist-2 runtime identity.`,
        },
        ...warnings.map((message) => ({
          severity: "warning" as const,
          message,
        })),
      ],
      checkpoint: {
        id: "research-idea-start",
        label: "Idea generation initialized",
        resumable: true,
        resume_hint:
          "Rerun run_idea_phase.py resume with the same run directory.",
      },
      next_actions: [
        "Run run_idea_phase.py synthesize",
        "Resume with run_idea_phase.py resume if interrupted",
        "Finish with manifest.json",
      ],
    },
  };
}

export function preToolCall(ctx: XlabHookContext): XlabHookResult {
  const event = eventValue(ctx);
  const toolName = typeof event.toolName === "string" ? event.toolName : "";
  const input = isRecord(event.input) ? event.input : {};
  if (toolName.startsWith("mcp__")) {
    return failure(
      "Use the packaged run_idea_phase.py CLI so idea artifacts, checkpoints, and diagnostics remain consistent.",
      "research_idea does not call MCP tools directly.",
    );
  }
  if (toolName !== "bash") {
    if (toolName === "read") {
      const rawPath = input.file_path ?? input.path;
      const path = typeof rawPath === "string" ? rawPath : "";
      if (
        path &&
        !hasUnsafePathSyntax(path) &&
        [ctx.runDir, ctx.packageDir].some((root) =>
          isWithin(resolveCommandPath(path, ctx), root),
        )
      ) {
        return { ok: true };
      }
      return failure(
        "Read only files inside the active run directory or research_idea package.",
        "Blocked read outside the research_idea run/package scope.",
      );
    }
    if (toolName === "write" || toolName === "edit") {
      if (mutationsAreRunScoped(input, ctx)) {
        return { ok: true };
      }
      return failure(
        "Write or edit only structurally contained paths inside the active research_idea run directory; include a file_path or path target.",
        "Blocked a non-Bash mutation without a safe run-owned target.",
      );
    }
    return failure(
      "Use read-only inspection, run-scoped write/edit, or the packaged run_idea_phase.py CLI.",
      `research_idea does not allow the ${toolName || "unknown"} tool during a run.`,
    );
  }
  const rawCommand = input.command ?? input.cmd;
  const command = typeof rawCommand === "string" ? rawCommand : "";
  if (
    /\b(?:curl|wget)\b|api\.openai\.com|chat\/completions|api\.xi-ai\.cn/i.test(
      command,
    )
  ) {
    return failure(
      "Use run_idea_phase.py and declared artifacts; do not bypass the research_idea runtime with direct provider calls.",
      "Direct provider HTTP commands bypass the research_idea contract.",
    );
  }
  if (
    /--(?:source|runtime|agent)-root\b|\/src\/agents\/idea_agent\/run\.py\b|\bsrc\.agents\.idea_agent\b/i.test(
      command,
    )
  ) {
    return failure(
      "Use only the package-local research_idea runtime; do not invoke an external checkout.",
      "Blocked a command that references an external research-agent checkout.",
    );
  }
  if (
    command.includes("run_idea_phase.py") &&
    !isAllowedPackagedCli(command, ctx)
  ) {
    return failure(
      "Use only run_idea_phase.py init, synthesize, resume, or audit in the slash-command runtime.",
      "Blocked a non-production research_idea CLI phase.",
    );
  }
  if (
    /\b(?:python(?:3)?\s+-c|node\s+-e|tee|cp|mv|touch)\b|>/.test(command) &&
    /(?:artifacts\/(?:research_idea\.json|idea_result\.json|idea_trace\.json|idea_report\.json)|manifest\.json)/.test(
      command,
    )
  ) {
    return failure(
      "Produce final idea artifacts through run_idea_phase.py so checkpoint signatures, audit status, and manifest paths remain canonical.",
      "Blocked an ad hoc command that could overwrite research_idea final artifacts.",
    );
  }
  if (
    !isAllowedPackagedCli(command, ctx) &&
    !isRunOrPackageInspection(command, ctx)
  ) {
    return failure(
      "Run the packaged research_idea phase CLI, or use read-only inspection scoped to the run/package directory.",
      "Blocked a Bash command outside the research_idea control plane.",
    );
  }
  return { ok: true };
}

export function postToolResult(ctx: XlabHookContext): XlabHookResult {
  const event = eventValue(ctx);
  const toolName = typeof event.toolName === "string" ? event.toolName : "";
  if (toolName !== "bash") {
    return { ok: true };
  }
  const input = isRecord(event.input) ? event.input : {};
  const rawCommand = input.command ?? input.cmd;
  const command = typeof rawCommand === "string" ? rawCommand : "";
  if (!command.includes("run_idea_phase.py")) {
    return { ok: true };
  }
  const phase = cliPhase(command, ctx);
  const report = existsSync(paths(ctx).report)
    ? readJson(paths(ctx).report)
    : undefined;
  const counters = report ? countersFromReport(report) : undefined;
  if (event.isError === true) {
    if (legalIncomplete(ctx, event)) {
      const incompleteReport = readJson(paths(ctx).report);
      return {
        ok: true,
        state_patch: {
          phase: {
            id: phase,
            label: `Research idea ${phase} produced an incomplete manifest`,
            status: "incomplete",
          },
          message:
            "The package-native XLab research-idea workflow runtime completed and wrote an incomplete manifest; this is a recoverable quality-gate outcome, not a crash.",
          counters: countersFromReport(incompleteReport),
          diagnostics: diagnosticsFromReport(
            incompleteReport,
            "Repair the source survey, provider configuration, or package-native XLab research-idea workflow resources, then rerun run_idea_phase.py resume.",
          ),
          next_actions: [
            "Rerun run_idea_phase.py resume after repairs",
            "Or finish with status incomplete using manifest.json",
          ],
        },
      };
    }
    const serialized = JSON.stringify(event);
    const timedOut = /timed?\s*out|timeout/i.test(serialized);
    return {
      ok: true,
      state_patch: {
        phase: {
          id: phase,
          label: `Research idea ${phase} phase`,
          status: "blocked",
        },
        counters,
        diagnostics: [
          {
            severity: "warning",
            message: timedOut
              ? `The research idea ${phase} phase exceeded the tool timeout.`
              : `The research idea ${phase} phase returned an error before writing a valid incomplete manifest.`,
            repair:
              "Inspect logs and rerun run_idea_phase.py resume; completed stage files and checkpoints are preserved.",
          },
        ],
      },
    };
  }
  return {
    ok: true,
    state_patch: {
      phase: {
        id: phase,
        label:
          phase === "synthesize"
            ? "Research idea artifacts generated"
            : phase === "audit"
              ? "Research idea audit complete"
              : "Research idea phase complete",
        status: phase === "audit" ? "success" : "running",
      },
      counters,
    },
  };
}

export function preFinish(ctx: XlabHookContext): XlabHookResult {
  const event = finishEvent(ctx);
  const artifacts = event.manifest?.artifacts ?? [];
  const researchIdeaPath = artifactPathFromManifest(
    ctx,
    artifacts,
    RESEARCH_IDEA_TYPE,
    "artifacts/research_idea.json",
  );
  const ideaResultPath = artifactPathFromManifest(
    ctx,
    artifacts,
    IDEA_RESULT_TYPE,
    "artifacts/idea_result.json",
  );
  const tracePath = artifactPathFromManifest(
    ctx,
    artifacts,
    TRACE_TYPE,
    "artifacts/idea_trace.json",
  );
  const reportPath = artifactPathFromManifest(
    ctx,
    artifacts,
    REPORT_TYPE,
    "artifacts/idea_report.json",
  );
  for (const [type, path, canonical] of [
    [RESEARCH_IDEA_TYPE, researchIdeaPath, paths(ctx).researchIdea],
    [IDEA_RESULT_TYPE, ideaResultPath, paths(ctx).ideaResult],
    [TRACE_TYPE, tracePath, paths(ctx).trace],
    [REPORT_TYPE, reportPath, paths(ctx).report],
  ] as const) {
    if (!isWithin(path, ctx.cwd) && !isWithin(path, ctx.runDir)) {
      return failure(
        "Use the run-local artifact paths written by run_idea_phase.py.",
        `Artifact ${type} points outside the project/run workspace: ${path}`,
      );
    }
    if (resolve(path) !== resolve(canonical)) {
      return failure(
        "Use the canonical manifest generated by run_idea_phase.py audit before calling xlab_finish.",
        `Artifact ${type} must point to ${canonical}.`,
      );
    }
  }
  for (const [name, path] of [
    ["research_idea.json", researchIdeaPath],
    ["idea_result.json", ideaResultPath],
    ["idea_trace.json", tracePath],
    ["idea_report.json", reportPath],
  ] as const) {
    if (!existsSync(path)) {
      return failure(
        `Run run_idea_phase.py synthesize before calling xlab_finish. Missing ${name}.`,
        `Missing ${name}.`,
      );
    }
  }
  if (!existsSync(paths(ctx).manifest)) {
    return failure(
      "Run run_idea_phase.py audit before calling xlab_finish.",
      "Missing finalized manifest.json.",
    );
  }
  let report: Record<string, unknown>;
  let finalizedManifest: Record<string, unknown>;
  try {
    report = readJson(reportPath);
    finalizedManifest = readJson(paths(ctx).manifest);
    readJson(researchIdeaPath);
    readJson(ideaResultPath);
    readJson(tracePath);
  } catch (error) {
    return failure(
      "Repair the research idea JSON artifacts and rerun run_idea_phase.py audit.",
      `Cannot read research idea artifacts: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  const counters = countersFromReport(report);
  const passed = report.passed === true;
  const blockers = Array.isArray(report.blocking_errors)
    ? report.blocking_errors.filter(
        (value): value is string => typeof value === "string",
      )
    : [];
  const warnings = Array.isArray(report.warnings)
    ? report.warnings.filter(
        (value): value is string => typeof value === "string",
      )
    : [];
  const diagnostics = [
    ...warnings.map((message) => ({ severity: "warning" as const, message })),
    ...blockers.map((message) => ({
      severity: "error" as const,
      message,
      repair:
        "Rerun run_idea_phase.py resume after repairing the source survey or provider configuration.",
    })),
  ];
  const finishStatus = event.finish?.status ?? "";
  const generatedStatus = passed ? "success" : "incomplete";
  const manifestStatus =
    typeof finalizedManifest.status === "string"
      ? finalizedManifest.status
      : "";
  if (manifestStatus !== generatedStatus) {
    return failure(
      "Run run_idea_phase.py audit to finalize a manifest consistent with idea_report.json.",
      `Finalized manifest status ${manifestStatus || "<missing>"} does not match report status ${generatedStatus}.`,
      counters,
    );
  }
  if (finishStatus === "success" && !passed) {
    return failure(
      "Resolve the audit blockers, or finish with status incomplete using the generated manifest.",
      blockers.join(" ") || "Research idea did not pass the success gates.",
      counters,
    );
  }
  if (finishStatus && finishStatus !== generatedStatus) {
    return failure(
      `Call xlab_finish with status ${generatedStatus}; the finalized manifest declares that status.`,
      `xlab_finish status ${finishStatus} does not match finalized status ${generatedStatus}.`,
      counters,
    );
  }
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "validate",
        label: passed ? "Research idea validated" : "Research idea incomplete",
        status: passed ? "success" : "incomplete",
        progress: 1,
      },
      message: `Validated ${counters.references} references, ${counters.rag_hits} RAG hits, ${counters.components} components, and ${counters.mcts_iterations} MCTS iterations.`,
      counters,
      diagnostics,
      artifacts: [
        {
          type: RESEARCH_IDEA_TYPE,
          name: "Research idea",
          path: runRelativePath(ctx, "artifacts/research_idea.json"),
          status: passed ? "ready" : "incomplete",
          summary: `${counters.evidence} evidence links; ${counters.components} components`,
        },
        {
          type: IDEA_RESULT_TYPE,
          name: "Research idea result",
          path: runRelativePath(ctx, "artifacts/idea_result.json"),
          status: passed ? "ready" : "incomplete",
          summary: `${counters.candidates} candidates; ${counters.mcts_iterations} MCTS iterations`,
        },
        {
          type: TRACE_TYPE,
          name: "Research idea trace",
          path: runRelativePath(ctx, "artifacts/idea_trace.json"),
          status: passed ? "ready" : "incomplete",
          summary: `${counters.rag_hits} RAG hits; ${counters.mcts_iterations} MCTS iterations`,
        },
        {
          type: REPORT_TYPE,
          name: "Research idea quality report",
          path: runRelativePath(ctx, "artifacts/idea_report.json"),
          status: passed ? "ready" : "incomplete",
          summary: `${blockers.length} blockers; ${warnings.length} warnings`,
        },
      ],
    },
  };
}

export function postFinish(ctx: XlabHookContext): XlabHookResult {
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "finish",
        label: "Research idea finished",
        status: ctx.run.status,
        progress: 1,
      },
      message:
        ctx.run.summary ??
        "Research idea artifacts are ready for downstream XLab skills.",
      next_actions: [
        "Use research_idea.json with /xlab check-idea-novelty",
        "Use research_idea.json with /xlab run-experiment",
      ],
    },
  };
}

export function onResume(ctx: XlabHookContext): XlabHookResult {
  let checkpoint: Record<string, unknown> | undefined;
  let report: Record<string, unknown> | undefined;
  const readDiagnostics: Array<{
    severity: "warning" | "error";
    message: string;
    repair?: string;
  }> = [];
  try {
    checkpoint = existsSync(paths(ctx).checkpoint)
      ? readJson(paths(ctx).checkpoint)
      : undefined;
  } catch (error) {
    readDiagnostics.push({
      severity: "error",
      message: `Cannot read idea checkpoint: ${error instanceof Error ? error.message : String(error)}`,
      repair:
        "Repair or remove state/checkpoint.json, then rerun run_idea_phase.py resume.",
    });
  }
  try {
    report = existsSync(paths(ctx).report)
      ? readJson(paths(ctx).report)
      : undefined;
  } catch (error) {
    readDiagnostics.push({
      severity: "error",
      message: `Cannot read idea report: ${error instanceof Error ? error.message : String(error)}`,
      repair:
        "Rerun run_idea_phase.py audit to regenerate artifacts/idea_report.json.",
    });
  }
  const completedStages = Array.isArray(checkpoint?.completed_stages)
    ? checkpoint.completed_stages.filter(
        (value): value is string => typeof value === "string",
      )
    : [];
  const lastError =
    typeof checkpoint?.last_error === "string"
      ? checkpoint.last_error
      : undefined;
  const counters = report
    ? countersFromReport(report)
    : isRecord(checkpoint?.counts)
      ? Object.fromEntries(
          Object.entries(checkpoint.counts)
            .map(([key, value]): [string, number] => [key, numeric(value) ?? 0])
            .filter(([, value]) => value > 0),
        )
      : undefined;
  const passed = report?.passed === true;
  const blockers =
    report && Array.isArray(report.blocking_errors)
      ? report.blocking_errors.filter(
          (value): value is string => typeof value === "string",
        )
      : [];
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "resume",
        label: passed
          ? "Research idea ready to finish"
          : blockers.length > 0 || lastError
            ? "Research idea blocked at quality gate"
            : "Resuming research idea",
        status: passed
          ? "success"
          : blockers.length > 0 || lastError
            ? "incomplete"
            : "running",
      },
      message:
        completedStages.length > 0
          ? `Completed stages: ${completedStages.join(", ")}.`
          : "No completed idea stages were found; rerun run_idea_phase.py synthesize or resume.",
      counters,
      diagnostics: [
        ...readDiagnostics,
        ...(report
          ? diagnosticsFromReport(
              report,
              "Repair the reported blocker, then rerun run_idea_phase.py resume.",
            )
          : []),
        ...(lastError
          ? [
              {
                severity: "error" as const,
                message: lastError,
                repair:
                  "Repair the source survey, provider configuration, or package-native XLab research-idea workflow resource and rerun run_idea_phase.py resume.",
              },
            ]
          : []),
      ],
      checkpoint: {
        id: "research-idea-resume",
        label:
          completedStages.length > 0
            ? `Last stage: ${completedStages.at(-1)}`
            : "No reusable stage recorded",
        resumable: true,
        resume_hint: passed
          ? "Run run_idea_phase.py audit and finish with status success."
          : "Rerun run_idea_phase.py resume with this run directory.",
        data: checkpoint,
      },
      next_actions: passed
        ? [
            "Run run_idea_phase.py audit",
            "Call xlab_finish with status success",
          ]
        : blockers.length > 0 || lastError
          ? [
              "Repair the reported blocker",
              "Rerun run_idea_phase.py resume",
              "Finish with status incomplete if the package-native algorithm contract checks cannot currently pass",
            ]
          : [
              "Run run_idea_phase.py resume",
              "Then run run_idea_phase.py audit",
            ],
    },
  };
}

export function onCancel(): XlabHookResult {
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "cancel",
        label: "Research idea cancelled",
        status: "incomplete",
      },
      message:
        "Partial idea artifacts, logs, and checkpoints were preserved for resume.",
    },
  };
}

export function onError(ctx: XlabHookContext): XlabHookResult {
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "error",
        label: "Research idea interrupted",
        status: "failed",
      },
      message:
        ctx.run.errorSummary ??
        ctx.run.lastRepair ??
        "Research idea generation stopped before completion.",
      checkpoint: {
        id: "research-idea-error",
        label: "Partial idea state preserved",
        resumable: true,
        resume_hint: "Rerun run_idea_phase.py resume with this run directory.",
      },
    },
  };
}
