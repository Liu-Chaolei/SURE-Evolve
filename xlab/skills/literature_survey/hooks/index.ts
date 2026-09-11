import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { basename, isAbsolute, join, relative, resolve } from "node:path";
import type {
  XlabHookContext,
  XlabHookResult,
} from "@earendil-works/pi-coding-agent";

const SURVEY_DOCUMENT_TYPE = "literature_survey_document";
const SURVEY_JSON_TYPE = "literature_survey_json";
const CITATION_TRACE_TYPE = "citation_trace";
const SURVEY_REPORT_TYPE = "survey_report";

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
        label: "Validating literature survey",
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
  surveyMarkdown: string;
  surveyJson: string;
  citations: string;
  report: string;
  manifest: string;
} {
  const artifacts = join(ctx.runDir, "artifacts");
  const state = join(ctx.runDir, "state");
  return {
    script: join(ctx.packageDir, "scripts", "run_survey_phase.py"),
    artifacts,
    state,
    checkpoint: join(state, "checkpoint.json"),
    surveyMarkdown: join(artifacts, "survey.md"),
    surveyJson: join(artifacts, "survey.json"),
    citations: join(artifacts, "citations.json"),
    report: join(artifacts, "survey_report.json"),
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
  return isAbsolute(path) ? resolve(path) : resolve(ctx.cwd, path);
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

const READ_ONLY_INSPECTION_TOOLS = new Set([
  "jq",
  "ls",
  "find",
  "rg",
  "grep",
  "test",
  "[",
]);
const FIND_MUTATING_PREDICATES = new Set([
  "-delete",
  "-exec",
  "-execdir",
  "-ok",
  "-okdir",
  "-fprint",
  "-fprint0",
  "-fprintf",
  "-fls",
]);

function inspectionOperands(args: string[]): string[] {
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

function findRoots(args: string[]): string[] {
  const roots: string[] = [];
  for (const value of args) {
    if (
      value === "--" ||
      value === "!" ||
      value === "(" ||
      value === ")" ||
      value.startsWith("-")
    ) {
      if (value !== "--") {
        break;
      }
      continue;
    }
    roots.push(value);
  }
  return roots;
}

function isScopedInspectionPath(value: string, ctx: XlabHookContext): boolean {
  if (!value || value.includes("$") || value.startsWith("~") || /[*?[\]{}]/.test(value)) {
    return false;
  }
  const path = isAbsolute(value) ? resolve(value) : resolve(ctx.cwd, value);
  return isWithin(path, ctx.runDir) || isWithin(path, ctx.packageDir);
}

function allInspectionPathsScoped(
  values: string[],
  ctx: XlabHookContext,
): boolean {
  return (
    values.length > 0 &&
    values.every((value) => isScopedInspectionPath(value, ctx))
  );
}

function isRunOrPackageInspection(
  command: string,
  ctx: XlabHookContext,
): boolean {
  if (
    hasShellComposition(command) ||
    /\b(?:curl|wget|python\s+-c|node\s+-e|perl\s+-e|nc|ssh|scp|rsync)\b|https?:\/\//i.test(
      command,
    )
  ) {
    return false;
  }
  const tokens = parseShellTokens(command);
  if (!tokens || tokens.length === 0) {
    return false;
  }
  const executable = basename(tokens[0]);
  const args = tokens.slice(1);
  if (
    (executable === "python" || executable === "python3") &&
    args[0] === "-m" &&
    args[1] === "json.tool"
  ) {
    return allInspectionPathsScoped(inspectionOperands(args.slice(2)), ctx);
  }
  if (!READ_ONLY_INSPECTION_TOOLS.has(executable) || executable !== tokens[0]) {
    return false;
  }
  if (executable === "find") {
    return (
      !args.some((value) => FIND_MUTATING_PREDICATES.has(value)) &&
      allInspectionPathsScoped(findRoots(args), ctx)
    );
  }
  if (executable === "rg" || executable === "grep") {
    if (
      args.some(
        (value) =>
          value === "--pre" ||
          value.startsWith("--pre=") ||
          value === "--pre-glob" ||
          value.startsWith("--pre-glob="),
      )
    ) {
      return false;
    }
    const operands = inspectionOperands(args);
    const patternFromFile = args.includes("-f") || args.includes("--file");
    return allInspectionPathsScoped(
      patternFromFile ? operands : operands.slice(1),
      ctx,
    );
  }
  if (executable === "jq") {
    const operands = inspectionOperands(args);
    const filterFromFile = args.includes("-f") || args.includes("--from-file");
    return allInspectionPathsScoped(
      filterFromFile ? operands : operands.slice(1),
      ctx,
    );
  }
  return allInspectionPathsScoped(inspectionOperands(args), ctx);
}

function commandExitCode(event: Record<string, unknown>): number | undefined {
  const serialized = JSON.stringify(event);
  const match = serialized.match(/Command exited with code (\d+)/);
  return match ? Number(match[1]) : undefined;
}

function legalIncomplete(
  ctx: XlabHookContext,
  event: Record<string, unknown>,
): boolean {
  const exitCode = commandExitCode(event);
  if (
    exitCode !== 2 ||
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
    papers: numeric(counts.papers) ?? 0,
    clusters: numeric(counts.clusters) ?? 0,
    sections: numeric(counts.sections) ?? 0,
    claims: numeric(counts.claims) ?? 0,
    gaps: numeric(counts.gaps) ?? 0,
    references: numeric(counts.references) ?? 0,
    traces: numeric(counts.traces) ?? 0,
    warnings,
  };
}

function parseShellTokens(command: string): string[] | undefined {
  const tokens: string[] = [];
  let current = "";
  let quote: "'" | '"' | undefined;
  let escaping = false;
  for (const char of command.trim()) {
    if (escaping) {
      current += char;
      escaping = false;
      continue;
    }
    if (char === "\\" && quote !== "'") {
      escaping = true;
      continue;
    }
    if ((char === "'" || char === '"') && (!quote || quote === char)) {
      quote = quote ? undefined : char;
      continue;
    }
    if (!quote && /\s/.test(char)) {
      if (current) {
        tokens.push(current);
        current = "";
      }
      continue;
    }
    current += char;
  }
  if (escaping || quote) {
    return undefined;
  }
  if (current) {
    tokens.push(current);
  }
  return tokens;
}

function cliPhase(
  command: string,
  ctx: XlabHookContext,
): "init" | "synthesize" | "resume" | "audit" | "smoke" | "other" {
  const parsed = parsePackagedCli(command, ctx);
  if (parsed) {
    return parsed.phase;
  }
  const tokens = parseShellTokens(command);
  const scriptIndex = tokens?.[0] && /^(?:python|python3)$/.test(basename(tokens[0]))
    ? 1
    : 0;
  const phase = tokens?.[scriptIndex + 1];
  return phase === "smoke" ? "smoke" : "other";
}

function hasShellComposition(command: string): boolean {
  let quote: "'" | '"' | undefined;
  let escaping = false;
  for (let index = 0; index < command.length; index += 1) {
    const char = command[index];
    if (escaping) {
      escaping = false;
      continue;
    }
    if (char === "\\" && quote !== "'") {
      escaping = true;
      continue;
    }
    if ((char === "'" || char === '"') && (!quote || quote === char)) {
      quote = quote ? undefined : char;
      continue;
    }
    if (quote) {
      continue;
    }
    const next = command[index + 1];
    if (
      char === ";" ||
      char === "|" ||
      char === ">" ||
      char === "<" ||
      char === "`" ||
      char === "&" ||
      char === "\n" ||
      char === "\r" ||
      (char === "$" && next === "(")
    ) {
      return true;
    }
  }
  return false;
}

function readFlag(tokens: string[], flag: string): string | undefined {
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (token === flag) {
      return tokens[index + 1];
    }
    if (token.startsWith(`${flag}=`)) {
      return token.slice(flag.length + 1);
    }
  }
  return undefined;
}

function parsePackagedCli(
  command: string,
  ctx: Pick<XlabHookContext, "cwd" | "packageDir" | "runDir" | "run">,
): { phase: "init" | "synthesize" | "resume" | "audit"; tokens: string[] } | undefined {
  if (hasShellComposition(command)) {
    return undefined;
  }
  const tokens = parseShellTokens(command);
  if (!tokens || tokens.length === 0) {
    return undefined;
  }
  let scriptIndex = 0;
  const executable = basename(tokens[0]);
  if (executable === "python" || executable === "python3") {
    scriptIndex = 1;
  }
  const scriptToken = tokens[scriptIndex];
  const phase = tokens[scriptIndex + 1];
  if (!scriptToken || !phase) {
    return undefined;
  }
  const allowedPhase =
    phase === "init" ||
    phase === "synthesize" ||
    phase === "resume" ||
    phase === "audit"
      ? phase
      : undefined;
  if (!allowedPhase) {
    return undefined;
  }
  const candidateScripts = isAbsolute(scriptToken)
    ? [resolve(scriptToken)]
    : [
        resolve(ctx.cwd || process.cwd(), scriptToken),
        resolve(ctx.packageDir || process.cwd(), scriptToken),
      ];
  if (!candidateScripts.some((candidate) => resolve(candidate) === resolve(paths(ctx as XlabHookContext).script))) {
    return undefined;
  }
  const runId = readFlag(tokens, "--run-id");
  if (runId && runId !== ctx.run.runId) {
    return undefined;
  }
  const runDir = readFlag(tokens, "--run-dir");
  if (runDir) {
    const resolvedRunDir = isAbsolute(runDir)
      ? resolve(runDir)
      : resolve(ctx.cwd || process.cwd(), runDir);
    if (resolve(resolvedRunDir) !== resolve(ctx.runDir)) {
      return undefined;
    }
  }
  return { phase: allowedPhase, tokens };
}

function isAllowedPackagedCli(command: string, ctx: XlabHookContext): boolean {
  return parsePackagedCli(command, ctx) !== undefined;
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
  const required = [
    join(ctx.packageDir, "scripts", "run_survey_phase.py"),
    join(ctx.packageDir, "scripts", "literature_survey_lib", "common.py"),
    join(ctx.packageDir, "scripts", "literature_survey_lib", "inputs.py"),
    join(
      ctx.packageDir,
      "scripts",
      "literature_survey_lib",
      "paper_sources.py",
    ),
    join(ctx.packageDir, "scripts", "literature_survey_lib", "manifest.py"),
    join(ctx.packageDir, "scripts", "literature_survey_lib", "audit.py"),
    join(ctx.packageDir, "scripts", "literature_survey_lib", "survey_agent.py"),
    join(
      ctx.packageDir,
      "scripts",
      "literature_survey_lib",
      "xcientist",
      "api.py",
    ),
    join(
      ctx.packageDir,
      "scripts",
      "literature_survey_lib",
      "xcientist",
      "engine.py",
    ),
    join(
      ctx.packageDir,
      "scripts",
      "literature_survey_lib",
      "xcientist",
      "adapters.py",
    ),
    join(
      ctx.packageDir,
      "scripts",
      "literature_survey_lib",
      "xcientist",
      "config.py",
    ),
    join(
      ctx.packageDir,
      "scripts",
      "literature_survey_lib",
      "xcientist",
      "modules",
      "work_collector.py",
    ),
    join(
      ctx.packageDir,
      "scripts",
      "literature_survey_lib",
      "xcientist",
      "modules",
      "data_manager.py",
    ),
    join(
      ctx.packageDir,
      "scripts",
      "literature_survey_lib",
      "xcientist",
      "modules",
      "database.py",
    ),
    join(
      ctx.packageDir,
      "scripts",
      "literature_survey_lib",
      "xcientist",
      "modules",
      "work_analyzer.py",
    ),
    join(
      ctx.packageDir,
      "scripts",
      "literature_survey_lib",
      "xcientist",
      "modules",
      "survey_generator.py",
    ),
    join(
      ctx.packageDir,
      "scripts",
      "literature_survey_lib",
      "xcientist",
      "modules",
      "judge.py",
    ),
    join(ctx.packageDir, "schemas", "request.schema.json"),
    join(ctx.packageDir, "schemas", "literature_survey.schema.json"),
    join(ctx.packageDir, "schemas", "citation_trace.schema.json"),
    join(ctx.packageDir, "schemas", "survey_report.schema.json"),
  ];
  const missing = required.filter((path) => !existsSync(path));
  if (missing.length > 0) {
    return failure(
      "Restore the missing files inside the literature_survey skill package.",
      `Literature survey package is incomplete: ${missing.join(", ")}`,
    );
  }
  for (const directory of [
    paths(ctx).artifacts,
    paths(ctx).state,
    join(ctx.runDir, "logs"),
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
      'Use /xlab write-literature-survey "<topic>" --graph <knowledge_graph manifest-or-run-dir>, or use compatibility input --input <paper_manifest>.',
      `Invalid literature survey arguments: ${detail}`,
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
      "Run run_survey_phase.py init manually and inspect its JSON output.",
      `Literature survey initialization returned invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  const warnings = Array.isArray(initialized.warnings)
    ? initialized.warnings.filter(
        (value): value is string => typeof value === "string",
      )
    : [];
  const api = isRecord(initialized.api) ? initialized.api : {};
  const contextWindow = numeric(api.llm_context_window);
  const contextWindowSummary = contextWindow
    ? `; context ${Math.round(contextWindow / 1000)}k`
    : "";
  const apiDiagnostic = {
    severity: "info",
    message: `LLM API: ${String(api.llm_model ?? "unknown")} via ${String(api.llm_base_url ?? "default")}${contextWindowSummary}; OPENAI_API_KEY ${api.llm_api_key_set === true ? "set" : "missing"}; SEMANTIC_SCHOLAR_API_KEY ${api.semantic_scholar_api_key_set === true ? "set" : "missing"}.`,
  } as const;
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "init",
        label: "Literature survey initialized",
        status: "running",
      },
      message: `Initialized literature survey for ${String(initialized.topic)}.`,
      counters: {
        papers: 0,
        clusters: 0,
        sections: 0,
        claims: 0,
        traces: 0,
        warnings: warnings.length,
      },
      diagnostics: [
        apiDiagnostic,
        ...warnings.map((message) => ({
          severity: "warning" as const,
          message,
        })),
      ],
      checkpoint: {
        id: "literature-survey-start",
        label: "Survey initialized",
        resumable: true,
        resume_hint:
          "Rerun run_survey_phase.py resume with the same run directory.",
      },
      next_actions: [
        "Run run_survey_phase.py synthesize",
        "Resume with run_survey_phase.py resume if interrupted",
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
      "Use the packaged run_survey_phase.py CLI so survey artifacts, checkpoints, and diagnostics remain consistent.",
      "literature_survey does not call MCP tools directly.",
    );
  }
  if (toolName !== "bash") {
    return { ok: true };
  }
  const rawCommand = input.command ?? input.cmd;
  const command = typeof rawCommand === "string" ? rawCommand : "";
  if (
    /\b(?:curl|wget)\b|api\.openai\.com|api\.semanticscholar\.org/i.test(
      command,
    )
  ) {
    return failure(
      "Use run_survey_phase.py and declared artifacts; do not bypass the survey runtime with direct provider calls.",
      "Direct provider HTTP commands bypass the literature_survey contract.",
    );
  }
  if (
    /--xcientist-root|Xcientist(?:-\d+)?|XAgora|XForge|PaperGraph/i.test(
      command,
    )
  ) {
    return failure(
      "Keep literature_survey self-contained under xlab/skills/literature_survey/scripts.",
      "Blocked a command that references an external research-agent checkout.",
    );
  }
  if (
    command.includes("run_survey_phase.py") &&
    !isAllowedPackagedCli(command, ctx)
  ) {
    return failure(
      "Use only run_survey_phase.py init, synthesize, resume, or audit in the slash-command runtime. The smoke command is test-only.",
      "Blocked a non-production literature_survey CLI phase.",
    );
  }
  if (
    /\b(?:python(?:3)?\s+-c|node\s+-e|tee|cp|mv|touch)\b|>/.test(command) &&
    /(?:artifacts\/(?:survey\.md|survey\.json|citations\.json|survey_report\.json)|manifest\.json)/.test(
      command,
    )
  ) {
    return failure(
      "Produce final survey artifacts through run_survey_phase.py so checkpoint signatures, audit status, and manifest paths remain canonical.",
      "Blocked an ad hoc command that could overwrite literature_survey final artifacts.",
    );
  }
  if (
    !isAllowedPackagedCli(command, ctx) &&
    !isRunOrPackageInspection(command, ctx)
  ) {
    return failure(
      "Run the packaged literature_survey phase CLI, or use read-only inspection scoped to the run/package directory.",
      "Blocked a Bash command outside the literature_survey control plane.",
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
  if (!command.includes("run_survey_phase.py")) {
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
            label: `Literature survey ${phase} produced an incomplete manifest`,
            status: "incomplete",
          },
          message:
            "The packaged survey runtime completed and wrote an incomplete manifest; this is a recoverable quality-gate outcome, not a crash.",
          counters: countersFromReport(incompleteReport),
          diagnostics: diagnosticsFromReport(
            incompleteReport,
            "Repair the upstream graph/input or provider configuration, then rerun run_survey_phase.py resume.",
          ),
          next_actions: [
            "Rerun run_survey_phase.py resume after repairs",
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
          label: `Literature survey ${phase} phase`,
          status: "blocked",
        },
        counters,
        diagnostics: [
          {
            severity: "warning",
            message: timedOut
              ? `The literature survey ${phase} phase exceeded the tool timeout.`
              : `The literature survey ${phase} phase returned an error before writing a valid incomplete manifest.`,
            repair: `Inspect logs and rerun run_survey_phase.py resume; completed stage files and checkpoints are preserved.`,
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
          phase === "synthesize" || phase === "smoke"
            ? "Literature survey artifacts generated"
            : phase === "audit"
              ? "Literature survey audit complete"
              : "Literature survey phase complete",
        status: phase === "audit" ? "success" : "running",
      },
      counters,
    },
  };
}

export function preFinish(ctx: XlabHookContext): XlabHookResult {
  const event = finishEvent(ctx);
  const artifacts = event.manifest?.artifacts ?? [];
  const surveyMarkdownPath = artifactPathFromManifest(
    ctx,
    artifacts,
    SURVEY_DOCUMENT_TYPE,
    "artifacts/survey.md",
  );
  const surveyJsonPath = artifactPathFromManifest(
    ctx,
    artifacts,
    SURVEY_JSON_TYPE,
    "artifacts/survey.json",
  );
  const citationTracePath = artifactPathFromManifest(
    ctx,
    artifacts,
    CITATION_TRACE_TYPE,
    "artifacts/citations.json",
  );
  const reportPath = artifactPathFromManifest(
    ctx,
    artifacts,
    SURVEY_REPORT_TYPE,
    "artifacts/survey_report.json",
  );
  for (const [type, path, canonical] of [
    [SURVEY_DOCUMENT_TYPE, surveyMarkdownPath, paths(ctx).surveyMarkdown],
    [SURVEY_JSON_TYPE, surveyJsonPath, paths(ctx).surveyJson],
    [CITATION_TRACE_TYPE, citationTracePath, paths(ctx).citations],
    [SURVEY_REPORT_TYPE, reportPath, paths(ctx).report],
  ] as const) {
    if (!isWithin(path, ctx.cwd) && !isWithin(path, ctx.runDir)) {
      return failure(
        "Use the run-local artifact paths written by run_survey_phase.py.",
        `Artifact ${type} points outside the project/run workspace: ${path}`,
      );
    }
    if (resolve(path) !== resolve(canonical)) {
      return failure(
        "Use the canonical manifest generated by run_survey_phase.py audit before calling xlab_finish.",
        `Artifact ${type} must point to ${canonical}.`,
      );
    }
  }
  for (const [name, path] of [
    ["survey.md", surveyMarkdownPath],
    ["survey.json", surveyJsonPath],
    ["citations.json", citationTracePath],
    ["survey_report.json", reportPath],
  ] as const) {
    if (!existsSync(path)) {
      return failure(
        `Run run_survey_phase.py synthesize before calling xlab_finish. Missing ${name}.`,
        `Missing ${name}.`,
      );
    }
  }
  const audit = runPython(
    ctx,
    [
      "audit",
      "--run-id",
      ctx.run.runId,
      "--run-dir",
      ctx.runDir,
      "--cwd",
      ctx.cwd,
    ],
    600_000,
  );
  const auditStatus = audit.status ?? 1;
  if (audit.error || (auditStatus !== 0 && auditStatus !== 2)) {
    return failure(
      "Run run_survey_phase.py audit manually, repair the reported artifact, and retry xlab_finish.",
      `Literature survey audit failed: ${audit.error?.message ?? audit.stderr.trim() ?? `exit ${audit.status}`}`,
    );
  }
  if (!existsSync(reportPath) || !existsSync(paths(ctx).manifest)) {
    return failure(
      "Run run_survey_phase.py audit manually and verify survey_report.json plus manifest.json were written.",
      "Literature survey audit completed without writing the required report or manifest.",
    );
  }
  let report: Record<string, unknown>;
  try {
    report = readJson(reportPath);
    readJson(surveyJsonPath);
    readJson(citationTracePath);
  } catch (error) {
    return failure(
      "Repair the survey JSON artifacts and rerun run_survey_phase.py audit.",
      `Cannot read survey artifacts: ${error instanceof Error ? error.message : String(error)}`,
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
    ...warnings.map((message) => ({ severity: "warning", message })),
    ...blockers.map((message) => ({
      severity: "error",
      message,
      repair:
        "Rerun run_survey_phase.py resume after repairing the source artifacts.",
    })),
  ];
  const finishStatus = event.finish?.status ?? "";
  const generatedStatus = passed ? "success" : "incomplete";
  if (finishStatus === "success" && !passed) {
    return failure(
      "Resolve the audit blockers, or finish with status incomplete using the generated manifest.",
      blockers.join(" ") || "Literature survey did not pass the success gates.",
      counters,
    );
  }
  if (finishStatus && finishStatus !== generatedStatus) {
    return failure(
      `Call xlab_finish with status ${generatedStatus}; the Python audit generated that manifest status.`,
      `xlab_finish status ${finishStatus} does not match audit status ${generatedStatus}.`,
      counters,
    );
  }
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "validate",
        label: passed
          ? "Literature survey validated"
          : "Literature survey incomplete",
        status: passed ? "success" : "incomplete",
        progress: 1,
      },
      message: `Validated ${counters.papers} papers, ${counters.clusters} clusters, and ${counters.traces} citation traces.`,
      counters,
      diagnostics,
      artifacts: [
        {
          type: SURVEY_DOCUMENT_TYPE,
          name: "Literature survey",
          path: runRelativePath(ctx, "artifacts/survey.md"),
          status: passed ? "ready" : "incomplete",
          summary: `${counters.sections} sections; ${counters.claims} claims`,
        },
        {
          type: SURVEY_JSON_TYPE,
          name: "Structured survey JSON",
          path: runRelativePath(ctx, "artifacts/survey.json"),
          status: passed ? "ready" : "incomplete",
          summary: `${counters.clusters} clusters; ${counters.references} references`,
        },
        {
          type: CITATION_TRACE_TYPE,
          name: "Citation trace",
          path: runRelativePath(ctx, "artifacts/citations.json"),
          status: passed ? "ready" : "incomplete",
          summary: `${counters.traces} traced claims`,
        },
        {
          type: SURVEY_REPORT_TYPE,
          name: "Survey quality report",
          path: runRelativePath(ctx, "artifacts/survey_report.json"),
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
        label: "Literature survey finished",
        status: ctx.run.status,
        progress: 1,
      },
      message:
        ctx.run.summary ??
        "Literature survey artifacts are ready for downstream XLab skills.",
      next_actions: [
        "Use survey.json in /xlab generate-research-ideas",
        "Use citations.json in /xlab check-idea-novelty",
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
      message: `Cannot read survey checkpoint: ${error instanceof Error ? error.message : String(error)}`,
      repair:
        "Repair or remove state/checkpoint.json, then rerun run_survey_phase.py resume.",
    });
  }
  try {
    report = existsSync(paths(ctx).report)
      ? readJson(paths(ctx).report)
      : undefined;
  } catch (error) {
    readDiagnostics.push({
      severity: "error",
      message: `Cannot read survey report: ${error instanceof Error ? error.message : String(error)}`,
      repair:
        "Rerun run_survey_phase.py audit to regenerate artifacts/survey_report.json.",
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
          ? "Literature survey ready to finish"
          : blockers.length > 0 || lastError
            ? "Literature survey blocked at quality gate"
            : "Resuming literature survey",
        status: passed
          ? "success"
          : blockers.length > 0 || lastError
            ? "incomplete"
            : "running",
      },
      message:
        completedStages.length > 0
          ? `Completed stages: ${completedStages.join(", ")}.`
          : "No completed survey stages were found; rerun run_survey_phase.py synthesize or resume.",
      counters,
      diagnostics: [
        ...readDiagnostics,
        ...(report
          ? diagnosticsFromReport(
              report,
              "Repair the reported blocker, then rerun run_survey_phase.py resume.",
            )
          : []),
        ...(lastError
          ? [
              {
                severity: "error" as const,
                message: lastError,
                repair:
                  "Repair the failing SurveyAgent/provider/input condition and rerun run_survey_phase.py resume.",
              },
            ]
          : []),
      ],
      checkpoint: {
        id: "literature-survey-resume",
        label:
          completedStages.length > 0
            ? `Last stage: ${completedStages.at(-1)}`
            : "No reusable stage recorded",
        resumable: true,
        resume_hint: passed
          ? "Run run_survey_phase.py audit and finish with status success."
          : "Rerun run_survey_phase.py resume with this run directory.",
        data: checkpoint,
      },
      next_actions: passed
        ? [
            "Run run_survey_phase.py audit",
            "Call xlab_finish with status success",
          ]
        : blockers.length > 0 || lastError
          ? [
              "Repair the reported blocker",
              "Rerun run_survey_phase.py resume",
              "Finish with status incomplete if the blocker is expected",
            ]
          : [
              "Run run_survey_phase.py resume",
              "Then run run_survey_phase.py audit",
            ],
    },
  };
}

export function onCancel(ctx: XlabHookContext): XlabHookResult {
  let checkpoint: Record<string, unknown> | undefined;
  let report: Record<string, unknown> | undefined;
  try {
    checkpoint = existsSync(paths(ctx).checkpoint)
      ? readJson(paths(ctx).checkpoint)
      : undefined;
  } catch {
    checkpoint = undefined;
  }
  try {
    report = existsSync(paths(ctx).report)
      ? readJson(paths(ctx).report)
      : undefined;
  } catch {
    report = undefined;
  }
  const completedStages = Array.isArray(checkpoint?.completed_stages)
    ? checkpoint.completed_stages.filter(
        (value): value is string => typeof value === "string",
      )
    : [];
  const readyArtifacts = [
    [SURVEY_DOCUMENT_TYPE, paths(ctx).surveyMarkdown],
    [SURVEY_JSON_TYPE, paths(ctx).surveyJson],
    [CITATION_TRACE_TYPE, paths(ctx).citations],
    [SURVEY_REPORT_TYPE, paths(ctx).report],
  ]
    .filter(([, path]) => existsSync(path))
    .map(([type, path]) => ({
      type,
      name: basename(path),
      path: relative(ctx.cwd, path).replace(/\\/g, "/"),
      status: "partial",
    }));
  const lastStage = completedStages.at(-1);
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "cancel",
        label: lastStage
          ? `Literature survey cancelled after ${lastStage}`
          : "Literature survey cancelled",
        status: "incomplete",
      },
      message:
        "Partial survey artifacts, logs, and checkpoints were preserved for resume.",
      counters: report ? countersFromReport(report) : undefined,
      diagnostics: report
        ? diagnosticsFromReport(
            report,
            "Repair any blockers and resume the same XLab run.",
          )
        : undefined,
      artifacts: readyArtifacts,
      checkpoint: {
        id: "literature-survey-cancelled",
        label: lastStage ? `Last completed stage: ${lastStage}` : "No completed stage recorded",
        resumable: true,
        resume_hint: "Use /xlab resume-run <run-id> for this run, then rerun run_survey_phase.py resume.",
        data: checkpoint,
      },
      next_actions: [
        "Resume this XLab run",
        "Rerun run_survey_phase.py resume",
        "Audit before calling xlab_finish",
      ],
    },
  };
}

export function onError(ctx: XlabHookContext): XlabHookResult {
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "error",
        label: "Literature survey interrupted",
        status: "failed",
      },
      message:
        ctx.run.errorSummary ??
        ctx.run.lastRepair ??
        "Literature survey stopped before completion.",
      checkpoint: {
        id: "literature-survey-error",
        label: "Partial survey preserved",
        resumable: true,
        resume_hint:
          "Rerun run_survey_phase.py resume with this run directory.",
      },
    },
  };
}
