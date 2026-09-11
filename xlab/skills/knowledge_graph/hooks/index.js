import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { basename, isAbsolute, join, relative, resolve } from "node:path";
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function numeric(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
}
function readJson(path) {
  const value = JSON.parse(readFileSync(path, "utf-8"));
  if (!isRecord(value)) {
    throw new Error(`${path} must contain a JSON object.`);
  }
  return value;
}
function failure(repair, message, counters) {
  return {
    ok: false,
    repair,
    state_patch: {
      phase: {
        id: "validate",
        label: "Validating knowledge graph",
        status: "blocked"
      },
      message,
      counters,
      diagnostics: [{ severity: "error", message, repair }]
    }
  };
}
function packagePaths(ctx) {
  return {
    script: join(ctx.packageDir, "scripts", "build_graph.py"),
    mineruManifest: join(ctx.runDir, "artifacts", "mineru.manifest.json"),
    structuresManifest: join(
      ctx.runDir,
      "artifacts",
      "paper_structures.manifest.json"
    ),
    extractionsManifest: join(
      ctx.runDir,
      "artifacts",
      "extractions.manifest.json"
    ),
    graph: join(ctx.runDir, "artifacts", "method_graph.json"),
    database: join(ctx.runDir, "artifacts", "graph.db"),
    report: join(ctx.runDir, "artifacts", "graph_report.json"),
    manifest: join(ctx.runDir, "manifest.json")
  };
}
function eventValue(ctx) {
  return isRecord(ctx.event) ? ctx.event : {};
}
function eventManifestPath(ctx) {
  const event = eventValue(ctx);
  const eventPath = event.manifestPath;
  if (typeof eventPath === "string" && eventPath.trim() !== "") {
    return resolveProjectOrRunPath(ctx, eventPath);
  }
  const finish = isRecord(event.finish) ? event.finish : {};
  const finishPath = finish.manifest_path ?? finish.manifestPath;
  return typeof finishPath === "string" && finishPath.trim() !== "" ? resolveProjectOrRunPath(ctx, finishPath) : void 0;
}
const SHELL_CONTROL_PATTERN = /(?:[;&|<>`]|\$\(|\n|\r)/;
const ALLOWED_CLI_PHASES = /* @__PURE__ */ new Set([
  "mineru",
  "structure",
  "extract",
  "build",
  "audit",
  "run",
  "resume"
]);
function splitCommand(command) {
  const tokens = [];
  let current = "";
  let quote;
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
        quote = void 0;
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
    return void 0;
  }
  if (current) {
    tokens.push(current);
  }
  return tokens;
}
function parseSimpleCommand(command) {
  if (SHELL_CONTROL_PATTERN.test(command)) {
    return void 0;
  }
  const tokens = splitCommand(command);
  if (!tokens || tokens.length === 0) {
    return void 0;
  }
  return { executable: tokens[0], args: tokens.slice(1) };
}
function cliPhase(command) {
  const match = command.match(
    /build_graph\.py\s+(mineru|structure|extract|build|audit|run|resume)\b/
  );
  if (match?.[1] === "mineru" || match?.[1] === "structure" || match?.[1] === "extract" || match?.[1] === "build" || match?.[1] === "audit") {
    return match[1];
  }
  if (match?.[1] === "run" || match?.[1] === "resume") {
    return "run";
  }
  return "other";
}
function isWithin(path, root) {
  const relativePath = relative(resolve(root), resolve(path));
  return relativePath === "" || !relativePath.startsWith("..") && !isAbsolute(relativePath);
}
function resolveProjectOrRunPath(ctx, path) {
  return isAbsolute(path) ? resolve(path) : resolve(ctx.cwd, path);
}
function resolveRunArgument(commandPath, ctx) {
  return isAbsolute(commandPath) ? resolve(commandPath) : resolve(ctx.cwd, commandPath);
}
function artifactPathFromManifest(ctx, artifacts, type, fallback) {
  const artifact = artifacts.find((entry) => entry.type === type);
  if (artifact?.path) {
    return resolveProjectOrRunPath(ctx, artifact.path);
  }
  return join(ctx.runDir, fallback);
}
function isPackagedScriptPath(commandPath, ctx) {
  return isAbsolute(commandPath) && resolve(commandPath) === resolve(packagePaths(ctx).script);
}
function isPythonExecutable(value) {
  return /^(?:python|python3(?:\.\d+)?)$/.test(basename(value));
}
function isBareExecutable(value) {
  return basename(value) === value;
}
function isAllowedPythonExecutable(value, ctx) {
  if (!isPythonExecutable(value)) {
    return false;
  }
  if (isBareExecutable(value)) {
    return true;
  }
  const configured = ctx.run.environment?.executable;
  return Boolean(configured && isAbsolute(value) && resolve(value) === resolve(configured));
}
function packagedScriptArgument(parsed, ctx) {
  if (isAllowedPythonExecutable(parsed.executable, ctx)) {
    const [script, ...args] = parsed.args;
    return script ? { script, args } : void 0;
  }
  return { script: parsed.executable, args: parsed.args };
}
function optionValues(args, name) {
  const values = [];
  for (let index = 0; index < args.length; index += 1) {
    const value = args[index];
    if (value === name && args[index + 1] !== void 0) {
      values.push(args[index + 1]);
      index += 1;
      continue;
    }
    if (value.startsWith(`${name}=`)) {
      values.push(value.slice(name.length + 1));
    }
  }
  return values;
}
function runDirValuesAreExpected(values, ctx) {
  return values.length === 1 && resolveRunArgument(values[0] ?? "", ctx) === resolve(ctx.runDir);
}
function parsePackagedCliCommand(command, ctx) {
  const parsed = parseSimpleCommand(command);
  if (!parsed) {
    return void 0;
  }
  const script = packagedScriptArgument(parsed, ctx);
  if (!script || !isPackagedScriptPath(script.script, ctx)) {
    return void 0;
  }
  const phase = script.args[0];
  if (!ALLOWED_CLI_PHASES.has(phase)) {
    return void 0;
  }
  if (!runDirValuesAreExpected(optionValues(script.args.slice(1), "--run-dir"), ctx)) {
    return void 0;
  }
  return { ...parsed, phase };
}
function isAllowedPackagedCli(command, ctx) {
  return parsePackagedCliCommand(command, ctx) !== void 0;
}
function packagedCliRejection(command, ctx) {
  const parsed = parseSimpleCommand(command);
  if (!parsed) {
    return {
      repair: "Run build_graph.py as one simple command without shell control operators or redirects.",
      message: "Blocked knowledge_graph CLI invocation with unsupported shell syntax."
    };
  }
  const script = packagedScriptArgument(parsed, ctx);
  if (!script || !isPackagedScriptPath(script.script, ctx)) {
    return {
      repair: `Use the packaged knowledge_graph CLI at ${packagePaths(ctx).script}.`,
      message: "Blocked build_graph.py invocation outside the active skill package."
    };
  }
  const phase = script.args[0];
  if (!ALLOWED_CLI_PHASES.has(phase)) {
    return {
      repair: "Use only build_graph.py mineru, structure, extract, build, audit, run, or resume in the slash-command runtime.",
      message: `Blocked unsupported knowledge_graph CLI phase: ${phase || "missing"}.`
    };
  }
  if (!runDirValuesAreExpected(optionValues(script.args.slice(1), "--run-dir"), ctx)) {
    return {
      repair: `Pass --run-dir ${ctx.runDir} for the active XLab run.`,
      message: "Blocked knowledge_graph CLI invocation for a different run directory."
    };
  }
  return void 0;
}
const READ_ONLY_INSPECTION_TOOLS = /* @__PURE__ */ new Set(["jq", "ls", "find", "rg", "grep", "test", "["]);
const FIND_MUTATING_PREDICATES = /* @__PURE__ */ new Set([
  "-delete",
  "-exec",
  "-execdir",
  "-ok",
  "-okdir",
  "-fprint",
  "-fprint0",
  "-fprintf",
  "-fls"
]);
function hasUnsafeShellExpansion(value) {
  return value.includes("$") || value.startsWith("~");
}
function hasPathGlob(value) {
  return /[*?[\]{}]/.test(value);
}
function isScopedInspectionPath(value, ctx) {
  if (!value || hasUnsafeShellExpansion(value) || hasPathGlob(value)) {
    return false;
  }
  const resolved = resolveRunArgument(value, ctx);
  return [ctx.runDir, ctx.packageDir].some((root) => isWithin(resolved, root));
}
function nonOptionOperands(args) {
  const operands = [];
  let afterOptions = false;
  for (const value of args) {
    if (!afterOptions && value === "--") {
      afterOptions = true;
      continue;
    }
    if (!afterOptions && value.startsWith("-") && value !== "-") {
      continue;
    }
    if (value === "]") {
      continue;
    }
    operands.push(value);
  }
  return operands;
}
function leadingFindPaths(args) {
  const paths = [];
  for (const value of args) {
    if (value === "--") {
      continue;
    }
    if (value.startsWith("-") || value === "!" || value === "(" || value === ")") {
      break;
    }
    paths.push(value);
  }
  return paths;
}
function allPathsAreScoped(values, ctx) {
  return values.length > 0 && values.every((value) => isScopedInspectionPath(value, ctx));
}
function areInspectionPathsScoped(parsed, ctx) {
  if (parsed.args.some(hasUnsafeShellExpansion)) {
    return false;
  }
  if (!isBareExecutable(parsed.executable) && !isAllowedPythonExecutable(parsed.executable, ctx)) {
    return false;
  }
  const executable = basename(parsed.executable);
  const pythonJsonTool = isAllowedPythonExecutable(parsed.executable, ctx) && parsed.args[0] === "-m" && parsed.args[1] === "json.tool";
  if (pythonJsonTool) {
    const paths = nonOptionOperands(parsed.args.slice(2));
    return paths.length === 1 && allPathsAreScoped(paths, ctx);
  }
  if (executable === "find") {
    return allPathsAreScoped(leadingFindPaths(parsed.args), ctx);
  }
  if (executable === "ls" || executable === "test" || executable === "[") {
    return allPathsAreScoped(nonOptionOperands(parsed.args), ctx);
  }
  if (executable === "jq") {
    const operands = nonOptionOperands(parsed.args);
    const filterFromFile = parsed.args.includes("-f") || parsed.args.includes("--from-file");
    return allPathsAreScoped(filterFromFile ? operands : operands.slice(1), ctx);
  }
  if (executable === "rg" || executable === "grep") {
    if (parsed.args.some(
      (value) => value === "--pre" || value.startsWith("--pre=") || value === "--pre-glob" || value.startsWith("--pre-glob=")
    )) {
      return false;
    }
    const operands = nonOptionOperands(parsed.args);
    const patternFromFile = parsed.args.includes("-f") || parsed.args.includes("--file");
    return allPathsAreScoped(patternFromFile ? operands : operands.slice(1), ctx);
  }
  return false;
}
function isRunOrPackageInspection(command, ctx) {
  const parsed = parseSimpleCommand(command);
  if (!parsed) {
    return false;
  }
  const executable = basename(parsed.executable);
  const pythonJsonTool = isAllowedPythonExecutable(parsed.executable, ctx) && parsed.args[0] === "-m" && parsed.args[1] === "json.tool";
  if (!pythonJsonTool && !READ_ONLY_INSPECTION_TOOLS.has(executable)) {
    return false;
  }
  if (/\b(?:curl|wget|python\s+-c|node\s+-e|perl\s+-e|nc|ssh|scp|rsync)\b|https?:\/\//i.test(
    command
  )) {
    return false;
  }
  if (executable === "find") {
    if (parsed.args.length === 0) {
      return false;
    }
    if (parsed.args.some((value) => FIND_MUTATING_PREDICATES.has(value))) {
      return false;
    }
  }
  return areInspectionPathsScoped(parsed, ctx);
}
function commandExitCode(event) {
  const serialized = JSON.stringify(event);
  const match = serialized.match(/Command exited with code (\d+)/);
  return match ? Number(match[1]) : void 0;
}
function legalIncomplete(ctx, event) {
  const paths = packagePaths(ctx);
  const exitCode = commandExitCode(event);
  if (exitCode !== 2 || !existsSync(paths.report) || !existsSync(paths.manifest)) {
    return false;
  }
  try {
    const report = readJson(paths.report);
    const manifest = readJson(paths.manifest);
    return report.passed !== true && manifest.status === "incomplete";
  } catch {
    return false;
  }
}
function diagnosticsFromReport(report, repair) {
  const warnings = Array.isArray(report.warnings) ? report.warnings.filter(
    (value) => typeof value === "string"
  ) : [];
  const blockers = Array.isArray(report.blocking_errors) ? report.blocking_errors.filter(
    (value) => typeof value === "string"
  ) : [];
  return [
    ...warnings.map((message) => ({ severity: "warning", message })),
    ...blockers.map((message) => ({
      severity: "error",
      message,
      repair
    }))
  ];
}
function preStart(ctx) {
  const requiredSecrets = [
    "KNOWLEDGE_GRAPH_LLM_API_URL",
    "KNOWLEDGE_GRAPH_LLM_API_KEY",
    "KNOWLEDGE_GRAPH_LLM_MODEL"
  ];
  const missingSecrets = requiredSecrets.filter(
    (name) => !process.env[name]?.trim()
  );
  if (missingSecrets.length > 0) {
    return failure(
      "Provide the missing values in the XLab initialization dialogs and start the command again.",
      `Missing required knowledge-graph LLM settings: ${missingSecrets.join(", ")}.`
    );
  }
  const required = [
    join(ctx.packageDir, "scripts", "build_graph.py"),
    join(ctx.packageDir, "scripts", "knowledge_graph_lib", "common.py"),
    join(ctx.packageDir, "scripts", "knowledge_graph_lib", "inputs.py"),
    join(ctx.packageDir, "scripts", "knowledge_graph_lib", "resources.py"),
    join(ctx.packageDir, "scripts", "knowledge_graph_lib", "mineru.py"),
    join(ctx.packageDir, "scripts", "knowledge_graph_lib", "mineru_cache.py"),
    join(ctx.packageDir, "scripts", "mineru_npu.py"),
    join(ctx.packageDir, "scripts", "knowledge_graph_lib", "structure.py"),
    join(ctx.packageDir, "scripts", "knowledge_graph_lib", "llm.py"),
    join(ctx.packageDir, "scripts", "knowledge_graph_lib", "extract.py"),
    join(ctx.packageDir, "scripts", "knowledge_graph_lib", "graph.py"),
    join(ctx.packageDir, "scripts", "knowledge_graph_lib", "audit.py"),
    join(ctx.packageDir, "schemas", "knowledge_graph_request.schema.json"),
    join(ctx.packageDir, "schemas", "mineru_manifest.schema.json"),
    join(ctx.packageDir, "schemas", "paper_documents.schema.json"),
    join(ctx.packageDir, "schemas", "paper_structures.schema.json"),
    join(ctx.packageDir, "schemas", "paper_extractions.schema.json"),
    join(ctx.packageDir, "schemas", "method_graph.schema.json"),
    join(ctx.packageDir, "schemas", "graph_report.schema.json")
  ];
  const missing = required.filter((path) => !existsSync(path));
  if (missing.length > 0) {
    return failure(
      "Restore the missing files inside the knowledge_graph package.",
      `Knowledge graph package is incomplete: ${missing.join(", ")}`
    );
  }
  for (const directory of [
    join(ctx.runDir, "artifacts"),
    join(ctx.runDir, "artifacts", "mineru"),
    join(ctx.runDir, "artifacts", "paper_structures"),
    join(ctx.runDir, "artifacts", "llm"),
    join(ctx.runDir, "artifacts", "extractions"),
    join(ctx.runDir, "artifacts", "logs")
  ]) {
    mkdirSync(directory, { recursive: true });
  }
  const python = ctx.run.environment?.executable ?? "python";
  const initialization = spawnSync(
    python,
    [
      packagePaths(ctx).script,
      "init",
      "--arguments",
      ctx.args,
      "--run-id",
      ctx.run.runId,
      "--run-dir",
      ctx.runDir,
      "--cwd",
      ctx.cwd
    ],
    {
      cwd: ctx.packageDir,
      encoding: "utf-8",
      maxBuffer: 10 * 1024 * 1024,
      timeout: 3e4
    }
  );
  if (initialization.error || initialization.status !== 0) {
    const detail = initialization.error?.message || initialization.stderr.trim() || initialization.stdout.trim() || `exit ${initialization.status}`;
    return failure(
      'Use /xlab build-knowledge-graph "<completed paper_collect directory>" and adjust --gpu or --mineru-backend if the resource probe rejects the defaults.',
      `Invalid knowledge graph invocation: ${detail}`
    );
  }
  let request;
  let resources;
  try {
    request = readJson(join(ctx.runDir, "artifacts", "request.json"));
    resources = readJson(join(ctx.runDir, "artifacts", "resource_plan.json"));
  } catch (error) {
    return failure(
      "Inspect artifacts/request.json and artifacts/resource_plan.json, then rerun the command.",
      `Knowledge graph initialization produced invalid state: ${error instanceof Error ? error.message : String(error)}`
    );
  }
  const mineru = isRecord(resources.mineru) ? resources.mineru : {};
  const selectedGpus = Array.isArray(mineru.selected_gpus) ? mineru.selected_gpus : [];
  const selectedNpus = Array.isArray(mineru.selected_npus) ? mineru.selected_npus : [];
  const resourceSummary = selectedNpus.length > 0 ? `${String(mineru.backend)} on NPU ${selectedNpus.join(", ")}` : selectedGpus.length > 0 ? `${String(mineru.backend)} on GPU ${selectedGpus.join(", ")}` : `${String(mineru.backend)} on CPU (${String(mineru.workers)} workers)`;
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "mineru",
        label: "Parsing PDFs with MinerU",
        status: "running"
      },
      message: `Initialized from ${String(request.input_directory)}; MinerU plan: ${resourceSummary}.`,
      counters: {
        papers: 0,
        downloaded_pdfs: 0,
        parsed_papers: 0,
        structured_papers: 0,
        extracted_papers: 0,
        nodes: 0,
        edges: 0
      },
      checkpoint: {
        id: "knowledge-graph-start",
        label: "Input and resource plan initialized",
        resumable: true,
        resume_hint: "Rerun the interrupted stage; MinerU, Step 1, and LLM pass checkpoints are checksum-validated."
      }
    }
  };
}
function preToolCall(ctx) {
  const event = eventValue(ctx);
  const toolName = typeof event.toolName === "string" ? event.toolName : "";
  const input = isRecord(event.input) ? event.input : {};
  if (toolName.startsWith("mcp__")) {
    return failure(
      "Use the packaged build_graph.py pipeline.",
      "knowledge_graph does not use MCP tools."
    );
  }
  if (toolName !== "bash") {
    if (toolName === "read") {
      const rawPath = input.file_path ?? input.path;
      const path = typeof rawPath === "string" ? rawPath : "";
      const resolved = path ? resolveRunArgument(path, ctx) : "";
      if (resolved && [ctx.runDir, ctx.packageDir].some((root) => isWithin(resolved, root))) {
        return { ok: true };
      }
      return failure(
        "Read only files inside the active run directory or knowledge_graph package.",
        "Blocked read outside the knowledge_graph run/package scope."
      );
    }
    return failure(
      "Use only read-only inspection plus the packaged build_graph.py CLI; final artifacts must be produced by the CLI.",
      `knowledge_graph does not allow the ${toolName || "unknown"} tool during a run.`
    );
  }
  const rawCommand = input.command ?? input.cmd;
  const command = typeof rawCommand === "string" ? rawCommand : "";
  if (/\b(?:curl|wget)\b|https?:\/\//i.test(command)) {
    return failure(
      "Use the packaged CLI; it handles MinerU, reference resolution, and the configured OpenAI-compatible LLM API.",
      "Ad hoc network calls bypass knowledge_graph retries, checkpoints, and secret handling."
    );
  }
  if (/Xcientist-2|XAgora|XForge|PaperGraph|Method-Evolution-Graph/i.test(command)) {
    return failure(
      "Use only code bundled inside xlab/skills/knowledge_graph.",
      "Runtime access to source repositories violates the self-contained package contract."
    );
  }
  if (command.includes("build_graph.py")) {
    const rejection = packagedCliRejection(command, ctx);
    if (rejection) {
      return failure(rejection.repair, rejection.message);
    }
  }
  if (/\b(?:python(?:3)?\s+-c|node\s+-e|tee|cp|mv|touch)\b|>/.test(command) && /(?:artifacts\/(?:mineru\.manifest\.json|paper_structures\.manifest\.json|extractions\.manifest\.json|method_graph\.json|graph\.db|graph_report\.json)|manifest\.json)/.test(
    command
  )) {
    return failure(
      "Produce final graph artifacts through build_graph.py so stage checksums, audit status, and manifest paths remain canonical.",
      "Blocked an ad hoc command that could overwrite knowledge_graph final artifacts."
    );
  }
  if (!isAllowedPackagedCli(command, ctx) && !isRunOrPackageInspection(command, ctx)) {
    return failure(
      "Run the packaged knowledge_graph phase CLI, or use read-only inspection scoped to the run/package directory.",
      "Blocked a Bash command outside the knowledge_graph control plane."
    );
  }
  return { ok: true };
}
function postToolResult(ctx) {
  const event = eventValue(ctx);
  const toolName = typeof event.toolName === "string" ? event.toolName : "";
  if (toolName !== "bash") {
    return { ok: true };
  }
  const input = isRecord(event.input) ? event.input : {};
  const rawCommand = input.command ?? input.cmd;
  const command = typeof rawCommand === "string" ? rawCommand : "";
  if (!command.includes("build_graph.py")) {
    return { ok: true };
  }
  const phase = cliPhase(command);
  const labels = {
    mineru: "MinerU PDF parsing complete",
    structure: "PaperGraph Step 1 complete",
    extract: "PaperGraph Step 2 complete",
    build: "Graph and database build complete",
    audit: "Knowledge graph audit complete",
    run: "Knowledge graph pipeline complete",
    other: "Knowledge graph command complete"
  };
  if (event.isError === true) {
    if (legalIncomplete(ctx, event)) {
      const incompleteReport = readJson(packagePaths(ctx).report);
      return {
        ok: true,
        state_patch: {
          phase: {
            id: phase,
            label: `${labels[phase]} with incomplete gates`,
            status: "incomplete"
          },
          message: "The packaged graph runtime completed and wrote an incomplete manifest; this is a recoverable quality-gate outcome, not a crash.",
          diagnostics: diagnosticsFromReport(
            incompleteReport,
            "Repair the reported graph artifact/resource/provider blocker, then rerun build_graph.py resume or the failed stage."
          ),
          next_actions: [
            "Rerun build_graph.py resume after repairs",
            "Or finish with status incomplete using manifest.json"
          ]
        }
      };
    }
    return {
      ok: true,
      state_patch: {
        phase: { id: phase, label: labels[phase], status: "blocked" },
        diagnostics: [
          {
            severity: "warning",
            message: `The knowledge graph ${phase} stage returned an error before writing a valid incomplete manifest.`,
            repair: "Inspect the packaged CLI error and rerun the same stage; completed per-paper checkpoints are preserved."
          }
        ]
      }
    };
  }
  return {
    ok: true,
    state_patch: {
      phase: { id: phase, label: labels[phase], status: "running" }
    }
  };
}
function preFinish(ctx) {
  const paths = packagePaths(ctx);
  if (!existsSync(paths.graph) || !existsSync(paths.database)) {
    return failure(
      "Run mineru, structure, extract, build, and audit before finishing.",
      "Missing artifacts/method_graph.json or artifacts/graph.db."
    );
  }
  const python = ctx.run.environment?.executable ?? "python";
  const audit = spawnSync(
    python,
    [paths.script, "audit", "--run-id", ctx.run.runId, "--run-dir", ctx.runDir],
    {
      cwd: ctx.packageDir,
      encoding: "utf-8",
      maxBuffer: 20 * 1024 * 1024,
      timeout: 6e5
    }
  );
  if (audit.error || audit.status !== 0 && audit.status !== 2 || !existsSync(paths.report)) {
    return failure(
      "Repair the reported artifact and rerun build_graph.py audit.",
      `Knowledge graph audit could not run: ${audit.error?.message || audit.stderr.trim() || `exit ${audit.status}`}`
    );
  }
  const submittedManifestPath = eventManifestPath(ctx);
  if (!submittedManifestPath || resolve(submittedManifestPath) !== resolve(paths.manifest)) {
    return failure(
      "Call xlab_finish with the canonical manifest generated by build_graph.py audit.",
      `knowledge_graph requires manifest_path ${paths.manifest}.`
    );
  }
  let report;
  let manifest;
  try {
    report = readJson(paths.report);
    manifest = readJson(submittedManifestPath);
  } catch (error) {
    return failure(
      "Repair artifacts/graph_report.json or manifest.json and rerun the audit.",
      `Cannot read graph audit artifacts: ${error instanceof Error ? error.message : String(error)}`
    );
  }
  const counts = isRecord(report.counts) ? report.counts : {};
  const counters = {
    papers: numeric(counts.input_papers),
    downloaded_pdfs: numeric(counts.downloaded_pdfs),
    parsed_papers: numeric(counts.parsed_papers),
    structured_papers: numeric(counts.structured_papers),
    extracted_papers: numeric(counts.extracted_papers),
    nodes: numeric(counts.nodes),
    edges: numeric(counts.edges),
    aliases: numeric(counts.aliases)
  };
  const manifestArtifacts = Array.isArray(manifest.artifacts) ? manifest.artifacts.filter(
    (entry) => isRecord(entry)
  ) : [];
  for (const [type, path, canonical] of [
    [
      "mineru_documents",
      artifactPathFromManifest(
        ctx,
        manifestArtifacts,
        "mineru_documents",
        "artifacts/mineru.manifest.json"
      ),
      paths.mineruManifest
    ],
    [
      "paper_structures",
      artifactPathFromManifest(
        ctx,
        manifestArtifacts,
        "paper_structures",
        "artifacts/paper_structures.manifest.json"
      ),
      paths.structuresManifest
    ],
    [
      "paper_extractions",
      artifactPathFromManifest(
        ctx,
        manifestArtifacts,
        "paper_extractions",
        "artifacts/extractions.manifest.json"
      ),
      paths.extractionsManifest
    ],
    [
      "method_graph",
      artifactPathFromManifest(
        ctx,
        manifestArtifacts,
        "method_graph",
        "artifacts/method_graph.json"
      ),
      paths.graph
    ],
    [
      "graph_db",
      artifactPathFromManifest(
        ctx,
        manifestArtifacts,
        "graph_db",
        "artifacts/graph.db"
      ),
      paths.database
    ],
    [
      "graph_report",
      artifactPathFromManifest(
        ctx,
        manifestArtifacts,
        "graph_report",
        "artifacts/graph_report.json"
      ),
      paths.report
    ]
  ]) {
    if (!isWithin(path, ctx.cwd) && !isWithin(path, ctx.runDir)) {
      return failure(
        "Use the run-local artifact paths written by build_graph.py audit.",
        `Artifact ${type} points outside the project/run workspace: ${path}`
      );
    }
    if (resolve(path) !== resolve(canonical)) {
      return failure(
        "Use the canonical manifest generated by build_graph.py audit before calling xlab_finish.",
        `Artifact ${type} must point to ${canonical}.`
      );
    }
  }
  const finishValue = eventValue(ctx).finish;
  const finish = isRecord(finishValue) ? finishValue : {};
  const finishStatus = typeof finish.status === "string" ? finish.status : "";
  const passed = report.passed === true;
  const reportStatus = passed ? "success" : "incomplete";
  const generatedStatus = typeof manifest.status === "string" ? manifest.status : reportStatus;
  if (generatedStatus !== reportStatus) {
    return failure(
      "Rerun build_graph.py audit so manifest status matches graph_report.json.",
      `manifest status ${generatedStatus} does not match report status ${reportStatus}.`,
      counters
    );
  }
  if (finishStatus === "success" && !passed) {
    const blockers = Array.isArray(report.blocking_errors) ? report.blocking_errors.filter(
      (value) => typeof value === "string"
    ) : [];
    return failure(
      "Resolve the audit blockers or finish with status incomplete.",
      blockers.join(" ") || "Knowledge graph did not pass its success gates.",
      counters
    );
  }
  if (finishStatus && finishStatus !== generatedStatus) {
    return failure(
      `Call xlab_finish with status ${generatedStatus}; the graph audit generated that status.`,
      `xlab_finish status ${finishStatus} does not match audit status ${generatedStatus}.`,
      counters
    );
  }
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "validate",
        label: passed ? "Knowledge graph validated" : "Knowledge graph incomplete",
        status: passed ? "success" : "incomplete",
        progress: 1
      },
      message: `Validated ${counters.nodes} nodes and ${counters.edges} edges from ${counters.extracted_papers} LLM-extracted papers.`,
      counters,
      artifacts: [
        {
          type: "mineru_documents",
          name: "MinerU Markdown and layout outputs",
          path: `.xlab/runs/${ctx.run.runId}/artifacts/mineru.manifest.json`,
          status: passed ? "ready" : "incomplete",
          summary: `${counters.parsed_papers} parsed PDFs`
        },
        {
          type: "paper_structures",
          name: "PaperGraph Step 1 structures",
          path: `.xlab/runs/${ctx.run.runId}/artifacts/paper_structures.manifest.json`,
          status: passed ? "ready" : "incomplete",
          summary: `${counters.structured_papers} structured papers`
        },
        {
          type: "paper_extractions",
          name: "PaperGraph Step 2 extractions",
          path: `.xlab/runs/${ctx.run.runId}/artifacts/extractions.manifest.json`,
          status: passed ? "ready" : "incomplete",
          summary: `${counters.extracted_papers} extracted papers`
        },
        {
          type: "method_graph",
          name: "Method evolution graph",
          path: `.xlab/runs/${ctx.run.runId}/artifacts/method_graph.json`,
          status: passed ? "ready" : "incomplete",
          summary: `${counters.nodes} nodes; ${counters.edges} edges`
        },
        {
          type: "graph_db",
          name: "Queryable SQLite graph",
          path: `.xlab/runs/${ctx.run.runId}/artifacts/graph.db`,
          status: passed ? "ready" : "incomplete",
          summary: `${counters.aliases} aliases`
        },
        {
          type: "graph_report",
          name: "Graph quality report",
          path: `.xlab/runs/${ctx.run.runId}/artifacts/graph_report.json`,
          status: passed ? "ready" : "incomplete"
        }
      ]
    }
  };
}
function postFinish(ctx) {
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "finish",
        label: "Knowledge graph finished",
        status: ctx.run.status,
        progress: 1
      },
      message: ctx.run.summary ?? "Knowledge graph run finished."
    }
  };
}
function onResume(ctx) {
  let report;
  const readDiagnostics = [];
  try {
    report = existsSync(packagePaths(ctx).report) ? readJson(packagePaths(ctx).report) : void 0;
  } catch (error) {
    readDiagnostics.push({
      severity: "error",
      message: `Cannot read graph report: ${error instanceof Error ? error.message : String(error)}`,
      repair: "Rerun build_graph.py audit to regenerate artifacts/graph_report.json."
    });
  }
  const counts = isRecord(report?.counts) ? report.counts : {};
  const counters = {
    papers: numeric(counts.input_papers),
    downloaded_pdfs: numeric(counts.downloaded_pdfs),
    parsed_papers: numeric(counts.parsed_papers),
    structured_papers: numeric(counts.structured_papers),
    extracted_papers: numeric(counts.extracted_papers),
    nodes: numeric(counts.nodes),
    edges: numeric(counts.edges)
  };
  const passed = report?.passed === true;
  const blockers = report && Array.isArray(report.blocking_errors) ? report.blocking_errors.filter(
    (value) => typeof value === "string"
  ) : [];
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "resume",
        label: passed ? "Knowledge graph ready to finish" : blockers.length > 0 ? "Knowledge graph blocked at quality gate" : "Resuming knowledge graph",
        status: passed ? "success" : blockers.length > 0 ? "incomplete" : "running"
      },
      message: "MinerU bundles, PaperGraph Step 1 structures, and validated per-pass LLM checkpoints will be reused.",
      counters,
      diagnostics: [
        ...readDiagnostics,
        ...report ? diagnosticsFromReport(
          report,
          "Repair the reported blocker, then rerun build_graph.py resume or the failed stage."
        ) : []
      ],
      checkpoint: {
        id: "knowledge-graph-resume",
        label: existsSync(packagePaths(ctx).extractionsManifest) ? "Extraction manifest present" : existsSync(packagePaths(ctx).structuresManifest) ? "Structure manifest present" : existsSync(packagePaths(ctx).mineruManifest) ? "MinerU manifest present" : "No reusable stage manifest",
        resumable: true,
        resume_hint: passed ? "Run build_graph.py audit and finish with status success." : "Rerun the interrupted mineru, structure, extract, build, or audit stage."
      },
      next_actions: passed ? ["Run build_graph.py audit", "Call xlab_finish with status success"] : blockers.length > 0 ? [
        "Repair the reported blocker",
        "Rerun build_graph.py resume or the failed stage",
        "Finish with status incomplete if the blocker is expected"
      ] : ["Rerun the interrupted stage", "Then run build_graph.py audit"]
    }
  };
}
function onCancel() {
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "cancel",
        label: "Knowledge graph cancelled",
        status: "incomplete"
      },
      message: "Completed MinerU, structure, and LLM checkpoints were preserved."
    }
  };
}
function onError(ctx) {
  return {
    ok: true,
    state_patch: {
      phase: {
        id: "error",
        label: "Knowledge graph interrupted",
        status: "failed"
      },
      message: ctx.run.errorSummary ?? ctx.run.lastRepair ?? "Knowledge graph stopped before completion.",
      checkpoint: {
        id: "knowledge-graph-error",
        label: "Partial graph work preserved",
        resumable: true,
        resume_hint: "Rerun the interrupted mineru, structure, extract, build, or audit stage."
      }
    }
  };
}
export {
  onCancel,
  onError,
  onResume,
  postFinish,
  postToolResult,
  preFinish,
  preStart,
  preToolCall
};
