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
        return Number.isFinite(parsed) ? parsed : undefined;
    }
    return undefined;
}
function failure(repair, message, counters) {
    return {
        ok: false,
        repair,
        state_patch: {
            phase: {
                id: "validate",
                label: "Validating paper collection",
                status: "blocked",
            },
            message,
            counters,
            diagnostics: [{ severity: "error", message, repair }],
        },
    };
}
function paths(ctx) {
    const artifacts = join(ctx.runDir, "artifacts");
    return {
        script: join(ctx.packageDir, "scripts", "paper_collect.py"),
        collection: join(artifacts, "papers.manifest.json"),
        edges: join(artifacts, "metadata", "edges.jsonl"),
        report: join(artifacts, "collection_report.json"),
        providerLog: join(artifacts, "logs", "provider_results.jsonl"),
        manifest: join(ctx.runDir, "manifest.json"),
    };
}
function providerStats(path) {
    if (!existsSync(path)) {
        return { logical: 0, successful: 0, failed: 0 };
    }
    const latest = new Map();
    for (const line of readFileSync(path, "utf-8").split(/\r?\n/)) {
        if (line.trim() === "") {
            continue;
        }
        try {
            const value = JSON.parse(line);
            if (!isRecord(value) || typeof value.operation !== "string") {
                continue;
            }
            const input = isRecord(value.input) ? { ...value.input } : {};
            if (value.operation === "semantic_scholar.get_paper_citations" ||
                value.operation === "semantic_scholar.get_paper_references" ||
                value.operation === "semantic_scholar.get_recommendations") {
                delete input.fields;
            }
            latest.set(JSON.stringify([value.operation, input]), value);
        }
        catch {
            continue;
        }
    }
    const records = [...latest.values()];
    const failed = records.filter((record) => record.is_error === true).length;
    return {
        logical: records.length,
        successful: records.length - failed,
        failed,
    };
}
function readJson(path) {
    const value = JSON.parse(readFileSync(path, "utf-8"));
    if (!isRecord(value)) {
        throw new Error(`${path} must contain a JSON object.`);
    }
    return value;
}
function eventManifestPath(ctx) {
    const manifestPath = eventValue(ctx).manifestPath;
    return typeof manifestPath === "string" && manifestPath.trim() !== ""
        ? resolveProjectOrRunPath(ctx, manifestPath)
        : undefined;
}
function eventValue(ctx) {
    return isRecord(ctx.event) ? ctx.event : {};
}
function cliPhase(command) {
    const match = command.match(/paper_collect\.py\s+(collect|resume|download|audit|run|smoke)\b/);
    if (match?.[1] === "collect" ||
        match?.[1] === "resume" ||
        match?.[1] === "run") {
        return "collect";
    }
    if (match?.[1] === "download") {
        return "download";
    }
    if (match?.[1] === "audit" || match?.[1] === "smoke") {
        return "audit";
    }
    return "other";
}
function isWithin(path, root) {
    const relativePath = relative(resolve(root), resolve(path));
    return (relativePath === "" ||
        (!relativePath.startsWith("..") && !isAbsolute(relativePath)));
}
function resolveProjectOrRunPath(ctx, path) {
    return isAbsolute(path) ? resolve(path) : resolve(ctx.cwd, path);
}
function artifactPathFromManifest(ctx, artifacts, type, fallback) {
    const artifact = artifacts.find((entry) => entry.type === type);
    if (artifact?.path) {
        return resolveProjectOrRunPath(ctx, artifact.path);
    }
    return join(ctx.runDir, fallback);
}
function diagnosticsFromReport(report, repair) {
    const warnings = Array.isArray(report.warnings)
        ? report.warnings.filter((value) => typeof value === "string")
        : [];
    const blockers = Array.isArray(report.blocking_errors)
        ? report.blocking_errors.filter((value) => typeof value === "string")
        : [];
    return [
        ...warnings.map((message) => ({ severity: "warning", message })),
        ...blockers.map((message) => ({
            severity: "error",
            message,
            repair,
        })),
    ];
}
const SHELL_CONTROL_PATTERN = /(?:[;&|<>`]|\$\(|\n|\r)/;
const ALLOWED_CLI_PHASES = new Set(["collect", "resume", "download", "audit"]);
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
                quote = undefined;
            }
            else {
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
function parseSimpleCommand(command) {
    if (SHELL_CONTROL_PATTERN.test(command)) {
        return undefined;
    }
    const tokens = splitCommand(command);
    if (!tokens || tokens.length === 0) {
        return undefined;
    }
    return { executable: tokens[0], args: tokens.slice(1) };
}
function resolveRunArgument(commandPath, ctx) {
    return isAbsolute(commandPath)
        ? resolve(commandPath)
        : resolve(ctx.cwd, commandPath);
}
function isPackagedScriptPath(commandPath, ctx) {
    return isAbsolute(commandPath) && resolve(commandPath) === resolve(paths(ctx).script);
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
        return script ? { script, args } : undefined;
    }
    return { script: parsed.executable, args: parsed.args };
}
function optionValues(args, name) {
    const values = [];
    for (let index = 0; index < args.length; index += 1) {
        const value = args[index];
        if (value === name && args[index + 1] !== undefined) {
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
    return (values.length === 1 && resolveRunArgument(values[0] ?? "", ctx) === resolve(ctx.runDir));
}
function parsePackagedCliCommand(command, ctx) {
    const parsed = parseSimpleCommand(command);
    if (!parsed) {
        return undefined;
    }
    const script = packagedScriptArgument(parsed, ctx);
    if (!script || !isPackagedScriptPath(script.script, ctx)) {
        return undefined;
    }
    const phase = script.args[0];
    if (!ALLOWED_CLI_PHASES.has(phase)) {
        return undefined;
    }
    if (!runDirValuesAreExpected(optionValues(script.args.slice(1), "--run-dir"), ctx)) {
        return undefined;
    }
    return { ...parsed, phase: phase };
}
function isAllowedPackagedCli(command, ctx) {
    return parsePackagedCliCommand(command, ctx) !== undefined;
}
function packagedCliRejection(command, ctx) {
    const parsed = parseSimpleCommand(command);
    if (!parsed) {
        return {
            repair: "Run paper_collect.py as one simple command without shell control operators or redirects.",
            message: "Blocked paper_collect CLI invocation with unsupported shell syntax.",
        };
    }
    const script = packagedScriptArgument(parsed, ctx);
    if (!script || !isPackagedScriptPath(script.script, ctx)) {
        return {
            repair: `Use the packaged paper_collect CLI at ${paths(ctx).script}.`,
            message: "Blocked paper_collect.py invocation outside the active skill package.",
        };
    }
    const phase = script.args[0];
    if (!ALLOWED_CLI_PHASES.has(phase)) {
        return {
            repair: "Use only paper_collect.py collect, resume, download, or audit in the slash-command runtime. The run and smoke commands are not workflow phases here.",
            message: `Blocked unsupported paper_collect CLI phase: ${phase || "missing"}.`,
        };
    }
    if (!runDirValuesAreExpected(optionValues(script.args.slice(1), "--run-dir"), ctx)) {
        return {
            repair: `Pass --run-dir ${ctx.runDir} for the active XLab run.`,
            message: "Blocked paper_collect CLI invocation for a different run directory.",
        };
    }
    return undefined;
}
const READ_ONLY_INSPECTION_TOOLS = new Set(["jq", "ls", "find", "rg", "grep", "test", "["]);
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
    const pythonJsonTool = isAllowedPythonExecutable(parsed.executable, ctx) &&
        parsed.args[0] === "-m" &&
        parsed.args[1] === "json.tool";
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
        if (parsed.args.some((value) => value === "--pre" || value.startsWith("--pre=") || value === "--pre-glob" || value.startsWith("--pre-glob="))) {
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
    const pythonJsonTool = isAllowedPythonExecutable(parsed.executable, ctx) &&
        parsed.args[0] === "-m" &&
        parsed.args[1] === "json.tool";
    if (!pythonJsonTool && !READ_ONLY_INSPECTION_TOOLS.has(executable)) {
        return false;
    }
    if (/\b(?:curl|wget|python\s+-c|node\s+-e|perl\s+-e|nc|ssh|scp|rsync)\b|https?:\/\//i.test(command)) {
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
export function preStart(ctx) {
    const required = [
        join(ctx.packageDir, "scripts", "paper_collect.py"),
        join(ctx.packageDir, "scripts", "paper_collect_lib", "providers.py"),
        join(ctx.packageDir, "scripts", "paper_collect_lib", "pipeline.py"),
        join(ctx.packageDir, "scripts", "paper_collect_lib", "metadata.py"),
        join(ctx.packageDir, "scripts", "paper_collect_lib", "download.py"),
        join(ctx.packageDir, "scripts", "paper_collect_lib", "audit.py"),
        join(ctx.packageDir, "schemas", "paper_collection.schema.json"),
    ];
    const missing = required.filter((path) => !existsSync(path));
    if (missing.length > 0) {
        return failure("Restore the missing files inside the paper_collect skill package.", `Paper collection package is incomplete: ${missing.join(", ")}`);
    }
    for (const directory of [
        join(ctx.runDir, "artifacts"),
        join(ctx.runDir, "artifacts", "metadata"),
        join(ctx.runDir, "artifacts", "pdfs"),
        join(ctx.runDir, "artifacts", "logs"),
    ]) {
        mkdirSync(directory, { recursive: true });
    }
    const python = ctx.run.environment?.executable ?? "python";
    const initialization = spawnSync(python, [
        paths(ctx).script,
        "init",
        "--arguments",
        ctx.args,
        "--run-id",
        ctx.run.runId,
        "--run-dir",
        ctx.runDir,
    ], {
        cwd: ctx.packageDir,
        encoding: "utf-8",
        maxBuffer: 10 * 1024 * 1024,
        timeout: 30_000,
    });
    if (initialization.error || initialization.status !== 0) {
        const detail = initialization.error?.message ||
            initialization.stderr.trim() ||
            initialization.stdout.trim() ||
            `exit ${initialization.status}`;
        return failure('Use /xlab collect-papers "<topic>" --target-papers N --max-papers N; repeat --facet "<facet>" as needed.', `Invalid paper collection arguments: ${detail}`);
    }
    let initialized;
    try {
        const value = JSON.parse(initialization.stdout);
        if (!isRecord(value)) {
            throw new Error("initialization output must be a JSON object");
        }
        initialized = value;
    }
    catch (error) {
        return failure("Run paper_collect.py init manually and inspect its JSON output.", `Paper collection initialization returned invalid JSON: ${error instanceof Error ? error.message : String(error)}`);
    }
    return {
        ok: true,
        state_patch: {
            phase: { id: "seed", label: "Planning seed searches", status: "running" },
            message: `Initialized paper collection for ${String(initialized.query)}.`,
            counters: {
                target_papers: numeric(initialized.target_papers) ?? 0,
                max_papers: numeric(initialized.max_papers) ?? 0,
                api_calls: 0,
            },
            checkpoint: {
                id: "paper-collect-start",
                label: "Collection initialized",
                resumable: true,
                resume_hint: "Rerun the interrupted collect, download, or audit phase; completed work is reused.",
            },
        },
    };
}
export function preToolCall(ctx) {
    const event = eventValue(ctx);
    const toolName = typeof event.toolName === "string" ? event.toolName : "";
    const input = isRecord(event.input) ? event.input : {};
    if (toolName.startsWith("mcp__")) {
        return failure("Run paper_collect.py run or resume. Provider access is implemented by the packaged Python CLI.", "MCP search is invoked by the packaged Python client, not as an injected tool call.");
    }
    if (toolName !== "bash") {
        if (toolName === "read") {
            const rawPath = input.file_path ?? input.path;
            const path = typeof rawPath === "string" ? rawPath : "";
            const resolved = path ? resolveRunArgument(path, ctx) : "";
            if (resolved &&
                [ctx.runDir, ctx.packageDir].some((root) => isWithin(resolved, root))) {
                return { ok: true };
            }
            return failure("Read only files inside the active run directory or paper_collect package.", "Blocked read outside the paper_collect run/package scope.");
        }
        return failure("Use only read-only inspection plus the packaged paper_collect.py CLI; final artifacts must be produced by the CLI.", `paper_collect does not allow the ${toolName || "unknown"} tool during a run.`);
    }
    const rawCommand = input.command ?? input.cmd;
    const command = typeof rawCommand === "string" ? rawCommand : "";
    if (/\b(?:curl|wget)\b/i.test(command) ||
        /api\.tavily\.com|api\.semanticscholar\.org/i.test(command)) {
        return failure("Use the packaged paper_collect.py CLI so retries, checkpoints, normalization, and audit evidence remain consistent.", "Direct provider HTTP commands bypass the collection contract.");
    }
    if (/\bmcp(?:__|[-_. ]?server|[-_. ]?config|\s+list|\s+inspect)/i.test(command)) {
        return failure("Use the configured WEB_SEARCH_MCP_URL through paper_collect.py; do not bypass the packaged provider client.", "MCP search configuration is consumed by the paper_collect provider client.");
    }
    if (command.includes("paper_collect.py")) {
        const rejection = packagedCliRejection(command, ctx);
        if (rejection) {
            return failure(rejection.repair, rejection.message);
        }
    }
    if (/\b(?:python(?:3)?\s+-c|node\s+-e|tee|cp|mv|touch)\b|>/.test(command) &&
        /(?:artifacts\/(?:papers\.manifest\.json|collection_report\.json|metadata\/edges\.jsonl)|manifest\.json)/.test(command)) {
        return failure("Produce final collection artifacts through paper_collect.py so provider evidence, audit counters, and manifest paths remain canonical.", "Blocked an ad hoc command that could overwrite paper_collect final artifacts.");
    }
    if (!isAllowedPackagedCli(command, ctx) &&
        !isRunOrPackageInspection(command, ctx)) {
        return failure("Run the packaged paper_collect phase CLI, or use read-only inspection scoped to the run/package directory.", "Blocked a Bash command outside the paper_collect control plane.");
    }
    return { ok: true };
}
export function postToolResult(ctx) {
    const event = eventValue(ctx);
    const toolName = typeof event.toolName === "string" ? event.toolName : "";
    if (toolName !== "bash") {
        return { ok: true };
    }
    const input = isRecord(event.input) ? event.input : {};
    const rawCommand = input.command ?? input.cmd;
    const command = typeof rawCommand === "string" ? rawCommand : "";
    if (!command.includes("paper_collect.py")) {
        return { ok: true };
    }
    const phase = cliPhase(command);
    const stats = providerStats(paths(ctx).providerLog);
    if (event.isError === true) {
        const serializedEvent = JSON.stringify(event);
        const timedOut = /timed?\s*out|timeout/i.test(serializedEvent);
        const repair = timedOut
            ? `Rerun the ${phase} phase with a longer bash timeout; its checkpoints make this safe.`
            : phase === "collect"
                ? "Repair the reported provider or authentication error, then rerun paper_collect.py resume."
                : `Inspect the reported ${phase} error and rerun the same phase.`;
        return {
            ok: true,
            state_patch: {
                phase: {
                    id: phase,
                    label: `Paper collection ${phase} phase`,
                    status: "blocked",
                },
                counters: { api_calls: stats.logical },
                diagnostics: [
                    {
                        severity: "warning",
                        message: timedOut
                            ? `The paper collection ${phase} phase exceeded the tool timeout.`
                            : `The paper collection ${phase} phase returned an error.`,
                        repair,
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
                label: phase === "collect"
                    ? "Provider metadata and relation collection complete"
                    : phase === "download"
                        ? "PDF download phase complete"
                        : "Paper collection audit complete",
                status: "running",
            },
            counters: { api_calls: stats.logical },
        },
    };
}
export function preFinish(ctx) {
    const packagePaths = paths(ctx);
    if (!existsSync(packagePaths.collection)) {
        return failure("Run paper_collect.py collect before download and audit.", "Missing artifacts/papers.manifest.json.");
    }
    const python = ctx.run.environment?.executable ?? "python";
    const result = spawnSync(python, [
        packagePaths.script,
        "audit",
        "--run-dir",
        ctx.runDir,
        "--run-id",
        ctx.run.runId,
    ], {
        cwd: ctx.packageDir,
        encoding: "utf-8",
        maxBuffer: 10 * 1024 * 1024,
        timeout: 600_000,
    });
    if (result.error || result.status !== 0) {
        return failure("Run the Python audit manually, repair the reported artifact, and retry xlab_finish.", `Paper collection audit failed: ${result.error?.message ?? result.stderr.trim() ?? `exit ${result.status}`}`);
    }
    const submittedManifestPath = eventManifestPath(ctx);
    if (!submittedManifestPath || resolve(submittedManifestPath) !== resolve(packagePaths.manifest)) {
        return failure("Call xlab_finish with the canonical manifest generated by paper_collect.py audit.", `paper_collect requires manifest_path ${packagePaths.manifest}.`);
    }
    let report;
    let collection;
    let manifest;
    try {
        report = readJson(packagePaths.report);
        collection = readJson(packagePaths.collection);
        manifest = readJson(submittedManifestPath);
    }
    catch (error) {
        return failure("Repair the collection JSON artifacts and rerun paper_collect.py audit.", `Cannot read collection audit artifacts: ${error instanceof Error ? error.message : String(error)}`);
    }
    const manifestArtifacts = Array.isArray(manifest.artifacts)
        ? manifest.artifacts.filter((entry) => isRecord(entry))
        : [];
    for (const [type, path, canonical] of [
        [
            "paper_set",
            artifactPathFromManifest(ctx, manifestArtifacts, "paper_set", "artifacts/papers.manifest.json"),
            packagePaths.collection,
        ],
        [
            "paper_edges",
            artifactPathFromManifest(ctx, manifestArtifacts, "paper_edges", "artifacts/metadata/edges.jsonl"),
            packagePaths.edges,
        ],
        [
            "collection_report",
            artifactPathFromManifest(ctx, manifestArtifacts, "collection_report", "artifacts/collection_report.json"),
            packagePaths.report,
        ],
    ]) {
        if (!isWithin(path, ctx.cwd) && !isWithin(path, ctx.runDir)) {
            return failure("Use the run-local artifact paths written by paper_collect.py audit.", `Artifact ${type} points outside the project/run workspace: ${path}`);
        }
        if (resolve(path) !== resolve(canonical)) {
            return failure("Use the canonical manifest generated by paper_collect.py audit before calling xlab_finish.", `Artifact ${type} must point to ${canonical}.`);
        }
    }
    const counts = isRecord(report.counts) ? report.counts : {};
    const counters = {
        api_calls: numeric(counts.logical_api_calls) ?? 0,
        collected_papers: numeric(counts.papers) ?? 0,
        target_papers: numeric(counts.target_papers) ?? 0,
        edges: numeric(counts.edges) ?? 0,
        downloaded_pdfs: numeric(counts.downloaded_pdfs) ?? 0,
        failed_downloads: numeric(counts.failed_downloads) ?? 0,
    };
    const finishValue = eventValue(ctx).finish;
    const finish = isRecord(finishValue) ? finishValue : {};
    const finishStatus = typeof finish.status === "string" ? finish.status : "";
    const passed = report.passed === true;
    if (finishStatus === "success" && !passed) {
        const blockers = Array.isArray(report.blocking_errors)
            ? report.blocking_errors.filter((value) => typeof value === "string")
            : [];
        return failure("Resolve the audit blockers, or finish with status incomplete using the generated manifest.", blockers.join(" ") || "Paper collection did not pass the success gates.", counters);
    }
    const reportStatus = passed ? "success" : "incomplete";
    const generatedStatus = typeof manifest.status === "string" ? manifest.status : reportStatus;
    if (generatedStatus !== reportStatus) {
        return failure("Rerun paper_collect.py audit so manifest status matches collection_report.json.", `manifest status ${generatedStatus} does not match report status ${reportStatus}.`, counters);
    }
    if (finishStatus && finishStatus !== generatedStatus) {
        return failure(`Call xlab_finish with status ${generatedStatus}; the Python audit generated that manifest status.`, `xlab_finish status ${finishStatus} does not match audit status ${generatedStatus}.`, counters);
    }
    const collectedCount = numeric(collection.collected_count) ?? counters.collected_papers;
    return {
        ok: true,
        state_patch: {
            phase: {
                id: "validate",
                label: passed
                    ? "Paper collection validated"
                    : "Paper collection incomplete",
                status: passed ? "success" : "incomplete",
                progress: 1,
            },
            message: `Validated ${collectedCount} papers and ${counters.edges} retained relations.`,
            counters,
            artifacts: [
                {
                    type: "paper_set",
                    name: "Graph-ready paper collection",
                    path: `.xlab/runs/${ctx.run.runId}/artifacts/papers.manifest.json`,
                    status: passed ? "ready" : "incomplete",
                    summary: `${collectedCount} papers; ${counters.downloaded_pdfs} PDFs`,
                },
                {
                    type: "paper_edges",
                    name: "Citation and recommendation edges",
                    path: `.xlab/runs/${ctx.run.runId}/artifacts/metadata/edges.jsonl`,
                    status: passed ? "ready" : "incomplete",
                    summary: `${counters.edges} retained relations`,
                },
                {
                    type: "collection_report",
                    name: "Collection quality report",
                    path: `.xlab/runs/${ctx.run.runId}/artifacts/collection_report.json`,
                    status: passed ? "ready" : "incomplete",
                    summary: `${counters.api_calls} logical API calls; ${counters.failed_downloads} failed downloads`,
                },
            ],
        },
    };
}
export function postFinish(ctx) {
    return {
        ok: true,
        state_patch: {
            phase: {
                id: "finish",
                label: "Paper collection finished",
                status: ctx.run.status,
                progress: 1,
            },
            message: ctx.run.summary ?? "Paper collection run finished.",
        },
    };
}
export function onResume(ctx) {
    let report;
    const readDiagnostics = [];
    try {
        report = existsSync(paths(ctx).report)
            ? readJson(paths(ctx).report)
            : undefined;
    }
    catch (error) {
        readDiagnostics.push({
            severity: "error",
            message: `Cannot read collection report: ${error instanceof Error ? error.message : String(error)}`,
            repair: "Rerun paper_collect.py audit to regenerate artifacts/collection_report.json.",
        });
    }
    const counts = isRecord(report?.counts) ? report.counts : {};
    const counters = {
        api_calls: numeric(counts.logical_api_calls) ??
            providerStats(paths(ctx).providerLog).logical,
        collected_papers: numeric(counts.papers) ?? 0,
        edges: numeric(counts.edges) ?? 0,
        downloaded_pdfs: numeric(counts.downloaded_pdfs) ?? 0,
        failed_downloads: numeric(counts.failed_downloads) ?? 0,
    };
    const passed = report?.passed === true;
    const blockers = report && Array.isArray(report.blocking_errors)
        ? report.blocking_errors.filter((value) => typeof value === "string")
        : [];
    return {
        ok: true,
        state_patch: {
            phase: {
                id: "resume",
                label: passed
                    ? "Paper collection ready to finish"
                    : blockers.length > 0
                        ? "Paper collection blocked at quality gate"
                        : "Resuming paper collection",
                status: passed
                    ? "success"
                    : blockers.length > 0
                        ? "incomplete"
                        : "running",
            },
            message: existsSync(paths(ctx).collection)
                ? "Existing paper manifest found; rerun collect, download, or audit as needed."
                : "No paper manifest was found; rerun paper_collect.py collect.",
            counters,
            diagnostics: [
                ...readDiagnostics,
                ...(report
                    ? diagnosticsFromReport(report, "Repair the reported blocker, then rerun paper_collect.py resume, download, or audit.")
                    : []),
            ],
            checkpoint: {
                id: "paper-collect-resume",
                label: existsSync(paths(ctx).collection)
                    ? "Collection manifest present"
                    : "No reusable collection manifest",
                resumable: true,
                resume_hint: passed
                    ? "Run paper_collect.py audit and finish with status success."
                    : "Rerun the interrupted collect, download, or audit phase with this run directory.",
            },
            next_actions: passed
                ? ["Run paper_collect.py audit", "Call xlab_finish with status success"]
                : blockers.length > 0
                    ? [
                        "Repair the reported blocker",
                        "Rerun paper_collect.py resume or audit",
                        "Finish with status incomplete if the blocker is expected",
                    ]
                    : [
                        "Run paper_collect.py resume",
                        "Run paper_collect.py download if PDFs are missing",
                        "Then run paper_collect.py audit",
                    ],
        },
    };
}
export function onCancel() {
    return {
        ok: true,
        state_patch: {
            phase: {
                id: "cancel",
                label: "Paper collection cancelled",
                status: "incomplete",
            },
            message: "Provider responses, partial metadata, and PDF part files were preserved for resume.",
        },
    };
}
export function onError(ctx) {
    return {
        ok: true,
        state_patch: {
            phase: {
                id: "error",
                label: "Paper collection interrupted",
                status: "failed",
            },
            message: ctx.run.errorSummary ??
                ctx.run.lastRepair ??
                "Paper collection stopped before completion.",
            checkpoint: {
                id: "paper-collect-error",
                label: "Partial collection preserved",
                resumable: true,
                resume_hint: "Rerun the interrupted collect, download, or audit phase with this run directory.",
            },
        },
    };
}
