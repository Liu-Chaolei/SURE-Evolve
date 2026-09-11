import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { validateXlabPackageSelfContained } from "./package-validator.ts";
import type {
	XlabCommandArgumentHint,
	XlabCommandDynamicSource,
	XlabConfigVariableHint,
	XlabHookDeclaration,
	XlabHookPoint,
	XlabRuntimeDeclaration,
	XlabRuntimeKind,
	XlabSkillDependency,
	XlabSkillManifest,
	XlabSkillPackage,
	XlabSkillPhaseHint,
	XlabSkillVisibility,
	XlabWorkflowStageDeclaration,
} from "./types.ts";
import { validateWorkflowStages } from "./workflow.ts";

const XLAB_SKILLS_DIR = ".xlab/skills";
const XLAB_REPOSITORY_SKILLS_DIR = "xlab/skills";
const XLAB_MANIFEST_NAME = "xlab.skill.json";

const HOOK_POINTS = new Set<XlabHookPoint>([
	"pre_start",
	"pre_stage",
	"post_stage",
	"pre_tool_call",
	"post_tool_result",
	"pre_finish",
	"post_finish",
	"on_error",
	"on_resume",
	"on_cancel",
]);

const VISIBILITIES = new Set<XlabSkillVisibility>(["public", "internal", "service-control"]);
const RUNTIME_KINDS = new Set<XlabRuntimeKind>([
	"agent",
	"node",
	"python",
	"container",
	"workflow",
	"native",
	"service-control",
]);
const DYNAMIC_SOURCES = new Set<XlabCommandDynamicSource>([
	"workspace",
	"graph",
	"survey",
	"idea",
	"run",
	"artifact",
	"file",
	"directory",
	"gpu",
]);

export interface XlabDiscoveryDiagnostic {
	type: "warning" | "error";
	message: string;
	path?: string;
}

export interface XlabDiscoveryResult {
	packages: XlabSkillPackage[];
	diagnostics: XlabDiscoveryDiagnostic[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readJson(path: string): unknown {
	return JSON.parse(readFileSync(path, "utf-8"));
}

function isPathInside(baseDir: string, candidate: string): boolean {
	const rel = relative(baseDir, candidate);
	return rel === "" || (!rel.startsWith("..") && !rel.startsWith("/") && rel !== "..");
}

function resolvePackagePath(packageDir: string, pathValue: string): string | undefined {
	const resolved = resolve(packageDir, pathValue);
	return isPathInside(packageDir, resolved) ? resolved : undefined;
}

function parseHookDeclaration(value: unknown): XlabHookDeclaration | undefined {
	if (!isRecord(value)) {
		return undefined;
	}
	const kind = value.kind === "command" ? "command" : "module";
	const module = typeof value.module === "string" && value.module.trim() !== "" ? value.module.trim() : undefined;
	const command = typeof value.command === "string" && value.command.trim() !== "" ? value.command.trim() : undefined;
	if ((kind === "module" && !module) || (kind === "command" && !command)) {
		return undefined;
	}
	return {
		kind,
		module,
		handler: typeof value.handler === "string" && value.handler.trim() !== "" ? value.handler : undefined,
		command,
		args: parseStringList(value.args),
		timeoutMs:
			typeof value.timeout_ms === "number" && Number.isFinite(value.timeout_ms) && value.timeout_ms > 0
				? value.timeout_ms
				: undefined,
		blocking: typeof value.blocking === "boolean" ? value.blocking : undefined,
	};
}

function parseHooks(value: unknown): Partial<Record<XlabHookPoint, XlabHookDeclaration[]>> | undefined {
	if (!isRecord(value)) {
		return undefined;
	}

	const hooks: Partial<Record<XlabHookPoint, XlabHookDeclaration[]>> = {};
	for (const [point, declarationValue] of Object.entries(value)) {
		if (!HOOK_POINTS.has(point as XlabHookPoint)) {
			continue;
		}

		const declarations = Array.isArray(declarationValue)
			? declarationValue
					.map(parseHookDeclaration)
					.filter((entry): entry is XlabHookDeclaration => entry !== undefined)
			: [parseHookDeclaration(declarationValue)].filter(
					(entry): entry is XlabHookDeclaration => entry !== undefined,
				);
		if (declarations.length > 0) {
			hooks[point as XlabHookPoint] = declarations;
		}
	}

	return Object.keys(hooks).length > 0 ? hooks : undefined;
}

function parseArtifacts(value: unknown): XlabSkillManifest["artifacts"] {
	if (!Array.isArray(value)) {
		return undefined;
	}
	return value.filter(isRecord).map((artifact) => ({
		type: typeof artifact.type === "string" ? artifact.type : undefined,
		schemaVersion: typeof artifact.schema_version === "string" ? artifact.schema_version : undefined,
		schema: typeof artifact.schema === "string" ? artifact.schema : undefined,
		path: typeof artifact.path === "string" ? artifact.path : undefined,
		required: typeof artifact.required === "boolean" ? artifact.required : undefined,
		description: typeof artifact.description === "string" ? artifact.description : undefined,
	}));
}

function parseStringList(value: unknown): string[] | undefined {
	if (!Array.isArray(value)) {
		return undefined;
	}
	const strings = value.filter((entry): entry is string => typeof entry === "string" && entry.trim() !== "");
	return strings.length > 0 ? strings.map((entry) => entry.trim()) : undefined;
}

function parseCommandArguments(value: unknown): XlabCommandArgumentHint[] | undefined {
	if (!Array.isArray(value)) {
		return undefined;
	}
	const arguments_: XlabCommandArgumentHint[] = value.flatMap((entry): XlabCommandArgumentHint[] => {
		if (
			!isRecord(entry) ||
			typeof entry.name !== "string" ||
			!/^--[a-z][a-z0-9-]*$|^[a-z][a-z0-9_]*$/.test(entry.name.trim()) ||
			typeof entry.description !== "string" ||
			entry.description.trim() === "" ||
			typeof entry.takesValue !== "boolean"
		) {
			return [];
		}
		const choices = parseStringList(entry.choices);
		return [
			{
				name: entry.name.trim(),
				description: entry.description.trim(),
				syntax: entry.name.trim().startsWith("--") ? "flag" : "assignment",
				valueHint:
					typeof entry.valueHint === "string" && entry.valueHint.trim() !== ""
						? entry.valueHint.trim()
						: undefined,
				choices: choices ? [...new Set(choices)] : undefined,
				dynamicSource:
					typeof entry.dynamicSource === "string" &&
					DYNAMIC_SOURCES.has(entry.dynamicSource as XlabCommandDynamicSource)
						? (entry.dynamicSource as XlabCommandDynamicSource)
						: undefined,
				takesValue: entry.takesValue,
				repeatable: typeof entry.repeatable === "boolean" ? entry.repeatable : undefined,
			},
		];
	});
	return arguments_.length > 0 ? arguments_ : undefined;
}

function parseUiHints(value: unknown): XlabSkillManifest["ui"] {
	if (!isRecord(value)) {
		return undefined;
	}
	const phases: XlabSkillPhaseHint[] | undefined = Array.isArray(value.phases)
		? value.phases.flatMap((entry): XlabSkillPhaseHint[] => {
				if (!isRecord(entry) || typeof entry.id !== "string" || entry.id.trim() === "") {
					return [];
				}
				return [
					{
						id: entry.id.trim(),
						label: typeof entry.label === "string" && entry.label.trim() !== "" ? entry.label.trim() : undefined,
						description:
							typeof entry.description === "string" && entry.description.trim() !== ""
								? entry.description.trim()
								: undefined,
					},
				];
			})
		: undefined;
	const ui = {
		primaryCounters: parseStringList(value.primaryCounters),
		artifactTypes: parseStringList(value.artifactTypes),
		defaultExpandedSections: parseStringList(value.defaultExpandedSections),
		phases: phases && phases.length > 0 ? phases : undefined,
		argumentHint:
			typeof value.argumentHint === "string" && value.argumentHint.trim() !== ""
				? value.argumentHint.trim()
				: undefined,
		arguments: parseCommandArguments(value.arguments),
	};
	return ui.primaryCounters ||
		ui.artifactTypes ||
		ui.defaultExpandedSections ||
		ui.phases ||
		ui.argumentHint ||
		ui.arguments
		? ui
		: undefined;
}

function parseSecretAliases(value: unknown): Record<string, string[]> | undefined {
	if (!isRecord(value)) {
		return undefined;
	}
	const aliases: Record<string, string[]> = {};
	for (const [canonical, rawAliases] of Object.entries(value)) {
		const name = canonical.trim();
		const values = parseStringList(rawAliases)?.filter((alias) => alias !== name) ?? [];
		if (name && values.length > 0) {
			aliases[name] = [...new Set(values)];
		}
	}
	return Object.keys(aliases).length > 0 ? aliases : undefined;
}

function parseConfigHints(value: unknown): XlabSkillManifest["config"] {
	if (!isRecord(value)) {
		return undefined;
	}
	const environment: XlabConfigVariableHint[] | undefined = Array.isArray(value.environment)
		? value.environment.flatMap((entry): XlabConfigVariableHint[] => {
				if (!isRecord(entry) || typeof entry.name !== "string" || entry.name.trim() === "") {
					return [];
				}
				return [
					{
						name: entry.name.trim(),
						description:
							typeof entry.description === "string" && entry.description.trim() !== ""
								? entry.description.trim()
								: undefined,
						secret: typeof entry.secret === "boolean" ? entry.secret : undefined,
						required: typeof entry.required === "boolean" ? entry.required : undefined,
						default: entry.default,
						aliases: parseStringList(entry.aliases),
					},
				];
			})
		: undefined;
	return environment && environment.length > 0 ? { environment } : undefined;
}

function parseRuntime(value: unknown): XlabRuntimeDeclaration | undefined {
	if (!isRecord(value) || typeof value.kind !== "string" || !RUNTIME_KINDS.has(value.kind as XlabRuntimeKind)) {
		return undefined;
	}
	return {
		kind: value.kind as XlabRuntimeKind,
		driver: typeof value.driver === "string" && value.driver.trim() !== "" ? value.driver.trim() : undefined,
		entrypoint: typeof value.entrypoint === "string" ? value.entrypoint.trim() : undefined,
		lockfile: typeof value.lockfile === "string" ? value.lockfile.trim() : undefined,
		containerfile: typeof value.containerfile === "string" ? value.containerfile.trim() : undefined,
		executables: parseStringList(value.executables),
		requiredSecrets: parseStringList(value.required_secrets),
		optionalSecrets: parseStringList(value.optional_secrets),
		secretAliases: parseSecretAliases(value.secret_aliases),
	};
}

function parseDependencies(value: unknown): XlabSkillDependency[] | undefined {
	if (!Array.isArray(value)) {
		return undefined;
	}
	const dependencies = value.flatMap((entry): XlabSkillDependency[] => {
		if (
			!isRecord(entry) ||
			typeof entry.name !== "string" ||
			entry.name.trim() === "" ||
			typeof entry.version !== "string" ||
			entry.version.trim() === ""
		) {
			return [];
		}
		return [{ name: entry.name.trim(), version: entry.version.trim() }];
	});
	return dependencies.length > 0 ? dependencies : undefined;
}

function parseWorkflow(value: unknown): XlabSkillManifest["workflow"] {
	if (!isRecord(value) || !Array.isArray(value.stages)) {
		return undefined;
	}
	const stages = value.stages.flatMap((entry): XlabWorkflowStageDeclaration[] => {
		if (
			!isRecord(entry) ||
			typeof entry.id !== "string" ||
			entry.id.trim() === "" ||
			typeof entry.skill !== "string" ||
			entry.skill.trim() === ""
		) {
			return [];
		}
		return [
			{
				id: entry.id.trim(),
				skill: entry.skill.trim(),
				needs: parseStringList(entry.needs),
				optional: typeof entry.optional === "boolean" ? entry.optional : undefined,
				maxAttempts:
					typeof entry.max_attempts === "number" && Number.isInteger(entry.max_attempts) && entry.max_attempts > 0
						? entry.max_attempts
						: undefined,
			},
		];
	});
	return stages.length > 0 ? { stages } : undefined;
}

function parseManifest(raw: unknown): XlabSkillManifest | undefined {
	if (!isRecord(raw)) {
		return undefined;
	}
	if (raw.schema_version !== "2") {
		return undefined;
	}
	if (typeof raw.name !== "string" || raw.name.trim() === "") {
		return undefined;
	}
	if (typeof raw.version !== "string" || raw.version.trim() === "") {
		return undefined;
	}
	if (typeof raw.visibility !== "string" || !VISIBILITIES.has(raw.visibility as XlabSkillVisibility)) {
		return undefined;
	}
	if (typeof raw.prompt !== "string" || raw.prompt.trim() === "") {
		return undefined;
	}
	const runtime = parseRuntime(raw.runtime);
	if (!runtime) {
		return undefined;
	}
	return {
		schemaVersion: "2",
		name: raw.name.trim(),
		version: raw.version.trim(),
		visibility: raw.visibility as XlabSkillVisibility,
		description: typeof raw.description === "string" ? raw.description.trim() : undefined,
		prompt: raw.prompt.trim(),
		runtime,
		permissions: isRecord(raw.permissions)
			? {
					tools: parseStringList(raw.permissions.tools),
					network: typeof raw.permissions.network === "boolean" ? raw.permissions.network : undefined,
					subprocess: typeof raw.permissions.subprocess === "boolean" ? raw.permissions.subprocess : undefined,
					externalMutations:
						typeof raw.permissions.external_mutations === "boolean"
							? raw.permissions.external_mutations
							: undefined,
				}
			: undefined,
		dependencies: parseDependencies(raw.dependencies),
		inputSchema: typeof raw.input_schema === "string" ? raw.input_schema.trim() : undefined,
		outputSchema: typeof raw.output_schema === "string" ? raw.output_schema.trim() : undefined,
		workflow: parseWorkflow(raw.workflow),
		hooks: parseHooks(raw.hooks),
		artifacts: parseArtifacts(raw.artifacts),
		ui: parseUiHints(raw.ui),
		config: parseConfigHints(raw.config),
	};
}

function loadPackage(
	manifestPath: string,
	source: XlabSkillPackage["source"],
	sourceRoot: string,
	diagnostics: XlabDiscoveryDiagnostic[],
): XlabSkillPackage | undefined {
	let parsed: XlabSkillManifest | undefined;
	try {
		parsed = parseManifest(readJson(manifestPath));
	} catch (error) {
		diagnostics.push({
			type: "error",
			message: `Invalid XLab skill manifest JSON: ${error instanceof Error ? error.message : String(error)}`,
			path: manifestPath,
		});
		return undefined;
	}

	if (!parsed) {
		diagnostics.push({
			type: "error",
			message:
				"Invalid XLab skill manifest v2: expected schema_version, name, version, visibility, prompt, and runtime",
			path: manifestPath,
		});
		return undefined;
	}

	const packageDir = dirname(manifestPath);
	const promptPath = resolvePackagePath(packageDir, parsed.prompt);
	if (!promptPath) {
		diagnostics.push({
			type: "error",
			message: "XLab skill prompt path must stay inside the skill package",
			path: manifestPath,
		});
		return undefined;
	}
	if (!existsSync(promptPath)) {
		diagnostics.push({
			type: "error",
			message: "XLab skill prompt path does not exist",
			path: promptPath,
		});
		return undefined;
	}

	const containedPaths = [
		parsed.inputSchema,
		parsed.outputSchema,
		parsed.runtime.entrypoint,
		parsed.runtime.lockfile,
		parsed.runtime.containerfile,
		...(parsed.artifacts ?? []).map((artifact) => artifact.schema),
	].filter((value): value is string => typeof value === "string" && value !== "");
	for (const containedPath of containedPaths) {
		const resolvedPath = resolvePackagePath(packageDir, containedPath);
		if (!resolvedPath || !existsSync(resolvedPath)) {
			diagnostics.push({
				type: "error",
				message: "XLab manifest referenced path must exist inside the skill package",
				path: resolvedPath ?? join(packageDir, containedPath),
			});
			return undefined;
		}
	}

	if (parsed.workflow) {
		const workflowError = validateWorkflowStages(parsed.workflow.stages);
		if (workflowError) {
			diagnostics.push({
				type: "error",
				message: workflowError,
				path: manifestPath,
			});
			return undefined;
		}
	}

	for (const declarations of Object.values(parsed.hooks ?? {})) {
		for (const declaration of declarations) {
			const hookTarget = declaration.kind === "command" ? declaration.command : declaration.module;
			if (!hookTarget) {
				continue;
			}
			const hookPath = resolvePackagePath(packageDir, hookTarget);
			if (!hookPath || !existsSync(hookPath)) {
				diagnostics.push({
					type: "error",
					message: "XLab hook module path must exist inside the skill package",
					path: hookPath ?? join(packageDir, hookTarget),
				});
				return undefined;
			}
		}
	}
	const selfContainedIssues = validateXlabPackageSelfContained(packageDir);
	if (selfContainedIssues.length > 0) {
		for (const issue of selfContainedIssues) {
			diagnostics.push({
				type: "error",
				message: issue.message,
				path: issue.path,
			});
		}
		return undefined;
	}

	return {
		manifest: parsed,
		manifestPath,
		packageDir,
		promptPath,
		prompt: readFileSync(promptPath, "utf-8").trim(),
		source,
		sourceRoot,
		dependencies: [],
	};
}

function discoverPackagesInRoot(
	root: string,
	source: XlabSkillPackage["source"],
	diagnostics: XlabDiscoveryDiagnostic[],
): XlabSkillPackage[] {
	const packages: XlabSkillPackage[] = [];
	if (!existsSync(root)) {
		return packages;
	}

	for (const entry of readdirSync(root)) {
		const packageDir = join(root, entry);
		if (!statSync(packageDir).isDirectory()) {
			continue;
		}
		const manifestPath = join(packageDir, XLAB_MANIFEST_NAME);
		if (!existsSync(manifestPath)) {
			continue;
		}
		const skillPackage = loadPackage(manifestPath, source, root, diagnostics);
		if (skillPackage) {
			packages.push(skillPackage);
		}
	}
	return packages;
}

export function discoverXlabSkillPackages(cwd: string): XlabDiscoveryResult {
	const diagnostics: XlabDiscoveryDiagnostic[] = [];
	const projectPackages = discoverPackagesInRoot(join(cwd, XLAB_SKILLS_DIR), "project", diagnostics);
	const repositoryPackages = discoverPackagesInRoot(join(cwd, XLAB_REPOSITORY_SKILLS_DIR), "repository", diagnostics);

	const projectNames = new Set(projectPackages.map((skillPackage) => skillPackage.manifest.name));
	const packagesWithoutDependencies = [
		...projectPackages,
		...repositoryPackages.filter((skillPackage) => {
			if (!projectNames.has(skillPackage.manifest.name)) {
				return true;
			}
			diagnostics.push({
				type: "warning",
				message: `Repository XLab skill "${skillPackage.manifest.name}" is overridden by project .xlab/skills.`,
				path: skillPackage.manifestPath,
			});
			return false;
		}),
	];
	const byName = new Map(
		packagesWithoutDependencies.map((skillPackage) => [skillPackage.manifest.name, skillPackage]),
	);
	const packages = packagesWithoutDependencies.filter((skillPackage) => {
		const resolvedDependencies: XlabSkillPackage[] = [];
		for (const dependency of skillPackage.manifest.dependencies ?? []) {
			const target = byName.get(dependency.name);
			if (!target || target.manifest.version !== dependency.version) {
				diagnostics.push({
					type: "error",
					message: `XLab skill "${skillPackage.manifest.name}" requires ${dependency.name}@${dependency.version}.`,
					path: skillPackage.manifestPath,
				});
				return false;
			}
			resolvedDependencies.push(target);
		}
		const workflowSkills = new Set(skillPackage.manifest.workflow?.stages.map((stage) => stage.skill) ?? []);
		for (const workflowSkill of workflowSkills) {
			if (
				workflowSkill !== skillPackage.manifest.name &&
				!resolvedDependencies.some((dependency) => dependency.manifest.name === workflowSkill)
			) {
				diagnostics.push({
					type: "error",
					message: `Workflow stage skill "${workflowSkill}" must be declared as an exact dependency.`,
					path: skillPackage.manifestPath,
				});
				return false;
			}
		}
		skillPackage.dependencies = resolvedDependencies;
		return true;
	});

	return {
		packages,
		diagnostics,
	};
}

export function resolveXlabPackagePath(packageDir: string, pathValue: string): string | undefined {
	return resolvePackagePath(packageDir, pathValue);
}
