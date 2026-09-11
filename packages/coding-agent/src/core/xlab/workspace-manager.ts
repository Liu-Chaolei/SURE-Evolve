import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join, relative } from "node:path";
import type {
	XlabArtifactReference,
	XlabManifestEnvelope,
	XlabRunRecord,
	XlabWorkspaceArtifactFamily,
	XlabWorkspaceCurrent,
	XlabWorkspaceHandleRecord,
	XlabWorkspaceIndex,
	XlabWorkspaceIndexEntry,
	XlabWorkspaceLineage,
	XlabWorkspaceRecord,
} from "./types.ts";

const XLAB_WORKSPACES_DIR = ".xlab/workspaces";
const INDEX_NAME = "index.json";
const WORKSPACE_NAME = "workspace.json";
const CURRENT_NAME = "current.json";
const LINEAGE_NAME = "lineage.json";
const DEFAULT_WORKSPACE_SLUG = "xlab-workspace";
const HANDLE_FAMILY_PREFIX: Record<XlabWorkspaceArtifactFamily, string> = {
	graph: "graph",
	survey: "survey",
	idea: "idea",
	other: "artifact",
};

export interface XlabWorkspaceSelection {
	slug: string;
	explicit: boolean;
	args: string;
}

export interface XlabWorkspaceRegisterResult {
	workspace: XlabWorkspaceRecord;
	handles: XlabWorkspaceHandleRecord[];
	current: XlabWorkspaceCurrent;
	lineage: XlabWorkspaceLineage;
	warnings: string[];
}

export interface XlabWorkspaceResolvedHandle {
	workspace: XlabWorkspaceRecord;
	handle: XlabWorkspaceHandleRecord;
	warnings: string[];
}

function nowIso(): string {
	return new Date().toISOString();
}

function writeJson(path: string, value: unknown): void {
	writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
}

function readJson<T>(path: string): T {
	return JSON.parse(readFileSync(path, "utf-8")) as T;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function uniqueStrings(values: string[]): string[] {
	return [...new Set(values.filter((value) => value.trim() !== ""))];
}

function titleFromSlug(slug: string): string {
	return slug
		.split("-")
		.filter(Boolean)
		.map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
		.join(" ");
}

function parseShellTokens(value: string): string[] {
	const tokens: string[] = [];
	let current = "";
	let quote: "'" | '"' | undefined;
	let escaping = false;
	for (const char of value) {
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
	if (current) {
		tokens.push(current);
	}
	return tokens;
}

function shellQuote(value: string): string {
	if (/^[A-Za-z0-9_./:@+=,-]+$/.test(value)) {
		return value;
	}
	return `'${value.replaceAll("'", `'\\''`)}'`;
}

function readFlag(tokens: string[], flag: string): string | undefined {
	for (let index = 0; index < tokens.length; index++) {
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

function positionalTokens(tokens: string[]): string[] {
	const valueFlags = new Set([
		"--workspace",
		"--topic",
		"--input",
		"--papers",
		"--paper-set",
		"--graph",
		"--survey",
		"--idea",
		"--mature-idea",
		"--refinement-scope",
		"--discussion",
		"--experiment-feedback",
		"--language",
		"--depth",
		"--max-papers",
		"--min-papers",
		"--facet",
	]);
	const positionals: string[] = [];
	for (let index = 0; index < tokens.length; index++) {
		const token = tokens[index];
		if (token.startsWith("--")) {
			if (!token.includes("=") && valueFlags.has(token)) {
				index += 1;
			}
			continue;
		}
		positionals.push(token);
	}
	return positionals;
}

function removeFlag(tokens: string[], flag: string): string[] {
	const next: string[] = [];
	for (let index = 0; index < tokens.length; index++) {
		const token = tokens[index];
		if (token === flag) {
			index += 1;
			continue;
		}
		if (token.startsWith(`${flag}=`)) {
			continue;
		}
		next.push(token);
	}
	return next;
}

function replaceFlag(tokens: string[], flag: string, value: string): string[] {
	let replaced = false;
	const next: string[] = [];
	for (let index = 0; index < tokens.length; index++) {
		const token = tokens[index];
		if (token === flag) {
			next.push(flag, value);
			index += 1;
			replaced = true;
			continue;
		}
		if (token.startsWith(`${flag}=`)) {
			next.push(flag, value);
			replaced = true;
			continue;
		}
		next.push(token);
	}
	if (!replaced) {
		next.push(flag, value);
	}
	return next;
}

function serializeTokens(tokens: string[]): string {
	return tokens.map(shellQuote).join(" ");
}

export function slugifyWorkspaceName(value: string): string {
	const normalized = value
		.normalize("NFKD")
		.replace(/[̀-ͯ]/g, "")
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.replace(/-{2,}/g, "-");
	return normalized || DEFAULT_WORKSPACE_SLUG;
}

export function xlabArtifactFamilyForSkill(
	skillName: string,
	artifactType?: string,
): XlabWorkspaceArtifactFamily | undefined {
	if (skillName === "knowledge_graph") {
		return artifactType === undefined || ["method_graph", "graph_db", "graph_report"].includes(artifactType)
			? "graph"
			: undefined;
	}
	if (skillName === "literature_survey") {
		return artifactType === undefined ||
			["literature_survey_document", "literature_survey_json", "citation_trace", "survey_report"].includes(
				artifactType,
			)
			? "survey"
			: undefined;
	}
	if (skillName === "research_idea") {
		return artifactType === undefined ||
			["research_idea", "research_idea_result", "research_idea_report"].includes(artifactType)
			? "idea"
			: undefined;
	}
	return undefined;
}

export function parseXlabWorkspaceSelection(args: string, fallbackTopic?: string): XlabWorkspaceSelection | undefined {
	const tokens = parseShellTokens(args);
	const explicit = readFlag(tokens, "--workspace");
	if (explicit) {
		return {
			slug: slugifyWorkspaceName(explicit),
			explicit: true,
			args: serializeTokens(removeFlag(tokens, "--workspace")),
		};
	}
	const topic = readFlag(tokens, "--topic") ?? fallbackTopic;
	const positionalTopic = positionalTokens(tokens).join(" ").trim() || undefined;
	const inferred = topic ?? positionalTopic;
	if (!inferred) {
		return undefined;
	}
	return { slug: slugifyWorkspaceName(inferred), explicit: false, args };
}

export class XlabWorkspaceManager {
	private cwd: string;

	constructor(cwd: string) {
		this.cwd = cwd;
	}

	get root(): string {
		return join(this.cwd, XLAB_WORKSPACES_DIR);
	}

	indexPath(): string {
		return join(this.root, INDEX_NAME);
	}

	workspaceDir(slug: string): string {
		return join(this.root, slugifyWorkspaceName(slug));
	}

	workspacePath(slug: string): string {
		return join(this.workspaceDir(slug), WORKSPACE_NAME);
	}

	currentPath(slug: string): string {
		return join(this.workspaceDir(slug), CURRENT_NAME);
	}

	lineagePath(slug: string): string {
		return join(this.workspaceDir(slug), LINEAGE_NAME);
	}

	readIndex(): XlabWorkspaceIndex {
		if (!existsSync(this.indexPath())) {
			return { schemaVersion: "1", workspaces: [] };
		}
		const raw = readJson<Partial<XlabWorkspaceIndex>>(this.indexPath());
		return {
			schemaVersion: "1",
			activeWorkspaceSlug: typeof raw.activeWorkspaceSlug === "string" ? raw.activeWorkspaceSlug : undefined,
			workspaces: Array.isArray(raw.workspaces)
				? raw.workspaces.flatMap((entry): XlabWorkspaceIndexEntry[] => {
						if (!isRecord(entry) || typeof entry.slug !== "string") {
							return [];
						}
						return [
							{
								slug: slugifyWorkspaceName(entry.slug),
								displayName:
									typeof entry.displayName === "string" ? entry.displayName : titleFromSlug(entry.slug),
								updatedAt: typeof entry.updatedAt === "string" ? entry.updatedAt : nowIso(),
							},
						];
					})
				: [],
		};
	}

	writeIndex(index: XlabWorkspaceIndex): void {
		mkdirSync(this.root, { recursive: true });
		const workspaces = [...index.workspaces].sort((left, right) => left.slug.localeCompare(right.slug));
		writeJson(this.indexPath(), { ...index, schemaVersion: "1", workspaces });
	}

	listWorkspaces(): XlabWorkspaceRecord[] {
		return this.readIndex().workspaces.flatMap((entry) => {
			const workspace = this.readWorkspace(entry.slug);
			return workspace ? [workspace] : [];
		});
	}

	readWorkspace(slug: string): XlabWorkspaceRecord | undefined {
		const normalized = slugifyWorkspaceName(slug);
		const path = this.workspacePath(normalized);
		if (!existsSync(path)) {
			return undefined;
		}
		return readJson<XlabWorkspaceRecord>(path);
	}

	ensureWorkspace(
		slug: string,
		options?: { displayName?: string; topic?: string; description?: string },
	): XlabWorkspaceRecord {
		const normalized = this.availableSlug(slug);
		const existing = this.readWorkspace(normalized);
		if (existing) {
			return existing;
		}
		const timestamp = nowIso();
		const workspace: XlabWorkspaceRecord = {
			schemaVersion: "1",
			id: randomUUID(),
			slug: normalized,
			displayName: options?.displayName?.trim() || titleFromSlug(normalized),
			createdAt: timestamp,
			updatedAt: timestamp,
			description: options?.description,
			topic: options?.topic,
		};
		mkdirSync(this.workspaceDir(normalized), { recursive: true });
		writeJson(this.workspacePath(normalized), workspace);
		writeJson(this.currentPath(normalized), {
			schemaVersion: "1",
			workspaceSlug: normalized,
			current: {},
			updatedAt: timestamp,
		} satisfies XlabWorkspaceCurrent);
		writeJson(this.lineagePath(normalized), {
			schemaVersion: "1",
			workspaceSlug: normalized,
			handles: [],
			updatedAt: timestamp,
		} satisfies XlabWorkspaceLineage);
		this.upsertIndexEntry(workspace);
		return workspace;
	}

	setActiveWorkspace(slug: string): XlabWorkspaceRecord | undefined {
		const workspace = this.readWorkspace(slug);
		if (!workspace) {
			return undefined;
		}
		const index = this.readIndex();
		this.writeIndex({ ...index, activeWorkspaceSlug: workspace.slug });
		return workspace;
	}

	activeWorkspace(): XlabWorkspaceRecord | undefined {
		const active = this.readIndex().activeWorkspaceSlug;
		return active ? this.readWorkspace(active) : undefined;
	}

	readCurrent(slug: string): XlabWorkspaceCurrent {
		const normalized = slugifyWorkspaceName(slug);
		if (!existsSync(this.currentPath(normalized))) {
			return {
				schemaVersion: "1",
				workspaceSlug: normalized,
				current: {},
				updatedAt: nowIso(),
			};
		}
		return readJson<XlabWorkspaceCurrent>(this.currentPath(normalized));
	}

	readLineage(slug: string): XlabWorkspaceLineage {
		const normalized = slugifyWorkspaceName(slug);
		if (!existsSync(this.lineagePath(normalized))) {
			return {
				schemaVersion: "1",
				workspaceSlug: normalized,
				handles: [],
				updatedAt: nowIso(),
			};
		}
		return readJson<XlabWorkspaceLineage>(this.lineagePath(normalized));
	}

	registerRunArtifacts(params: {
		run: XlabRunRecord;
		manifest: XlabManifestEnvelope;
		artifactRefs: XlabArtifactReference[];
		workspaceSlug?: string;
	}): XlabWorkspaceRegisterResult | undefined {
		const family = xlabArtifactFamilyForSkill(params.run.skillName);
		if (!family || params.artifactRefs.length === 0) {
			return undefined;
		}
		const workspaceSlug =
			params.workspaceSlug ??
			params.run.workspaceSlug ??
			this.activeWorkspace()?.slug ??
			this.inferRunWorkspaceSlug(params.run, params.manifest);
		const workspace = this.ensureWorkspace(workspaceSlug, {
			displayName: titleFromSlug(workspaceSlug),
			topic: this.inferTopic(params.run, params.manifest),
		});
		const lineage = this.readLineage(workspace.slug);
		const current = this.readCurrent(workspace.slug);
		const familyRefs = params.artifactRefs.filter(
			(ref) => xlabArtifactFamilyForSkill(params.run.skillName, ref.type) === family,
		);
		if (familyRefs.length === 0) {
			return {
				workspace,
				handles: [],
				current,
				lineage,
				warnings: this.staleWarnings(workspace.slug),
			};
		}
		const artifactIds = uniqueStrings(familyRefs.map((ref) => ref.artifactId));
		const existingHandles = this.handlesForArtifactIds(
			lineage,
			params.manifest.artifacts.flatMap((artifact) => artifact.parents ?? []),
		);
		const parents = uniqueStrings([
			...params.manifest.artifacts.flatMap((artifact) => artifact.parents ?? []),
			...existingHandles.flatMap((handle) => handle.artifactIds),
		]);
		const parentHandles = uniqueStrings(existingHandles.map((handle) => handle.handle));
		const primaryRef = this.primaryArtifactRef(params.run.skillName, familyRefs) ?? familyRefs[0];
		const handle: XlabWorkspaceHandleRecord = {
			schemaVersion: "1",
			workspaceSlug: workspace.slug,
			handle: this.nextHandle(lineage, family),
			family,
			artifactId: primaryRef.artifactId,
			artifactIds,
			artifactRefs: familyRefs,
			runId: params.run.runId,
			runDir: relative(this.cwd, params.run.runDir) || ".",
			manifestPath: params.run.manifestPath
				? relative(this.cwd, params.run.manifestPath) || basename(params.run.manifestPath)
				: undefined,
			skillName: params.run.skillName,
			skillVersion: params.run.skillVersion,
			status: params.run.status,
			parents,
			parentHandles,
			createdAt: nowIso(),
			updatedAt: nowIso(),
			validation: params.manifest.validation,
			summary: params.run.summary,
		};
		const nextLineage: XlabWorkspaceLineage = {
			schemaVersion: "1",
			workspaceSlug: workspace.slug,
			handles: [...lineage.handles, handle],
			updatedAt: nowIso(),
		};
		const nextCurrent: XlabWorkspaceCurrent = {
			schemaVersion: "1",
			workspaceSlug: workspace.slug,
			current: params.run.status === "success" ? { ...current.current, [family]: handle.handle } : current.current,
			updatedAt: nowIso(),
		};
		writeJson(this.lineagePath(workspace.slug), nextLineage);
		writeJson(this.currentPath(workspace.slug), nextCurrent);
		const updatedWorkspace = {
			...workspace,
			updatedAt: nowIso(),
			topic: workspace.topic ?? this.inferTopic(params.run, params.manifest),
		};
		writeJson(this.workspacePath(workspace.slug), updatedWorkspace);
		this.upsertIndexEntry(updatedWorkspace);
		return {
			workspace: updatedWorkspace,
			handles: [handle],
			current: nextCurrent,
			lineage: nextLineage,
			warnings: this.staleWarnings(workspace.slug),
		};
	}

	resolveHandle(
		slug: string,
		handleOrFamily: string,
		family?: XlabWorkspaceArtifactFamily,
	): XlabWorkspaceResolvedHandle | undefined {
		const workspace = this.readWorkspace(slug);
		if (!workspace) {
			return undefined;
		}
		const lineage = this.readLineage(workspace.slug);
		const current = this.readCurrent(workspace.slug);
		const requested = handleOrFamily.trim();
		const handleName = requested.includes("-v")
			? requested
			: family
				? current.current[family]
				: current.current[this.familyFromHandle(requested) ?? "other"];
		if (!handleName) {
			return undefined;
		}
		const handle = lineage.handles.find((entry) => entry.handle === handleName);
		return handle
			? {
					workspace,
					handle,
					warnings: this.staleWarnings(workspace.slug, handle),
				}
			: undefined;
	}

	resolveSkillArgs(
		skillName: string,
		args: string,
		options?: { workspaceSlug?: string },
	): {
		args: string;
		workspaceSlug?: string;
		warnings: string[];
		error?: string;
	} {
		const tokens = parseShellTokens(args);
		const explicitWorkspace = readFlag(tokens, "--workspace");
		let nextTokens = removeFlag(tokens, "--workspace");
		const activeWorkspace = this.activeWorkspace();
		const workspaceSlug =
			options?.workspaceSlug ??
			(explicitWorkspace ? slugifyWorkspaceName(explicitWorkspace) : activeWorkspace?.slug);
		const warnings: string[] = [];
		if (!workspaceSlug) {
			return { args: serializeTokens(nextTokens), warnings };
		}
		if (!this.readWorkspace(workspaceSlug)) {
			return {
				args: serializeTokens(nextTokens),
				workspaceSlug,
				warnings,
				error: `XLab workspace not found: ${workspaceSlug}. Run /xlab list-workspaces to see available workspaces.`,
			};
		}

		if (skillName === "literature_survey") {
			const graphValue = readFlag(nextTokens, "--graph");
			const hasCompatibilityInput = Boolean(
				readFlag(nextTokens, "--input") ?? readFlag(nextTokens, "--papers") ?? readFlag(nextTokens, "--paper-set"),
			);
			if (graphValue && this.looksLikeHandle(graphValue)) {
				const resolved = this.resolveHandle(workspaceSlug, graphValue, "graph");
				if (!resolved) {
					return {
						args,
						workspaceSlug,
						warnings,
						error: this.unresolvedHandleMessage(workspaceSlug, graphValue, "graph"),
					};
				}
				nextTokens = replaceFlag(nextTokens, "--graph", this.pathForHandle(resolved.handle));
				warnings.push(...resolved.warnings);
			} else if (!graphValue) {
				const resolved = this.resolveHandle(workspaceSlug, "graph", "graph");
				if (resolved) {
					nextTokens = replaceFlag(nextTokens, "--graph", this.pathForHandle(resolved.handle));
					warnings.push(`Using ${resolved.handle.handle} from workspace ${workspaceSlug}.`, ...resolved.warnings);
				} else if (!hasCompatibilityInput) {
					warnings.push(
						`No current graph handle is registered in workspace ${workspaceSlug}; run /xlab build-knowledge-graph first or pass --graph graph-vN. Compatibility inputs --input, --papers, and --paper-set remain supported.`,
					);
				}
			}
		}

		if (skillName === "research_idea") {
			const surveyValue = readFlag(nextTokens, "--survey");
			if (surveyValue && this.looksLikeHandle(surveyValue)) {
				const resolved = this.resolveHandle(workspaceSlug, surveyValue, "survey");
				if (!resolved) {
					return {
						args,
						workspaceSlug,
						warnings,
						error: this.unresolvedHandleMessage(workspaceSlug, surveyValue, "survey"),
					};
				}
				nextTokens = replaceFlag(nextTokens, "--survey", this.pathForHandle(resolved.handle));
				warnings.push(...resolved.warnings);
			} else if (!surveyValue) {
				const resolved = this.resolveHandle(workspaceSlug, "survey", "survey");
				if (resolved) {
					nextTokens = replaceFlag(nextTokens, "--survey", this.pathForHandle(resolved.handle));
					warnings.push(`Using ${resolved.handle.handle} from workspace ${workspaceSlug}.`, ...resolved.warnings);
				} else {
					return {
						args: serializeTokens(nextTokens),
						workspaceSlug,
						warnings,
						error: this.missingCurrentHandleMessage(workspaceSlug, "survey", "--survey"),
					};
				}
			}
		}
		if (skillName === "run_experiment") {
			const ideaValue = readFlag(nextTokens, "--idea");
			if (ideaValue && this.looksLikeHandle(ideaValue)) {
				const resolved = this.resolveHandle(workspaceSlug, ideaValue, "idea");
				if (!resolved) {
					return {
						args,
						workspaceSlug,
						warnings,
						error: this.unresolvedHandleMessage(workspaceSlug, ideaValue, "idea"),
					};
				}
				const payloadPath = this.payloadPathForHandle(resolved.handle);
				if (!payloadPath) {
					return {
						args,
						workspaceSlug,
						warnings,
						error: `Workspace handle ${ideaValue} does not reference a Research Idea payload.`,
					};
				}
				nextTokens = replaceFlag(nextTokens, "--idea", payloadPath);
				warnings.push(...resolved.warnings);
			} else if (!ideaValue) {
				const resolved = this.resolveHandle(workspaceSlug, "idea", "idea");
				if (resolved) {
					const payloadPath = this.payloadPathForHandle(resolved.handle);
					if (!payloadPath) {
						return {
							args: serializeTokens(nextTokens),
							workspaceSlug,
							warnings,
							error: `Workspace handle ${resolved.handle.handle} does not reference a Research Idea payload.`,
						};
					}
					nextTokens = replaceFlag(nextTokens, "--idea", payloadPath);
					warnings.push(`Using ${resolved.handle.handle} from workspace ${workspaceSlug}.`, ...resolved.warnings);
				}
			}
		}
		return { args: serializeTokens(nextTokens), workspaceSlug, warnings };
	}

	staleWarnings(slug: string, target?: XlabWorkspaceHandleRecord): string[] {
		const current = this.readCurrent(slug).current;
		const lineage = this.readLineage(slug);
		const handles = target ? [target] : lineage.handles;
		const warnings: string[] = [];
		for (const handle of handles) {
			if (handle.family === "survey" && current.graph && !handle.parentHandles.includes(current.graph)) {
				warnings.push(
					`${handle.handle} may be stale: current graph is ${current.graph}, but its parents are ${
						handle.parentHandles.join(", ") || "untracked"
					}.`,
				);
			}
			if (handle.family === "idea" && current.survey && !handle.parentHandles.includes(current.survey)) {
				warnings.push(
					`${handle.handle} may be stale: current survey is ${current.survey}, but its parents are ${
						handle.parentHandles.join(", ") || "untracked"
					}.`,
				);
			}
		}
		return uniqueStrings(warnings);
	}

	private availableSlug(seed: string): string {
		const base = slugifyWorkspaceName(seed);
		if (!this.readWorkspace(base)) {
			return base;
		}
		return base;
	}

	private inferRunWorkspaceSlug(run: XlabRunRecord, manifest: XlabManifestEnvelope): string {
		return slugifyWorkspaceName(this.inferTopic(run, manifest) ?? `${run.skillName}-workspace`);
	}

	private inferTopic(run: XlabRunRecord, manifest: XlabManifestEnvelope): string | undefined {
		for (const source of [manifest.inputs, manifest.outputs]) {
			for (const key of ["topic", "query", "title", "name"]) {
				const value = source[key];
				if (typeof value === "string" && value.trim()) {
					return value.trim();
				}
			}
		}
		const topic = readFlag(parseShellTokens(run.args), "--topic");
		if (topic) {
			return topic;
		}
		const positional = positionalTokens(parseShellTokens(run.args)).join(" ").trim();
		return positional || undefined;
	}

	private upsertIndexEntry(workspace: XlabWorkspaceRecord): void {
		const index = this.readIndex();
		const entries = index.workspaces.filter((entry) => entry.slug !== workspace.slug);
		entries.push({
			slug: workspace.slug,
			displayName: workspace.displayName,
			updatedAt: workspace.updatedAt,
		});
		this.writeIndex({
			...index,
			workspaces: entries,
			activeWorkspaceSlug: index.activeWorkspaceSlug ?? workspace.slug,
		});
	}

	private nextHandle(lineage: XlabWorkspaceLineage, family: XlabWorkspaceArtifactFamily): string {
		const prefix = HANDLE_FAMILY_PREFIX[family];
		const maxVersion = lineage.handles
			.filter((handle) => handle.family === family)
			.map((handle) => {
				const match = new RegExp(`^${prefix}-v(\\d+)$`).exec(handle.handle);
				return match ? Number.parseInt(match[1], 10) : 0;
			})
			.reduce((max, value) => Math.max(max, value), 0);
		return `${prefix}-v${maxVersion + 1}`;
	}

	private handlesForArtifactIds(lineage: XlabWorkspaceLineage, artifactIds: string[]): XlabWorkspaceHandleRecord[] {
		const wanted = new Set(artifactIds);
		return lineage.handles.filter((handle) => handle.artifactIds.some((artifactId) => wanted.has(artifactId)));
	}

	private primaryArtifactRef(skillName: string, refs: XlabArtifactReference[]): XlabArtifactReference | undefined {
		const preferred: Record<string, string[]> = {
			knowledge_graph: ["graph_db", "method_graph"],
			literature_survey: ["literature_survey_json", "literature_survey_document"],
			research_idea: ["research_idea", "research_idea_result"],
		};
		for (const type of preferred[skillName] ?? []) {
			const found = refs.find((ref) => ref.type === type);
			if (found) {
				return found;
			}
		}
		return refs[0];
	}

	private familyFromHandle(handle: string): XlabWorkspaceArtifactFamily | undefined {
		if (handle.startsWith("graph")) {
			return "graph";
		}
		if (handle.startsWith("survey")) {
			return "survey";
		}
		if (handle.startsWith("idea")) {
			return "idea";
		}
		return undefined;
	}

	private looksLikeHandle(value: string): boolean {
		return /^(graph|survey|idea|artifact)-v\d+$/.test(value);
	}

	private payloadPathForHandle(handle: XlabWorkspaceHandleRecord): string | undefined {
		if (handle.status !== "success") {
			return undefined;
		}
		const primary = handle.artifactRefs.find((reference) => reference.artifactId === handle.artifactId);
		if (primary && ["research_idea", "research_idea_result"].includes(primary.type)) {
			return primary.payloadPath;
		}
		return undefined;
	}

	private pathForHandle(handle: XlabWorkspaceHandleRecord): string {
		return handle.manifestPath ?? join(handle.runDir, "manifest.json");
	}

	private missingCurrentHandleMessage(slug: string, family: XlabWorkspaceArtifactFamily, flag: string): string {
		const available = this.availableHandleList(slug, family);
		return `No current ${family} handle is registered in workspace ${slug}. Run the upstream XLab skill first or pass ${flag} <path>${
			available ? ` / ${available}.` : "."
		}`;
	}

	private unresolvedHandleMessage(slug: string, handle: string, family: XlabWorkspaceArtifactFamily): string {
		const available = this.availableHandleList(slug, family);
		return `Cannot resolve ${handle} in workspace ${slug}.${
			available ? ` Available ${family} handles: ${available}.` : ` No ${family} handles are registered.`
		}`;
	}

	private availableHandleList(slug: string, family: XlabWorkspaceArtifactFamily): string {
		return this.readLineage(slug)
			.handles.filter((entry) => entry.family === family)
			.map((entry) => entry.handle)
			.join(", ");
	}
}
