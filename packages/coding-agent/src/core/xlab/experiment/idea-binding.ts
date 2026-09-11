import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import { isAbsolute, resolve } from "node:path";
import { type StoredArtifactLookup, XlabArtifactStore } from "../artifact-store.ts";
import { XlabRunManager } from "../run-manager.ts";
import type { XlabRunRecord, XlabWorkspaceHandleRecord } from "../types.ts";
import { XlabWorkspaceManager } from "../workspace-manager.ts";

const RESEARCH_IDEA_TYPES = new Set(["research_idea", "research_idea_result"]);
const RESEARCH_IDEA_SCHEMAS = new Set(["xlab.research_idea.v2", "xlab.research_idea.result.v2"]);

export type XlabIdeaResolutionMethod = "path" | "artifact-id" | "producer-run" | "workspace-handle" | "title";

export interface XlabIdeaBindingMetadata {
	schema_version: "xlab.idea_binding.v1";
	resolution_method: XlabIdeaResolutionMethod;
	reference: string;
	source_artifact_id?: string;
	source_artifact_type?: string;
	source_schema_version: string;
	source_digest: string;
	canonical_digest: string;
	producer?: {
		run_id: string;
		skill: string;
		version: string;
	};
	workspace?: {
		slug: string;
		handle: string;
	};
}

export interface XlabIdeaBinding {
	canonical: Record<string, unknown>;
	canonicalBytes: Buffer;
	metadata: XlabIdeaBindingMetadata;
	warnings: string[];
}

export type XlabIdeaBindingResult = { ok: true; binding: XlabIdeaBinding } | { ok: false; error: string };

interface ResolvedSource {
	bytes: Buffer;
	method: XlabIdeaResolutionMethod;
	reference: string;
	artifact?: StoredArtifactLookup;
	run?: XlabRunRecord;
	handle?: XlabWorkspaceHandleRecord;
	workspaceSlug?: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function digest(bytes: Buffer): string {
	return createHash("sha256").update(bytes).digest("hex");
}

function flagValue(args: string, flag: string): string | undefined {
	const pattern = new RegExp(`(?:^|\\s)${flag}(?:=|\\s+)(?:'([^']*)'|"([^"]*)"|([^\\s]+))`);
	const match = pattern.exec(args);
	return match?.[1] ?? match?.[2] ?? match?.[3];
}

function handleForPayload(
	workspaceManager: XlabWorkspaceManager,
	workspaceSlug: string | undefined,
	payloadPath: string,
): XlabWorkspaceHandleRecord | undefined {
	if (!workspaceSlug) {
		return undefined;
	}
	const target = resolve(payloadPath);
	return workspaceManager
		.readLineage(workspaceSlug)
		.handles.find((handle) =>
			handle.artifactRefs.some(
				(reference) => RESEARCH_IDEA_TYPES.has(reference.type) && resolve(reference.payloadPath) === target,
			),
		);
}

function artifactLookup(
	artifactStore: XlabArtifactStore,
	artifactId: string,
): StoredArtifactLookup | string | undefined {
	try {
		return artifactStore.get(artifactId);
	} catch (error) {
		return error instanceof Error ? error.message : String(error);
	}
}

function sourceFromTitle(params: {
	artifactStore: XlabArtifactStore;
	reference: string;
}): ResolvedSource | string | undefined {
	let candidates: Array<{ artifact_id: string; type: string; metadata: Record<string, unknown> }>;
	try {
		candidates = params.artifactStore
			.list()
			.filter(
				(artifact) =>
					artifact.type === "research_idea_result" &&
					typeof artifact.metadata.title === "string" &&
					artifact.metadata.title.trim() === params.reference.trim(),
			);
	} catch (error) {
		return error instanceof Error ? error.message : String(error);
	}
	const artifactIds = [...new Set(candidates.map((artifact) => artifact.artifact_id))];
	if (artifactIds.length > 1) {
		return `Idea title ${params.reference} is ambiguous; use an artifact ID, producer run ID, workspace handle, or path.`;
	}
	if (artifactIds.length === 0) {
		return undefined;
	}
	const artifact = artifactLookup(params.artifactStore, artifactIds[0]);
	if (typeof artifact === "string") {
		return artifact;
	}
	return artifact ? sourceFromArtifact(artifact, "title", params.reference) : undefined;
}

function sourceFromArtifact(
	artifact: StoredArtifactLookup,
	method: XlabIdeaResolutionMethod,
	reference: string,
): ResolvedSource | undefined {
	if (!RESEARCH_IDEA_TYPES.has(artifact.type) || artifact.payload_kind !== "file") {
		return undefined;
	}
	return {
		bytes: readFileSync(artifact.payloadPath),
		method,
		reference,
		artifact,
	};
}

function resolveSource(params: {
	cwd: string;
	args: string;
	workspaceSlug?: string;
	workspaceManager: XlabWorkspaceManager;
	artifactStore: XlabArtifactStore;
	runManager: XlabRunManager;
}): ResolvedSource | string {
	const reference = flagValue(params.args, "--idea");
	if (!reference) {
		return "run-experiment requires --idea <reference> or a current workspace Idea.";
	}

	if (reference.startsWith("sha256:")) {
		const artifact = artifactLookup(params.artifactStore, reference);
		if (typeof artifact === "string") {
			return artifact;
		}
		const source = artifact && sourceFromArtifact(artifact, "artifact-id", reference);
		return source ?? `Artifact ${reference} is not a Research Idea payload.`;
	}

	if (/^idea-v\d+$/.test(reference) && params.workspaceSlug) {
		const resolvedHandle = params.workspaceManager.resolveHandle(params.workspaceSlug, reference, "idea");
		if (!resolvedHandle || resolvedHandle.handle.status !== "success") {
			return `Cannot resolve successful ${reference} in workspace ${params.workspaceSlug}.`;
		}
		const primary = resolvedHandle.handle.artifactRefs.find(
			(item) => item.artifactId === resolvedHandle.handle.artifactId && RESEARCH_IDEA_TYPES.has(item.type),
		);
		if (!primary || !existsSync(primary.payloadPath)) {
			return `Workspace handle ${reference} does not reference a Research Idea payload.`;
		}
		const artifact = artifactLookup(params.artifactStore, primary.artifactId);
		if (typeof artifact === "string") {
			return artifact;
		}
		return {
			bytes: readFileSync(primary.payloadPath),
			method: "workspace-handle",
			reference,
			handle: resolvedHandle.handle,
			workspaceSlug: params.workspaceSlug,
			artifact,
		};
	}

	const producerRun = params.runManager.readRun(reference);
	if (producerRun) {
		if (producerRun.status !== "success") {
			return `Producer run ${reference} is not successful.`;
		}
		const candidates = (producerRun.artifactRefs ?? []).filter((item) => RESEARCH_IDEA_TYPES.has(item.type));
		const preferred = candidates.filter((item) => item.type !== "research_idea");
		const selected = preferred.length > 0 ? preferred : candidates;
		if (selected.length !== 1) {
			return `Producer run ${reference} must resolve uniquely to one primary Research Idea payload.`;
		}
		const candidate = selected[0];
		if (!existsSync(candidate.payloadPath)) {
			return `Research Idea payload from producer run ${reference} is missing.`;
		}
		const artifact = artifactLookup(params.artifactStore, candidate.artifactId);
		if (typeof artifact === "string") {
			return artifact;
		}
		return {
			bytes: readFileSync(candidate.payloadPath),
			method: "producer-run",
			reference,
			run: producerRun,
			artifact,
		};
	}

	const path = isAbsolute(reference) ? resolve(reference) : resolve(params.cwd, reference);
	if (!existsSync(path) || !statSync(path).isFile()) {
		const titleSource = sourceFromTitle({ artifactStore: params.artifactStore, reference });
		return titleSource ?? `Cannot resolve Idea reference ${reference} to an existing Research Idea JSON file.`;
	}
	const handle = handleForPayload(params.workspaceManager, params.workspaceSlug, path);
	if (handle && handle.status !== "success") {
		return `Workspace handle ${handle.handle} is not successful.`;
	}
	const artifactReference = handle?.artifactRefs.find(
		(item) => resolve(item.payloadPath) === path && RESEARCH_IDEA_TYPES.has(item.type),
	);
	const artifact = artifactReference ? artifactLookup(params.artifactStore, artifactReference.artifactId) : undefined;
	if (typeof artifact === "string") {
		return artifact;
	}
	return {
		bytes: readFileSync(path),
		method: handle ? "workspace-handle" : "path",
		reference,
		handle,
		workspaceSlug: handle ? params.workspaceSlug : undefined,
		artifact,
	};
}

function canonicalize(source: ResolvedSource): Record<string, unknown> | string {
	let payload: unknown;
	try {
		payload = JSON.parse(source.bytes.toString("utf-8"));
	} catch (error) {
		return `Research Idea payload is not valid JSON: ${error instanceof Error ? error.message : String(error)}`;
	}
	if (!isRecord(payload) || !RESEARCH_IDEA_SCHEMAS.has(String(payload.schema_version ?? ""))) {
		return "Research Idea payload has an unsupported schema_version.";
	}
	if (payload.status !== "success" || !Array.isArray(payload.blockers) || payload.blockers.length > 0) {
		return "Research Idea payload must have status success and no blockers.";
	}
	const result = payload.schema_version === "xlab.research_idea.v2" ? payload.idea_result : payload;
	if (!isRecord(result)) {
		return "Research Idea payload does not contain a structured idea result.";
	}
	if (result.status !== undefined && result.status !== "success") {
		return "Research Idea result is not successful.";
	}
	if (Array.isArray(result.blockers) && result.blockers.length > 0) {
		return "Research Idea result contains blockers.";
	}
	if (!Array.isArray(result.components) || result.components.length === 0) {
		return "Research Idea components must be a non-empty list.";
	}
	const seen = new Set<string>();
	const components: Array<Record<string, string>> = [];
	for (let index = 0; index < result.components.length; index += 1) {
		const item = result.components[index];
		if (!isRecord(item)) {
			return `Research Idea components[${index}] must be an object.`;
		}
		const component = typeof item.component === "string" ? item.component.trim() : "";
		if (!component) {
			return `Research Idea components[${index}] is missing component.`;
		}
		if (seen.has(component)) {
			return `Research Idea components contains duplicate component ${component}.`;
		}
		seen.add(component);
		const explanation = [item.explanation, item.description, item.summary].find(
			(value): value is string => typeof value === "string" && value.trim() !== "",
		);
		components.push({ component, explanation: explanation?.trim() ?? "", index: String(index + 1) });
	}
	return {
		...result,
		schema_version: "xlab.experiment_idea.v1",
		status: "success",
		blockers: [],
		components,
	};
}

export function bindXlabExperimentIdea(params: {
	cwd: string;
	args: string;
	workspaceSlug?: string;
	workspaceManager?: XlabWorkspaceManager;
	artifactStore?: XlabArtifactStore;
	runManager?: XlabRunManager;
}): XlabIdeaBindingResult {
	const workspaceManager = params.workspaceManager ?? new XlabWorkspaceManager(params.cwd);
	const artifactStore = params.artifactStore ?? new XlabArtifactStore(params.cwd);
	const runManager = params.runManager ?? new XlabRunManager(params.cwd);
	const source = resolveSource({ ...params, workspaceManager, artifactStore, runManager });
	if (typeof source === "string") {
		return { ok: false, error: source };
	}
	const canonical = canonicalize(source);
	if (typeof canonical === "string") {
		return { ok: false, error: canonical };
	}
	const canonicalBytes = Buffer.from(`${JSON.stringify(canonical, null, 2)}\n`, "utf-8");
	const producer = source.run
		? { run_id: source.run.runId, skill: source.run.skillName, version: source.run.skillVersion }
		: source.handle
			? { run_id: source.handle.runId, skill: source.handle.skillName, version: source.handle.skillVersion }
			: source.artifact?.producer;
	return {
		ok: true,
		binding: {
			canonical,
			canonicalBytes,
			metadata: {
				schema_version: "xlab.idea_binding.v1",
				resolution_method: source.method,
				reference: source.reference,
				source_artifact_id: source.artifact?.artifact_id ?? source.handle?.artifactId,
				source_artifact_type:
					source.artifact?.type ??
					source.handle?.artifactRefs.find((item) => item.artifactId === source.handle?.artifactId)?.type,
				source_schema_version: String(JSON.parse(source.bytes.toString("utf-8")).schema_version),
				source_digest: digest(source.bytes),
				canonical_digest: digest(canonicalBytes),
				producer,
				workspace:
					source.handle && source.workspaceSlug
						? { slug: source.workspaceSlug, handle: source.handle.handle }
						: undefined,
			},
			warnings: [],
		},
	};
}
