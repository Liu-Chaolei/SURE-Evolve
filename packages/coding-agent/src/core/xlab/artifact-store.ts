import { createHash } from "node:crypto";
import {
	cpSync,
	existsSync,
	lstatSync,
	mkdirSync,
	mkdtempSync,
	readdirSync,
	readFileSync,
	renameSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { dirname, join, relative } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type {
	XlabArtifactReference,
	XlabArtifactRequirement,
	XlabManifestArtifact,
	XlabManifestEnvelope,
	XlabRunRecord,
} from "./types.ts";

export interface StoredArtifactMetadata {
	artifact_id: string;
	type: string;
	schema_version: string;
	digest: string;
	created_at: string;
	producer: {
		skill: string;
		version: string;
		run_id: string;
	};
	parents: string[];
	validation: Record<string, unknown>;
	metadata: Record<string, unknown>;
	payload_kind: "file" | "directory";
	occurrences?: StoredArtifactOccurrence[];
}

export interface StoredArtifactLookup extends StoredArtifactMetadata {
	payloadPath: string;
	metadataPath: string;
}

export interface StoredArtifactLineage {
	child_id?: string;
	session_id?: string;
	work_unit?: string;
	execution_attempt?: number;
	generation?: number;
	review_round?: number;
	parent_artifact_digests: string[];
}

interface StoredArtifactIndexRow {
	artifact_id: string;
	digest: string;
	type: string;
	schema_version: string;
	payload_path: string;
	metadata_path: string;
	metadata_json: string;
}

interface StoredArtifactOccurrence {
	producer_run_id: string;
	type: string;
	schema_version: string;
	producer_skill: string;
	producer_version: string;
	created_at: string;
	lineage?: StoredArtifactLineage;
}

function optionalString(value: unknown): string | undefined {
	return typeof value === "string" ? value : undefined;
}

function optionalNonNegativeInteger(value: unknown): number | undefined {
	return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : undefined;
}

function occurrenceLineage(metadata: Record<string, unknown>, parents: string[]): StoredArtifactLineage | undefined {
	const nested = metadata.lineage;
	const hasNestedLineage = typeof nested === "object" && nested !== null && !Array.isArray(nested);
	const source = hasNestedLineage ? (nested as Record<string, unknown>) : metadata;
	const lineageKeys = [
		"child_id",
		"session_id",
		"work_unit",
		"execution_attempt",
		"generation",
		"review_round",
		"parent_artifact_digests",
		"parent_digests",
	];
	if (!hasNestedLineage && !lineageKeys.some((key) => key in source)) {
		return undefined;
	}
	const parentDigestsValue = source.parent_artifact_digests ?? source.parent_digests;
	const parentArtifactDigests = Array.isArray(parentDigestsValue)
		? parentDigestsValue.filter((value): value is string => typeof value === "string")
		: parents;
	const lineage: StoredArtifactLineage = {
		child_id: optionalString(source.child_id),
		session_id: optionalString(source.session_id),
		work_unit: optionalString(source.work_unit),
		execution_attempt: optionalNonNegativeInteger(source.execution_attempt),
		generation: optionalNonNegativeInteger(source.generation),
		review_round: optionalNonNegativeInteger(source.review_round),
		parent_artifact_digests: parentArtifactDigests,
	};
	return Object.values(lineage).some((value) => value !== undefined && (!Array.isArray(value) || value.length > 0))
		? lineage
		: undefined;
}

function contentMetadata(metadata: Record<string, unknown>, hasLineage: boolean): Record<string, unknown> {
	if (!hasLineage) {
		return metadata;
	}
	const result = { ...metadata };
	for (const key of [
		"lineage",
		"child_id",
		"session_id",
		"work_unit",
		"execution_attempt",
		"generation",
		"review_round",
		"parent_artifact_digests",
		"parent_digests",
	]) {
		delete result[key];
	}
	return result;
}

function writeJson(path: string, value: unknown): void {
	writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
}

function hashPath(path: string): {
	digest: string;
	contentDigest: string;
	bytes: number;
	kind: "file" | "directory";
} {
	const stat = lstatSync(path);
	if (stat.isSymbolicLink()) {
		throw new Error(`Artifact payload must not be a symbolic link: ${path}`);
	}
	if (stat.isFile()) {
		const bytes = readFileSync(path);
		return {
			digest: createHash("sha256").update("file\0").update(bytes).digest("hex"),
			contentDigest: createHash("sha256").update(bytes).digest("hex"),
			bytes: stat.size,
			kind: "file",
		};
	}
	if (!stat.isDirectory()) {
		throw new Error(`Artifact payload must be a file or directory: ${path}`);
	}

	const hash = createHash("sha256").update("directory\0");
	let bytes = 0;
	const visit = (root: string, current: string): void => {
		for (const name of readdirSync(current).sort()) {
			const child = join(current, name);
			const childStat = lstatSync(child);
			if (childStat.isSymbolicLink()) {
				throw new Error(`Artifact directory must not contain symbolic links: ${child}`);
			}
			const childRelative = relative(root, child).replaceAll("\\", "/");
			if (childStat.isDirectory()) {
				hash.update(`dir:${childRelative}\n`);
				visit(root, child);
			} else if (childStat.isFile()) {
				bytes += childStat.size;
				hash.update(`file:${childRelative}:${childStat.size}\n`);
				hash.update(readFileSync(child));
			}
		}
	};
	visit(path, path);
	const digest = hash.digest("hex");
	return { digest, contentDigest: digest, bytes, kind: "directory" };
}

interface PreflightArtifact {
	artifact: XlabManifestArtifact;
	sourcePath: string;
	digest: string;
	kind: "file" | "directory";
	artifactId: string;
	objectDir: string;
	payloadPath: string;
	metadataPath: string;
	metadata: StoredArtifactMetadata;
	occurrenceMetadata: StoredArtifactMetadata;
}

function matchesRequirement(artifact: XlabManifestArtifact, requirement: XlabArtifactRequirement): boolean {
	return (
		(!requirement.type || artifact.type === requirement.type) &&
		(!requirement.schemaVersion || artifact.schema_version === requirement.schemaVersion) &&
		(!requirement.path || artifact.path === requirement.path)
	);
}

export class XlabArtifactStore {
	readonly root: string;
	private indexPath: string;

	constructor(cwd: string) {
		this.root = join(cwd, ".xlab", "artifacts");
		this.indexPath = join(this.root, "index.sqlite");
		mkdirSync(join(this.root, "objects", "sha256"), { recursive: true });
		this.initializeIndex();
	}

	commitManifestArtifacts(
		run: XlabRunRecord,
		manifest: XlabManifestEnvelope,
		resolvePath: (path: string) => string | undefined,
		requiredArtifacts: XlabArtifactRequirement[] = [],
	): XlabArtifactReference[] {
		if (
			manifest.schema_version !== "2" ||
			manifest.run_id !== run.runId ||
			manifest.skill_name !== run.skillName ||
			manifest.skill_version !== run.skillVersion
		) {
			throw new Error("Artifact manifest does not match the run or schema-2 contract.");
		}
		for (const requirement of requiredArtifacts.filter((item) => item.required)) {
			if (!manifest.artifacts.some((artifact) => matchesRequirement(artifact, requirement))) {
				throw new Error(
					`Artifact manifest is missing required artifact ${requirement.type ?? requirement.path ?? "declaration"}.`,
				);
			}
		}
		const createdAt = new Date().toISOString();
		const preflight = manifest.artifacts.map((artifact) =>
			this.preflightArtifact(run, manifest, artifact, resolvePath, createdAt),
		);
		for (const item of preflight) this.stageObject(item);

		const database = new DatabaseSync(this.indexPath);
		try {
			database.exec("BEGIN IMMEDIATE");
			try {
				for (const item of preflight) this.commitPreflight(database, run, manifest, item);
				const manifestJson = JSON.stringify(manifest);
				const manifestDigest = createHash("sha256").update(manifestJson).digest("hex");
				const existing = database
					.prepare("SELECT manifest_digest, manifest_json FROM manifest_occurrences WHERE producer_run_id = ?")
					.get(run.runId) as { manifest_digest: string; manifest_json: string } | undefined;
				if (existing && (existing.manifest_digest !== manifestDigest || existing.manifest_json !== manifestJson)) {
					throw new Error(`Conflicting artifact manifest occurrence for run ${run.runId}.`);
				}
				if (!existing)
					database
						.prepare(
							"INSERT INTO manifest_occurrences (producer_run_id, manifest_digest, manifest_json, created_at) VALUES (?, ?, ?, ?)",
						)
						.run(run.runId, manifestDigest, manifestJson, createdAt);
				database.exec("COMMIT");
			} catch (error) {
				database.exec("ROLLBACK");
				throw error;
			}
		} finally {
			database.close();
		}
		return preflight.map((item) => ({
			artifactId: item.artifactId,
			type: item.artifact.type,
			schemaVersion: item.artifact.schema_version,
			digest: item.digest,
			payloadPath: item.payloadPath,
			metadataPath: item.metadataPath,
		}));
	}

	list(): StoredArtifactMetadata[] {
		const database = new DatabaseSync(this.indexPath, { readOnly: true });
		try {
			const rows = database
				.prepare("SELECT metadata_json FROM artifacts ORDER BY created_at DESC, artifact_id")
				.all() as unknown as Array<{ metadata_json: string }>;
			return rows.map((row) => JSON.parse(row.metadata_json) as StoredArtifactMetadata);
		} finally {
			database.close();
		}
	}

	get(artifactId: string): StoredArtifactLookup | undefined {
		const database = new DatabaseSync(this.indexPath, { readOnly: true });
		try {
			const row = database
				.prepare(
					`SELECT artifact_id, digest, type, schema_version, payload_path, metadata_path, metadata_json
					FROM artifacts WHERE artifact_id = ?`,
				)
				.get(artifactId) as StoredArtifactIndexRow | undefined;
			if (!row?.metadata_json) {
				return undefined;
			}
			if (!existsSync(row.payload_path) || !existsSync(row.metadata_path)) {
				throw new Error(`Artifact object is missing for ${artifactId}`);
			}
			const metadata = JSON.parse(row.metadata_json) as StoredArtifactMetadata;
			const persistedMetadata = JSON.parse(readFileSync(row.metadata_path, "utf-8")) as StoredArtifactMetadata;
			const payload = hashPath(row.payload_path);
			const expectedKind = metadata.payload_kind;
			if (
				row.artifact_id !== artifactId ||
				row.digest !== metadata.digest ||
				row.type !== metadata.type ||
				row.schema_version !== metadata.schema_version ||
				metadata.artifact_id !== artifactId ||
				persistedMetadata.artifact_id !== artifactId ||
				persistedMetadata.digest !== row.digest ||
				persistedMetadata.type !== row.type ||
				persistedMetadata.schema_version !== row.schema_version ||
				persistedMetadata.payload_kind !== expectedKind ||
				payload.digest !== row.digest ||
				payload.kind !== expectedKind
			) {
				throw new Error(`Artifact object failed integrity validation: ${artifactId}`);
			}
			const occurrenceRows = database
				.prepare(
					`SELECT producer_run_id, type, schema_version, producer_skill, producer_version,
						metadata_json, created_at
					FROM artifact_occurrences WHERE artifact_id = ? ORDER BY created_at, producer_run_id`,
				)
				.all(artifactId) as unknown as Array<StoredArtifactOccurrence & { metadata_json: string }>;
			const occurrences = occurrenceRows.map(({ metadata_json, ...occurrence }) => {
				const occurrenceMetadata = JSON.parse(metadata_json) as StoredArtifactMetadata;
				const lineage = occurrenceLineage(occurrenceMetadata.metadata, occurrenceMetadata.parents);
				return lineage ? { ...occurrence, lineage } : occurrence;
			});
			const parents = database
				.prepare("SELECT parent_id FROM artifact_parents WHERE artifact_id = ? ORDER BY parent_id")
				.all(artifactId) as unknown as Array<{ parent_id: string }>;
			return {
				...metadata,
				parents: parents.map((parent) => parent.parent_id),
				occurrences,
				payloadPath: row.payload_path,
				metadataPath: row.metadata_path,
			};
		} finally {
			database.close();
		}
	}

	private preflightArtifact(
		run: XlabRunRecord,
		manifest: XlabManifestEnvelope,
		artifact: XlabManifestArtifact,
		resolvePath: (path: string) => string | undefined,
		createdAt: string,
	): PreflightArtifact {
		if (!artifact.type.trim() || !artifact.schema_version.trim())
			throw new Error("Artifact type and schema_version must be non-empty.");
		if ((artifact.parents ?? []).some((parent) => !/^sha256:[0-9a-f]{64}$/.test(parent)))
			throw new Error(`Artifact ${artifact.type} has invalid parent lineage.`);
		const sourcePath = resolvePath(artifact.path);
		if (!sourcePath || !existsSync(sourcePath))
			throw new Error(`Artifact path does not exist or escapes the run workspace: ${artifact.path}`);
		const { digest, contentDigest, bytes, kind } = hashPath(sourcePath);
		const declaredDigest =
			artifact.digest ?? (typeof artifact.metadata?.sha256 === "string" ? artifact.metadata.sha256 : undefined);
		if (artifact.bytes !== undefined && artifact.bytes !== bytes)
			throw new Error(`Artifact byte count does not match ${artifact.path}.`);
		if (declaredDigest !== undefined && declaredDigest.replace(/^sha256:/, "") !== contentDigest)
			throw new Error(`Artifact digest does not match ${artifact.path}.`);
		const objectDir = join(this.root, "objects", "sha256", digest.slice(0, 2), digest);
		const artifactId = `sha256:${digest}`;
		const artifactMetadata = artifact.metadata ?? {};
		const lineage = occurrenceLineage(artifactMetadata, artifact.parents ?? []);
		const metadata: StoredArtifactMetadata = {
			artifact_id: artifactId,
			type: artifact.type,
			schema_version: artifact.schema_version,
			digest,
			created_at: createdAt,
			producer: { skill: run.skillName, version: manifest.skill_version, run_id: run.runId },
			parents: lineage ? [] : (artifact.parents ?? []),
			validation: manifest.validation,
			metadata: contentMetadata(artifactMetadata, lineage !== undefined),
			payload_kind: kind,
		};
		return {
			artifact,
			sourcePath,
			digest,
			kind,
			artifactId,
			objectDir,
			payloadPath: join(objectDir, "payload"),
			metadataPath: join(objectDir, "artifact.json"),
			metadata,
			occurrenceMetadata: {
				...metadata,
				producer: { ...metadata.producer },
				parents: artifact.parents ?? [],
				metadata: artifactMetadata,
			},
		};
	}

	private stageObject(item: PreflightArtifact): void {
		if (existsSync(item.objectDir)) {
			if (!existsSync(item.payloadPath) || !existsSync(item.metadataPath)) {
				throw new Error(`Artifact object is incomplete for ${item.artifactId}.`);
			}
			const persisted = JSON.parse(readFileSync(item.metadataPath, "utf-8")) as StoredArtifactMetadata;
			const payload = hashPath(item.payloadPath);
			if (
				persisted.artifact_id !== item.artifactId ||
				persisted.digest !== item.digest ||
				persisted.type !== item.artifact.type ||
				persisted.schema_version !== item.artifact.schema_version ||
				persisted.payload_kind !== item.kind ||
				payload.digest !== item.digest ||
				payload.kind !== item.kind
			) {
				throw new Error(`Artifact object conflicts with ${item.artifactId}.`);
			}
			item.metadata = persisted;
			return;
		}
		mkdirSync(dirname(item.objectDir), { recursive: true });
		const stagingObject = mkdtempSync(join(dirname(item.objectDir), ".staging-"));
		try {
			cpSync(item.sourcePath, join(stagingObject, "payload"), {
				recursive: item.kind === "directory",
				errorOnExist: true,
			});
			writeJson(join(stagingObject, "artifact.json"), item.metadata);
			try {
				renameSync(stagingObject, item.objectDir);
			} catch (error) {
				const code = (error as NodeJS.ErrnoException).code;
				if ((code !== "EEXIST" && code !== "ENOTEMPTY") || !existsSync(item.objectDir)) throw error;
			}
		} finally {
			rmSync(stagingObject, { recursive: true, force: true });
		}
	}

	private commitPreflight(
		database: DatabaseSync,
		run: XlabRunRecord,
		manifest: XlabManifestEnvelope,
		item: PreflightArtifact,
	): void {
		const existingArtifact = database
			.prepare("SELECT type, schema_version FROM artifacts WHERE artifact_id = ?")
			.get(item.artifactId) as { type: string; schema_version: string } | undefined;
		if (
			existingArtifact &&
			(existingArtifact.type !== item.artifact.type ||
				existingArtifact.schema_version !== item.artifact.schema_version)
		) {
			throw new Error(`Artifact ${item.artifactId} already exists with conflicting type or schema.`);
		}
		if (!existingArtifact)
			database
				.prepare(`INSERT INTO artifacts (
			artifact_id, digest, type, schema_version, producer_skill, producer_version, producer_run_id,
			payload_path, metadata_path, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
				.run(
					item.artifactId,
					item.digest,
					item.artifact.type,
					item.artifact.schema_version,
					run.skillName,
					manifest.skill_version,
					run.runId,
					item.payloadPath,
					item.metadataPath,
					JSON.stringify(item.metadata),
					item.metadata.created_at,
				);
		const existingOccurrence = database
			.prepare(`SELECT schema_version, producer_skill, producer_version, metadata_json
			FROM artifact_occurrences WHERE artifact_id = ? AND producer_run_id = ? AND type = ?`)
			.get(item.artifactId, run.runId, item.artifact.type) as
			| { schema_version: string; producer_skill: string; producer_version: string; metadata_json: string }
			| undefined;
		const occurrenceMetadata = existingOccurrence
			? {
					...item.occurrenceMetadata,
					created_at: (JSON.parse(existingOccurrence.metadata_json) as StoredArtifactMetadata).created_at,
				}
			: item.occurrenceMetadata;
		const occurrenceJson = JSON.stringify(occurrenceMetadata);
		if (
			existingOccurrence &&
			(existingOccurrence.schema_version !== item.artifact.schema_version ||
				existingOccurrence.producer_skill !== run.skillName ||
				existingOccurrence.producer_version !== manifest.skill_version ||
				existingOccurrence.metadata_json !== occurrenceJson)
		) {
			throw new Error(`Conflicting artifact occurrence for ${item.artifactId} in run ${run.runId}.`);
		}
		if (!existingOccurrence)
			database
				.prepare(`INSERT INTO artifact_occurrences (
			artifact_id, producer_run_id, type, schema_version, producer_skill, producer_version, metadata_json, created_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
				.run(
					item.artifactId,
					run.runId,
					item.artifact.type,
					item.artifact.schema_version,
					run.skillName,
					manifest.skill_version,
					occurrenceJson,
					item.metadata.created_at,
				);
		const insertParent = database.prepare(
			"INSERT OR IGNORE INTO artifact_parents (artifact_id, parent_id) VALUES (?, ?)",
		);
		for (const parent of item.metadata.parents) insertParent.run(item.artifactId, parent);
	}

	private initializeIndex(): void {
		const database = new DatabaseSync(this.indexPath);
		try {
			database.exec(`
				PRAGMA journal_mode = WAL;
				CREATE TABLE IF NOT EXISTS artifacts (
					artifact_id TEXT PRIMARY KEY,
					digest TEXT NOT NULL,
					type TEXT NOT NULL,
					schema_version TEXT NOT NULL,
					producer_skill TEXT NOT NULL,
					producer_version TEXT NOT NULL,
					producer_run_id TEXT NOT NULL,
					payload_path TEXT NOT NULL,
					metadata_path TEXT NOT NULL,
					metadata_json TEXT NOT NULL,
					created_at TEXT NOT NULL
				) STRICT;
				CREATE TABLE IF NOT EXISTS artifact_parents (
					artifact_id TEXT NOT NULL,
					parent_id TEXT NOT NULL,
					PRIMARY KEY (artifact_id, parent_id)
				) STRICT;
				CREATE TABLE IF NOT EXISTS artifact_occurrences (
					artifact_id TEXT NOT NULL,
					producer_run_id TEXT NOT NULL,
					type TEXT NOT NULL,
					schema_version TEXT NOT NULL,
					producer_skill TEXT NOT NULL,
					producer_version TEXT NOT NULL,
					metadata_json TEXT NOT NULL,
					created_at TEXT NOT NULL,
					PRIMARY KEY (artifact_id, producer_run_id, type)
				) STRICT;
				CREATE TABLE IF NOT EXISTS manifest_occurrences (
					producer_run_id TEXT PRIMARY KEY,
					manifest_digest TEXT NOT NULL,
					manifest_json TEXT NOT NULL,
					created_at TEXT NOT NULL
				) STRICT;
			`);
		} finally {
			database.close();
		}
	}
}
