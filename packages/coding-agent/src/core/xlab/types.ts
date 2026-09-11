import type { ImageContent, TextContent } from "@earendil-works/pi-ai";

export type XlabRunStatus = "pending" | "running" | "paused" | "success" | "failed" | "incomplete" | "cancelled";

export type XlabHookPoint =
	| "pre_start"
	| "pre_stage"
	| "post_stage"
	| "pre_tool_call"
	| "post_tool_result"
	| "pre_finish"
	| "post_finish"
	| "on_error"
	| "on_resume"
	| "on_cancel";

export type XlabSkillVisibility = "public" | "internal" | "service-control";

export type XlabRuntimeKind = "agent" | "node" | "python" | "container" | "workflow" | "native" | "service-control";

export interface XlabHookDeclaration {
	kind?: "module" | "command";
	module?: string;
	handler?: string;
	command?: string;
	args?: string[];
	timeoutMs?: number;
	blocking?: boolean;
}

export interface XlabArtifactRequirement {
	type?: string;
	schemaVersion?: string;
	schema?: string;
	path?: string;
	required?: boolean;
	description?: string;
}

export interface XlabRuntimeDeclaration {
	kind: XlabRuntimeKind;
	driver?: string;
	entrypoint?: string;
	lockfile?: string;
	containerfile?: string;
	executables?: string[];
	requiredSecrets?: string[];
	optionalSecrets?: string[];
	secretAliases?: Record<string, string[]>;
}

export interface XlabConfigVariableHint {
	name: string;
	description?: string;
	secret?: boolean;
	required?: boolean;
	default?: unknown;
	aliases?: string[];
}

export interface XlabSkillConfigHints {
	environment?: XlabConfigVariableHint[];
}

export interface XlabSkillPhaseHint {
	id: string;
	label?: string;
	description?: string;
}

export type XlabCommandDynamicSource =
	| "workspace"
	| "graph"
	| "survey"
	| "idea"
	| "run"
	| "artifact"
	| "file"
	| "directory"
	| "gpu";

export interface XlabCommandArgumentHint {
	name: string;
	description: string;
	syntax?: "flag" | "assignment";
	valueHint?: string;
	choices?: string[];
	dynamicSource?: XlabCommandDynamicSource;
	takesValue: boolean;
	repeatable?: boolean;
}

export interface XlabPermissionDeclaration {
	tools?: string[];
	network?: boolean;
	subprocess?: boolean;
	externalMutations?: boolean;
}

export interface XlabSkillDependency {
	name: string;
	version: string;
}

export interface XlabWorkflowStageDeclaration {
	id: string;
	skill: string;
	needs?: string[];
	optional?: boolean;
	maxAttempts?: number;
}

export interface XlabWorkflowDeclaration {
	stages: XlabWorkflowStageDeclaration[];
}

export interface XlabSkillUiHints {
	primaryCounters?: string[];
	artifactTypes?: string[];
	defaultExpandedSections?: string[];
	phases?: XlabSkillPhaseHint[];
	argumentHint?: string;
	arguments?: XlabCommandArgumentHint[];
}

export interface XlabSkillManifest {
	schemaVersion: "2";
	name: string;
	version: string;
	visibility: XlabSkillVisibility;
	description?: string;
	prompt: string;
	runtime: XlabRuntimeDeclaration;
	permissions?: XlabPermissionDeclaration;
	dependencies?: XlabSkillDependency[];
	inputSchema?: string;
	outputSchema?: string;
	workflow?: XlabWorkflowDeclaration;
	hooks?: Partial<Record<XlabHookPoint, XlabHookDeclaration[]>>;
	artifacts?: XlabArtifactRequirement[];
	ui?: XlabSkillUiHints;
	config?: XlabSkillConfigHints;
}

export interface XlabSkillPackage {
	manifest: XlabSkillManifest;
	manifestPath: string;
	packageDir: string;
	promptPath: string;
	prompt: string;
	source: "project" | "repository";
	sourceRoot: string;
	dependencies: XlabSkillPackage[];
}

export interface XlabManifestArtifact {
	type: string;
	schema_version: string;
	path: string;
	bytes?: number;
	digest?: string;
	parents?: string[];
	metadata?: Record<string, unknown>;
}

export interface XlabArtifactReference {
	artifactId: string;
	type: string;
	schemaVersion: string;
	digest: string;
	payloadPath: string;
	metadataPath: string;
}

export type XlabWorkspaceArtifactFamily = "graph" | "survey" | "idea" | "other";

export interface XlabWorkspaceRecord {
	schemaVersion: "1";
	id: string;
	slug: string;
	displayName: string;
	createdAt: string;
	updatedAt: string;
	description?: string;
	topic?: string;
}

export interface XlabWorkspaceIndexEntry {
	slug: string;
	displayName: string;
	updatedAt: string;
}

export interface XlabWorkspaceIndex {
	schemaVersion: "1";
	activeWorkspaceSlug?: string;
	workspaces: XlabWorkspaceIndexEntry[];
}

export interface XlabWorkspaceHandleRecord {
	schemaVersion: "1";
	workspaceSlug: string;
	handle: string;
	family: XlabWorkspaceArtifactFamily;
	artifactId: string;
	artifactIds: string[];
	artifactRefs: XlabArtifactReference[];
	runId: string;
	runDir: string;
	manifestPath?: string;
	skillName: string;
	skillVersion: string;
	status: XlabRunStatus;
	parents: string[];
	parentHandles: string[];
	createdAt: string;
	updatedAt: string;
	validation?: Record<string, unknown>;
	summary?: string;
}

export interface XlabWorkspaceCurrent {
	schemaVersion: "1";
	workspaceSlug: string;
	current: Partial<Record<XlabWorkspaceArtifactFamily, string>>;
	updatedAt: string;
}

export interface XlabWorkspaceLineage {
	schemaVersion: "1";
	workspaceSlug: string;
	handles: XlabWorkspaceHandleRecord[];
	updatedAt: string;
}

export interface XlabEnvironmentRecord {
	kind: "python-system" | "python-venv";
	environmentId: string;
	path: string;
	executable: string;
	lockfile: string;
	lockDigest: string;
	runtimeVersion: string;
}

export interface XlabManifestEnvelope {
	schema_version: "2";
	run_id: string;
	skill_name: string;
	skill_version: string;
	status: XlabRunStatus;
	created_at: string;
	inputs: Record<string, unknown>;
	outputs: Record<string, unknown>;
	validation: Record<string, unknown>;
	artifacts: XlabManifestArtifact[];
}

export interface XlabRunRecord {
	runId: string;
	skillName: string;
	skillVersion: string;
	command: string;
	invokedCommand?: string;
	status: XlabRunStatus;
	cwd: string;
	packageDir: string;
	runDir: string;
	args: string;
	startedAt: string;
	updatedAt: string;
	finishedAt?: string;
	manifestPath?: string;
	summary?: string;
	errorSummary?: string;
	lastRepair?: string;
	publicationRecoveryReason?: string;
	artifacts?: unknown;
	artifactRefs?: XlabArtifactReference[];
	workspaceSlug?: string;
	workspaceHandle?: string;
	environment?: XlabEnvironmentRecord;
	resumedFrom?: string;
	resumeCount?: number;
}

export type XlabWorkflowStageStatus = "pending" | "running" | "success" | "failed" | "blocked" | "skipped";

export interface XlabWorkflowStageState {
	id: string;
	skill: string;
	needs: string[];
	optional: boolean;
	maxAttempts: number;
	attempts: number;
	status: XlabWorkflowStageStatus;
	startedAt?: string;
	finishedAt?: string;
	checkpoint?: unknown;
	artifacts?: unknown;
	error?: string;
}

export interface XlabWorkflowState {
	schemaVersion: "1";
	runId: string;
	stages: XlabWorkflowStageState[];
	updatedAt: string;
}

export type XlabDisplayPhaseStatus =
	| "pending"
	| "running"
	| "success"
	| "failed"
	| "incomplete"
	| "skipped"
	| "blocked";

export interface XlabDisplayPhase {
	id?: string;
	label?: string;
	status?: XlabDisplayPhaseStatus;
	progress?: number;
}

export interface XlabDisplayDiagnostic {
	severity?: "info" | "warning" | "error";
	code?: string;
	message: string;
	repair?: string;
	data?: unknown;
}

export interface XlabDisplayArtifact {
	type?: string;
	name?: string;
	path?: string;
	status?: "draft" | "ready" | "failed" | "incomplete";
	summary?: string;
	metadata?: Record<string, unknown>;
}

export interface XlabDisplayCheckpoint {
	id?: string;
	label?: string;
	resumable?: boolean;
	resume_hint?: string;
	data?: unknown;
}

export interface XlabDisplayState {
	phase?: XlabDisplayPhase;
	message?: string;
	progress?: number;
	counters?: Record<string, number>;
	diagnostics?: XlabDisplayDiagnostic[];
	artifacts?: XlabDisplayArtifact[];
	checkpoint?: XlabDisplayCheckpoint;
	next_actions?: string[];
}

export interface XlabHookContext {
	point: XlabHookPoint;
	run: XlabRunRecord;
	skill: XlabSkillManifest;
	cwd: string;
	packageDir: string;
	runDir: string;
	args: string;
	event?: unknown;
}

export interface XlabHookResult {
	ok?: boolean;
	message?: string;
	repair?: string;
	patch?: unknown;
	artifacts?: unknown;
	diagnostics?: unknown;
	state_patch?: unknown;
}

export interface XlabFinishParams {
	status: "success" | "incomplete" | "failed";
	manifest_path: string;
	summary: string;
	artifacts?: unknown;
	error_summary?: string;
	next_actions?: string[];
}

export interface XlabFinishDetails {
	runId?: string;
	status?: XlabRunStatus;
	accepted: boolean;
	repair?: string;
	diagnostics?: unknown;
}

export interface XlabUpdateStateDetails {
	runId?: string;
	accepted: boolean;
	state?: XlabDisplayState;
	repair?: string;
}

export type XlabToolContent = TextContent | ImageContent;
