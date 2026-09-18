import { existsSync, readdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { type Static, type TSchema, Type } from "typebox";
import type { AgentSession } from "../../agent-session.ts";
import { type AgentToolResult, defineTool, type ToolDefinition } from "../../extensions/index.ts";
import { DefaultResourceLoader } from "../../resource-loader.ts";
import { type CreateAgentSessionOptions, createAgentSession } from "../../sdk.ts";
import { SessionManager } from "../../session-manager.ts";
import { SettingsManager } from "../../settings-manager.ts";
import { createEditToolDefinition } from "../../tools/edit.ts";
import { createReadToolDefinition } from "../../tools/read.ts";
import { createWriteToolDefinition } from "../../tools/write.ts";
import {
	atomicWriteFile,
	type CanonicalJson,
	type ExperimentAssignment,
	type ExperimentChildIdentity,
	type ExperimentChildRole,
	type ExperimentOwnerFence,
	type ExperimentProtocolStore,
	type ExperimentSubmission,
	type ExperimentSubmissionKind,
	resolveContainedPath,
} from "./protocol.ts";

const SUBMISSION_TOOL_NAMES: Record<ExperimentSubmissionKind, string> = {
	plan: "submit_plan",
	worker_result: "submit_worker_result",
	review: "submit_review",
	final_review: "submit_final_review",
};

const ROLE_KINDS: Record<ExperimentChildRole, ExperimentSubmissionKind> = {
	planner: "plan",
	worker: "worker_result",
	reviewer: "review",
	final_reviewer: "final_review",
};

const submissionParameters = Type.Object({
	result: Type.Optional(Type.Unknown()),
	bound_plan: Type.Optional(
		Type.Boolean({
			description:
				"For planner assignments only, submit the complete immutable executor-bound plan without retranscribing it.",
		}),
	),
	output_paths: Type.Optional(Type.Array(Type.String())),
});

type SubmissionParameters = Static<typeof submissionParameters>;

export interface ExperimentChildRuntimeOptions {
	runDir: string;
	agentDir: string;
	protocol: ExperimentProtocolStore;
	parentSession?: string;
	settingsManager?: SettingsManager;
	clock?: () => string;
	createSession?: typeof createAgentSession;
	systemPrompt?: (assignment: ExperimentAssignment) => string;
	validateResult?: (assignment: ExperimentAssignment, result: CanonicalJson) => string | undefined;
	assignmentTools?: (assignment: ExperimentAssignment) => NonNullable<CreateAgentSessionOptions["customTools"]>;
	boundPlan?: (assignment: ExperimentAssignment) => CanonicalJson;
	allowEdits?: (assignment: ExperimentAssignment) => boolean;
	agentOptions?: Pick<CreateAgentSessionOptions, "authStorage" | "modelRegistry" | "model" | "thinkingLevel">;
	ownerFence?: () => ExperimentOwnerFence | undefined;
}

export interface CreateExperimentChildOptions {
	child_id: string;
	parent_run_id: string;
	role: ExperimentChildRole;
	stage: ExperimentChildIdentity["stage"];
	work_unit?: string;
	execution_attempt: number;
	generation: number;
	review_round: number;
	cwd: string;
}

export interface AdvanceExperimentChildOptions {
	child_id: string;
	execution_attempt: number;
	generation: number;
	review_round: number;
}

export interface ExperimentChildHandle {
	identity: ExperimentChildIdentity;
	assignment: ExperimentAssignment;
	session: AgentSession;
	resumed: boolean;
}

function toolText(text: string): Array<{ type: "text"; text: string }> {
	return [{ type: "text", text }];
}

function expectedKind(role: ExperimentChildRole): ExperimentSubmissionKind {
	return ROLE_KINDS[role];
}

function assertAssignmentRole(assignment: ExperimentAssignment): void {
	const kind = expectedKind(assignment.child.role);
	if (assignment.kind !== kind) {
		throw new Error(`Role ${assignment.child.role} requires a ${kind} assignment, not ${assignment.kind}.`);
	}
}

function allowedPath(scopes: string[], pathValue: string, requireExisting: boolean): string {
	for (const scope of scopes) {
		try {
			return resolveContainedPath(scope, pathValue, requireExisting);
		} catch {
			// Try the next immutable scope.
		}
	}
	throw new Error(`Path is outside the assignment scope: ${pathValue}`);
}

function constrainPathTool<TParams extends TSchema, TDetails, TState>(
	tool: ToolDefinition<TParams, TDetails, TState>,
	scopes: string[],
	requireExisting: boolean,
): ToolDefinition<TParams, TDetails, TState> {
	return defineTool({
		...tool,
		async execute(toolCallId, params, signal, onUpdate, ctx) {
			const value = params as Record<string, unknown>;
			if (typeof value.path !== "string") {
				throw new Error(`${tool.name} requires a path.`);
			}
			const path = allowedPath(scopes, value.path, requireExisting);
			if (tool.name === "read" && statSync(path).isDirectory()) {
				return {
					content: toolText(
						JSON.stringify(
							readdirSync(path, { withFileTypes: true }).map((entry) => ({
								name: entry.name,
								directory: entry.isDirectory(),
							})),
						),
					),
					details: {} as TDetails,
				};
			}
			return tool.execute(toolCallId, { ...value, path } as Static<TParams>, signal, onUpdate, ctx);
		},
	});
}

interface SubmissionToolDetails {
	accepted: boolean;
	error?: string;
	disposition?: "accepted" | "idempotent";
	record?: unknown;
}

function createSubmissionTool(
	assignment: ExperimentAssignment,
	token: string,
	protocol: ExperimentProtocolStore,
	clock: () => string,
	ownerFence?: () => ExperimentOwnerFence | undefined,
	validateResult?: ExperimentChildRuntimeOptions["validateResult"],
	boundPlan?: ExperimentChildRuntimeOptions["boundPlan"],
): ToolDefinition<typeof submissionParameters, SubmissionToolDetails> {
	const name = SUBMISSION_TOOL_NAMES[assignment.kind];
	return defineTool({
		name,
		label: name,
		description: `Submit the structured ${assignment.kind} for the current immutable assignment.`,
		promptSnippet: `Submit the completed ${assignment.kind}`,
		promptGuidelines: [`Call ${name} exactly once after completing and checking the assignment.`],
		executionMode: "sequential",
		parameters: submissionParameters,
		async execute(_toolCallId, params: SubmissionParameters): Promise<AgentToolResult<SubmissionToolDetails>> {
			if (params.bound_plan && (assignment.kind !== "plan" || !boundPlan))
				throw new Error("No executor-bound plan is available for this assignment.");
			const result = params.bound_plan ? boundPlan!(assignment) : (params.result as CanonicalJson);
			if (result === undefined) throw new Error("Submission requires result or a bound plan.");
			const resultError = validateResult?.(assignment, result);
			if (resultError) {
				return {
					content: toolText(resultError),
					details: { accepted: false, error: resultError },
				};
			}
			const submission: ExperimentSubmission = {
				schema_version: "xlab.experiment_submission.v1",
				submission_id: `${assignment.assignment_id}-submission`,
				assignment_id: assignment.assignment_id,
				assignment_token: token,
				kind: assignment.kind,
				child_id: assignment.child.child_id,
				role: assignment.child.role,
				stage: assignment.child.stage,
				...(assignment.child.work_unit === undefined ? {} : { work_unit: assignment.child.work_unit }),
				execution_attempt: assignment.child.execution_attempt,
				generation: assignment.child.generation,
				review_round: assignment.child.review_round,
				input_digest: assignment.input_digest,
				target_digest: assignment.target_digest,
				policy_digest: assignment.policy_digest,
				result,
				output_paths: params.output_paths ?? [],
				submitted_at: clock(),
			};
			const accepted = protocol.acceptSubmission(submission, ownerFence?.());
			if (!accepted.ok) {
				return {
					content: toolText(accepted.error),
					details: { accepted: false, error: accepted.error },
				};
			}
			return {
				content: toolText(`Accepted ${assignment.kind} for ${assignment.assignment_id}.`),
				terminate: true,
				details: { accepted: true, disposition: accepted.disposition, record: accepted.accepted },
			};
		},
	});
}

function persistNewSessionHeader(manager: SessionManager): SessionManager {
	const sessionFile = manager.getSessionFile();
	const header = manager.getHeader();
	if (!sessionFile || !header) {
		throw new Error("Experiment child sessions must be durable.");
	}
	atomicWriteFile(sessionFile, Buffer.from(`${JSON.stringify(header)}\n`, "utf-8"));
	return SessionManager.open(sessionFile, manager.getSessionDir(), manager.getCwd());
}

function isolatedSystemPrompt(assignment: ExperimentAssignment): string {
	return [
		`You are the ${assignment.child.role} child for XLab experiment run ${assignment.child.parent_run_id}.`,
		`Complete only assignment ${assignment.assignment_id} for stage ${assignment.child.stage}${assignment.child.work_unit ? `, work unit ${assignment.child.work_unit}` : ""}.`,
		"The assignment digests and identity are immutable. Do not attempt to update parent workflow, aggregate, manifest, or symbolic memory state.",
		`Finish by calling ${SUBMISSION_TOOL_NAMES[assignment.kind]}.`,
	].join("\n");
}

export class ExperimentChildRuntime {
	private readonly live = new Map<string, AgentSession>();
	private readonly clock: () => string;
	private readonly createSession: typeof createAgentSession;
	private readonly options: ExperimentChildRuntimeOptions;

	constructor(options: ExperimentChildRuntimeOptions) {
		this.options = options;
		this.clock = options.clock ?? (() => new Date().toISOString());
		this.createSession = options.createSession ?? createAgentSession;
	}

	createIdentity(options: CreateExperimentChildOptions): ExperimentChildIdentity {
		const sessionDir = join(this.options.runDir, "experiment", "sessions");
		const created = SessionManager.create(resolve(options.cwd), sessionDir, {
			id: options.child_id,
			parentSession: this.options.parentSession,
		});
		const durable = persistNewSessionHeader(created);
		const sessionFile = durable.getSessionFile();
		if (!sessionFile) {
			throw new Error("Experiment child session file was not allocated.");
		}
		const identity: ExperimentChildIdentity = {
			child_id: options.child_id,
			session_id: durable.getSessionId(),
			session_file: sessionFile,
			parent_run_id: options.parent_run_id,
			role: options.role,
			stage: options.stage,
			...(options.work_unit === undefined ? {} : { work_unit: options.work_unit }),
			execution_attempt: options.execution_attempt,
			generation: options.generation,
			review_round: options.review_round,
		};
		this.options.protocol.registerChild(identity, this.options.ownerFence?.());
		return identity;
	}

	advanceIdentity(options: AdvanceExperimentChildOptions): ExperimentChildIdentity {
		if (this.live.has(options.child_id)) {
			throw new Error(`Child session is still open: ${options.child_id}`);
		}
		const current = this.options.protocol.read().children[options.child_id];
		if (!current) {
			throw new Error(`Unknown child identity ${options.child_id}.`);
		}
		if (
			options.execution_attempt < current.execution_attempt ||
			options.generation < current.generation ||
			options.review_round < current.review_round
		) {
			throw new Error(`Child identity cannot move backwards: ${options.child_id}.`);
		}
		const identity: ExperimentChildIdentity = {
			...current,
			execution_attempt: options.execution_attempt,
			generation: options.generation,
			review_round: options.review_round,
		};
		this.options.protocol.advanceChild(identity, this.options.ownerFence?.());
		return identity;
	}

	async openAssignment(assignment: ExperimentAssignment, token: string): Promise<ExperimentChildHandle> {
		assertAssignmentRole(assignment);
		const child = assignment.child;
		if (this.live.has(child.child_id)) {
			throw new Error(`Child session is already open: ${child.child_id}`);
		}
		if (!existsSync(child.session_file)) {
			throw new Error(`Durable child session is missing: ${child.session_file}`);
		}
		const sessionDir = dirname(child.session_file);
		const manager = SessionManager.open(child.session_file, sessionDir);
		if (manager.getSessionId() !== child.session_id) {
			throw new Error(`Child session ID does not match durable identity ${child.child_id}.`);
		}
		const resumed = manager.buildSessionContext().messages.length > 0;
		const settingsManager =
			this.options.settingsManager ?? SettingsManager.create(manager.getCwd(), this.options.agentDir);
		const resourceLoader = new DefaultResourceLoader({
			cwd: manager.getCwd(),
			agentDir: this.options.agentDir,
			settingsManager,
			noExtensions: true,
			noSkills: true,
			noPromptTemplates: true,
			noThemes: true,
			noContextFiles: true,
			systemPrompt: this.options.systemPrompt?.(assignment) ?? isolatedSystemPrompt(assignment),
		});
		await resourceLoader.reload();

		const customTools: ToolDefinition<any, any, any>[] = [];
		const activeTools: string[] = [];
		if (assignment.read_scope.length > 0) {
			customTools.push(constrainPathTool(createReadToolDefinition(manager.getCwd()), assignment.read_scope, true));
			activeTools.push("read");
		}
		if (
			child.role === "worker" &&
			(this.options.allowEdits?.(assignment) ?? true) &&
			assignment.write_scope.length > 0
		) {
			customTools.push(constrainPathTool(createEditToolDefinition(manager.getCwd()), assignment.write_scope, true));
			customTools.push(
				constrainPathTool(createWriteToolDefinition(manager.getCwd()), assignment.write_scope, false),
			);
			activeTools.push("edit", "write");
		}
		const submitTool = createSubmissionTool(
			assignment,
			token,
			this.options.protocol,
			this.clock,
			this.options.ownerFence,
			this.options.validateResult,
			this.options.boundPlan,
		);
		customTools.push(submitTool);
		activeTools.push(submitTool.name);
		for (const tool of this.options.assignmentTools?.(assignment) ?? []) {
			if (activeTools.includes(tool.name)) throw new Error(`Duplicate assignment tool ${tool.name}.`);
			customTools.push(tool);
			activeTools.push(tool.name);
		}

		const { session } = await this.createSession({
			cwd: manager.getCwd(),
			agentDir: this.options.agentDir,
			settingsManager,
			sessionManager: manager,
			resourceLoader,
			customTools,
			tools: activeTools,
			...this.options.agentOptions,
		});
		this.live.set(child.child_id, session);
		return { identity: child, assignment, session, resumed };
	}

	async abortChild(childId: string): Promise<boolean> {
		const session = this.live.get(childId);
		if (!session) {
			return false;
		}
		await session.abort();
		return true;
	}

	async abortAll(): Promise<void> {
		await Promise.all([...this.live.values()].map((session) => session.abort()));
	}

	release(childId: string): void {
		const session = this.live.get(childId);
		if (!session) {
			return;
		}
		this.live.delete(childId);
		session.dispose();
	}
}
