import { isAbsolute, normalize, resolve } from "node:path";
import type { AgentSession } from "../../agent-session.ts";
import { createExperimentAssignmentTokenFactory } from "./assignment-token.ts";
import {
	type CreateExperimentChildOptions,
	type ExperimentChildHandle,
	ExperimentChildRuntime,
	type ExperimentChildRuntimeOptions,
} from "./child-runtime.ts";
import {
	type CanonicalJson,
	canonicalDigest,
	canonicalJson,
	codeReviewRoles,
	EXPERIMENT_STAGES,
	type ExperimentAcceptedResult,
	type ExperimentAssignment,
	type ExperimentChildIdentity,
	type ExperimentChildRole,
	type ExperimentOwnerFence,
	type ExperimentPlan,
	type ExperimentProtocolSnapshot,
	type ExperimentProtocolStore,
	type ExperimentResultValidationContext,
	type ExperimentReviewReport,
	type ExperimentReviewValidationContext,
	type ExperimentStage,
	type ExperimentSubmissionKind,
	type ExperimentWorkUnit,
	SCIENCE_REVIEW_ROLES,
	validateExperimentResult,
	validateReviewMatrix,
} from "./protocol.ts";

const DEFAULT_LEASE_DURATION_MS = 5 * 60 * 1000;

export interface ExperimentCoordinatorOptions {
	runId: string;
	runDir: string;
	cwd: string;
	agentDir: string;
	ideaDigest: string;
	policyDigest: string;
	ownerId: string;
	protocol: ExperimentProtocolStore;
	tokenFactory?: (assignmentId: string, allowKeyCreation: boolean) => string;
	childRuntime?: ExperimentChildRuntime;
	childRuntimeOptions?: Omit<ExperimentChildRuntimeOptions, "runDir" | "agentDir" | "protocol" | "ownerFence">;
	leaseDurationMs?: number;
	maxAssignmentPromptAttempts?: number;
	clock?: () => string;
	onStageTransition?: (stage: ExperimentStage, action: "start" | "complete") => void;
}

export interface ExperimentAssignmentSpec {
	assignmentId: string;
	childId: string;
	role: ExperimentChildRole;
	stage: ExperimentStage;
	workUnit?: string;
	kind: ExperimentSubmissionKind;
	executionAttempt: number;
	generation: number;
	reviewRound: number;
	inputDigest: string;
	targetDigest: string;
	readScope: string[];
	writeScope: string[];
	resultValidation?: ExperimentResultValidationContext;
	prompt: string;
}

export interface ExperimentReviewSpec {
	stage: "code" | "science";
	workUnit: string;
	reviewRound: number;
	inputDigest: string;
	targetDigest: string;
	readScope: string[];
	prompt: (role: string) => string;
	validationContext?: ExperimentReviewValidationContext;
}

export interface ExperimentReviewMatrixResult {
	reports: ExperimentReviewReport[];
	assignmentIds: string[];
	inputDigest: string;
	targetDigest: string;
	reviewRound: number;
	blockingIssues: ExperimentReviewReport["issues"];
}

export interface ExperimentReviewedWorkerSpec {
	worker: ExperimentAssignmentSpec;
	maxReviewRounds: number;
	review: (
		accepted: ExperimentAcceptedResult,
		reviewRound: number,
	) => Omit<ExperimentReviewSpec, "reviewRound" | "targetDigest">;
	repair: (
		previous: ExperimentAcceptedResult,
		issues: ExperimentReviewReport["issues"],
		reviewRound: number,
	) => ExperimentAssignmentSpec;
}

export interface ExperimentReviewedWorkerResult {
	worker: ExperimentAcceptedResult;
	review: ExperimentReviewMatrixResult;
	reviewRound: number;
}

export interface ExperimentStageDriver {
	stage: ExperimentStage;
	run: (coordinator: ExperimentCoordinator) => Promise<void>;
}

export interface ExperimentWorkUnitDriver {
	workUnit: ExperimentWorkUnit;
	run: (coordinator: ExperimentCoordinator, workUnit: ExperimentWorkUnit) => Promise<void>;
}

function assignmentKindForRole(role: ExperimentChildRole): ExperimentSubmissionKind {
	switch (role) {
		case "planner":
			return "plan";
		case "worker":
			return "worker_result";
		case "reviewer":
			return "review";
		case "final_reviewer":
			return "final_review";
	}
}

function stablePart(value: string): string {
	const readable =
		value
			.trim()
			.replace(/[^A-Za-z0-9_.-]+/g, "-")
			.replace(/^-+|-+$/g, "")
			.slice(0, 48) || "unit";
	return `${readable}-${canonicalDigest(value).slice(0, 16)}`;
}

function canonicalScope(cwd: string, value: string): string {
	const absolute = isAbsolute(value) ? normalize(value) : resolve(cwd, value);
	if (!isAbsolute(absolute) || normalize(absolute) !== absolute) {
		throw new Error(`Experiment assignment scope is not canonical: ${value}`);
	}
	return absolute;
}

function resultObject(result: CanonicalJson, assignmentId: string): Record<string, unknown> {
	if (typeof result !== "object" || result === null || Array.isArray(result)) {
		throw new Error(`Assignment ${assignmentId} did not submit an object-shaped result.`);
	}
	return result;
}

export class ExperimentCoordinator {
	private readonly options: ExperimentCoordinatorOptions;
	private readonly protocol: ExperimentProtocolStore;
	private readonly childRuntime: ExperimentChildRuntime;
	private readonly tokenFactory: (assignmentId: string, allowKeyCreation: boolean) => string;
	private readonly clock: () => string;
	private readonly leaseDurationMs: number;
	private fence?: ExperimentOwnerFence;
	private ownershipLost?: Error;

	constructor(options: ExperimentCoordinatorOptions) {
		if (!options.ownerId.trim()) {
			throw new Error("Experiment coordinator owner ID must be non-empty.");
		}
		this.options = options;
		this.protocol = options.protocol;
		this.tokenFactory =
			options.tokenFactory ?? createExperimentAssignmentTokenFactory(options.agentDir, options.runId);
		this.clock = options.clock ?? (() => new Date().toISOString());
		this.leaseDurationMs = options.leaseDurationMs ?? DEFAULT_LEASE_DURATION_MS;
		if (!Number.isFinite(this.leaseDurationMs) || this.leaseDurationMs <= 0) {
			throw new Error("Experiment coordinator lease duration must be positive.");
		}
		this.childRuntime =
			options.childRuntime ??
			new ExperimentChildRuntime({
				runDir: options.runDir,
				agentDir: options.agentDir,
				protocol: options.protocol,
				ownerFence: () => this.fence,
				...options.childRuntimeOptions,
			});
	}

	initialize(): ExperimentProtocolSnapshot {
		return this.protocol.initialize({
			runId: this.options.runId,
			ideaDigest: this.options.ideaDigest,
			policyDigest: this.options.policyDigest,
		});
	}

	read(): ExperimentProtocolSnapshot {
		return this.protocol.read();
	}

	listAcceptedResults(): ExperimentAcceptedResult[] {
		return this.protocol.listAcceptedResults();
	}

	acquireOwnership(): ExperimentOwnerFence {
		this.assertOwnershipRetained();
		const now = this.clock();
		const expiresAt = new Date(Date.parse(now) + this.leaseDurationMs).toISOString();
		const snapshot = this.protocol.acquireOwner(this.options.ownerId, expiresAt);
		if (!snapshot.owner || snapshot.owner.owner_id !== this.options.ownerId) {
			throw new Error("Experiment coordinator failed to acquire ownership.");
		}
		this.fence = {
			owner_id: snapshot.owner.owner_id,
			generation: snapshot.owner.generation,
		};
		return this.fence;
	}

	async run(drivers: ExperimentStageDriver[]): Promise<ExperimentProtocolSnapshot> {
		const byStage = new Map(drivers.map((driver) => [driver.stage, driver]));
		if (
			drivers.length !== EXPERIMENT_STAGES.length ||
			byStage.size !== EXPERIMENT_STAGES.length ||
			EXPERIMENT_STAGES.some((stage) => !byStage.has(stage))
		) {
			throw new Error(
				`Experiment coordinator requires exactly these stage drivers: ${EXPERIMENT_STAGES.join(", ")}.`,
			);
		}
		this.initialize();
		for (const stage of EXPERIMENT_STAGES) {
			let snapshot = this.read();
			if (snapshot.completed_stages.includes(stage)) {
				continue;
			}
			if (snapshot.status === "paused") {
				return snapshot;
			}
			if (snapshot.terminal_generation !== undefined) {
				return snapshot;
			}
			this.acquireOwnership();
			const stageAlreadyRunning = snapshot.current_stage === stage && snapshot.status === "running";
			snapshot = this.protocol.start(stage, this.requireFence());
			if (snapshot.current_stage !== stage) {
				throw new Error(`Experiment protocol did not start stage ${stage}.`);
			}
			if (!stageAlreadyRunning) {
				this.options.onStageTransition?.(stage, "start");
			}
			await this.withLeaseHeartbeat(() => byStage.get(stage)!.run(this));
			snapshot = this.read();
			if (snapshot.status === "paused" || snapshot.terminal_generation !== undefined) {
				return snapshot;
			}
			if (snapshot.current_stage !== stage || snapshot.status !== "running") {
				throw new Error(`Experiment stage ${stage} changed unexpectedly while its driver was running.`);
			}
			this.acquireOwnership();
			this.protocol.completeStage(stage, this.requireFence());
			this.options.onStageTransition?.(stage, "complete");
		}
		this.acquireOwnership();
		return this.protocol.finish("success", "All native experiment stages completed.", this.requireFence());
	}

	async runPlanWorkUnits(plan: ExperimentPlan, drivers: ExperimentWorkUnitDriver[]): Promise<void> {
		if (drivers.length !== plan.work_units.length) {
			throw new Error("Experiment work-unit drivers must exactly cover the validated plan.");
		}
		const workUnits = new Map(plan.work_units.map((workUnit) => [workUnit.id, workUnit]));
		const byId = new Map<string, ExperimentWorkUnitDriver>();
		for (const driver of drivers) {
			if (!workUnits.has(driver.workUnit.id) || byId.has(driver.workUnit.id)) {
				throw new Error(`Experiment work-unit driver does not uniquely match plan unit ${driver.workUnit.id}.`);
			}
			if (canonicalDigest(driver.workUnit) !== canonicalDigest(workUnits.get(driver.workUnit.id)!)) {
				throw new Error(`Experiment work-unit driver has stale plan data for ${driver.workUnit.id}.`);
			}
			byId.set(driver.workUnit.id, driver);
		}
		this.initialize();
		const completed = new Set<string>();
		while (completed.size < plan.work_units.length) {
			const ready = plan.work_units.filter(
				(workUnit) =>
					!completed.has(workUnit.id) && (workUnit.needs ?? []).every((dependency) => completed.has(dependency)),
			);
			if (ready.length === 0) {
				throw new Error("Experiment work-unit dependency graph is cyclic or references an unknown unit.");
			}
			const execute = async (workUnit: ExperimentWorkUnit) => {
				if (this.read().terminal_generation !== undefined) throw new Error("Experiment is cancelled or terminal.");
				const driver = byId.get(workUnit.id);
				if (!driver) {
					throw new Error(`Experiment plan is missing a driver for work unit ${workUnit.id}.`);
				}
				await driver.run(this, workUnit);
				completed.add(workUnit.id);
			};
			if (plan.stage === "science") {
				const results = await Promise.allSettled(ready.map(execute));
				const failures = results.filter((result): result is PromiseRejectedResult => result.status === "rejected");
				if (failures.length)
					throw new AggregateError(
						failures.map((result) => result.reason),
						"Science conditions remain incomplete.",
					);
			} else {
				for (const workUnit of ready) await execute(workUnit);
			}
		}
	}

	async executeAssignment(spec: ExperimentAssignmentSpec): Promise<ExperimentAcceptedResult> {
		spec = {
			...spec,
			readScope: spec.readScope.map((scope) => canonicalScope(this.options.cwd, scope)),
			writeScope: spec.writeScope.map((scope) => canonicalScope(this.options.cwd, scope)),
		};
		if (assignmentKindForRole(spec.role) !== spec.kind) {
			throw new Error(`Role ${spec.role} cannot execute a ${spec.kind} assignment.`);
		}
		this.acquireOwnership();
		let snapshot = this.read();
		if (snapshot.current_stage !== spec.stage || snapshot.status !== "running") {
			throw new Error(`Experiment stage ${spec.stage} is not running.`);
		}
		const existingAssignment = snapshot.assignments[spec.assignmentId];
		if (existingAssignment) {
			this.assertStableAssignment(existingAssignment, spec);
		}
		if (existingAssignment?.status === "accepted") {
			const accepted = this.protocol.readAcceptedResult(spec.assignmentId);
			if (!accepted) {
				throw new Error(`Accepted assignment ${spec.assignmentId} is missing its durable result.`);
			}
			const resultError = validateExperimentResult(existingAssignment, accepted.submission.result);
			if (resultError) {
				throw new Error(`Accepted assignment ${spec.assignmentId} failed semantic validation: ${resultError}`);
			}
			return accepted;
		}
		const token = this.tokenFactory(spec.assignmentId, existingAssignment === undefined);
		if (!token) {
			throw new Error(`Assignment token factory returned an empty token for ${spec.assignmentId}.`);
		}
		if (existingAssignment && existingAssignment.status !== "active" && existingAssignment.status !== "submitted") {
			throw new Error(`Assignment ${spec.assignmentId} is ${existingAssignment.status}.`);
		}

		let identity = snapshot.children[spec.childId];
		if (!identity) {
			identity = this.childRuntime.createIdentity(this.createChildOptions(spec));
		} else {
			this.assertStableIdentity(identity, spec);
			if (
				identity.execution_attempt !== spec.executionAttempt ||
				identity.generation !== spec.generation ||
				identity.review_round !== spec.reviewRound
			) {
				identity = this.childRuntime.advanceIdentity({
					child_id: identity.child_id,
					execution_attempt: spec.executionAttempt,
					generation: spec.generation,
					review_round: spec.reviewRound,
				});
			}
		}
		snapshot = this.read();
		const assignment = this.protocol.createAssignment(
			{
				schema_version: "xlab.experiment_assignment.v1",
				assignment_id: spec.assignmentId,
				child: identity,
				kind: spec.kind,
				input_digest: spec.inputDigest,
				target_digest: spec.targetDigest,
				policy_digest: snapshot.policy_digest,
				read_scope: spec.readScope,
				write_scope: spec.writeScope,
				...(spec.resultValidation ? { result_validation: spec.resultValidation } : {}),
			},
			token,
			this.requireFence(),
		);

		let handle: ExperimentChildHandle | undefined;
		let promptError: unknown;
		try {
			handle = await this.childRuntime.openAssignment(assignment, token);
			const attempts = this.options.maxAssignmentPromptAttempts ?? 1;
			if (!Number.isInteger(attempts) || attempts < 1)
				throw new Error("Assignment attempts must be a positive integer.");
			for (let attempt = 0; attempt < attempts; attempt++) {
				const current = this.read();
				if (current.terminal_generation !== undefined || current.status === "paused")
					throw new Error("Assignment interrupted by run state.");
				promptError = undefined;
				const prompt = attempt
					? `${spec.prompt}\nThe previous turn did not produce an accepted submission. Resume this same work unit, reuse existing evidence, and finish with the correct submit tool. Do not repeat completed execution. Required validation context: ${JSON.stringify(spec.resultValidation ?? {})}`
					: spec.prompt;
				try {
					await this.withLeaseHeartbeat(() => handle!.session.prompt(prompt, { expandPromptTemplates: false }));
				} catch (error) {
					promptError = error;
				}
				const accepted = this.protocol.readAcceptedResult(assignment.assignment_id);
				if (accepted) return accepted;
				if (attempt + 1 < attempts) await new Promise<void>((done) => setTimeout(done, 1500 * (attempt + 1)));
			}
			if (promptError) {
				throw promptError;
			}
			throw new Error(`Child completed without submitting assignment ${assignment.assignment_id}.`);
		} finally {
			if (handle) {
				try {
					this.childRuntime.release(handle.identity.child_id);
				} catch {
					// Durable protocol state and the primary assignment outcome are authoritative.
				}
			}
		}
	}

	async runReviewMatrix(spec: ExperimentReviewSpec): Promise<ExperimentReviewMatrixResult> {
		const roles =
			spec.stage === "code" ? codeReviewRoles(spec.validationContext?.execution_profile) : SCIENCE_REVIEW_ROLES;
		const reports: ExperimentReviewReport[] = [];
		const assignmentIds: string[] = [];
		for (let index = 0; index < roles.length; index += 1) {
			const role = roles[index];
			const lineage = canonicalDigest({
				input_digest: spec.inputDigest,
				target_digest: spec.targetDigest,
			});
			const childId = `reviewer-${spec.stage}-${stablePart(spec.workUnit)}-${role}-r${spec.reviewRound}-${lineage}`;
			const assignmentId = `assignment-${childId}`;
			const accepted = await this.executeAssignment({
				assignmentId,
				childId,
				role: "reviewer",
				stage: spec.stage,
				workUnit: spec.workUnit,
				kind: "review",
				executionAttempt: 1,
				generation: 1,
				reviewRound: spec.reviewRound,
				inputDigest: spec.inputDigest,
				targetDigest: spec.targetDigest,
				readScope: spec.readScope,
				writeScope: [],
				resultValidation: {
					reviewer_role: role,
					...(spec.validationContext ? { review_context: spec.validationContext } : {}),
				},
				prompt: spec.prompt(role),
			});
			assignmentIds.push(accepted.accepted.assignment_id);
			const report = resultObject(accepted.submission.result, assignmentId) as unknown as ExperimentReviewReport;
			reports.push(report);
		}
		const validationError = validateReviewMatrix(spec.stage, reports, spec.validationContext);
		if (validationError) {
			throw new Error(validationError);
		}
		return {
			reports,
			assignmentIds,
			inputDigest: spec.inputDigest,
			targetDigest: spec.targetDigest,
			reviewRound: spec.reviewRound,
			blockingIssues: reports
				.filter((report) => report.blocking && report.verdict === "FAIL")
				.flatMap((report) => report.issues),
		};
	}

	async runReviewedWorker(spec: ExperimentReviewedWorkerSpec): Promise<ExperimentReviewedWorkerResult> {
		if (!Number.isInteger(spec.maxReviewRounds) || spec.maxReviewRounds <= 0) {
			throw new Error("Experiment reviewed worker requires a positive review-round budget.");
		}
		let workerSpec = spec.worker;
		let accepted = await this.executeAssignment(workerSpec);
		for (let reviewRound = 1; reviewRound <= spec.maxReviewRounds; reviewRound += 1) {
			const reviewSpec = spec.review(accepted, reviewRound);
			const review = await this.runReviewMatrix({
				...reviewSpec,
				reviewRound,
				targetDigest: accepted.accepted.result_digest,
			});
			if (review.blockingIssues.length === 0) {
				return { worker: accepted, review, reviewRound };
			}
			if (reviewRound === spec.maxReviewRounds) {
				throw new Error(`Experiment review repair budget exhausted after round ${reviewRound}.`);
			}
			workerSpec = spec.repair(accepted, review.blockingIssues, reviewRound);
			accepted = await this.repairWorker(accepted.accepted.assignment_id, workerSpec, review.blockingIssues);
		}
		throw new Error("Experiment reviewed worker terminated without a review result.");
	}

	async repairWorker(
		previousAssignmentId: string,
		spec: ExperimentAssignmentSpec,
		issues: ExperimentReviewReport["issues"],
	): Promise<ExperimentAcceptedResult> {
		if (spec.role !== "worker" || spec.kind !== "worker_result") {
			throw new Error("Experiment repair must target a worker assignment.");
		}
		if (issues.length === 0) {
			throw new Error("Experiment repair requires structured review issues.");
		}
		this.acquireOwnership();
		const previous = this.protocol.readAcceptedResult(previousAssignmentId);
		if (!previous) {
			throw new Error(`Experiment repair requires accepted prior assignment ${previousAssignmentId}.`);
		}
		this.assertRepairProgress(spec, previous, previous.submission.review_round + 1);
		const previousAssignment = this.read().assignments[previousAssignmentId];
		if (
			!previousAssignment ||
			(previousAssignment.status !== "accepted" && previousAssignment.status !== "superseded")
		) {
			throw new Error(`Experiment repair requires accepted prior assignment ${previousAssignmentId}.`);
		}
		const repairPrompt = [
			spec.prompt,
			"Repair the exact prior worker result using these structured reviewer issues:",
			JSON.stringify(canonicalJson(issues), null, 2),
		].join("\n\n");
		return this.executeAssignment({ ...spec, prompt: repairPrompt });
	}

	pause(reason: string): ExperimentProtocolSnapshot {
		this.acquireOwnership();
		return this.protocol.pause(reason, this.requireFence());
	}

	resume(): ExperimentProtocolSnapshot {
		this.acquireOwnership();
		return this.protocol.resume(this.requireFence());
	}

	async cancel(reason: string): Promise<ExperimentProtocolSnapshot> {
		const cancelled = this.protocol.cancel(reason);
		try {
			await this.childRuntime.abortAll();
		} catch {
			// Cancellation is durable before abort fan-out; cleanup failure cannot undo it.
		}
		return cancelled;
	}

	finish(
		status: "success" | "incomplete" | "failed",
		reason: string,
		publicationRecovery = false,
	): ExperimentProtocolSnapshot {
		this.acquireOwnership();
		return this.protocol.finish(status, reason, this.requireFence(), publicationRecovery);
	}

	private async withLeaseHeartbeat<T>(operation: () => Promise<T>): Promise<T> {
		this.assertOwnershipRetained();
		const intervalMs = Math.max(1, Math.floor(this.leaseDurationMs / 3));
		const timer = setInterval(() => {
			try {
				this.acquireOwnership();
			} catch (error) {
				this.ownershipLost = error instanceof Error ? error : new Error(String(error));
				void this.childRuntime.abortAll().catch(() => undefined);
			}
		}, intervalMs);
		try {
			const result = await operation();
			this.assertOwnershipRetained();
			this.acquireOwnership();
			return result;
		} finally {
			clearInterval(timer);
		}
	}

	private assertOwnershipRetained(): void {
		if (this.ownershipLost) {
			throw new Error(`Experiment coordinator lost ownership: ${this.ownershipLost.message}`);
		}
	}

	private createChildOptions(spec: ExperimentAssignmentSpec): CreateExperimentChildOptions {
		return {
			child_id: spec.childId,
			parent_run_id: this.options.runId,
			role: spec.role,
			stage: spec.stage,
			...(spec.workUnit === undefined ? {} : { work_unit: spec.workUnit }),
			execution_attempt: spec.executionAttempt,
			generation: spec.generation,
			review_round: spec.reviewRound,
			cwd: this.options.cwd,
		};
	}

	private assertRepairProgress(
		spec: ExperimentAssignmentSpec,
		previous: ExperimentAcceptedResult,
		reviewRound: number,
	): void {
		const prior = previous.submission;
		if (
			spec.role !== "worker" ||
			spec.kind !== "worker_result" ||
			spec.childId !== prior.child_id ||
			spec.stage !== prior.stage ||
			spec.workUnit !== prior.work_unit ||
			spec.targetDigest !== prior.target_digest
		) {
			throw new Error("Experiment repair must resume the exact prior worker identity.");
		}
		if (
			spec.assignmentId === prior.assignment_id ||
			spec.executionAttempt <= prior.execution_attempt ||
			spec.generation <= prior.generation ||
			spec.reviewRound !== reviewRound ||
			spec.reviewRound <= prior.review_round
		) {
			throw new Error("Experiment repair must advance assignment, attempt, generation, and review round.");
		}
	}

	private assertStableAssignment(assignment: ExperimentAssignment, spec: ExperimentAssignmentSpec): void {
		if (
			assignment.assignment_id !== spec.assignmentId ||
			assignment.kind !== spec.kind ||
			assignment.input_digest !== spec.inputDigest ||
			assignment.target_digest !== spec.targetDigest ||
			assignment.child.child_id !== spec.childId ||
			assignment.child.execution_attempt !== spec.executionAttempt ||
			assignment.child.generation !== spec.generation ||
			assignment.child.review_round !== spec.reviewRound ||
			canonicalDigest(assignment.read_scope) !== canonicalDigest(spec.readScope) ||
			canonicalDigest(assignment.write_scope) !== canonicalDigest(spec.writeScope) ||
			canonicalDigest(assignment.result_validation ?? null) !== canonicalDigest(spec.resultValidation ?? null)
		) {
			throw new Error(`Assignment ${spec.assignmentId} does not match its immutable coordinator specification.`);
		}
		this.assertStableIdentity(assignment.child, spec);
	}

	private assertStableIdentity(identity: ExperimentChildIdentity, spec: ExperimentAssignmentSpec): void {
		if (
			identity.parent_run_id !== this.options.runId ||
			identity.role !== spec.role ||
			identity.stage !== spec.stage ||
			identity.work_unit !== spec.workUnit
		) {
			throw new Error(`Child ${identity.child_id} does not match its stable coordinator identity.`);
		}
		if (
			spec.executionAttempt < identity.execution_attempt ||
			spec.generation < identity.generation ||
			spec.reviewRound < identity.review_round
		) {
			throw new Error(`Child identity cannot move backwards: ${identity.child_id}.`);
		}
	}

	private requireFence(): ExperimentOwnerFence {
		if (!this.fence) {
			throw new Error("Experiment coordinator does not own the protocol.");
		}
		return this.fence;
	}
}

export function acceptedPlan(result: ExperimentAcceptedResult): ExperimentPlan {
	return resultObject(result.submission.result, result.accepted.assignment_id) as unknown as ExperimentPlan;
}

export type ExperimentCoordinatorSession = Pick<AgentSession, "prompt" | "abort" | "dispose">;
