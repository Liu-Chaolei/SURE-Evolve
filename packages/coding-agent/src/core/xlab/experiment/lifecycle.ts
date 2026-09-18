import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { XlabRunStagingContext } from "../run-manager.ts";
import { XlabRunManager } from "../run-manager.ts";
import type { XlabArtifactReference, XlabRunRecord, XlabRunStatus } from "../types.ts";
import { ExperimentCoordinator, type ExperimentCoordinatorOptions, type ExperimentStageDriver } from "./coordinator.ts";
import type { XlabIdeaBinding } from "./idea-binding.ts";
import {
	atomicWriteFile,
	CODE_REVIEW_ROLES,
	canonicalDigest,
	canonicalJsonBytes,
	codeReviewRoles,
	EXPERIMENT_STAGES,
	type ExperimentExecutionProfile,
	type ExperimentProtocolSnapshot,
	ExperimentProtocolStore,
	SCIENCE_REVIEW_ROLES,
	sha256,
} from "./protocol.ts";

export const NATIVE_EXPERIMENT_POLICY = {
	schema_version: "xlab.experiment_policy.v1",
	stages: EXPERIMENT_STAGES,
	code_review_roles: CODE_REVIEW_ROLES,
	science_review_roles: SCIENCE_REVIEW_ROLES,
} as const;

export const NATIVE_EXPERIMENT_POLICY_DIGEST = canonicalDigest(NATIVE_EXPERIMENT_POLICY);

export function experimentPolicy(profile?: ExperimentExecutionProfile) {
	return profile
		? { ...NATIVE_EXPERIMENT_POLICY, execution_profile: profile, code_review_roles: codeReviewRoles(profile) }
		: NATIVE_EXPERIMENT_POLICY;
}

export interface XlabExperimentLifecycleOptions {
	cwd: string;
	agentDir: string;
	runId: string;
	ownerId: string;
	runManager?: XlabRunManager;
	clock?: () => string;
	leaseDurationMs?: number;
	coordinatorOptions?: Pick<
		ExperimentCoordinatorOptions,
		"tokenFactory" | "childRuntime" | "childRuntimeOptions" | "maxAssignmentPromptAttempts"
	>;
	finalizeSuccess?: (record: XlabRunRecord) => { artifactRefs: XlabArtifactReference[]; manifestPath: string };
}

export interface ReconstructedXlabExperiment {
	record: XlabRunRecord;
	protocol: ExperimentProtocolStore;
	snapshot: ExperimentProtocolSnapshot;
	coordinator: ExperimentCoordinator;
}

function policyPath(runDir: string): string {
	return join(runDir, "experiment", "policy.json");
}

export function initializeXlabExperimentStaging(
	context: XlabRunStagingContext,
	ideaBinding: XlabIdeaBinding,
	clock?: () => string,
	profile?: ExperimentExecutionProfile,
): ExperimentProtocolSnapshot {
	const policy = experimentPolicy(profile);
	atomicWriteFile(policyPath(context.stagingDir), canonicalJsonBytes(policy));
	return new ExperimentProtocolStore(context.stagingDir, clock).initialize({
		runId: context.record.runId,
		ideaDigest: ideaBinding.metadata.canonical_digest,
		policyDigest: canonicalDigest(policy),
	});
}

export function createXlabExperimentStagingMaterializer(
	ideaBinding: XlabIdeaBinding,
	clock?: () => string,
	profile?: ExperimentExecutionProfile,
): (context: XlabRunStagingContext) => void {
	return (context) => {
		initializeXlabExperimentStaging(context, ideaBinding, clock, profile);
	};
}

export function projectExperimentProtocolToRun(
	record: XlabRunRecord,
	snapshot: ExperimentProtocolSnapshot,
): XlabRunRecord {
	const status = snapshot.status as XlabRunStatus;
	const terminal = status === "success" || status === "failed" || status === "incomplete" || status === "cancelled";
	return {
		...record,
		status,
		updatedAt: snapshot.updated_at,
		finishedAt: terminal ? snapshot.updated_at : undefined,
		publicationRecoveryReason: snapshot.publication_recovery_reason,
	};
}

export class XlabExperimentLifecycle {
	private readonly options: XlabExperimentLifecycleOptions;
	private readonly runManager: XlabRunManager;

	constructor(options: XlabExperimentLifecycleOptions) {
		this.options = options;
		this.runManager = options.runManager ?? new XlabRunManager(options.cwd);
	}

	reconstruct(): ReconstructedXlabExperiment {
		const record = this.runManager.readRun(this.options.runId);
		if (!record) {
			throw new Error(`Cannot reconstruct unknown XLab run ${this.options.runId}.`);
		}
		if (record.runId !== this.options.runId) {
			throw new Error("XLab run record does not match its immutable run ID.");
		}
		const policy = policyPath(record.runDir);
		if (!existsSync(policy)) {
			throw new Error(`Native experiment policy is missing for run ${record.runId}.`);
		}
		let parsedPolicy: unknown;
		try {
			parsedPolicy = JSON.parse(readFileSync(policy, "utf-8"));
		} catch (error) {
			throw new Error(
				`Native experiment policy is invalid: ${error instanceof Error ? error.message : String(error)}`,
			);
		}
		const profile = (parsedPolicy as { execution_profile?: ExperimentExecutionProfile }).execution_profile;
		if (
			profile &&
			(!["standard", "direct_formal"].includes(profile.mode) || !["research", "baseline"].includes(profile.kind))
		)
			throw new Error("Unknown experiment execution profile.");
		const policyDigest = canonicalDigest(experimentPolicy(profile));
		if (canonicalDigest(parsedPolicy) !== policyDigest) {
			throw new Error("Native experiment policy digest does not match the canonical policy.");
		}
		const ideaPath = join(record.runDir, "inputs", "idea.json");
		if (!existsSync(ideaPath)) {
			throw new Error(`Canonical experiment Idea is missing for run ${record.runId}.`);
		}
		const ideaDigest = sha256(readFileSync(ideaPath));
		const protocol = new ExperimentProtocolStore(record.runDir, this.options.clock);
		const snapshot = protocol.read();
		if (
			snapshot.run_id !== record.runId ||
			snapshot.idea_digest !== ideaDigest ||
			snapshot.policy_digest !== policyDigest
		) {
			throw new Error("Native experiment protocol does not match immutable run inputs.");
		}
		const coordinator = new ExperimentCoordinator({
			runId: record.runId,
			runDir: record.runDir,
			cwd: record.cwd,
			agentDir: this.options.agentDir,
			ideaDigest,
			policyDigest,
			ownerId: this.options.ownerId,
			protocol,
			clock: this.options.clock,
			leaseDurationMs: this.options.leaseDurationMs,
			onStageTransition: (stage, action) => {
				const transition = this.runManager.transitionStage(record, stage, action);
				if (!transition.ok) {
					throw new Error(transition.repair ?? `Failed to ${action} workflow stage ${stage}.`);
				}
			},
			...this.options.coordinatorOptions,
		});
		return { record, protocol, snapshot, coordinator };
	}

	async start(drivers: ExperimentStageDriver[]): Promise<XlabRunRecord> {
		const state = this.reconstruct();
		this.reconcileOneWorkflowTransition(state.record, state.snapshot);
		const snapshot = await state.coordinator.run(drivers);
		return this.publishProjection(state.record, snapshot, "native_experiment_status");
	}

	async resume(drivers: ExperimentStageDriver[]): Promise<XlabRunRecord> {
		const state = this.reconstruct();
		if (state.snapshot.status === "incomplete" && Boolean(state.snapshot.publication_recovery_reason)) {
			return this.retryPublication(state);
		}
		const ownerIsLive =
			state.snapshot.owner?.lease_expires_at !== undefined &&
			state.snapshot.owner.lease_expires_at > (this.options.clock?.() ?? new Date().toISOString());
		if (state.snapshot.status !== "paused" && !(state.snapshot.status === "running" && !ownerIsLive)) {
			throw new Error(
				ownerIsLive
					? `Native experiment run ${state.record.runId} has a live owner.`
					: `Native experiment run ${state.record.runId} is not resumable.`,
			);
		}
		this.reconcileOneWorkflowTransition(state.record, state.snapshot);
		if (state.snapshot.status === "paused") {
			state.coordinator.resume();
		}
		const snapshot = await state.coordinator.run(drivers);
		return this.publishProjection(state.record, snapshot, "native_experiment_resumed");
	}

	async cancel(reason: string): Promise<XlabRunRecord> {
		const state = this.reconstruct();
		const snapshot = await state.coordinator.cancel(reason);
		return this.publishProjection(state.record, snapshot, "native_experiment_cancelled");
	}

	status(): XlabRunRecord {
		const state = this.reconstruct();
		return projectExperimentProtocolToRun(state.record, state.snapshot);
	}

	private retryPublication(state: ReconstructedXlabExperiment): XlabRunRecord {
		if (!this.options.finalizeSuccess) {
			throw new Error(`Native experiment run ${state.record.runId} has no publication indexer.`);
		}
		try {
			const { artifactRefs, manifestPath } = this.options.finalizeSuccess(state.record);
			const snapshot = state.coordinator.finish("success", "Native experiment publication recovered.", true);
			return this.runManager.updateRun(
				state.record,
				{
					status: "success",
					finishedAt: snapshot.updated_at,
					artifactRefs,
					manifestPath,
					lastRepair: undefined,
					publicationRecoveryReason: undefined,
				},
				"native_experiment_publication_recovered",
				{ status: "success" },
			);
		} catch (error) {
			const reason = `Native experiment artifact indexing incomplete: ${error instanceof Error ? error.message : String(error)}`;
			return this.runManager.updateRun(
				state.record,
				{
					status: "incomplete",
					lastRepair: reason,
					publicationRecoveryReason: state.snapshot.publication_recovery_reason,
				},
				"native_experiment_publication_retry_failed",
				{ status: "incomplete" },
			);
		}
	}

	settleFailure(reason: string): XlabRunRecord {
		const state = this.reconstruct();
		let snapshot = state.snapshot;
		if (snapshot.terminal_generation === undefined) {
			snapshot = state.coordinator.finish("failed", reason);
		}
		return this.publishProjection(state.record, snapshot, "native_experiment_failed", { reason });
	}

	private reconcileOneWorkflowTransition(record: XlabRunRecord, snapshot: ExperimentProtocolSnapshot): void {
		const workflow = this.runManager.readWorkflow(record);
		if (!workflow) return;
		const expected = EXPERIMENT_STAGES.map((stage) =>
			snapshot.completed_stages.includes(stage)
				? "success"
				: snapshot.current_stage === stage
					? "running"
					: "pending",
		);
		const differences = workflow.stages.flatMap((stage, index) =>
			stage.status === expected[index] ? [] : [{ stage, expected: expected[index] }],
		);
		if (differences.length === 0) return;
		if (differences.length !== 1) {
			throw new Error("Native experiment workflow differs from validated protocol by more than one transition.");
		}
		const difference = differences[0]!;
		const action =
			difference.expected === "running" && difference.stage.status === "pending"
				? "start"
				: difference.expected === "success" && difference.stage.status === "running"
					? "complete"
					: undefined;
		if (!action) {
			throw new Error(`Native experiment workflow cannot reconcile stage ${difference.stage.id}.`);
		}
		const result = this.runManager.transitionStage(record, difference.stage.id, action);
		if (!result.ok) throw new Error(result.repair ?? "Native experiment workflow reconciliation failed.");
	}

	private publishProjection(
		record: XlabRunRecord,
		snapshot: ExperimentProtocolSnapshot,
		eventType: string,
		data?: unknown,
	): XlabRunRecord {
		const projected = projectExperimentProtocolToRun(record, snapshot);
		let artifactRefs: XlabArtifactReference[] | undefined;
		let manifestPath: string | undefined;
		let status = projected.status;
		let lastRepair: string | undefined;
		let publicationRecoveryReason = projected.publicationRecoveryReason;
		if (status === "success" && this.options.finalizeSuccess) {
			try {
				({ artifactRefs, manifestPath } = this.options.finalizeSuccess(record));
			} catch (error) {
				lastRepair = `Native experiment artifact indexing incomplete: ${error instanceof Error ? error.message : String(error)}`;
				status = "incomplete";
				publicationRecoveryReason = lastRepair;
				new ExperimentProtocolStore(record.runDir, this.options.clock).finish(
					"incomplete",
					lastRepair,
					undefined,
					true,
				);
			}
		}
		return this.runManager.updateRun(
			record,
			{
				status,
				finishedAt: projected.finishedAt,
				artifactRefs,
				manifestPath,
				lastRepair,
				publicationRecoveryReason,
			},
			eventType,
			data ?? { status },
		);
	}
}
