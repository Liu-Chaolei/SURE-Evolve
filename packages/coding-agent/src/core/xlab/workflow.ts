import type {
	XlabSkillManifest,
	XlabWorkflowStageDeclaration,
	XlabWorkflowStageState,
	XlabWorkflowState,
} from "./types.ts";

function nowIso(): string {
	return new Date().toISOString();
}

export function createWorkflowState(runId: string, manifest: XlabSkillManifest): XlabWorkflowState | undefined {
	if (!manifest.workflow) {
		return undefined;
	}
	return {
		schemaVersion: "1",
		runId,
		stages: manifest.workflow.stages.map((stage) => ({
			id: stage.id,
			skill: stage.skill,
			needs: stage.needs ?? [],
			optional: stage.optional ?? false,
			maxAttempts: stage.maxAttempts ?? 1,
			attempts: 0,
			status: "pending",
		})),
		updatedAt: nowIso(),
	};
}

export function validateWorkflowStages(stages: XlabWorkflowStageDeclaration[]): string | undefined {
	const ids = new Set<string>();
	for (const stage of stages) {
		if (ids.has(stage.id)) {
			return `Duplicate workflow stage id: ${stage.id}`;
		}
		ids.add(stage.id);
	}
	for (const stage of stages) {
		for (const dependency of stage.needs ?? []) {
			if (!ids.has(dependency)) {
				return `Workflow stage "${stage.id}" needs unknown stage "${dependency}".`;
			}
			if (dependency === stage.id) {
				return `Workflow stage "${stage.id}" cannot depend on itself.`;
			}
		}
	}

	const visiting = new Set<string>();
	const visited = new Set<string>();
	const byId = new Map(stages.map((stage) => [stage.id, stage]));
	const visit = (id: string): boolean => {
		if (visiting.has(id)) {
			return false;
		}
		if (visited.has(id)) {
			return true;
		}
		visiting.add(id);
		for (const dependency of byId.get(id)?.needs ?? []) {
			if (!visit(dependency)) {
				return false;
			}
		}
		visiting.delete(id);
		visited.add(id);
		return true;
	};
	for (const stage of stages) {
		if (!visit(stage.id)) {
			return "Workflow stages must form an acyclic graph.";
		}
	}
	return undefined;
}

export function transitionWorkflowStage(
	state: XlabWorkflowState,
	stageId: string,
	action: "start" | "complete" | "fail" | "block" | "skip",
	data?: { checkpoint?: unknown; artifacts?: unknown; error?: string },
): { ok: boolean; state: XlabWorkflowState; repair?: string; stage?: XlabWorkflowStageState } {
	const stage = state.stages.find((candidate) => candidate.id === stageId);
	if (!stage) {
		return { ok: false, state, repair: `Unknown workflow stage: ${stageId}` };
	}
	const nextStage: XlabWorkflowStageState = { ...stage };
	if (action === "start") {
		const missing = stage.needs.filter(
			(dependency) => state.stages.find((candidate) => candidate.id === dependency)?.status !== "success",
		);
		if (missing.length > 0) {
			return {
				ok: false,
				state,
				repair: `Complete prerequisite stages before "${stageId}": ${missing.join(", ")}`,
			};
		}
		if (stage.status === "success" || stage.status === "skipped") {
			return { ok: false, state, repair: `Workflow stage "${stageId}" is already terminal.` };
		}
		if (stage.attempts >= stage.maxAttempts) {
			return { ok: false, state, repair: `Workflow stage "${stageId}" exhausted ${stage.maxAttempts} attempts.` };
		}
		nextStage.status = "running";
		nextStage.attempts += 1;
		nextStage.startedAt = nowIso();
		nextStage.error = undefined;
	} else {
		if (stage.status !== "running" && action !== "skip") {
			return { ok: false, state, repair: `Start workflow stage "${stageId}" before marking it ${action}.` };
		}
		nextStage.status =
			action === "complete" ? "success" : action === "fail" ? "failed" : action === "block" ? "blocked" : "skipped";
		nextStage.finishedAt = nowIso();
		nextStage.checkpoint = data?.checkpoint;
		nextStage.artifacts = data?.artifacts;
		nextStage.error = data?.error;
	}
	const next: XlabWorkflowState = {
		...state,
		stages: state.stages.map((candidate) => (candidate.id === stageId ? nextStage : candidate)),
		updatedAt: nowIso(),
	};
	return { ok: true, state: next, stage: nextStage };
}
