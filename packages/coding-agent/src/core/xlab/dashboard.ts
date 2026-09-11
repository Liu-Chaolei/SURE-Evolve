import type {
	XlabDisplayArtifact,
	XlabDisplayDiagnostic,
	XlabDisplayState,
	XlabRunRecord,
	XlabSkillPackage,
	XlabWorkflowStageStatus,
	XlabWorkflowState,
} from "./types.ts";

export interface XlabDashboardOptions {
	run: XlabRunRecord;
	state?: XlabDisplayState;
	workflow?: XlabWorkflowState;
	skillPackage?: XlabSkillPackage;
	maxLines?: number;
}

const DEFAULT_MAX_LINES = 10;
const MAX_LINE_LENGTH = 120;
const SECRET_NAME_PATTERN = /(api[_-]?key|token|secret|password|credential)/i;

function truncate(value: string, maxLength = MAX_LINE_LENGTH): string {
	const normalized = value.replace(/[\r\n\t]+/g, " ").trim();
	if (normalized.length <= maxLength) {
		return normalized;
	}
	return `${normalized.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`;
}

function redactSecrets(value: string): string {
	let redacted = value;
	for (const [name, secret] of Object.entries(process.env)) {
		const normalized = secret?.trim();
		if (!normalized || normalized.length < 4 || !SECRET_NAME_PATTERN.test(name)) {
			continue;
		}
		redacted = redacted.split(normalized).join("••••");
	}
	return redacted;
}

function safeLine(value: string): string {
	return truncate(redactSecrets(value));
}

function formatPercent(value: number | undefined): string | undefined {
	if (value === undefined || !Number.isFinite(value)) {
		return undefined;
	}
	const normalized = value <= 1 ? value * 100 : value;
	return `${Math.max(0, Math.min(100, Math.round(normalized)))}%`;
}

function phaseStatusIcon(status: string | undefined): string {
	switch (status) {
		case "success":
			return "✓";
		case "running":
			return "▶";
		case "failed":
			return "✗";
		case "blocked":
			return "!";
		case "incomplete":
			return "…";
		case "skipped":
			return "↷";
		case "cancelled":
			return "⊘";
		default:
			return "·";
	}
}

function stageStatusIcon(status: XlabWorkflowStageStatus): string {
	return phaseStatusIcon(status);
}

function formatPhaseLine(state: XlabDisplayState | undefined): string | undefined {
	const phase = state?.phase;
	const label = phase?.label ?? phase?.id;
	const phaseProgress = formatPercent(phase?.progress);
	const runProgress = formatPercent(state?.progress);
	const progress = phaseProgress ?? runProgress;
	const status = phase?.status ? ` ${phaseStatusIcon(phase.status)}` : "";
	const message = state?.message?.trim();
	const parts = [label ? `${label}${status}` : undefined, progress, message].filter((entry): entry is string =>
		Boolean(entry),
	);
	return parts.length > 0 ? safeLine(`Phase: ${parts.join(" · ")}`) : undefined;
}

function formatWorkflowLine(workflow: XlabWorkflowState | undefined): string | undefined {
	if (!workflow || workflow.stages.length === 0) {
		return undefined;
	}
	const stages = workflow.stages.slice(0, 6).map((stage) => `${stage.id} ${stageStatusIcon(stage.status)}`);
	const suffix = workflow.stages.length > stages.length ? ` +${workflow.stages.length - stages.length}` : "";
	return safeLine(`Workflow: ${stages.join("  ")}${suffix}`);
}

function counterNames(state: XlabDisplayState, skillPackage: XlabSkillPackage | undefined): string[] {
	const counters = state.counters ?? {};
	const hinted = skillPackage?.manifest.ui?.primaryCounters?.filter((name) => Object.hasOwn(counters, name)) ?? [];
	const remaining = Object.keys(counters)
		.filter((name) => !hinted.includes(name))
		.sort((left, right) => left.localeCompare(right));
	return [...hinted, ...remaining].slice(0, 5);
}

function formatCountersLine(
	state: XlabDisplayState | undefined,
	skillPackage: XlabSkillPackage | undefined,
): string | undefined {
	if (!state?.counters) {
		return undefined;
	}
	const names = counterNames(state, skillPackage);
	if (names.length === 0) {
		return undefined;
	}
	const counters = names.map((name) => `${name}=${state.counters?.[name] ?? 0}`);
	const hidden = Object.keys(state.counters).length - names.length;
	return safeLine(`Counters: ${counters.join("  ")}${hidden > 0 ? `  +${hidden}` : ""}`);
}

function severityRank(severity: XlabDisplayDiagnostic["severity"]): number {
	switch (severity) {
		case "error":
			return 3;
		case "warning":
			return 2;
		case "info":
			return 1;
		default:
			return 0;
	}
}

function selectDiagnostic(diagnostics: XlabDisplayDiagnostic[]): XlabDisplayDiagnostic | undefined {
	return diagnostics
		.map((diagnostic, index) => ({ diagnostic, index }))
		.sort(
			(left, right) =>
				severityRank(right.diagnostic.severity) - severityRank(left.diagnostic.severity) ||
				right.index - left.index,
		)[0]?.diagnostic;
}

function formatDiagnosticLine(state: XlabDisplayState | undefined): string | undefined {
	const diagnostic = state?.diagnostics?.length ? selectDiagnostic(state.diagnostics) : undefined;
	if (!diagnostic) {
		return undefined;
	}
	const severity = diagnostic.severity ?? "info";
	const repair = diagnostic.repair ? ` Repair: ${diagnostic.repair}` : "";
	return safeLine(`Blocker: ${severity}: ${diagnostic.message}${repair}`);
}

function artifactLabel(artifact: XlabDisplayArtifact): string {
	return artifact.name ?? artifact.path ?? artifact.type ?? "artifact";
}

function formatArtifactsLine(
	state: XlabDisplayState | undefined,
	skillPackage: XlabSkillPackage | undefined,
): string | undefined {
	const artifacts = state?.artifacts;
	if (!artifacts || artifacts.length === 0) {
		return undefined;
	}
	const preferredTypes = skillPackage?.manifest.ui?.artifactTypes ?? [];
	const sorted = [...artifacts].sort((left, right) => {
		const leftIndex = left.type ? preferredTypes.indexOf(left.type) : -1;
		const rightIndex = right.type ? preferredTypes.indexOf(right.type) : -1;
		if (leftIndex !== rightIndex) {
			if (leftIndex === -1) return 1;
			if (rightIndex === -1) return -1;
			return leftIndex - rightIndex;
		}
		return artifactLabel(left).localeCompare(artifactLabel(right));
	});
	const counts = artifacts.reduce<Record<string, number>>((accumulator, artifact) => {
		const status = artifact.status ?? "draft";
		accumulator[status] = (accumulator[status] ?? 0) + 1;
		return accumulator;
	}, {});
	const countText = Object.entries(counts)
		.sort(([left], [right]) => left.localeCompare(right))
		.map(([status, count]) => `${status}=${count}`)
		.join(" ");
	const labels = sorted.slice(0, 2).map(artifactLabel).join(", ");
	return safeLine(`Artifacts: ${countText}${labels ? ` · ${labels}` : ""}`);
}

function formatCheckpointLine(state: XlabDisplayState | undefined): string | undefined {
	const checkpoint = state?.checkpoint;
	if (!checkpoint?.resumable) {
		return undefined;
	}
	const label = checkpoint.label ?? checkpoint.id ?? "checkpoint";
	const hint = checkpoint.resume_hint ? ` · ${checkpoint.resume_hint}` : "";
	return safeLine(`Checkpoint: ${label}${hint}`);
}

function formatNextActionsLine(state: XlabDisplayState | undefined): string | undefined {
	const actions = state?.next_actions?.filter((action) => action.trim() !== "") ?? [];
	if (actions.length === 0) {
		return undefined;
	}
	const visible = actions.slice(0, 2).join("; ");
	const suffix = actions.length > 2 ? `; +${actions.length - 2}` : "";
	return safeLine(`Next: ${visible}${suffix}`);
}

export function formatXlabDashboard(options: XlabDashboardOptions): string[] {
	const { run, state, workflow, skillPackage } = options;
	const maxLines = Math.max(1, options.maxLines ?? DEFAULT_MAX_LINES);
	const workspace = run.workspaceSlug ? ` · workspace ${run.workspaceSlug}` : "";
	const displayCommand = run.invokedCommand ?? run.command;
	const title = safeLine(
		`XLab /${displayCommand} · ${run.runId} · ${phaseStatusIcon(run.status)} ${run.status}${workspace}`,
	);
	const lines = [
		title,
		formatPhaseLine(state),
		formatWorkflowLine(workflow),
		formatCountersLine(state, skillPackage),
		formatDiagnosticLine(state),
		formatArtifactsLine(state, skillPackage),
		formatCheckpointLine(state),
		formatNextActionsLine(state),
		safeLine("Setup: /xlab check-xlab-setup"),
	].filter((line): line is string => Boolean(line));
	return lines.slice(0, maxLines);
}
