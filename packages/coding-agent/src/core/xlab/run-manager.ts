import { randomUUID } from "node:crypto";
import {
	appendFileSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readdirSync,
	readFileSync,
	renameSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import lockfile from "proper-lockfile";
import type { XlabIdeaBinding } from "./experiment/idea-binding.ts";
import { xlabProductCommandForSkill } from "./product-commands.ts";
import { mergeXlabDisplayState } from "./state.ts";
import type { XlabDisplayState, XlabRunRecord, XlabRunStatus, XlabSkillPackage, XlabWorkflowState } from "./types.ts";
import { createWorkflowState, transitionWorkflowStage } from "./workflow.ts";

const XLAB_RUNS_DIR = ".xlab/runs";

function nowIso(): string {
	return new Date().toISOString();
}

function safeTimestamp(): string {
	return nowIso().replace(/[-:]/g, "").replace(/\..+$/, "").replace("T", "-");
}

function writeJson(path: string, value: unknown): void {
	const temporary = `${path}.${process.pid}.${randomUUID()}.tmp`;
	writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
	renameSync(temporary, path);
}

const TERMINAL_RUN_STATUSES = new Set<XlabRunStatus>(["success", "failed", "incomplete", "cancelled"]);

function replaceIdeaReference(args: string, reference: string): string {
	return args.replace(
		/(^|\s)--idea(?:=|\s+)(?:'[^']*'|"[^"]*"|[^\s]+)/,
		(_match, prefix: string) => `${prefix}--idea ${reference}`,
	);
}

function isPathInside(baseDir: string, candidate: string): boolean {
	const rel = relative(baseDir, candidate);
	return rel === "" || (!rel.startsWith("..") && !rel.startsWith("/") && rel !== "..");
}

export interface XlabRunEvent {
	type: string;
	timestamp: string;
	data?: unknown;
}

export interface XlabRunStagingContext {
	stagingDir: string;
	runDir: string;
	record: XlabRunRecord;
}

export class XlabRunManager {
	private cwd: string;

	constructor(cwd: string) {
		this.cwd = cwd;
	}

	get runsRoot(): string {
		return join(this.cwd, XLAB_RUNS_DIR);
	}

	createRun(
		skillPackage: XlabSkillPackage,
		args: string,
		options?: {
			workspaceSlug?: string;
			invokedCommand?: string;
			ideaBinding?: XlabIdeaBinding;
			materializeStaging?: (context: XlabRunStagingContext) => void;
		},
	): XlabRunRecord {
		const runId = `${safeTimestamp()}-${randomUUID().slice(0, 8)}`;
		const runDir = join(this.runsRoot, runId);
		mkdirSync(this.runsRoot, { recursive: true });
		const stagingDir = mkdtempSync(join(dirname(runDir), `.staging-${runId}-`));
		const runArgs = options?.ideaBinding ? replaceIdeaReference(args, `.xlab/runs/${runId}/inputs/idea.json`) : args;
		const record: XlabRunRecord = {
			runId,
			skillName: skillPackage.manifest.name,
			skillVersion: skillPackage.manifest.version,
			command: xlabProductCommandForSkill(skillPackage.manifest.name)?.task ?? skillPackage.manifest.name,
			invokedCommand: options?.invokedCommand,
			status: "pending",
			cwd: this.cwd,
			packageDir: skillPackage.packageDir,
			runDir,
			args: runArgs,
			startedAt: nowIso(),
			updatedAt: nowIso(),
			workspaceSlug: options?.workspaceSlug,
		};
		try {
			mkdirSync(join(stagingDir, "logs"), { recursive: true });
			mkdirSync(join(stagingDir, "artifacts"), { recursive: true });
			if (options?.ideaBinding) {
				mkdirSync(join(stagingDir, "inputs"), { recursive: true });
				writeFileSync(join(stagingDir, "inputs", "idea.json"), options.ideaBinding.canonicalBytes);
				writeJson(join(stagingDir, "inputs", "idea-binding.json"), options.ideaBinding.metadata);
			}
			writeJson(join(stagingDir, "run.json"), record);
			const workflow = createWorkflowState(runId, skillPackage.manifest);
			if (workflow) {
				writeJson(join(stagingDir, "workflow.json"), workflow);
			}
			const event: XlabRunEvent = { type: "created", timestamp: nowIso(), data: record };
			writeFileSync(join(stagingDir, "events.jsonl"), `${JSON.stringify(event)}\n`, "utf-8");
			options?.materializeStaging?.({ stagingDir, runDir, record });
			renameSync(stagingDir, runDir);
			return record;
		} finally {
			rmSync(stagingDir, { recursive: true, force: true });
		}
	}

	readRun(runId: string): XlabRunRecord | undefined {
		const runPath = join(this.runsRoot, runId, "run.json");
		if (!existsSync(runPath)) {
			return undefined;
		}
		return JSON.parse(readFileSync(runPath, "utf-8")) as XlabRunRecord;
	}

	listRuns(): XlabRunRecord[] {
		if (!existsSync(this.runsRoot)) {
			return [];
		}
		return readdirSync(this.runsRoot, { withFileTypes: true })
			.filter((entry) => entry.isDirectory())
			.flatMap((entry) => {
				const record = this.readRun(entry.name);
				return record ? [record] : [];
			})
			.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
	}

	readState(record: XlabRunRecord): XlabDisplayState | undefined {
		const statePath = join(record.runDir, "state.json");
		if (!existsSync(statePath)) {
			return undefined;
		}
		return JSON.parse(readFileSync(statePath, "utf-8")) as XlabDisplayState;
	}

	readWorkflow(record: XlabRunRecord): XlabWorkflowState | undefined {
		const workflowPath = join(record.runDir, "workflow.json");
		if (!existsSync(workflowPath)) {
			return undefined;
		}
		return JSON.parse(readFileSync(workflowPath, "utf-8")) as XlabWorkflowState;
	}

	updateRun(record: XlabRunRecord, patch: Partial<XlabRunRecord>, eventType: string, data?: unknown): XlabRunRecord {
		const runDir = join(this.runsRoot, record.runId);
		const release = lockfile.lockSync(runDir, { realpath: false });
		try {
			const latest = this.readRun(record.runId);
			if (!latest) {
				throw new Error(`Cannot update unknown XLab run ${record.runId}.`);
			}
			if (TERMINAL_RUN_STATUSES.has(latest.status) && patch.status !== undefined && patch.status !== latest.status) {
				const markingPublicationIncomplete =
					latest.status === "success" &&
					patch.status === "incomplete" &&
					typeof patch.publicationRecoveryReason === "string" &&
					Boolean(patch.publicationRecoveryReason.trim());
				const recoveringPublication =
					latest.status === "incomplete" &&
					patch.status === "success" &&
					typeof latest.publicationRecoveryReason === "string" &&
					Boolean(latest.publicationRecoveryReason.trim()) &&
					Object.hasOwn(patch, "publicationRecoveryReason") &&
					patch.publicationRecoveryReason === undefined;
				if (!markingPublicationIncomplete && !recoveringPublication) return latest;
			}
			const next: XlabRunRecord = {
				...latest,
				...patch,
				updatedAt: nowIso(),
			};
			this.writeRecord(next);
			this.appendEvent(next.runId, {
				type: eventType,
				timestamp: nowIso(),
				data,
			});
			return next;
		} finally {
			release();
		}
	}

	setStatus(record: XlabRunRecord, status: XlabRunStatus, eventType = "status"): XlabRunRecord {
		const finishedAt =
			status === "success" || status === "failed" || status === "incomplete" || status === "cancelled"
				? nowIso()
				: record.finishedAt;
		return this.updateRun(record, { status, finishedAt }, eventType, {
			status,
		});
	}

	resumeRun(record: XlabRunRecord): XlabRunRecord {
		return this.updateRun(
			record,
			{
				status: "running",
				finishedAt: undefined,
				resumedFrom: record.updatedAt,
				resumeCount: (record.resumeCount ?? 0) + 1,
			},
			"resumed",
		);
	}

	transitionStage(
		record: XlabRunRecord,
		stageId: string,
		action: "start" | "complete" | "fail" | "block" | "skip",
		data?: { checkpoint?: unknown; artifacts?: unknown; error?: string },
	): { ok: boolean; state?: XlabWorkflowState; repair?: string } {
		const workflow = this.readWorkflow(record);
		if (!workflow) {
			return {
				ok: false,
				repair: `XLab run ${record.runId} has no workflow declaration.`,
			};
		}
		const result = transitionWorkflowStage(workflow, stageId, action, data);
		if (!result.ok) {
			return { ok: false, state: workflow, repair: result.repair };
		}
		this.writeWorkflow(record, result.state);
		if (action === "start" && result.stage) {
			mkdirSync(join(record.runDir, "children", stageId, `attempt-${result.stage.attempts}`), { recursive: true });
		}
		this.appendEvent(record.runId, {
			type: `stage_${action}`,
			timestamp: nowIso(),
			data: { stageId, ...data, stage: result.stage },
		});
		return { ok: true, state: result.state };
	}

	updateState(record: XlabRunRecord, patch: XlabDisplayState, eventType = "state_patch"): XlabDisplayState {
		const state = mergeXlabDisplayState(this.readState(record), patch);
		this.writeState(record, state);
		this.appendEvent(record.runId, {
			type: eventType,
			timestamp: nowIso(),
			data: { patch, state },
		});
		return state;
	}

	resolveRunPath(record: XlabRunRecord, pathValue: string): string | undefined {
		if (!pathValue.trim()) {
			return undefined;
		}
		if (isAbsolute(pathValue)) {
			const absolutePath = resolve(pathValue);
			return isPathInside(record.cwd, absolutePath) || isPathInside(record.runDir, absolutePath)
				? absolutePath
				: undefined;
		}
		const pathParts = pathValue.replaceAll("\\", "/").split("/");
		if (pathParts.includes("..")) {
			return undefined;
		}
		const legacyRunPrefix = `.xlab/runs/${record.runId}/`;
		const resolved = pathValue.replaceAll("\\", "/").startsWith(legacyRunPrefix)
			? resolve(record.cwd, pathValue)
			: resolve(record.runDir, pathValue);
		return isPathInside(record.runDir, resolved) ? resolved : undefined;
	}

	private writeRecord(record: XlabRunRecord): void {
		mkdirSync(record.runDir, { recursive: true });
		writeJson(join(record.runDir, "run.json"), record);
	}

	private writeState(record: XlabRunRecord, state: XlabDisplayState): void {
		mkdirSync(record.runDir, { recursive: true });
		writeJson(join(record.runDir, "state.json"), state);
	}

	private writeWorkflow(record: XlabRunRecord, state: XlabWorkflowState): void {
		mkdirSync(record.runDir, { recursive: true });
		writeJson(join(record.runDir, "workflow.json"), state);
	}

	private appendEvent(runId: string, event: XlabRunEvent): void {
		const runDir = join(this.runsRoot, runId);
		mkdirSync(runDir, { recursive: true });
		appendFileSync(join(runDir, "events.jsonl"), `${JSON.stringify(event)}\n`, "utf-8");
	}
}
