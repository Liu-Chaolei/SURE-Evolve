import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { XlabHookContext, XlabHookResult } from "@earendil-works/pi-coding-agent";

function parseArgs(value: string): Record<string, string> {
	const result: Record<string, string> = {};
	const pattern = /(\w+)\s*=\s*(?:"([^"]*)"|(\S+))/g;
	let match = pattern.exec(value);
	while (match) {
		result[match[1]] = match[2] ?? match[3] ?? "";
		match = pattern.exec(value);
	}
	if (!result.scholar_name && value.trim() && !value.includes("=")) {
		result.scholar_name = value.trim();
	}
	return result;
}

export function preStart(ctx: XlabHookContext): XlabHookResult {
	const args = parseArgs(ctx.args);
	if (!args.scholar_name) {
		return {
			ok: false,
			repair: 'Provide a scholar name, for example /xlab build-scholar-profile scholar_name="Yoshua Bengio".',
		};
	}
	return {
		ok: true,
		state_patch: {
			phase: { id: "source_discovery", label: "Discovering scholar evidence", status: "running" },
			counters: { completed_stages: 0, total_stages: 5 },
		},
	};
}

export function preFinish(ctx: XlabHookContext): XlabHookResult {
	const profilePath = join(ctx.runDir, "scholar_profile.json");
	const promptPath = join(ctx.runDir, "system_prompt.md");
	if (!existsSync(profilePath) || !existsSync(promptPath)) {
		return {
			ok: false,
			repair: "Create scholar_profile.json and system_prompt.md in the run directory before finishing.",
		};
	}
	try {
		const profile = JSON.parse(readFileSync(profilePath, "utf-8")) as {
			scholar_name?: unknown;
			mainline_graph?: unknown;
			representative_papers?: unknown;
			system_prompt_path?: unknown;
		};
		const prompt = readFileSync(promptPath, "utf-8").trim();
		if (
			typeof profile.scholar_name !== "string" ||
			!profile.mainline_graph ||
			!Array.isArray(profile.representative_papers) ||
			typeof profile.system_prompt_path !== "string"
		) {
			return { ok: false, repair: "Fill all required scholar_profile@1 fields." };
		}
		if (prompt.length < 1000) {
			return { ok: false, repair: "Generate a more complete evidence-grounded system_prompt.md." };
		}
		return {
			ok: true,
			state_patch: {
				phase: { id: "validate", label: "Scholar profile validated", status: "success", progress: 1 },
				counters: { completed_stages: 5, total_stages: 5 },
			},
		};
	} catch (error) {
		return {
			ok: false,
			repair: `Repair scholar_profile.json: ${error instanceof Error ? error.message : String(error)}`,
		};
	}
}

export function onError(ctx: XlabHookContext): XlabHookResult {
	const checkpoint = {
		run_id: ctx.run.runId,
		status: ctx.run.status,
		repair: ctx.run.lastRepair,
		updated_at: new Date().toISOString(),
	};
	writeFileSync(join(ctx.runDir, "scholar_profile.checkpoint.json"), `${JSON.stringify(checkpoint, null, 2)}\n`);
	return {
		ok: true,
		state_patch: {
			phase: { id: "interrupted", label: "Scholar profile interrupted", status: "incomplete" },
			checkpoint: {
				id: "scholar-profile",
				label: "Scholar profile stage checkpoint",
				resumable: true,
				resume_hint: `Run /xlab resume-run ${ctx.run.runId}.`,
				data: checkpoint,
			},
		},
	};
}

export function onResume(): XlabHookResult {
	return { ok: true };
}

export function onCancel(): XlabHookResult {
	return { ok: true };
}

export function postFinish(ctx: XlabHookContext): XlabHookResult {
	return {
		ok: true,
		state_patch: {
			phase: { id: "finish", label: "Scholar profile finished", status: ctx.run.status, progress: 1 },
		},
	};
}
