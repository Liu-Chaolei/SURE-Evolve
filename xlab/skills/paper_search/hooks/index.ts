import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { XlabHookContext, XlabHookResult } from "@earendil-works/pi-coding-agent";

interface CandidateSet {
	candidates?: unknown;
}

export function preStart(ctx: XlabHookContext): XlabHookResult {
	return existsSync(join(ctx.packageDir, "scripts", "search.mjs"))
		? { ok: true }
		: { ok: false, repair: "Restore scripts/search.mjs inside paper_search." };
}

export function preFinish(ctx: XlabHookContext): XlabHookResult {
	const path = join(ctx.runDir, "artifacts", "paper_candidates.json");
	if (!existsSync(path)) {
		return { ok: false, repair: "Create artifacts/paper_candidates.json before finishing paper_search." };
	}
	try {
		const value = JSON.parse(readFileSync(path, "utf-8")) as CandidateSet;
		if (!Array.isArray(value.candidates)) {
			return { ok: false, repair: "paper_candidates.json must contain a candidates array." };
		}
		const keys = value.candidates.flatMap((candidate) =>
			typeof candidate === "object" &&
			candidate !== null &&
			"dedupe_key" in candidate &&
			typeof candidate.dedupe_key === "string"
				? [candidate.dedupe_key]
				: [],
		);
		if (keys.length !== value.candidates.length || new Set(keys).size !== keys.length) {
			return { ok: false, repair: "Fill unique dedupe_key values for every paper candidate." };
		}
		return { ok: true };
	} catch (error) {
		return {
			ok: false,
			repair: `Repair paper_candidates.json: ${error instanceof Error ? error.message : String(error)}`,
		};
	}
}
