import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { XlabHookContext, XlabHookResult } from "@earendil-works/pi-coding-agent";

export function preStart(ctx: XlabHookContext): XlabHookResult {
	return existsSync(join(ctx.packageDir, "scripts", "fetch.mjs"))
		? { ok: true }
		: { ok: false, repair: "Restore scripts/fetch.mjs inside paper_fetch." };
}

export function preFinish(ctx: XlabHookContext): XlabHookResult {
	const path = join(ctx.runDir, "artifacts", "fetch_manifest.json");
	if (!existsSync(path)) return { ok: false, repair: "Create artifacts/fetch_manifest.json." };
	try {
		const value = JSON.parse(readFileSync(path, "utf-8")) as { papers?: unknown };
		return Array.isArray(value.papers)
			? { ok: true }
			: { ok: false, repair: "fetch_manifest.json must contain every input paper in papers." };
	} catch (error) {
		return { ok: false, repair: `Repair fetch_manifest.json: ${error instanceof Error ? error.message : String(error)}` };
	}
}
