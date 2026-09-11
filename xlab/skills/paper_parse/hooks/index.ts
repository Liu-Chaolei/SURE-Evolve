import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { XlabHookContext, XlabHookResult } from "@earendil-works/pi-coding-agent";

export function preStart(ctx: XlabHookContext): XlabHookResult {
	return existsSync(join(ctx.packageDir, "scripts", "parse_documents.py"))
		? { ok: true }
		: { ok: false, repair: "Restore scripts/parse_documents.py inside paper_parse." };
}

export function preFinish(ctx: XlabHookContext): XlabHookResult {
	const path = join(ctx.runDir, "artifacts", "paper_documents.json");
	if (!existsSync(path)) return { ok: false, repair: "Create artifacts/paper_documents.json." };
	try {
		const value = JSON.parse(readFileSync(path, "utf-8")) as { documents?: unknown };
		return Array.isArray(value.documents)
			? { ok: true }
			: { ok: false, repair: "paper_documents.json must contain a documents array." };
	} catch (error) {
		return { ok: false, repair: `Repair paper_documents.json: ${error instanceof Error ? error.message : String(error)}` };
	}
}
