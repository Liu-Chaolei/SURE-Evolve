import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { preFinish, preToolCall } from "../../../../xlab/skills/knowledge_graph/hooks/index.ts";

function makeContext(tempDir: string, event: Record<string, unknown> = {}): any {
	const runId = "kg-test-run";
	return {
		cwd: tempDir,
		packageDir: join(tempDir, "xlab", "skills", "knowledge_graph"),
		runDir: join(tempDir, ".xlab", "runs", runId),
		args: `"${join(tempDir, "collection")}"`,
		run: {
			runId,
			status: "running",
			environment: {},
		},
		event,
	};
}

function runToolGate(tempDir: string, toolName: string, input: Record<string, unknown>) {
	return preToolCall(makeContext(tempDir, { toolName, input }));
}

function writeFakeAuditScript(tempDir: string): void {
	const ctx = makeContext(tempDir);
	const scriptsDir = join(ctx.packageDir, "scripts");
	const artifactsDir = join(ctx.runDir, "artifacts");
	mkdirSync(scriptsDir, { recursive: true });
	mkdirSync(artifactsDir, { recursive: true });
	writeFileSync(join(artifactsDir, "method_graph.json"), "{}\n", "utf-8");
	writeFileSync(join(artifactsDir, "graph.db"), "sqlite-placeholder", "utf-8");
	writeFileSync(
		join(scriptsDir, "build_graph.py"),
		`#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from pathlib import Path
args = sys.argv[1:]
run_dir = Path(args[args.index("--run-dir") + 1])
artifacts = run_dir / "artifacts"
artifacts.mkdir(parents=True, exist_ok=True)
(artifacts / "graph_report.json").write_text(json.dumps({
    "passed": True,
    "counts": {
        "input_papers": 1,
        "downloaded_pdfs": 1,
        "parsed_papers": 1,
        "structured_papers": 1,
        "extracted_papers": 1,
        "nodes": 1,
        "edges": 1,
        "aliases": 0,
    },
    "warnings": [],
    "blocking_errors": [],
}) + "\\n", encoding="utf-8")
base = f".xlab/runs/{run_dir.name}/artifacts"
(run_dir / "manifest.json").write_text(json.dumps({
    "schema_version": "2",
    "run_id": run_dir.name,
    "skill_name": "knowledge_graph",
    "skill_version": "3.0.0",
    "status": "success",
    "created_at": "2026-01-01T00:00:00Z",
    "inputs": {},
    "outputs": {},
    "validation": {"passed": True},
    "artifacts": [
        {"type": "mineru_documents", "schema_version": "1", "path": f"{base}/mineru.manifest.json"},
        {"type": "paper_structures", "schema_version": "1", "path": f"{base}/paper_structures.manifest.json"},
        {"type": "paper_extractions", "schema_version": "2", "path": f"{base}/extractions.manifest.json"},
        {"type": "method_graph", "schema_version": "2", "path": f"{base}/method_graph.json"},
        {"type": "graph_db", "schema_version": "1", "path": f"{base}/graph.db"},
        {"type": "graph_report", "schema_version": "2", "path": f"{base}/graph_report.json"},
    ],
}) + "\\n", encoding="utf-8")
`,
		"utf-8",
	);
}

describe("knowledge_graph hook hardening", () => {
	const cleanups: Array<() => void> = [];

	afterEach(() => {
		while (cleanups.length > 0) {
			cleanups.pop()?.();
		}
	});

	function tempProject(): string {
		const tempDir = mkdtempSync(join(tmpdir(), "xlab-kg-hook-"));
		cleanups.push(() => rmSync(tempDir, { recursive: true, force: true }));
		return tempDir;
	}

	it("allows only the canonical packaged CLI for the active run directory", () => {
		const tempDir = tempProject();
		const ctx = makeContext(tempDir);
		const scriptPath = join(ctx.packageDir, "scripts", "build_graph.py");
		const canonical = runToolGate(tempDir, "bash", {
			command: `python ${scriptPath} audit --run-id ${ctx.run.runId} --run-dir ${ctx.runDir}`,
		});
		expect(canonical.ok).toBe(true);

		for (const command of [
			`python ${scriptPath} audit --run-id ${ctx.run.runId} --run-dir ${join(tempDir, "other-run")}`,
			`python ${scriptPath} audit --run-id ${ctx.run.runId} --run-dir ${ctx.runDir} --run-dir ${join(tempDir, "other-run")}`,
			`python ${scriptPath} audit --run-id ${ctx.run.runId} --run-dir ${ctx.runDir}; touch ${join(ctx.runDir, "artifacts", "forged")}`,
			`python xlab/skills/knowledge_graph/scripts/build_graph.py audit --run-id ${ctx.run.runId} --run-dir ${ctx.runDir}`,
			`build_graph.py audit --run-id ${ctx.run.runId} --run-dir ${ctx.runDir}`,
			`./python ${scriptPath} audit --run-id ${ctx.run.runId} --run-dir ${ctx.runDir}`,
		]) {
			const result = runToolGate(tempDir, "bash", { command });
			expect(result.ok, command).toBe(false);
		}
	});

	it("allows only scoped read-only inspection", () => {
		const tempDir = tempProject();
		const ctx = makeContext(tempDir);
		for (const command of [
			`ls ${ctx.runDir}`,
			`find ${ctx.runDir} -maxdepth 2 -type f`,
			`rg GraphMixer ${join(ctx.runDir, "artifacts", "request.json")}`,
			`python -m json.tool ${join(ctx.runDir, "artifacts", "request.json")}`,
		]) {
			const result = runToolGate(tempDir, "bash", { command });
			expect(result.ok, command).toBe(true);
		}
		for (const command of [
			`./grep graph ${join(ctx.runDir, "artifacts", "request.json")}`,
			`rg --pre python graph ${join(ctx.runDir, "artifacts", "request.json")}`,
			"rg PRIVATE_KEY $HOME",
			`find ${ctx.runDir} -delete`,
		]) {
			const result = runToolGate(tempDir, "bash", { command });
			expect(result.ok, command).toBe(false);
		}
	});

	it("blocks write tools and read paths outside run/package scope", () => {
		const tempDir = tempProject();
		const ctx = makeContext(tempDir);
		expect(runToolGate(tempDir, "write", { path: join(ctx.runDir, "manifest.json") }).ok).toBe(false);
		expect(runToolGate(tempDir, "read", { path: join(tempDir, "outside.json") }).ok).toBe(false);
		expect(runToolGate(tempDir, "read", { path: join(ctx.runDir, "artifacts", "request.json") }).ok).toBe(true);
		expect(runToolGate(tempDir, "read", { path: join(ctx.packageDir, "SKILL.md") }).ok).toBe(true);
	});

	it("requires xlab_finish to submit the canonical run manifest", () => {
		const tempDir = tempProject();
		writeFakeAuditScript(tempDir);
		const ctx = makeContext(tempDir, {
			finish: { status: "success" },
			manifestPath: join(tempDir, ".xlab", "runs", "kg-test-run", "alt-manifest.json"),
		});
		const result = preFinish(ctx);
		expect(result.ok).toBe(false);
		expect(JSON.stringify(result)).toContain("canonical manifest");
	});
});
