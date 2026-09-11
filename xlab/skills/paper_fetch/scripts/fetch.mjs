#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";

function parseArgs(argv) {
	const value = { input: "", outputDir: "", manifest: "", offline: false, maxBytes: 50 * 1024 * 1024 };
	for (let index = 0; index < argv.length; index += 1) {
		const arg = argv[index];
		if (arg === "--offline") value.offline = true;
		else if (arg === "--input") value.input = argv[++index] ?? "";
		else if (arg === "--output-dir") value.outputDir = argv[++index] ?? "";
		else if (arg === "--manifest") value.manifest = argv[++index] ?? "";
		else if (arg === "--max-bytes") value.maxBytes = Number.parseInt(argv[++index] ?? "", 10);
	}
	if (!value.input || !value.outputDir || !value.manifest) throw new Error("--input, --output-dir, and --manifest are required");
	return value;
}

async function download(url, maxBytes) {
	const response = await fetch(url, {
		headers: { Accept: "application/pdf", "User-Agent": "XLab/1.0 paper-fetch" },
		redirect: "follow",
		signal: AbortSignal.timeout(60_000),
	});
	if (!response.ok) throw new Error(`HTTP ${response.status}`);
	const declared = Number.parseInt(response.headers.get("content-length") ?? "0", 10);
	if (declared > maxBytes) throw new Error(`payload exceeds ${maxBytes} bytes`);
	const bytes = new Uint8Array(await response.arrayBuffer());
	if (bytes.byteLength > maxBytes) throw new Error(`payload exceeds ${maxBytes} bytes`);
	if (new TextDecoder().decode(bytes.slice(0, 4)) !== "%PDF") throw new Error("response is not a PDF");
	return bytes;
}

const options = parseArgs(process.argv.slice(2));
const input = JSON.parse(readFileSync(options.input, "utf-8"));
if (!Array.isArray(input.candidates)) throw new Error("input candidates array is required");
mkdirSync(options.outputDir, { recursive: true });
const papers = [];
for (const candidate of input.candidates) {
	const result = { ...candidate, download_status: "unavailable", pdf_path: null, sha256: null, blocker: null };
	if (options.offline) {
		result.download_status = "skipped_offline";
		result.blocker = "offline mode";
	} else if (!candidate.open_access_pdf) {
		result.blocker = "no open-access PDF URL";
	} else {
		try {
			const bytes = await download(candidate.open_access_pdf, options.maxBytes);
			const name = `${candidate.id || basename(candidate.open_access_pdf)}.pdf`;
			const path = join(options.outputDir, name);
			writeFileSync(path, bytes);
			result.download_status = "downloaded";
			result.pdf_path = path;
			result.sha256 = createHash("sha256").update(bytes).digest("hex");
		} catch (error) {
			result.download_status = "failed";
			result.blocker = error instanceof Error ? error.message : String(error);
		}
	}
	papers.push(result);
}
const output = {
	schema_version: "xlab.paper_fetch_manifest.v1",
	generated_at: new Date().toISOString(),
	input_path: options.input,
	papers,
	summary: {
		total: papers.length,
		downloaded: papers.filter((paper) => paper.download_status === "downloaded").length,
		failed: papers.filter((paper) => paper.download_status === "failed").length,
	},
};
writeFileSync(options.manifest, `${JSON.stringify(output, null, 2)}\n`, "utf-8");
console.log(JSON.stringify({ manifest: options.manifest, ...output.summary }));
