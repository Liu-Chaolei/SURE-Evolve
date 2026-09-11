#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

function args(argv) {
	const parsed = { query: "", limit: 20, output: "", offline: false };
	for (let index = 0; index < argv.length; index += 1) {
		const value = argv[index];
		if (value === "--offline") parsed.offline = true;
		else if (value === "--query") parsed.query = argv[++index] ?? "";
		else if (value === "--limit") parsed.limit = Number.parseInt(argv[++index] ?? "20", 10);
		else if (value === "--output") parsed.output = argv[++index] ?? "";
	}
	if (!parsed.query || !parsed.output) throw new Error("--query and --output are required");
	parsed.limit = Math.max(1, Math.min(Number.isFinite(parsed.limit) ? parsed.limit : 20, 100));
	return parsed;
}

function normalized(value) {
	return String(value ?? "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().replace(/\s+/g, " ");
}

function candidateId(value) {
	return `paper_${createHash("sha256").update(value).digest("hex").slice(0, 16)}`;
}

function normalizePaper(paper, rank, provider) {
	const externalIds = paper.externalIds ?? {};
	const title = String(paper.title ?? "").trim();
	const doi = typeof externalIds.DOI === "string" ? externalIds.DOI : null;
	const arxivId = typeof externalIds.ArXiv === "string" ? externalIds.ArXiv : null;
	const dedupeKey = doi ? `doi:${doi.toLowerCase()}` : arxivId ? `arxiv:${arxivId.toLowerCase()}` : `title:${normalized(title)}`;
	return {
		id: candidateId(dedupeKey),
		title,
		authors: Array.isArray(paper.authors) ? paper.authors.map((author) => String(author.name ?? "")).filter(Boolean) : [],
		year: Number.isInteger(paper.year) ? paper.year : null,
		venue: String(paper.venue ?? ""),
		abstract: typeof paper.abstract === "string" ? paper.abstract : null,
		url: typeof paper.url === "string" ? paper.url : null,
		doi,
		arxiv_id: arxivId,
		open_access_pdf:
			paper.openAccessPdf && typeof paper.openAccessPdf.url === "string" ? paper.openAccessPdf.url : null,
		citation_count: Number.isInteger(paper.citationCount) ? paper.citationCount : 0,
		provider,
		source_rank: rank,
		dedupe_key: dedupeKey,
	};
}

function offlineCandidates(query, limit) {
	return Array.from({ length: limit }, (_, index) => {
		const title = `${query} evidence study ${index + 1}`;
		return normalizePaper(
			{
				title,
				authors: [{ name: `Researcher ${index + 1}` }],
				year: 2026 - (index % 5),
				venue: "XLab Offline Fixture",
				abstract: `Deterministic fixture candidate for ${query}.`,
				url: `https://example.org/${candidateId(title)}`,
				externalIds: { DOI: `10.0000/xlab.${index + 1}` },
				citationCount: limit - index,
			},
			index + 1,
			"offline",
		);
	});
}

async function semanticScholar(query, limit) {
	const fields = "title,authors,year,venue,abstract,url,citationCount,externalIds,openAccessPdf";
	const url = new URL("https://api.semanticscholar.org/graph/v1/paper/search");
	url.searchParams.set("query", query);
	url.searchParams.set("limit", String(limit));
	url.searchParams.set("fields", fields);
	const headers = { Accept: "application/json", "User-Agent": "XLab/1.0 paper-search" };
	if (process.env.SEMANTIC_SCHOLAR_API_KEY) headers["x-api-key"] = process.env.SEMANTIC_SCHOLAR_API_KEY;
	const response = await fetch(url, { headers, signal: AbortSignal.timeout(30_000) });
	if (!response.ok) throw new Error(`Semantic Scholar HTTP ${response.status}`);
	const payload = await response.json();
	return Array.isArray(payload.data)
		? payload.data.map((paper, index) => normalizePaper(paper, index + 1, "semantic_scholar"))
		: [];
}

const options = args(process.argv.slice(2));
const errors = [];
let candidates = [];
if (options.offline) {
	candidates = offlineCandidates(options.query, options.limit);
} else {
	try {
		candidates = await semanticScholar(options.query, options.limit);
	} catch (error) {
		errors.push({ provider: "semantic_scholar", message: error instanceof Error ? error.message : String(error) });
	}
}
const unique = new Map();
for (const candidate of candidates) {
	if (candidate.title && !unique.has(candidate.dedupe_key)) unique.set(candidate.dedupe_key, candidate);
}
const output = {
	schema_version: "xlab.paper_candidates.v1",
	query: options.query,
	generated_at: new Date().toISOString(),
	providers: options.offline ? ["offline"] : ["semantic_scholar"],
	errors,
	candidates: Array.from(unique.values()).slice(0, options.limit),
};
mkdirSync(dirname(options.output), { recursive: true });
writeFileSync(options.output, `${JSON.stringify(output, null, 2)}\n`, "utf-8");
console.log(JSON.stringify({ output: options.output, count: output.candidates.length, errors: errors.length }));
