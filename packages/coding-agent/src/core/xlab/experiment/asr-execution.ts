import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync } from "node:fs";
import { basename, join } from "node:path";
import { promisify } from "node:util";
import { Type } from "typebox";
import { defineTool } from "../../extensions/index.ts";
import type { CreateAgentSessionOptions } from "../../sdk.ts";
import { atomicWriteFile, atomicWriteJson, canonicalDigest, type ExperimentAssignment } from "./protocol.ts";

const executeFile = promisify(execFile);
export type SlurmCommandRunner = (command: string, args: string[]) => Promise<{ stdout: string; stderr: string }>;

export interface AsrEvolutionConfig {
	deployment_root?: string;
	research_script?: string;
	schema_version: "xlab.asr_evolution.v1";
	workspace: string;
	env_file: string;
	python: string;
	runtime: string;
	icefall: string;
	recipe: string;
	data: string;
	survey: string;
	resource_manifest: string;
	sure_pythonpath: string;
	pipeline_id: string;
	ports_dir: string;
	refs: { regular: string; selection: string; test: string };
	training: { epochs: 30; world_size: 4; max_duration: 900; precision: "fp32"; seed: 42 };
	slurm: {
		partition: string;
		gres: string;
		cpus: number;
		memory: string;
		temporary: string;
		time: string;
		image: string;
	};
	rounds: 6;
}

export interface AsrConditionResult {
	status: "success";
	job_id: string;
	condition_id: string;
	disabled_components: string[];
	inherited_disabled_components: string[];
	enabled_components: string[];
	source_digest: string;
	checkpoint: string;
	checkpoint_dir: string;
	epoch: number;
	split: string;
	metric: "wer";
	score: number;
	coverage: number;
	sure_report: string;
	output: string;
	project: string;
}

interface JobRecord {
	identity: string;
	status: "submitting" | "submitted" | "success" | "failed" | "cancelled";
	job_id?: string;
	comment: string;
}

export function readJson<T>(path: string): T {
	return JSON.parse(readFileSync(path, "utf-8")) as T;
}

export function sourceDigest(project: string): string {
	const hash = createHash("sha256");
	const visit = (directory: string) => {
		for (const entry of readdirSync(directory, { withFileTypes: true }).sort((a, b) =>
			a.name.localeCompare(b.name),
		)) {
			if (entry.name === "__pycache__") continue;
			const path = join(directory, entry.name);
			if (entry.isDirectory()) visit(path);
			else if (entry.isFile()) {
				hash.update(path.slice(project.length));
				hash.update(readFileSync(path));
			} else throw new Error(`Frozen source cannot contain links: ${path}`);
		}
	};
	for (const name of ["recipe", "library"]) visit(join(project, name));
	hash.update("method.py");
	hash.update(readFileSync(join(project, "method.py")));
	return hash.digest("hex");
}

export function validateAsrConfig(config: AsrEvolutionConfig): void {
	if (
		config.schema_version !== "xlab.asr_evolution.v1" ||
		config.rounds !== 6 ||
		canonicalDigest(config.training) !==
			canonicalDigest({ epochs: 30, world_size: 4, max_duration: 900, precision: "fp32", seed: 42 })
	) {
		throw new Error("Expected the authorized six-round, full-data, 30-epoch, four-NPU, duration-900 protocol.");
	}
	if (config.slurm.gres !== "gpu:ascend910b3:4" || !config.slurm.image.includes("@sha256:"))
		throw new Error("Slurm requires four NPUs and an immutable image digest.");
	for (const path of [
		config.python,
		config.runtime,
		config.recipe,
		config.icefall,
		config.data,
		config.survey,
		config.resource_manifest,
		config.sure_pythonpath,
		...Object.values(config.refs),
	]) {
		if (!path.startsWith("/shared/") || !existsSync(path)) throw new Error(`Missing shared resource: ${path}`);
	}
	const allIds = new Set<string>();
	const allTalks = new Set<string>();
	for (const [split, expected] of [
		["regular", 200],
		["selection", 196],
		["test", 1155],
	] as const) {
		const lines = readFileSync(config.refs[split], "utf-8").trimEnd().split("\n");
		if (lines.length !== expected) throw new Error(`Unexpected ${split} reference count.`);
		const talks = new Set<string>();
		for (const line of lines) {
			const [id, text] = line.split("\t");
			if (!id || text === undefined || allIds.has(id))
				throw new Error("Missing, duplicate or overlapping reference IDs.");
			allIds.add(id);
			const talk = id.replace(/-\d+$/, "");
			if (allTalks.has(talk)) throw new Error("Evaluation splits overlap at talk level.");
			talks.add(talk);
		}
		for (const talk of talks) allTalks.add(talk);
	}
	const prepared = readJson<{ features_ready: boolean; splits: { train: { utterances: number } } }>(
		join(config.data, "preparation.json"),
	);
	if (!prepared.features_ready || prepared.splits.train.utterances !== 268263)
		throw new Error("Full TEDLIUM3 prepared training data is required.");
}

function shellQuote(value: string): string {
	return `'${value.replace(/'/g, `'"'"'`)}'`;
}

export async function runLocalPython(
	config: AsrEvolutionConfig,
	operation: "prepare" | "static",
	requestPath: string,
): Promise<void> {
	await executeFile(config.python, ["-P", join(config.runtime, "asr_runtime/runner.py"), operation, requestPath], {
		maxBuffer: 4 * 1024 * 1024,
		env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" },
	});
}

export function freezeProject(project: string, runDir: string): { project: string; digest: string } {
	const digest = sourceDigest(project);
	const frozen = join(runDir, "sources", digest);
	if (!existsSync(frozen)) {
		mkdirSync(frozen, { recursive: true });
		for (const name of ["recipe", "library", "method.py"])
			cpSync(join(project, name), join(frozen, name), { recursive: true });
	}
	if (sourceDigest(frozen) !== digest) throw new Error("Frozen source verification failed.");
	return { project: frozen, digest };
}

export class AsrSlurmExecutor {
	readonly config: AsrEvolutionConfig;
	readonly runDir: string;
	private readonly command: SlurmCommandRunner;
	private readonly failedHere = new Set<string>();
	constructor(config: AsrEvolutionConfig, runDir: string, command: SlurmCommandRunner = executeFile) {
		this.config = config;
		this.runDir = runDir;
		this.command = command;
	}

	async runCondition(options: {
		id: string;
		project: string;
		sourceDigest: string;
		components: string[];
		disabled: string[];
		inheritedDisabled?: string[];
		split?: "regular" | "selection" | "test";
		checkpointDir?: string;
		signal?: AbortSignal;
	}): Promise<AsrConditionResult> {
		if (!/^[A-Za-z0-9_.-]+$/.test(options.id)) throw new Error("Unsafe condition identifier.");
		if (options.disabled.some((name) => !options.components.includes(name)))
			throw new Error("Unknown disabled component.");
		if (sourceDigest(options.project) !== options.sourceDigest)
			throw new Error("Scientific source changed after freeze.");
		const output = join(this.runDir, "project/results/science", options.id);
		mkdirSync(output, { recursive: true });
		const request = {
			config: this.config,
			project: options.project,
			output,
			source_digest: options.sourceDigest,
			condition_id: options.id,
			enabled_components: options.components.filter((name) => !options.disabled.includes(name)),
			disabled_components: options.disabled,
			inherited_disabled_components: options.inheritedDisabled ?? [],
			split: options.split ?? "regular",
			...(options.checkpointDir ? { checkpoint_dir: options.checkpointDir, evaluation_only: true } : {}),
		};
		const identity = canonicalDigest(request);
		const requestPath = join(output, "request.json");
		const jobPath = join(output, "job.json");
		const comment = `xlab:${basename(this.runDir)}:${identity}`;
		let job: JobRecord;
		if (existsSync(jobPath)) {
			const previous = readJson<JobRecord>(jobPath);
			if (previous.identity !== identity) throw new Error("Changed condition cannot reuse old training evidence.");
			if (previous.status === "cancelled") throw new Error("Cancelled conditions cannot be resumed.");
			if (previous.status === "failed" && !this.failedHere.has(identity)) {
				const queue = previous.job_id
					? await this.command("squeue", ["--noheader", "--jobs", previous.job_id, "--format=%T"])
					: { stdout: "" };
				if (queue.stdout.trim())
					throw new Error("Previous failed job is still allocated; reconcile before retrying.");
				renameSync(jobPath, join(output, `job-${previous.job_id ?? "unknown"}.json`));
				for (const name of ["exit.json", "result.json"]) rmSync(join(output, name), { force: true });
			}
		}
		if (existsSync(jobPath)) {
			job = readJson<JobRecord>(jobPath);
			if (job.identity !== identity)
				throw new Error("Condition identity changed; old checkpoints cannot be reused.");
		} else {
			this.assertActive(options.signal);
			atomicWriteJson(requestPath, request);
			job = { identity, status: "submitting", comment };
			const inner = [
				"srun",
				"--ntasks=1",
				`--gres=${this.config.slurm.gres}`,
				"sudo",
				"-n",
				"slurm-docker-run",
				"--pull",
				"missing",
				"--network",
				"host",
				"--shm-size",
				"128g",
				"--mount",
				"/shared/chaolei.liu:/shared/chaolei.liu:ro",
				"--mount",
				`${output}:${output}:rw`,
				"--mount",
				`${options.project}:${options.project}:ro`,
				"--mount",
				`${this.config.ports_dir}:${this.config.ports_dir}:rw`,
				"--workdir",
				output,
				"--env",
				"PYTHONDONTWRITEBYTECODE=1",
				"--env",
				"TMPDIR=/local/job",
				this.config.slurm.image,
				"python",
				"-P",
				join(this.config.runtime, "asr_runtime/runner.py"),
				"execute",
				requestPath,
			];
			const script = join(output, "condition.sbatch");
			const exitPath = shellQuote(join(output, "exit.json"));
			const body = `#!/bin/bash\nset -uo pipefail\n${inner.map(shellQuote).join(" ")}\nrc=$?\nprintf '{"exit_code":%s,"job_id":"%s"}\\n' "$rc" "$SLURM_JOB_ID" > ${exitPath}.tmp\nmv ${exitPath}.tmp ${exitPath}\nexit "$rc"\n`;
			// Persist the intent before submission; an uncertain acknowledgement is reconciled, never resubmitted.
			atomicWriteFile(script, Buffer.from(body));
			mkdirSync(this.config.ports_dir, { recursive: true });
			atomicWriteJson(jobPath, job);
			const answer = await this.command("sbatch", [
				"--parsable",
				"--partition",
				this.config.slurm.partition,
				"--nodes=1",
				"--ntasks=1",
				`--cpus-per-task=${this.config.slurm.cpus}`,
				`--mem=${this.config.slurm.memory}`,
				`--tmp=${this.config.slurm.temporary}`,
				`--gres=${this.config.slurm.gres}`,
				`--time=${this.config.slurm.time}`,
				`--comment=${comment}`,
				`--job-name=xlab-${identity.slice(0, 12)}`,
				`--output=${join(output, "slurm-%j.log")}`,
				script,
			]);
			const id = answer.stdout.trim().split(";")[0];
			if (!/^\d+$/.test(id)) throw new Error("Slurm acknowledgement did not contain a job ID.");
			job = { ...job, job_id: id, status: "submitted" };
			atomicWriteJson(jobPath, job);
		}
		if (!job.job_id) {
			if (existsSync(join(output, "exit.json")))
				job.job_id = readJson<{ job_id: string }>(join(output, "exit.json")).job_id;
			else {
				const queue = await this.command("squeue", ["--noheader", "--me", "--format=%i|%k"]);
				const matches = queue.stdout
					.trim()
					.split("\n")
					.filter((line) => line.split("|")[1] === comment);
				if (matches.length !== 1)
					throw new Error("Uncertain Slurm submission; reconcile the persisted intent before resuming.");
				job.job_id = matches[0].split("|")[0].trim();
			}
			atomicWriteJson(jobPath, job);
		}
		for (;;) {
			this.assertActive(options.signal);
			const queue = await this.command("squeue", ["--noheader", "--jobs", job.job_id, "--format=%T"]);
			if (!queue.stdout.trim()) break;
			await new Promise<void>((done) => setTimeout(done, 15000));
		}
		const terminalPath = join(output, "exit.json");
		if (!existsSync(terminalPath) || readJson<{ exit_code: number }>(terminalPath).exit_code !== 0) {
			this.failedHere.add(identity);
			atomicWriteJson(jobPath, { ...job, status: "failed" });
			throw new Error(`Formal condition ${options.id} failed; inspect ${output}. No scientific result accepted.`);
		}
		const result = readJson<AsrConditionResult>(join(output, "result.json"));
		const split = options.split ?? "regular";
		const expectedCoverage = { regular: 200, selection: 196, test: 1155 }[split];
		if (
			result.status !== "success" ||
			result.source_digest !== options.sourceDigest ||
			result.epoch !== 30 ||
			!Number.isFinite(result.score) ||
			result.score < 0 ||
			result.job_id !== job.job_id ||
			result.split !== split ||
			result.metric !== "wer" ||
			result.coverage !== expectedCoverage ||
			result.project !== options.project ||
			result.output !== output ||
			result.checkpoint !== join(options.checkpointDir ?? join(output, "models"), "epoch-30.pt") ||
			result.sure_report !== join(output, `sure_${split}/score.json`) ||
			canonicalDigest(result.enabled_components) !== canonicalDigest(request.enabled_components) ||
			canonicalDigest(result.disabled_components) !== canonicalDigest(options.disabled) ||
			canonicalDigest(result.inherited_disabled_components) !==
				canonicalDigest(request.inherited_disabled_components) ||
			!existsSync(result.checkpoint)
		) {
			throw new Error("Formal result failed protocol, score or checkpoint validation.");
		}
		const scoring = readJson<{ score: number; coverage: number; pipeline_id: string; reference_sha256: string }>(
			result.sure_report,
		);
		if (
			scoring.score !== result.score ||
			scoring.coverage !== expectedCoverage ||
			scoring.pipeline_id !== this.config.pipeline_id ||
			scoring.reference_sha256 !== createHash("sha256").update(readFileSync(this.config.refs[split])).digest("hex")
		)
			throw new Error("Condition metrics do not match the frozen SURE report.");
		atomicWriteJson(jobPath, { ...job, status: "success" });
		return result;
	}

	async cancel(): Promise<void> {
		atomicWriteJson(join(this.runDir, "cancel_requested.json"), { requested_at: new Date().toISOString() });
		const root = join(this.runDir, "project/results/science");
		if (!existsSync(root)) return;
		for (const entry of readdirSync(root)) {
			const path = join(root, entry, "job.json");
			if (!existsSync(path)) continue;
			const job = readJson<JobRecord>(path);
			if (!job.job_id || job.status === "success") continue;
			const current = await this.command("squeue", ["--noheader", "--jobs", job.job_id, "--format=%k"]);
			if (current.stdout.trim() === job.comment) await this.command("scancel", [job.job_id]);
			atomicWriteJson(path, { ...job, status: "cancelled" });
		}
	}

	private assertActive(signal?: AbortSignal): void {
		if (signal?.aborted || existsSync(join(this.runDir, "cancel_requested.json")))
			throw new Error("Experiment execution cancelled.");
	}
}

export function executionTools(
	execute: (assignment: ExperimentAssignment, signal?: AbortSignal) => Promise<unknown>,
	assignment: ExperimentAssignment,
): NonNullable<CreateAgentSessionOptions["customTools"]> {
	if (assignment.child.role !== "worker") return [];
	return [
		defineTool({
			name: "experiment_execute",
			label: "Execute bound experiment work",
			description:
				"Execute the current immutable prepared/static/formal work unit and return real evidence. Science waits for its Slurm job without model polling; replay reattaches to the same job.",
			parameters: Type.Object({}),
			executionMode: "sequential",
			async execute(_id, _params, signal) {
				const result = await execute(assignment, signal);
				return { content: [{ type: "text", text: JSON.stringify(result) }], details: { result } };
			},
		}),
	];
}
