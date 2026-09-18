import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	type AsrEvolutionConfig,
	AsrSlurmExecutor,
	readJson,
	type SlurmCommandRunner,
	sourceDigest,
} from "../src/core/xlab/experiment/asr-execution.ts";
import { atomicWriteJson } from "../src/core/xlab/experiment/protocol.ts";

const roots: string[] = [];
afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

function fixture() {
	const root = mkdtempSync(join(tmpdir(), "xlab-slurm-unit-"));
	roots.push(root);
	const project = join(root, "source");
	for (const name of ["recipe", "library"]) mkdirSync(join(project, name), { recursive: true });
	writeFileSync(join(project, "method.py"), "# offline scheduler fixture\n");
	const ref = join(root, "regular.txt");
	writeFileSync(ref, "utterance\twords\n");
	const config = {
		runtime: "/runtime",
		ports_dir: join(root, "ports"),
		refs: { regular: ref },
		pipeline_id: "asr.en.wer",
		training: { epochs: 30, world_size: 4, max_duration: 900, precision: "fp32", seed: 42 },
		slurm: {
			gres: "gpu:ascend910b3:4",
			partition: "compute",
			cpus: 80,
			memory: "500G",
			temporary: "1T",
			time: "3-00:00:00",
			image: "image@sha256:abc",
		},
	} as AsrEvolutionConfig;
	let submissions = 0;
	let lostAcknowledgement = false;
	let fail = false;
	const command: SlurmCommandRunner = async (name, args) => {
		if (name === "squeue") return { stdout: "", stderr: "" };
		if (name !== "sbatch") throw new Error(`Unexpected command ${name}`);
		submissions++;
		const output = dirname(args.find((arg) => arg.startsWith("--output="))!.slice("--output=".length));
		const request = readJson<Record<string, unknown>>(join(output, "request.json"));
		const job = String(100 + submissions);
		mkdirSync(join(output, "models"), { recursive: true });
		writeFileSync(join(output, "models/epoch-30.pt"), "offline fixture");
		atomicWriteJson(join(output, "exit.json"), { exit_code: fail ? 1 : 0, job_id: job });
		atomicWriteJson(join(output, "sure_regular/score.json"), {
			score: 0.1,
			coverage: 200,
			pipeline_id: config.pipeline_id,
			reference_sha256: createHash("sha256").update(readFileSync(ref)).digest("hex"),
		});
		atomicWriteJson(join(output, "result.json"), {
			status: "success",
			job_id: job,
			condition_id: request.condition_id,
			disabled_components: [],
			inherited_disabled_components: request.inherited_disabled_components,
			enabled_components: [],
			source_digest: request.source_digest,
			checkpoint: join(output, "models/epoch-30.pt"),
			checkpoint_dir: join(output, "models"),
			epoch: 30,
			split: "regular",
			metric: "wer",
			score: 0.1,
			coverage: 200,
			sure_report: join(output, "sure_regular/score.json"),
			output,
			project,
		});
		if (lostAcknowledgement) {
			lostAcknowledgement = false;
			throw new Error("acknowledgement lost");
		}
		return { stdout: job, stderr: "" };
	};
	return {
		root,
		project,
		config,
		command,
		options: { id: "all-components", project, sourceDigest: sourceDigest(project), components: [], disabled: [] },
		submissions: () => submissions,
		loseAcknowledgement: () => {
			lostAcknowledgement = true;
		},
		fail: (value: boolean) => {
			fail = value;
		},
	};
}

describe("durable Slurm condition ownership", () => {
	it("replays a completed condition without submitting another job", async () => {
		const f = fixture();
		const first = await new AsrSlurmExecutor(f.config, f.root, f.command).runCondition(f.options);
		const second = await new AsrSlurmExecutor(f.config, f.root, f.command).runCondition(f.options);
		expect(second).toEqual(first);
		expect(f.submissions()).toBe(1);
	});
	it("recovers a lost acknowledgement from the durable terminal receipt", async () => {
		const f = fixture();
		f.loseAcknowledgement();
		await expect(new AsrSlurmExecutor(f.config, f.root, f.command).runCondition(f.options)).rejects.toThrow(
			"acknowledgement lost",
		);
		const result = await new AsrSlurmExecutor(f.config, f.root, f.command).runCondition(f.options);
		expect(result.job_id).toBe("101");
		expect(f.submissions()).toBe(1);
	});
	it("keeps failure explicit and permits a new controller to retry the same frozen condition", async () => {
		const f = fixture();
		f.fail(true);
		const executor = new AsrSlurmExecutor(f.config, f.root, f.command);
		await expect(executor.runCondition(f.options)).rejects.toThrow("failed");
		await expect(executor.runCondition(f.options)).rejects.toThrow("failed");
		expect(f.submissions()).toBe(1);
		f.fail(false);
		await new AsrSlurmExecutor(f.config, f.root, f.command).runCondition(f.options);
		expect(f.submissions()).toBe(2);
	});
	it("rejects changed sources and never cancels a job with another ownership tag", async () => {
		const f = fixture();
		writeFileSync(join(f.project, "method.py"), "# changed scientific source\n");
		await expect(new AsrSlurmExecutor(f.config, f.root, f.command).runCondition(f.options)).rejects.toThrow(
			"changed after freeze",
		);
		atomicWriteJson(join(f.root, "project/results/science/other/job.json"), {
			identity: "test",
			status: "submitted",
			job_id: "999",
			comment: "owned-tag",
		});
		const commands: string[] = [];
		await new AsrSlurmExecutor(f.config, f.root, async (name) => {
			commands.push(name);
			return { stdout: "someone-else", stderr: "" };
		}).cancel();
		expect(commands).not.toContain("scancel");
		expect(f.submissions()).toBe(0);
	});
});
