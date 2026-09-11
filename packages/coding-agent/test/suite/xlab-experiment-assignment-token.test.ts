import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { ExperimentAssignmentTokenFactory } from "../../src/core/xlab/experiment/assignment-token.ts";

const KEY_FILE = join("secrets", "experiment-assignment-token-v1.key");

describe("experiment assignment token factory", () => {
	const cleanups: Array<() => void> = [];

	afterEach(() => {
		while (cleanups.length > 0) cleanups.pop()?.();
	});

	function agentDir(): string {
		const path = mkdtempSync(join(tmpdir(), "xlab-assignment-token-"));
		cleanups.push(() => rmSync(path, { recursive: true, force: true }));
		return path;
	}

	it("persists one restrictive key and derives restart-stable domain-separated tokens", () => {
		const dir = agentDir();
		const first = new ExperimentAssignmentTokenFactory(dir, "run-1");
		const token = first.create("assignment-1", true);
		const restarted = new ExperimentAssignmentTokenFactory(dir, "run-1");

		expect(restarted.create("assignment-1", false)).toBe(token);
		expect(restarted.create("assignment-2", false)).not.toBe(token);
		expect(new ExperimentAssignmentTokenFactory(dir, "run-2").create("assignment-1", false)).not.toBe(token);
		expect(statSync(join(dir, "secrets")).mode & 0o777).toBe(0o700);
		expect(statSync(join(dir, KEY_FILE)).mode & 0o777).toBe(0o600);
		expect(readFileSync(join(dir, KEY_FILE), "utf-8")).not.toContain(token);
	});

	it("fails closed when an active assignment key is missing or unsafe", () => {
		const missing = agentDir();
		expect(() => new ExperimentAssignmentTokenFactory(missing, "run-1").create("assignment-1", false)).toThrow(
			"active assignments cannot be resumed",
		);

		const malformed = agentDir();
		mkdirSync(join(malformed, "secrets"), { mode: 0o700 });
		writeFileSync(join(malformed, KEY_FILE), "not-a-key\n", { mode: 0o600 });
		expect(() => new ExperimentAssignmentTokenFactory(malformed, "run-1").create("assignment-1", false)).toThrow(
			"key is invalid",
		);

		const permissive = agentDir();
		const factory = new ExperimentAssignmentTokenFactory(permissive, "run-1");
		factory.create("assignment-1", true);
		chmodSync(join(permissive, KEY_FILE), 0o644);
		expect(() => new ExperimentAssignmentTokenFactory(permissive, "run-1").create("assignment-1", false)).toThrow(
			"unsafe permissions",
		);
	});
});
