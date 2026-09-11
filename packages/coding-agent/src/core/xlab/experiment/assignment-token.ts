import { createHmac, randomBytes, randomUUID } from "node:crypto";
import {
	chmodSync,
	closeSync,
	existsSync,
	fsyncSync,
	linkSync,
	lstatSync,
	mkdirSync,
	openSync,
	readFileSync,
	unlinkSync,
	writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";

const TOKEN_KEY_FILE = "experiment-assignment-token-v1.key";
const TOKEN_KEY_BYTES = 32;
const TOKEN_CONTEXT = "xlab.experiment.assignment-token.v1";

function errorCode(error: unknown): string | undefined {
	return typeof error === "object" && error !== null && "code" in error
		? String((error as { code?: unknown }).code)
		: undefined;
}

function fsyncDirectory(path: string): void {
	const fd = openSync(path, "r");
	try {
		fsyncSync(fd);
	} finally {
		closeSync(fd);
	}
}

function readKey(path: string): Buffer {
	const metadata = lstatSync(path);
	if (!metadata.isFile() || metadata.isSymbolicLink() || (metadata.mode & 0o077) !== 0) {
		throw new Error("Experiment assignment token key is invalid or has unsafe permissions.");
	}
	const encoded = readFileSync(path, "utf-8");
	if (!/^[A-Za-z0-9_-]{43}\n$/.test(encoded)) {
		throw new Error("Experiment assignment token key is invalid.");
	}
	const key = Buffer.from(encoded.slice(0, -1), "base64url");
	if (key.length !== TOKEN_KEY_BYTES || `${key.toString("base64url")}\n` !== encoded) {
		throw new Error("Experiment assignment token key is invalid.");
	}
	return key;
}

function createKey(path: string): Buffer {
	const parent = dirname(path);
	mkdirSync(parent, { recursive: true, mode: 0o700 });
	chmodSync(parent, 0o700);
	const key = randomBytes(TOKEN_KEY_BYTES);
	const temporary = join(parent, `.${TOKEN_KEY_FILE}.${process.pid}.${randomUUID()}.tmp`);
	let fd: number | undefined;
	try {
		fd = openSync(temporary, "wx", 0o600);
		writeFileSync(fd, `${key.toString("base64url")}\n`, "utf-8");
		fsyncSync(fd);
		closeSync(fd);
		fd = undefined;
		try {
			linkSync(temporary, path);
		} catch (error) {
			if (errorCode(error) !== "EEXIST") {
				throw error;
			}
			return readKey(path);
		}
		fsyncDirectory(parent);
		return key;
	} finally {
		if (fd !== undefined) {
			try {
				closeSync(fd);
			} catch {
				// Preserve the original key creation error.
			}
		}
		try {
			if (existsSync(temporary)) unlinkSync(temporary);
		} catch {
			// Preserve the original key creation error.
		}
	}
}

export class ExperimentAssignmentTokenFactory {
	private key?: Buffer;
	private readonly keyPath: string;
	private readonly runId: string;

	constructor(agentDir: string, runId: string) {
		this.keyPath = join(agentDir, "secrets", TOKEN_KEY_FILE);
		this.runId = runId;
	}

	create(assignmentId: string, allowKeyCreation: boolean): string {
		if (!this.key) {
			if (existsSync(this.keyPath)) {
				this.key = readKey(this.keyPath);
			} else if (allowKeyCreation) {
				this.key = createKey(this.keyPath);
			} else {
				throw new Error("Experiment assignment token key is missing; active assignments cannot be resumed.");
			}
		}
		return createHmac("sha256", this.key)
			.update(TOKEN_CONTEXT)
			.update("\0")
			.update(this.runId)
			.update("\0")
			.update(assignmentId)
			.digest("base64url");
	}
}

export function createExperimentAssignmentTokenFactory(
	agentDir: string,
	runId: string,
): (assignmentId: string, allowKeyCreation: boolean) => string {
	const factory = new ExperimentAssignmentTokenFactory(agentDir, runId);
	return (assignmentId, allowKeyCreation) => factory.create(assignmentId, allowKeyCreation);
}
