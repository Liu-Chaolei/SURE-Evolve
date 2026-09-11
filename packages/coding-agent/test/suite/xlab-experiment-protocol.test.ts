import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	CODE_REVIEW_ROLES,
	canonicalDigest,
	canonicalJsonBytes,
	type ExperimentAssignment,
	type ExperimentChildIdentity,
	ExperimentProtocolStore,
	type ExperimentReviewReport,
	type ExperimentSubmission,
	PREPARE_WORK_UNITS,
	resolveContainedPath,
	SCIENCE_REVIEW_ROLES,
	validateCodePlan,
	validatePreparePlan,
	validateReviewMatrix,
	validateSciencePlan,
} from "../../src/core/xlab/experiment/protocol.ts";

function digest(value: string): string {
	return createHash("sha256").update(value).digest("hex");
}

function child(runId: string): ExperimentChildIdentity {
	return {
		child_id: "worker-code-integration",
		session_id: "session-1",
		session_file: "/sessions/session-1.jsonl",
		parent_run_id: runId,
		role: "worker",
		stage: "code",
		work_unit: "final_integration_smoke",
		execution_attempt: 1,
		generation: 1,
		review_round: 0,
	};
}

function assignment(
	runId: string,
	policyDigest: string,
): Omit<ExperimentAssignment, "assignment_token_digest" | "status" | "created_at" | "updated_at"> {
	return {
		schema_version: "xlab.experiment_assignment.v1",
		assignment_id: "assignment-1",
		child: child(runId),
		kind: "worker_result",
		input_digest: digest("input"),
		target_digest: digest("target"),
		policy_digest: policyDigest,
		read_scope: [],
		write_scope: [],
	};
}

function submission(
	record: ExperimentAssignment,
	result: Record<string, unknown> = { artifact_role: "worker_result", status: "success" },
): ExperimentSubmission {
	return {
		schema_version: "xlab.experiment_submission.v1",
		submission_id: "submission-1",
		assignment_id: record.assignment_id,
		assignment_token: "secret-token",
		kind: record.kind,
		child_id: record.child.child_id,
		role: record.child.role,
		stage: record.child.stage,
		work_unit: record.child.work_unit,
		execution_attempt: record.child.execution_attempt,
		generation: record.child.generation,
		review_round: record.child.review_round,
		input_digest: record.input_digest,
		target_digest: record.target_digest,
		policy_digest: record.policy_digest,
		result,
		output_paths: [],
		submitted_at: "2026-08-08T00:00:00.000Z",
	};
}

function review(role: string, structuredFindings: Record<string, unknown> = { checked: true }): ExperimentReviewReport {
	return {
		artifact_role: "reviewer_report",
		role,
		reviewer_kind: "agent",
		verdict: "PASS",
		blocking: true,
		summary: `${role} passed`,
		checked_artifacts: ["project/result.json"],
		issues: [],
		structured_findings: structuredFindings,
	};
}

function managedCompletion(artifactIds: string[]): string {
	return `Use managed artifact tools and artifact ledger proof for ${artifactIds.join(", ")}.`;
}

function preparePlan() {
	const artifacts: Record<string, string[]> = {
		repos: ["prepare.discovery", "prepare.repos"],
		dataset: ["prepare.dataset"],
		model: ["prepare.model"],
		env: ["prepare.env"],
		synthesis: ["prepare.idea", "prepare.target_inventory"],
	};
	return {
		stage: "prepare" as const,
		work_units: PREPARE_WORK_UNITS.map((id) => ({
			id,
			goal: `Prepare ${id}`,
			input_paths: { idea: "inputs/idea.json" },
			repos_policy: "reference_or_copy" as const,
			project_must_be_self_contained: true,
			research_required: false,
			acquisition_required: false,
			existing_local_hints: [],
			artifact_ids: artifacts[id],
			done_condition: managedCompletion(artifacts[id]),
		})),
	};
}

const components = ["Evidence memory", "Verifier"];

function codeUnit(id: string, artifactId: string) {
	return {
		id,
		goal: `Implement ${id}`,
		input_paths: { idea: "inputs/idea.json" },
		repos_policy: "reference_or_copy" as const,
		project_must_be_self_contained: true,
		component_scope: components,
		code_artifacts: [
			{
				path: `src/${id}.ts`,
				artifact_type: "source",
				symbols: [id],
				responsibility: `Implement ${id}`,
				dependencies: [],
				config_keys: [],
				entrypoint_role: "runtime",
			},
		],
		interface_contract: { api: id },
		implementation_requirements: { executable: true },
		experiment_bindings: { components },
		component_disable_hooks: [{ config_key: "disabled_component" }],
		write_scope: "project",
		verify_command: `node project/${id}.js`,
		artifact_ids: [artifactId],
		done_condition: `${managedCompletion([artifactId])} Verify a real dataset with metric evaluation.`,
	};
}

function scienceCondition(id: string, disabled: string[] = []) {
	const artifactId = `science.${id}.evidence`;
	return {
		id,
		kind: disabled.length === 0 ? "all_components_reference" : "component_disabled",
		goal: `Run ${id}`,
		full_run: true,
		run_level: "full",
		disabled_component: disabled[0],
		enabled_components: components.filter((component) => !disabled.includes(component)),
		disabled_components: disabled,
		reference_condition_id: disabled.length === 0 ? undefined : "reference",
		output_dir: `results/science/${id}`,
		command: `node project/run.js --condition ${id}${disabled.length ? ` --disabled-component "${disabled[0]}"` : ""}`,
		setup_rationale: "Canonical full comparison",
		runtime_probe_summary: "Runtime and resources verified",
		source_basis: [{ source: "idea" }],
		training_protocol: { epochs: 1 },
		evaluation_protocol: { metric: "accuracy" },
		train_dataset_binding: { id: "train" },
		evaluation_dataset_bindings: [{ id: "test" }],
		metric_bindings: [{ id: "accuracy" }],
		raw_evidence: [{ path: `${id}/metrics.json` }],
		artifact_ids: [artifactId],
		pass_condition: `${managedCompletion([artifactId])} Require metric, log, and evidence files.`,
	};
}

describe("native experiment protocol", () => {
	const cleanups: Array<() => void> = [];

	afterEach(() => {
		while (cleanups.length > 0) {
			cleanups.pop()?.();
		}
	});

	it("canonicalizes objects without reordering semantic arrays", () => {
		const left = canonicalJsonBytes({ z: 1, nested: { b: 2, a: 1 }, order: ["b", "a"] });
		const right = canonicalJsonBytes({ order: ["b", "a"], nested: { a: 1, b: 2 }, z: 1 });
		expect(left).toEqual(right);
		expect(left.at(-1)).toBe(10);
		expect(() => canonicalJsonBytes({ invalid: Number.NaN })).toThrow("non-finite");
		expect(canonicalDigest(JSON.parse(left.toString("utf-8")))).toMatch(/^[a-f0-9]{64}$/);
	});

	it("keeps the journal chained and the current projection atomic", () => {
		const runDir = mkdtempSync(join(tmpdir(), "xlab-native-protocol-"));
		cleanups.push(() => rmSync(runDir, { recursive: true, force: true }));
		const policyDigest = digest("policy");
		const store = new ExperimentProtocolStore(
			runDir,
			(() => {
				let tick = 0;
				return () => `2026-08-08T00:00:0${tick++}.000Z`;
			})(),
		);
		store.initialize({ runId: "run-1", ideaDigest: digest("idea"), policyDigest });
		const staleProjection = store.acquireOwner("owner-1", "2026-08-08T01:00:00.000Z");
		const renewed = store.acquireOwner("owner-1", "2026-08-08T02:00:00.000Z");
		expect(renewed.owner?.generation).toBe(1);
		store.registerChild(child("run-1"));
		const created = store.createAssignment(assignment("run-1", policyDigest), "secret-token");
		expect(store.createAssignment(assignment("run-1", policyDigest), "secret-token")).toEqual(created);

		const snapshot = store.read();
		expect(snapshot.last_event_sequence).toBe(4);
		expect(snapshot.assignments["assignment-1"].status).toBe("active");
		const events = readFileSync(store.eventsPath, "utf-8")
			.trim()
			.split("\n")
			.map((line) => JSON.parse(line));
		expect(events.map((event) => event.sequence)).toEqual([1, 2, 3, 4]);
		expect(events[1].previous_digest).toBe(events[0].digest);
		expect(events[2].previous_digest).toBe(events[1].digest);
		expect(events[3].previous_digest).toBe(events[2].digest);

		writeFileSync(store.protocolPath, canonicalJsonBytes(staleProjection));
		const recovered = store.read();
		expect(recovered.last_event_sequence).toBe(4);
		expect(recovered.assignments["assignment-1"].status).toBe("active");

		writeFileSync(store.protocolPath, "{broken", "utf-8");
		expect(store.read()).toEqual(recovered);
		rmSync(store.protocolPath);
		expect(store.read()).toEqual(recovered);
	});

	it("binds terminal recovery to a durable publication marker", () => {
		const createCompletedStore = (suffix: string) => {
			const runDir = mkdtempSync(join(tmpdir(), `xlab-native-publication-${suffix}-`));
			cleanups.push(() => rmSync(runDir, { recursive: true, force: true }));
			const store = new ExperimentProtocolStore(runDir);
			store.initialize({ runId: `run-${suffix}`, ideaDigest: digest("idea"), policyDigest: digest("policy") });
			for (const stage of ["prepare", "code", "science", "finalize"] as const) {
				store.start(stage);
				store.completeStage(stage);
			}
			return store;
		};

		const ordinary = createCompletedStore("ordinary");
		ordinary.finish("incomplete", "scientific result is incomplete");
		expect(() => ordinary.finish("success", "must not promote", undefined, true)).toThrow("already terminal");

		const publication = createCompletedStore("indexing");
		publication.finish("success", "stages complete");
		const marked = publication.finish("incomplete", "artifact index unavailable", undefined, true);
		expect(marked.publication_recovery_reason).toBe("artifact index unavailable");
		expect(publication.read().publication_recovery_reason).toBe("artifact index unavailable");
		const recovered = publication.finish("success", "index recovered", undefined, true);
		expect(recovered.publication_recovery_reason).toBeUndefined();
		expect(publication.read().publication_recovery_reason).toBeUndefined();
	});

	it("fails closed for corrupted, torn, and conflicting journal state", () => {
		const createStore = (suffix: string) => {
			const runDir = mkdtempSync(join(tmpdir(), `xlab-native-journal-${suffix}-`));
			cleanups.push(() => rmSync(runDir, { recursive: true, force: true }));
			const store = new ExperimentProtocolStore(runDir);
			store.initialize({ runId: "run-1", ideaDigest: digest("idea"), policyDigest: digest("policy") });
			store.acquireOwner("owner-1", "2099-08-08T01:00:00.000Z");
			store.registerChild(child("run-1"));
			return store;
		};

		const corrupted = createStore("corrupt");
		const events = readFileSync(corrupted.eventsPath, "utf-8")
			.trimEnd()
			.split("\n")
			.map((line) => JSON.parse(line));
		events[1].previous_digest = digest("wrong");
		writeFileSync(corrupted.eventsPath, `${events.map((event) => JSON.stringify(event)).join("\n")}\n`);
		expect(() => corrupted.read()).toThrow("digest chain");

		const torn = createStore("torn");
		writeFileSync(torn.eventsPath, readFileSync(torn.eventsPath, "utf-8").trimEnd());
		expect(() => torn.read()).toThrow("torn final record");

		const conflicting = createStore("conflict");
		const projection = JSON.parse(readFileSync(conflicting.protocolPath, "utf-8"));
		projection.status = "failed";
		writeFileSync(conflicting.protocolPath, canonicalJsonBytes(projection));
		expect(() => conflicting.read()).toThrow("does not match its journal");
	});

	it("accepts identical replay and rejects conflicting or late submissions", () => {
		const runDir = mkdtempSync(join(tmpdir(), "xlab-native-submission-"));
		cleanups.push(() => rmSync(runDir, { recursive: true, force: true }));
		const policyDigest = digest("policy");
		const store = new ExperimentProtocolStore(runDir);
		store.initialize({ runId: "run-1", ideaDigest: digest("idea"), policyDigest });
		store.registerChild(child("run-1"));
		const record = store.createAssignment(assignment("run-1", policyDigest), "secret-token");
		const value = submission(record);

		expect(store.acceptSubmission(value)).toMatchObject({ ok: true, disposition: "accepted" });
		expect(store.acceptSubmission(value)).toMatchObject({ ok: true, disposition: "idempotent" });

		const assignmentPath = join(store.assignmentsDir, "assignment-1.json");
		const acceptedPath = join(store.acceptedDir, "assignment-1.json");
		const submissionPath = join(store.submissionsDir, "submission-1.json");
		const expectedAssignment = readFileSync(assignmentPath);
		const expectedAccepted = readFileSync(acceptedPath);
		const expectedSubmission = readFileSync(submissionPath);
		rmSync(assignmentPath);
		rmSync(acceptedPath);
		rmSync(submissionPath);
		store.read();
		expect(readFileSync(assignmentPath)).toEqual(expectedAssignment);
		expect(readFileSync(acceptedPath)).toEqual(expectedAccepted);
		expect(readFileSync(submissionPath)).toEqual(expectedSubmission);
		expect(JSON.parse(expectedSubmission.toString("utf-8")).assignment_token).toBe("[redacted]");
		expect(readFileSync(store.eventsPath, "utf-8")).not.toContain("secret-token");

		expect(
			store.acceptSubmission(
				submission(record, { artifact_role: "worker_result", status: "success", changed: true }),
			),
		).toMatchObject({
			ok: false,
			error: expect.stringContaining("Conflicting replay"),
		});

		const second = store.createAssignment(
			{ ...assignment("run-1", policyDigest), assignment_id: "assignment-2" },
			"secret-token",
		);
		store.cancel("user requested cancellation");
		expect(store.cancel("duplicate cancellation").terminal_generation).toBeDefined();
		expect(JSON.parse(readFileSync(join(store.assignmentsDir, "assignment-2.json"), "utf-8")).status).toBe(
			"cancelled",
		);
		expect(store.acceptSubmission({ ...submission(second), assignment_id: "assignment-2" })).toMatchObject({
			ok: false,
		});
	});

	it("serializes mutations from independent store instances", async () => {
		const runDir = mkdtempSync(join(tmpdir(), "xlab-native-concurrent-"));
		cleanups.push(() => rmSync(runDir, { recursive: true, force: true }));
		const policyDigest = digest("policy");
		const first = new ExperimentProtocolStore(runDir);
		const second = new ExperimentProtocolStore(runDir);
		first.initialize({ runId: "run-1", ideaDigest: digest("idea"), policyDigest });

		const children = Array.from({ length: 20 }, (_, index) => ({
			...child("run-1"),
			child_id: `worker-${index}`,
			session_id: `session-${index}`,
			session_file: `/sessions/session-${index}.jsonl`,
			work_unit: `unit-${index}`,
		}));
		await Promise.all(
			children.map(
				(value, index) =>
					new Promise<void>((resolvePromise, reject) => {
						setImmediate(() => {
							try {
								(index % 2 === 0 ? first : second).registerChild(value);
								resolvePromise();
							} catch (error) {
								reject(error);
							}
						});
					}),
			),
		);

		const snapshot = first.read();
		expect(Object.keys(snapshot.children)).toHaveLength(children.length);
		const events = readFileSync(first.eventsPath, "utf-8")
			.trim()
			.split("\n")
			.map((line) => JSON.parse(line));
		expect(events.map((event) => event.sequence)).toEqual(
			Array.from({ length: children.length }, (_, index) => index + 1),
		);
		for (let index = 1; index < events.length; index += 1) {
			expect(events[index].previous_digest).toBe(events[index - 1].digest);
		}
	});

	it("rejects a submission callback fenced by a stale owner generation", () => {
		const runDir = mkdtempSync(join(tmpdir(), "xlab-native-stale-owner-"));
		cleanups.push(() => rmSync(runDir, { recursive: true, force: true }));
		const policyDigest = digest("policy");
		let now = "2026-08-08T00:00:00.000Z";
		const store = new ExperimentProtocolStore(runDir, () => now);
		store.initialize({ runId: "run-1", ideaDigest: digest("idea"), policyDigest });
		const original = store.acquireOwner("owner-a", "2026-08-08T00:00:00.001Z");
		store.registerChild(child("run-1"));
		const record = store.createAssignment(assignment("run-1", policyDigest), "secret-token");
		now = "2026-08-08T00:00:00.002Z";
		store.acquireOwner("owner-b", "2026-08-08T01:00:00.000Z");

		expect(
			store.acceptSubmission(submission(record), {
				owner_id: original.owner!.owner_id,
				generation: original.owner!.generation,
			}),
		).toMatchObject({ ok: false, error: expect.stringContaining("ownership fence") });
		expect(store.read().assignments[record.assignment_id].status).toBe("active");
	});

	it("reconciles crashes at durable publication boundaries", () => {
		const points = [
			"after_assignment_record",
			"after_submission_record",
			"after_accepted_record",
			"after_assignment_acceptance_record",
			"after_journal_append",
			"after_projection_write",
		] as const;
		for (const point of points) {
			const runDir = mkdtempSync(join(tmpdir(), `xlab-native-fault-${point}-`));
			cleanups.push(() => rmSync(runDir, { recursive: true, force: true }));
			const policyDigest = digest("policy");
			let armed = false;
			const crashing = new ExperimentProtocolStore(runDir, undefined, (current) => {
				if (armed && current === point) {
					armed = false;
					throw new Error(`fault:${point}`);
				}
			});
			crashing.initialize({ runId: "run-1", ideaDigest: digest("idea"), policyDigest });
			crashing.registerChild(child("run-1"));
			armed = true;
			let record: ExperimentAssignment;
			if (
				point === "after_assignment_record" ||
				point === "after_journal_append" ||
				point === "after_projection_write"
			) {
				expect(() => crashing.createAssignment(assignment("run-1", policyDigest), "secret-token")).toThrow(
					`fault:${point}`,
				);
				const recovered = new ExperimentProtocolStore(runDir);
				const snapshot = recovered.read();
				if (point === "after_assignment_record") {
					expect(snapshot.assignments["assignment-1"]).toBeUndefined();
					record = recovered.createAssignment(assignment("run-1", policyDigest), "secret-token");
				} else {
					expect(snapshot.assignments["assignment-1"]).toBeDefined();
					record = snapshot.assignments["assignment-1"];
				}
				expect(record.status).toBe("active");
				continue;
			}
			record = crashing.createAssignment(assignment("run-1", policyDigest), "secret-token");
			expect(() => crashing.acceptSubmission(submission(record))).toThrow(`fault:${point}`);
			const recovered = new ExperimentProtocolStore(runDir);
			const snapshot = recovered.read();
			expect(snapshot.assignments["assignment-1"].status).toBe("active");
			expect(recovered.acceptSubmission(submission(snapshot.assignments["assignment-1"]))).toMatchObject({
				ok: true,
				disposition: "accepted",
			});
		}
	});

	it("reconciles cancellation intent across every publication boundary", () => {
		const points = [
			"after_cancellation_intent",
			"after_cancellation_assignments",
			"after_journal_append",
			"after_projection_write",
		] as const;
		for (const point of points) {
			const runDir = mkdtempSync(join(tmpdir(), `xlab-native-cancel-${point}-`));
			cleanups.push(() => rmSync(runDir, { recursive: true, force: true }));
			const policyDigest = digest("policy");
			let armed = false;
			const crashing = new ExperimentProtocolStore(runDir, undefined, (current) => {
				if (armed && current === point) {
					armed = false;
					throw new Error(`fault:${point}`);
				}
			});
			crashing.initialize({ runId: "run-1", ideaDigest: digest("idea"), policyDigest });
			crashing.registerChild(child("run-1"));
			const record = crashing.createAssignment(assignment("run-1", policyDigest), "secret-token");
			armed = true;
			expect(() => crashing.cancel("stop")).toThrow(`fault:${point}`);

			const recovered = new ExperimentProtocolStore(runDir);
			const snapshot = recovered.read();
			expect(snapshot.status).toBe("cancelled");
			expect(snapshot.terminal_generation).toBe(1);
			expect(snapshot.assignments[record.assignment_id].status).toBe("cancelled");
			expect(
				JSON.parse(readFileSync(join(recovered.assignmentsDir, `${record.assignment_id}.json`), "utf-8")).status,
			).toBe("cancelled");
			expect(recovered.acceptSubmission(submission(record))).toMatchObject({ ok: false });
		}
	});

	it("rejects invalid tokens and stale assignment identity", () => {
		const runDir = mkdtempSync(join(tmpdir(), "xlab-native-token-"));
		cleanups.push(() => rmSync(runDir, { recursive: true, force: true }));
		const policyDigest = digest("policy");
		const store = new ExperimentProtocolStore(runDir);
		store.initialize({ runId: "run-1", ideaDigest: digest("idea"), policyDigest });
		store.registerChild(child("run-1"));
		const record = store.createAssignment(assignment("run-1", policyDigest), "secret-token");
		expect(store.acceptSubmission({ ...submission(record), assignment_token: "wrong" })).toMatchObject({
			ok: false,
			error: expect.stringContaining("token"),
		});
		expect(store.acceptSubmission({ ...submission(record), generation: 2 })).toMatchObject({
			ok: false,
			error: expect.stringContaining("generation"),
		});
	});

	it("enforces realpath containment and rejects symlink escapes", () => {
		const root = mkdtempSync(join(tmpdir(), "xlab-native-scope-"));
		cleanups.push(() => rmSync(root, { recursive: true, force: true }));
		const allowed = join(root, "allowed");
		const outside = join(root, "outside");
		mkdirSync(allowed);
		mkdirSync(outside);
		writeFileSync(join(allowed, "ok.txt"), "ok");
		writeFileSync(join(outside, "secret.txt"), "secret");
		symlinkSync(join(outside, "secret.txt"), join(allowed, "escape.txt"));

		expect(resolveContainedPath(allowed, "ok.txt")).toBe(join(allowed, "ok.txt"));
		expect(() => resolveContainedPath(allowed, "../outside/secret.txt")).toThrow("escapes");
		expect(() => resolveContainedPath(allowed, "escape.txt")).toThrow("escapes");
	});

	it("validates canonical prepare, code, and science plans", () => {
		expect(validatePreparePlan(preparePlan())).toBeUndefined();
		expect(validatePreparePlan({ stage: "prepare", work_units: [{ id: "repos" }] })).toContain("exactly");
		const incompletePrepare = preparePlan();
		incompletePrepare.work_units[0].done_condition = "artifact ledger only";
		expect(validatePreparePlan(incompletePrepare)).toContain("managed artifact");

		const implementation = codeUnit("implementation", "code.implementation.handoff");
		const smoke = {
			...codeUnit("final_integration_smoke", "code.final_integration_smoke.handoff"),
			needs: ["implementation"],
			evidence: ["integrated", "component_disabled"],
		};
		expect(validateCodePlan({ stage: "code", work_units: [implementation, smoke] }, components)).toBeUndefined();
		expect(
			validateCodePlan(
				{ stage: "code", work_units: [{ ...smoke, needs: [], verify_command: "dry-run" }] },
				components,
			),
		).toContain("evidentiary");

		const scienceUnits = [
			scienceCondition("reference"),
			scienceCondition("without-memory", [components[0]]),
			scienceCondition("without-verifier", [components[1]]),
		];
		expect(validateSciencePlan({ stage: "science", work_units: scienceUnits }, components)).toBeUndefined();
		expect(
			validateSciencePlan(
				{ stage: "science", work_units: [scienceUnits[1], scienceUnits[0], scienceUnits[2]] },
				components,
			),
		).toContain("begin");
	});

	it("enforces fixed reviewer ordering and statistical authority", () => {
		expect(
			validateReviewMatrix(
				"code",
				CODE_REVIEW_ROLES.map((role) => review(role)),
			),
		).toBeUndefined();
		const conditions = [
			{
				id: "reference",
				kind: "all_components_reference" as const,
				enabled_components: components,
				disabled_components: [],
			},
			{
				id: "without-memory",
				kind: "component_disabled" as const,
				enabled_components: [components[1]],
				disabled_components: [components[0]],
			},
			{
				id: "without-verifier",
				kind: "component_disabled" as const,
				enabled_components: [components[0]],
				disabled_components: [components[1]],
			},
		];
		const componentResults = {
			[components[0]]: {
				condition_id: "without-memory",
				enabled_components: [components[1]],
				disabled_components: [components[0]],
				classification: "supported",
			},
			[components[1]]: {
				condition_id: "without-verifier",
				enabled_components: [components[0]],
				disabled_components: [components[1]],
				classification: "supported",
			},
		};
		const scienceReports = SCIENCE_REVIEW_ROLES.map((role) =>
			review(
				role,
				role === "statistical_interpretation" ? { component_results: componentResults } : { checked: true },
			),
		);
		expect(
			validateReviewMatrix("science", scienceReports, {
				component_names: components,
				science_conditions: conditions,
			}),
		).toBeUndefined();
		expect(
			validateReviewMatrix(
				"code",
				[...CODE_REVIEW_ROLES].reverse().map((role) => review(role)),
			),
		).toContain(CODE_REVIEW_ROLES[0]);
		expect(
			validateReviewMatrix(
				"science",
				SCIENCE_REVIEW_ROLES.map((role) => review(role)),
				{ component_names: components, science_conditions: conditions },
			),
		).toContain("component_results");
		const failed = review(CODE_REVIEW_ROLES[0]);
		failed.verdict = "FAIL";
		expect(
			validateReviewMatrix("code", [failed, ...CODE_REVIEW_ROLES.slice(1).map((role) => review(role))]),
		).toContain("inconsistent");

		const invalidVerdict = review(CODE_REVIEW_ROLES[0]) as unknown as Record<string, unknown>;
		invalidVerdict.verdict = "APPROVE";
		expect(
			validateReviewMatrix("code", [
				invalidVerdict,
				...CODE_REVIEW_ROLES.slice(1).map((role) => review(role)),
			] as ExperimentReviewReport[]),
		).toContain("incomplete reviewer report contract");
		for (const issue of [null, "bad", 1]) {
			const invalidIssue = review(CODE_REVIEW_ROLES[0]) as unknown as Record<string, unknown>;
			invalidIssue.verdict = "FAIL";
			invalidIssue.issues = [issue];
			expect(
				validateReviewMatrix("code", [
					invalidIssue,
					...CODE_REVIEW_ROLES.slice(1).map((role) => review(role)),
				] as ExperimentReviewReport[]),
			).toContain("typed structured issues");
		}
	});
});
