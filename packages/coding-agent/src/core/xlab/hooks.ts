import { spawn } from "node:child_process";
import { extname } from "node:path";
import { createJiti } from "jiti/static";
import { isBunBinary } from "../../config.ts";
import { resolveXlabPackagePath } from "./manifest.ts";
import { getXlabHookAliases, getXlabHookVirtualModules } from "./module-loader.ts";
import type { XlabHookContext, XlabHookDeclaration, XlabHookPoint, XlabHookResult, XlabSkillPackage } from "./types.ts";

type HookFunction = (context: XlabHookContext) => XlabHookResult | Promise<XlabHookResult | undefined> | undefined;

export interface XlabGateResult {
	ok: boolean;
	message?: string;
	repair?: string;
	diagnostics?: unknown;
	state_patch?: unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function normalizeHookResult(value: unknown): XlabHookResult {
	if (!isRecord(value)) {
		return {};
	}
	return {
		ok: typeof value.ok === "boolean" ? value.ok : undefined,
		message: typeof value.message === "string" ? value.message : undefined,
		repair: typeof value.repair === "string" ? value.repair : undefined,
		patch: value.patch,
		artifacts: value.artifacts,
		diagnostics: value.diagnostics,
		state_patch: value.state_patch,
	};
}

function toGateFailure(result: XlabHookResult, fallbackMessage: string): XlabGateResult | undefined {
	if (result.ok !== false && !result.repair) {
		return undefined;
	}
	return {
		ok: false,
		message: result.message ?? fallbackMessage,
		repair: result.repair ?? result.message ?? fallbackMessage,
		diagnostics: result.diagnostics,
		state_patch: result.state_patch,
	};
}

export class XlabHookRunner {
	private skillPackage: XlabSkillPackage;
	private jiti = createJiti(import.meta.url, {
		moduleCache: false,
		...(isBunBinary
			? { virtualModules: getXlabHookVirtualModules(), tryNative: false }
			: { alias: getXlabHookAliases() }),
	});

	constructor(skillPackage: XlabSkillPackage) {
		this.skillPackage = skillPackage;
	}

	async run(point: XlabHookPoint, context: Omit<XlabHookContext, "point">): Promise<XlabGateResult> {
		const declarations = this.skillPackage.manifest.hooks?.[point] ?? [];
		let statePatch: unknown;
		for (const declaration of declarations) {
			const result = await this.runHook(point, declaration, context);
			if (result.state_patch !== undefined) {
				statePatch = result.state_patch;
			}
			const failure = toGateFailure(result, `XLab ${point} hook rejected the action.`);
			if (failure && declaration.blocking !== false) {
				return { ...failure, state_patch: failure.state_patch ?? statePatch };
			}
		}
		return { ok: true, state_patch: statePatch };
	}

	private async runHook(
		point: XlabHookPoint,
		declaration: XlabHookDeclaration,
		context: Omit<XlabHookContext, "point">,
	): Promise<XlabHookResult> {
		if (declaration.kind === "command") {
			return this.runCommandHook(point, declaration, context);
		}
		if (!declaration.module) {
			return {
				ok: false,
				message: `XLab ${point} module hook is missing its module path.`,
				repair: "Set hooks.<point>[].module to a file inside the skill package.",
			};
		}
		const hookPath = resolveXlabPackagePath(this.skillPackage.packageDir, declaration.module);
		if (!hookPath) {
			return {
				ok: false,
				message: `Hook module path escapes skill package: ${declaration.module}`,
				repair: "Fix xlab.skill.json so hook module paths stay inside the skill package.",
			};
		}

		try {
			const loaded = await this.jiti.import(hookPath);
			const moduleValue = loaded as Record<string, unknown>;
			const handlerName = declaration.handler ?? point;
			const handler =
				(typeof moduleValue[handlerName] === "function" ? moduleValue[handlerName] : undefined) ??
				(typeof moduleValue.default === "function" ? moduleValue.default : undefined);
			if (!handler) {
				return {
					ok: false,
					message: `Hook handler not found: ${handlerName}`,
					repair: `Export a function named "${handlerName}" from ${declaration.module}, or provide a default export.`,
				};
			}

			const value = await (handler as HookFunction)({
				...context,
				point,
			});
			return normalizeHookResult(value);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			return {
				ok: false,
				message: `XLab ${point} hook failed: ${message}`,
				repair: `Fix the hook failure in ${declaration.module} and retry the action.`,
			};
		}
	}

	private async runCommandHook(
		point: XlabHookPoint,
		declaration: XlabHookDeclaration,
		context: Omit<XlabHookContext, "point">,
	): Promise<XlabHookResult> {
		if (!declaration.command) {
			return {
				ok: false,
				message: `XLab ${point} command hook is missing its command path.`,
				repair: "Set hooks.<point>[].command to a file inside the skill package.",
			};
		}
		const hookPath = resolveXlabPackagePath(this.skillPackage.packageDir, declaration.command);
		if (!hookPath) {
			return {
				ok: false,
				message: `Hook command path escapes skill package: ${declaration.command}`,
				repair: "Keep command hook entrypoints inside the skill package.",
			};
		}
		const extension = extname(hookPath).toLowerCase();
		const executable =
			extension === ".py"
				? (context.run.environment?.executable ?? "python")
				: extension === ".js" || extension === ".mjs"
					? process.execPath
					: hookPath;
		const args = [...(executable === hookPath ? [] : [hookPath]), ...(declaration.args ?? [])];
		const payload = JSON.stringify({ ...context, point });
		const timeoutMs = declaration.timeoutMs ?? 30_000;

		return new Promise((resolve) => {
			const child = spawn(executable, args, {
				cwd: this.skillPackage.packageDir,
				env: {
					...process.env,
					XLAB_HOOK_POINT: point,
					XLAB_RUN_ID: context.run.runId,
				},
				stdio: ["pipe", "pipe", "pipe"],
			});
			let stdout = "";
			let stderr = "";
			let settled = false;
			const finish = (result: XlabHookResult): void => {
				if (settled) {
					return;
				}
				settled = true;
				clearTimeout(timer);
				resolve(result);
			};
			const timer = setTimeout(() => {
				child.kill("SIGTERM");
				finish({
					ok: false,
					message: `XLab ${point} command hook timed out after ${timeoutMs}ms.`,
					repair: `Fix or increase timeout_ms for ${declaration.command}.`,
				});
			}, timeoutMs);
			child.stdout.setEncoding("utf-8");
			child.stderr.setEncoding("utf-8");
			child.stdout.on("data", (chunk: string) => {
				stdout += chunk;
			});
			child.stderr.on("data", (chunk: string) => {
				stderr += chunk;
			});
			child.on("error", (error) => {
				finish({
					ok: false,
					message: `XLab ${point} command hook failed to start: ${error.message}`,
					repair: `Fix the command hook runtime for ${declaration.command}.`,
				});
			});
			child.on("close", (code) => {
				if (settled) {
					return;
				}
				const output = stdout.trim();
				if (code !== 0) {
					finish({
						ok: false,
						message: `XLab ${point} command hook exited with code ${code}: ${stderr.trim() || output}`,
						repair: `Fix ${declaration.command} and retry the action.`,
					});
					return;
				}
				try {
					finish(normalizeHookResult(output ? JSON.parse(output) : {}));
				} catch (error) {
					finish({
						ok: false,
						message: `XLab ${point} command hook returned invalid JSON: ${
							error instanceof Error ? error.message : String(error)
						}`,
						repair: `Make ${declaration.command} write one JSON object to stdout.`,
					});
				}
			});
			child.stdin.end(payload);
		});
	}
}
