import { readFile, readdir } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { commandGroups, commands } from "../src/data/commands.ts";
import { forbiddenClaims } from "../src/data/claims.ts";
import { pipeline, sourceQuickstart } from "../src/data/site.ts";
import {
  XLAB_CONTROL_COMMANDS,
  XLAB_PRODUCT_COMMANDS,
  validateXlabProductCommands,
} from "../../packages/coding-agent/src/core/xlab/product-commands.ts";

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(webRoot, "..");

async function filesBelow(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? filesBelow(path) : [path];
  }));
  return nested.flat();
}

export async function validateContent(): Promise<string[]> {
  const errors = validateXlabProductCommands();
  const productByTask = new Map<string, (typeof XLAB_PRODUCT_COMMANDS)[number]>(XLAB_PRODUCT_COMMANDS.map((entry) => [entry.task, entry]));
  const controlByTask = new Map<string, (typeof XLAB_CONTROL_COMMANDS)[number]>(XLAB_CONTROL_COMMANDS.map((entry) => [entry.task, entry]));
  const commandByTask = new Map(commands.map((entry) => [entry.task, entry]));
  const groups = new Set(commandGroups.map((entry) => entry.id));
  const productCommands = commands.filter((entry) => entry.kind === "product");
  const controlCommands = commands.filter((entry) => entry.kind === "control");

  if (commands.length !== 25) errors.push(`Expected 25 curated commands, found ${commands.length}`);
  if (productCommands.length !== 14) errors.push(`Expected 14 product commands, found ${productCommands.length}`);
  if (controlCommands.length !== 11) errors.push(`Expected 11 control commands, found ${controlCommands.length}`);
  if (commandByTask.size !== commands.length) errors.push("Curated command tasks are not unique");
  if (new Set(commands.map((entry) => entry.anchor)).size !== commands.length) errors.push("Curated command anchors are not unique");

  for (const canonical of [...XLAB_PRODUCT_COMMANDS, ...XLAB_CONTROL_COMMANDS]) {
    if (!commandByTask.has(canonical.task)) errors.push(`Canonical command is missing: ${canonical.task}`);
  }

  for (const command of commands) {
    const product = productByTask.get(command.task);
    const control = controlByTask.get(command.task);
    const canonical = product ?? control;
    if (!canonical) {
      errors.push(`Unknown curated command: ${command.task}`);
      continue;
    }
    if (canonical.description !== command.description) errors.push(`Description differs from canonical catalog: ${command.task}`);
    if (command.anchor !== `command-${command.task}`) errors.push(`Invalid command anchor: ${command.task}`);
    if (!groups.has(command.group)) errors.push(`Invalid command group: ${command.task}`);
    if (!command.usage.trim() || !command.example.trim() || !command.result.trim()) errors.push(`Incomplete command documentation: ${command.task}`);

    if (product) {
      if (command.kind !== "product") errors.push(`Canonical product has the wrong kind: ${command.task}`);
      if (command.skillName !== product.skillName) errors.push(`Product skill mapping differs from canonical catalog: ${command.task}`);
      if (command.sourcePath !== `xlab/skills/${product.skillName}/xlab.skill.json`) errors.push(`Invalid product source path: ${command.task}`);
      if (!command.usage.endsWith("[--workspace <slug>]")) errors.push(`Product usage omits the workspace option: ${command.task}`);

      try {
        const manifest = JSON.parse(await readFile(join(repositoryRoot, command.sourcePath), "utf8")) as {
          schema_version?: string;
          name?: string;
          visibility?: string;
          runtime?: { kind?: string };
          ui?: { argumentHint?: string; arguments?: unknown[] };
        };
        if (manifest.schema_version !== "2") errors.push(`Unsupported manifest schema: ${command.task}`);
        if (manifest.name !== product.skillName) errors.push(`Manifest name mismatch: ${command.task}`);
        const expectedVisibility = product.skillName === "science_gateway" ? "service-control" : "public";
        if (manifest.visibility !== expectedVisibility) errors.push(`Product skill visibility differs from canonical exposure: ${command.task}`);
        if (manifest.runtime?.kind !== command.runtimeKind) errors.push(`Runtime kind differs from manifest: ${command.task}`);
        if ((manifest.ui?.argumentHint ?? "") !== (command.argumentHint ?? "")) errors.push(`Argument hint differs from manifest: ${command.task}`);
        if (!isDeepStrictEqual(manifest.ui?.arguments ?? [], command.arguments)) errors.push(`Argument metadata differs from manifest: ${command.task}`);
      } catch (error) {
        errors.push(`Cannot read product manifest ${command.sourcePath}: ${String(error)}`);
      }
    } else {
      if (command.kind !== "control") errors.push(`Canonical control has the wrong kind: ${command.task}`);
      if (command.sourcePath !== "README.md") errors.push(`Invalid control source path: ${command.task}`);
      if (command.skillName !== undefined || command.runtimeKind !== undefined || command.argumentHint !== undefined || command.arguments.length > 0) errors.push(`Control command declares product metadata: ${command.task}`);
    }
  }

  for (const stage of pipeline) {
    const product = XLAB_PRODUCT_COMMANDS.find((entry) => entry.task === stage.command.replace("/xlab ", ""));
    if (product?.skillName !== stage.skillName) {
      errors.push(`Pipeline command mapping is invalid: ${stage.command}`);
    }
    try {
      const manifest = JSON.parse(await readFile(join(repositoryRoot, stage.source.path), "utf8")) as {
        name?: string;
        visibility?: string;
        artifacts?: Array<{ type?: string }>;
      };
      if (manifest.name !== stage.skillName) errors.push(`Manifest name mismatch: ${stage.source.path}`);
      if (manifest.visibility !== "public") errors.push(`Pipeline skill is not public: ${stage.skillName}`);
      if (!manifest.artifacts?.length) errors.push(`Pipeline skill declares no artifacts: ${stage.skillName}`);
    } catch (error) {
      errors.push(`Cannot read pipeline manifest ${stage.source.path}: ${String(error)}`);
    }
  }

  const readme = await readFile(join(repositoryRoot, "README.md"), "utf8");
  const readmeTasks = new Set([...readme.matchAll(/`\/xlab ([a-z][a-z0-9-]*)/g)].map((match) => match[1]));
  for (const canonical of [...XLAB_PRODUCT_COMMANDS, ...XLAB_CONTROL_COMMANDS]) {
    if (!readmeTasks.has(canonical.task)) errors.push(`Canonical command is absent from README.md: ${canonical.task}`);
  }
  for (const command of controlCommands) {
    if (!readme.includes(`\`${command.usage}\``)) errors.push(`Control usage differs from README.md: ${command.task}`);
  }
  for (const command of sourceQuickstart) {
    if (!readme.includes(command)) errors.push(`Quickstart command is absent from README.md: ${command}`);
  }

  const contentRoots = ["src/pages", "src/components", "src/layouts", "public"];
  for (const root of contentRoots) {
    for (const file of await filesBelow(join(webRoot, root))) {
      const content = (await readFile(file, "utf8")).toLowerCase();
      for (const claim of forbiddenClaims) {
        if (content.includes(claim)) {
          errors.push(`Forbidden claim "${claim}" in ${relative(webRoot, file)}`);
        }
      }
    }
  }

  return errors;
}

const errors = await validateContent();
if (errors.length) {
  console.error(errors.map((error) => `- ${error}`).join("\n"));
  process.exitCode = 1;
} else {
  console.log("Repository-backed website content is valid.");
}
