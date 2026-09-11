import type { XlabSkillPackage } from "./types.ts";

export interface XlabProductCommand {
	task: string;
	skillName: string;
	description: string;
}

export interface XlabControlCommand {
	task: string;
	description: string;
}

export const XLAB_PRODUCT_COMMANDS = [
	{
		task: "collect-papers",
		skillName: "paper_collect",
		description: "Find, download, and organize research papers.",
	},
	{
		task: "build-knowledge-graph",
		skillName: "knowledge_graph",
		description: "Build a traceable method knowledge graph from papers.",
	},
	{
		task: "write-literature-survey",
		skillName: "literature_survey",
		description: "Write a cited literature survey from papers and graph evidence.",
	},
	{
		task: "generate-research-ideas",
		skillName: "research_idea",
		description: "Generate evidence-grounded, experiment-ready research ideas.",
	},
	{
		task: "check-idea-novelty",
		skillName: "novelty_check",
		description: "Check an idea for prior work, overlap, and uncertainty.",
	},
	{
		task: "run-research-workflow",
		skillName: "research_workflow",
		description: "Run the complete papers-to-novelty research workflow.",
	},
	{
		task: "build-research-dataset",
		skillName: "data_workflow",
		description: "Build and validate a provenance-preserving research dataset.",
	},
	{
		task: "run-experiment",
		skillName: "run_experiment",
		description: "Prepare, execute, and review a research experiment.",
	},
	{
		task: "run-sure-master",
		skillName: "sure_master",
		description: "Run SURE-Master speech-model self-evolution with XLab research feedback.",
	},
	{
		task: "evaluate-model",
		skillName: "model_eval",
		description: "Evaluate a model on a frozen dataset and metric contract.",
	},
	{
		task: "create-model-mcp-server",
		skillName: "model_mcp",
		description: "Package and validate a model as an MCP server.",
	},
	{
		task: "reproduce-paper",
		skillName: "paper_reproduction",
		description: "Reproduce a paper with official code and explicit metrics.",
	},
	{
		task: "conduct-research-discussion",
		skillName: "research_discussion",
		description: "Conduct a structured, evidence-linked research discussion.",
	},
	{
		task: "build-scholar-profile",
		skillName: "scholar_profile",
		description: "Build an evidence-backed profile of a researcher.",
	},
	{
		task: "manage-research-communications",
		skillName: "science_gateway",
		description: "Manage local research threads, messages, and synchronization.",
	},
] as const satisfies readonly XlabProductCommand[];

export type XlabProductTask = (typeof XLAB_PRODUCT_COMMANDS)[number]["task"];

export const XLAB_CONTROL_COMMANDS = [
	{
		task: "show-help",
		description: "Show all XLab commands or detailed help for one task.",
	},
	{
		task: "check-xlab-setup",
		description: "Check XLab configuration, packages, and required integrations.",
	},
	{
		task: "configure-xlab",
		description: "Configure XLab runtime and research integrations.",
	},
	{
		task: "list-workspaces",
		description: "List available XLab research workspaces.",
	},
	{
		task: "show-workspace",
		description: "Show details for an XLab workspace.",
	},
	{
		task: "select-workspace",
		description: "Select the active XLab workspace.",
	},
	{ task: "list-runs", description: "List recent XLab runs." },
	{
		task: "show-run",
		description: "Show the state and workflow of an XLab run.",
	},
	{ task: "resume-run", description: "Resume an incomplete XLab run." },
	{ task: "cancel-run", description: "Cancel an XLab run." },
	{
		task: "show-artifact",
		description: "Show metadata and lineage for an XLab artifact.",
	},
] as const satisfies readonly XlabControlCommand[];

export const XLAB_CONTROL_WORDS = XLAB_CONTROL_COMMANDS.map((entry) => entry.task);

const PRODUCT_BY_TASK = new Map<string, XlabProductCommand>(XLAB_PRODUCT_COMMANDS.map((entry) => [entry.task, entry]));
const PRODUCT_BY_SKILL = new Map<string, XlabProductCommand>(
	XLAB_PRODUCT_COMMANDS.map((entry) => [entry.skillName, entry]),
);

export function xlabProductCommandForTask(task: string): XlabProductCommand | undefined {
	return PRODUCT_BY_TASK.get(task);
}

export function xlabProductCommandForSkill(skillName: string): XlabProductCommand | undefined {
	return PRODUCT_BY_SKILL.get(skillName);
}

export function resolveXlabProductPackage(packages: XlabSkillPackage[], task: string): XlabSkillPackage | undefined {
	const product = xlabProductCommandForTask(task);
	return product ? packages.find((candidate) => candidate.manifest.name === product.skillName) : undefined;
}

export function validateXlabProductCommands(): string[] {
	const errors: string[] = [];
	const tasks = new Set<string>();
	const skills = new Set<string>();
	const controls = new Set<string>();
	const validTask = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/;
	for (const entry of XLAB_CONTROL_COMMANDS) {
		if (!validTask.test(entry.task)) {
			errors.push(`Invalid XLab control command name: ${entry.task}`);
		}
		if (!entry.description.trim()) {
			errors.push(`Missing XLab control command description: ${entry.task}`);
		}
		if (controls.has(entry.task)) {
			errors.push(`Duplicate XLab control command name: ${entry.task}`);
		}
		controls.add(entry.task);
	}
	for (const entry of XLAB_PRODUCT_COMMANDS) {
		if (!validTask.test(entry.task)) {
			errors.push(`Invalid XLab product task name: ${entry.task}`);
		}
		if (!entry.description.trim()) {
			errors.push(`Missing XLab product task description: ${entry.task}`);
		}
		if (!entry.skillName.trim()) {
			errors.push(`Missing XLab product skill mapping: ${entry.task}`);
		}
		if (tasks.has(entry.task)) {
			errors.push(`Duplicate XLab product task name: ${entry.task}`);
		}
		if (skills.has(entry.skillName)) {
			errors.push(`Duplicate XLab product skill mapping: ${entry.skillName}`);
		}
		if (controls.has(entry.task)) {
			errors.push(`XLab product task conflicts with a control word: ${entry.task}`);
		}
		tasks.add(entry.task);
		skills.add(entry.skillName);
	}
	return errors;
}
