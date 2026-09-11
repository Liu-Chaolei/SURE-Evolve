export const REPOSITORY_URL = "https://github.com/daqige/XLab";
export const REPOSITORY_BRANCH = "main";

export function sourceUrl(path: string): string {
  return `${REPOSITORY_URL}/blob/${REPOSITORY_BRANCH}/${path}`;
}

export interface SourceReference {
  path: string;
  symbol?: string;
}

export interface SiteClaim {
  title: string;
  body: string;
  source: SourceReference;
}

export interface PipelineStage {
  label: string;
  command: string;
  skillName: string;
  artifact: string;
  source: SourceReference;
}

export const navItems = [
  { label: "Home", href: "/" },
  { label: "Docs", href: "/docs/" },
] as const;

export const sourceQuickstart = ["npm install --ignore-scripts", "./pi-test.sh"] as const;

export const pipeline: readonly PipelineStage[] = [
  { label: "Papers", command: "/xlab collect-papers", skillName: "paper_collect", artifact: "Paper set", source: { path: "xlab/skills/paper_collect/xlab.skill.json" } },
  { label: "Knowledge graph", command: "/xlab build-knowledge-graph", skillName: "knowledge_graph", artifact: "Method graph", source: { path: "xlab/skills/knowledge_graph/xlab.skill.json" } },
  { label: "Survey", command: "/xlab write-literature-survey", skillName: "literature_survey", artifact: "Citation trace", source: { path: "xlab/skills/literature_survey/xlab.skill.json" } },
  { label: "Ideas", command: "/xlab generate-research-ideas", skillName: "research_idea", artifact: "Research idea", source: { path: "xlab/skills/research_idea/xlab.skill.json" } },
  { label: "Novelty analysis", command: "/xlab check-idea-novelty", skillName: "novelty_check", artifact: "Overlap-risk report", source: { path: "xlab/skills/novelty_check/xlab.skill.json" } },
] as const;

export const reliabilityClaims: readonly SiteClaim[] = [
  { title: "Durable runs", body: "Run metadata, display state, workflow state, and an append-only event stream are written under .xlab/runs.", source: { path: "packages/coding-agent/src/core/xlab/run-manager.ts", symbol: "XlabRunManager" } },
  { title: "Typed artifacts", body: "Every accepted output carries a schema version, producer, validation metadata, and parent relationships.", source: { path: "packages/coding-agent/src/core/xlab/artifact-store.ts", symbol: "StoredArtifactMetadata" } },
  { title: "Content addressed", body: "Files and directories are stored by SHA-256 digest, with symlinks rejected from artifact payloads.", source: { path: "packages/coding-agent/src/core/xlab/artifact-store.ts", symbol: "hashPath" } },
  { title: "Acceptance gates", body: "Skill hooks validate task-specific requirements before the harness accepts completion.", source: { path: "README.md", symbol: "Hooks And Gates" } },
  { title: "Checkpoint and resume", body: "A run can persist checkpoint hints and resume from its recorded state, while keeping a resume count.", source: { path: "packages/coding-agent/src/core/xlab/run-manager.ts", symbol: "resumeRun" } },
  { title: "Workspace lineage", body: "Versioned graph, survey, and idea handles preserve parent links and surface stale-dependency warnings.", source: { path: "packages/coding-agent/src/core/xlab/workspace-manager.ts", symbol: "staleWarnings" } },
] as const;
