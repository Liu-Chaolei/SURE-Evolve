# EvoMaster Self-Evolution Guide

EvoMaster self-evolution is an optional post-run wrapper that lets an agent learn from its own completed trajectory. It runs the original agent, analyzes the run records and environment feedback, generates reusable local skills and prompt patches, then runs the same agent again with the evolved overlay.

The feature is designed as a plugin-like layer. Normal EvoMaster runs are unchanged unless you pass `--evolve`.

## Quick Start

From the repository root:

```bash
python run.py \
  --agent minimal \
  --config configs/minimal/gpt-5-example.yaml \
  --task "Discover a pattern: Given sequence 1, 4, 9, 16, 25... find the formula" \
  --run-dir runs/evo_test \
  --evolve \
  --evolve-iterations 2
```

This command performs:

1. `iter_000_baseline`: run the original `minimal` agent once.
2. Analyze the baseline trajectory, logs, tool calls, errors, artifacts, and workspace manifest.
3. Generate an evolution overlay with skills, prompt patches, and tool proposals.
4. `iter_001_evolved`: run the same agent with the evolved overlay.
5. Repeat analysis and overlay generation.
6. `iter_002_evolved`: run the agent again with the second evolved overlay.

## What Gets Evolved

Self-evolution can generate three kinds of outputs:

| Output | Automatically Applied | Purpose |
| --- | --- | --- |
| Skills | Yes | Reusable procedures, debugging workflows, domain facts, and successful strategies extracted from a run. |
| Prompt patches | Yes | Additional instructions appended to an agent's system or user prompt through generated prompt overlay files. |
| Tool proposals | No | JSONL proposals for new tools. Tool code is not generated or enabled automatically because tool implementation has higher long-term risk. |

The generated assets are run-local. EvoMaster does not overwrite your original config, original prompts, or built-in skill directory.

## CLI Options

### `--evolve`

Enables the self-evolution wrapper.

Without this flag, EvoMaster uses the normal execution path.

### `--evolve-iterations N`

Runs `N` evolution rounds after the baseline.

For example, `--evolve-iterations 2` means:

```text
baseline run -> analyze/apply -> evolved run 1 -> analyze/apply -> evolved run 2
```

`--evolve-iterations 0` runs only the baseline inside the evolution run directory and records evolution metadata.

### `--evolve-disable-llm-analyzer`

Disables the LLM analyzer and uses heuristic evolution only.

Use this when you want deterministic fallback behavior, when the analyzer model is unavailable, or when you want to test the wrapper without spending analyzer LLM calls.

### Supported Run Modes

The current implementation supports:

- One task per evolution run.
- Serial execution only. Do not pass `--parallel`.
- Existing agents such as `minimal`, serial `minimal_multi_agent`, `minimal_kaggle`, `x_master`, and `browse_master`, as long as the selected config can run normally without evolution.

Parallel evolution for exp-level parallel agents is intentionally not enabled yet.

## Output Directory Layout

For `--run-dir runs/evo_test`, the important files are:

```text
runs/evo_test/
├── evolution_state.json
├── logs/
│   └── evolution.log
├── iterations/
│   ├── iter_000_baseline/
│   │   ├── logs/
│   │   ├── trajectories/
│   │   └── workspace/
│   ├── iter_001_evolved/
│   │   ├── logs/
│   │   ├── trajectories/
│   │   └── workspace/
│   └── iter_002_evolved/
│       ├── logs/
│       ├── trajectories/
│       └── workspace/
└── evolution_artifacts/
    ├── iter_001/
    │   ├── config.yaml
    │   ├── overlay_summary.json
    │   ├── tool_proposals.jsonl
    │   ├── skills/
    │   │   └── <generated-skill>/SKILL.md
    │   └── prompts/
    │       └── <agent-name>/<prompt-type>_prompt.evolved.txt
    └── iter_002/
        └── ...
```

## Reading the Results

### Evolution State

`evolution_state.json` summarizes every baseline and evolved run:

- run directory
- return code
- compact metrics
- overlay config path
- comparison to the previous run
- overlay summary path

### Evolution Log

`logs/evolution.log` is the main control-plane log for the wrapper. It records:

- evolution configuration
- baseline and evolved subprocess output
- trace digest collection
- metrics and comparisons
- analyzer LLM system prompt
- analyzer LLM user prompt
- analyzer LLM raw response
- parsed skill, prompt patch, and tool proposal candidates
- generated overlay paths

The per-agent logs remain under each iteration directory, such as:

```text
runs/evo_test/iterations/iter_000_baseline/logs/evomaster.log
runs/evo_test/iterations/iter_001_evolved/logs/evomaster.log
```

### Overlay Summary

Each `overlay_summary.json` records the concrete assets created for that iteration:

```json
{
  "source_config": "...",
  "overlay_config": ".../evolution_artifacts/iter_001/config.yaml",
  "skills_dir": ".../evolution_artifacts/iter_001/skills",
  "prompts_dir": ".../evolution_artifacts/iter_001/prompts",
  "applied_skills": ["..."],
  "applied_prompt_patches": ["general:system"],
  "tool_proposals": ["..."],
  "analysis_summary": "..."
}
```

### Generated Skills

Generated skills are standard EvoMaster skills with `SKILL.md` files. The evolved config points `skills.extra_roots` to the generated skill directory and adds the selected skill names to the target agent.

You can inspect a generated skill directly:

```bash
find runs/evo_test/evolution_artifacts -path "*/SKILL.md" -print
```

### Prompt Patches

Prompt patches are written into generated prompt files under:

```text
runs/evo_test/evolution_artifacts/iter_<N>/prompts/<agent-name>/
```

The evolved overlay config points the agent's `system_prompt_file` or `user_prompt_file` to those generated files.

### Tool Proposals

Tool proposals are written to:

```text
runs/evo_test/evolution_artifacts/iter_<N>/tool_proposals.jsonl
```

They are review artifacts only. EvoMaster does not auto-enable proposed tools.

## How the Analyzer Works

After each completed run, EvoMaster collects a compact `TraceDigest` from:

- trajectory JSON files
- log excerpts
- tool call counts
- tool error counts
- detected issues
- score-like artifacts such as `result.json`, `metric*.json`, or `grade*.json`
- workspace file manifest

If LLM analysis is enabled, the analyzer asks the configured default LLM to return structured JSON containing `skills`, `prompt_patches`, and `tool_proposals`. If the LLM call fails or returns invalid JSON, EvoMaster falls back to heuristic candidates.

The analyzer is intentionally aggressive about turning useful single-run observations into agent-local skills, while still avoiding secrets, private gold labels, and overfit one-off paths.

## Configuration Notes

The evolved overlay config contains:

- `evolution.base_config_dir`: keeps relative prompt, MCP, and custom-tool paths anchored to the original config directory.
- `skills.extra_roots`: loads generated run-local skills.
- Updated agent skill lists when skills are applied.
- Updated prompt file paths when prompt patches are applied.

Because the overlay config is generated under `runs/`, your original config remains unchanged.

## Practical Workflow

1. Run a baseline command without `--evolve` first if the agent or config is new.
2. Enable `--evolve` with `--evolve-iterations 1`.
3. Inspect `logs/evolution.log` and `evolution_artifacts/iter_001/overlay_summary.json`.
4. If the generated skills or prompt patches look useful, run more iterations.
5. Manually review tool proposals before turning any proposal into a real tool.

## Troubleshooting

### No skills were applied

Check:

- `logs/evolution.log` for analyzer errors.
- `overlay_summary.json` for `applied_skills`.
- Whether the target agent names in the analyzer output match configured agent names.

If LLM analysis fails, rerun with:

```bash
python run.py ... --evolve --evolve-disable-llm-analyzer
```

### Prompt patches were not applied

The applier can only patch prompts that are configured through `system_prompt_file` or `user_prompt_file` and can be resolved from the original config/playground directory.

Check `logs/evolution.log` for warnings such as missing prompt files.

### The evolved run behaves worse

The system preserves every iteration separately. Compare:

```text
runs/evo_test/iterations/iter_000_baseline/
runs/evo_test/iterations/iter_001_evolved/
runs/evo_test/evolution_artifacts/iter_001/config.yaml
```

Then remove or edit the generated skill/prompt overlay before reusing it.

### The log contains sensitive information

Evolution logs include LLM prompts, LLM responses, trajectory excerpts, tool outputs, and workspace manifests. Do not commit `runs/` directories if your tasks contain private data.

