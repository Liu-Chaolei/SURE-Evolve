# EvoMaster 自进化使用指南

EvoMaster 自进化是一个可选的运行后包装层。它会先运行原始智能体，读取这次运行产生的轨迹、日志和环境反馈，再把其中有价值的方法、问题和经验提炼成可复用的本地 skill、prompt patch 和 tool proposal，随后用生成的 overlay 配置再次运行同一个智能体。

这个功能按类似插件的方式接入。只要不传 `--evolve`，EvoMaster 的普通运行路径不会变化。

## 快速开始

在仓库根目录运行：

```bash
python run.py \
  --agent minimal \
  --config configs/minimal/gpt-5-example.yaml \
  --task "Discover a pattern: Given sequence 1, 4, 9, 16, 25... find the formula" \
  --run-dir runs/evo_test \
  --evolve \
  --evolve-iterations 2
```

这条命令会执行：

1. `iter_000_baseline`：先运行一次原始 `minimal` 智能体。
2. 分析 baseline 的轨迹、日志、工具调用、错误、产物和 workspace 文件清单。
3. 生成 evolution overlay，包括 skill、prompt patch 和 tool proposal。
4. `iter_001_evolved`：用 evolution overlay 再运行一次同一个智能体。
5. 再次分析并生成下一轮 overlay。
6. `iter_002_evolved`：用第二轮 overlay 继续运行智能体。

## 会进化什么

自进化目前会生成三类内容：

| 输出 | 是否自动应用 | 用途 |
| --- | --- | --- |
| Skills | 是 | 从运行轨迹中提炼出的可复用流程、调试方法、领域事实和成功策略。 |
| Prompt patches | 是 | 追加到 agent system prompt 或 user prompt 的补充指令，通过生成的 prompt overlay 文件生效。 |
| Tool proposals | 否 | 新工具的 JSONL 提案。工具代码不会自动生成或启用，因为长期工具的实现风险更高。 |

这些内容都是 run-local 的。EvoMaster 不会覆盖你的原始 config、原始 prompt 文件或内置 skill 目录。

## CLI 参数

### `--evolve`

开启自进化包装层。

不传这个参数时，EvoMaster 走原来的普通运行逻辑。

### `--evolve-iterations N`

在 baseline 之后运行 `N` 轮进化。

例如 `--evolve-iterations 2` 表示：

```text
baseline run -> analyze/apply -> evolved run 1 -> analyze/apply -> evolved run 2
```

`--evolve-iterations 0` 表示只在 evolution 目录结构下运行 baseline，并记录 evolution 元信息。

### `--evolve-disable-llm-analyzer`

关闭 LLM analyzer，只使用启发式 evolution。

当 analyzer 模型不可用、想降低 LLM 调用成本，或只想测试包装层流程时，可以使用这个参数。

### 当前支持的运行模式

当前实现支持：

- 每次 evolution 只运行一个任务。
- 串行执行，不要传 `--parallel`。
- `minimal`、串行 `minimal_multi_agent`、`minimal_kaggle`、`x_master`、`browse_master` 等已有 agent，只要所选 config 在普通模式下能正常运行。

Exp 级并行 agent 的并行自进化暂未启用。

## 输出目录结构

如果使用 `--run-dir runs/evo_test`，关键文件如下：

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

## 如何查看结果

### Evolution State

`evolution_state.json` 汇总了 baseline 和每轮 evolved run：

- run 目录
- 返回码
- 紧凑 metrics
- overlay config 路径
- 与上一轮的对比
- overlay summary 路径

### Evolution Log

`logs/evolution.log` 是 evolution 包装层的主控日志，记录：

- evolution 配置
- baseline 和 evolved 子进程输出
- trace digest 收集过程
- metrics 与对比结果
- analyzer LLM system prompt
- analyzer LLM user prompt
- analyzer LLM raw response
- 解析后的 skill、prompt patch 和 tool proposal 候选
- 生成的 overlay 路径

每个 agent 自己的日志仍然保留在对应 iteration 目录下，例如：

```text
runs/evo_test/iterations/iter_000_baseline/logs/evomaster.log
runs/evo_test/iterations/iter_001_evolved/logs/evomaster.log
```

### Overlay Summary

每轮的 `overlay_summary.json` 会记录这一轮实际生成和应用的资产：

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

### 生成的 Skills

生成的 skill 是标准 EvoMaster skill，包含 `SKILL.md`。evolved config 会把 `skills.extra_roots` 指向本轮生成的 skill 目录，并把选中的 skill 名称加入目标 agent 配置。

可以用下面的命令查看所有生成的 skill：

```bash
find runs/evo_test/evolution_artifacts -path "*/SKILL.md" -print
```

### Prompt Patches

Prompt patch 会写入生成的 prompt 文件：

```text
runs/evo_test/evolution_artifacts/iter_<N>/prompts/<agent-name>/
```

evolved overlay config 会把 agent 的 `system_prompt_file` 或 `user_prompt_file` 指向这些生成文件。

### Tool Proposals

Tool proposal 会写入：

```text
runs/evo_test/evolution_artifacts/iter_<N>/tool_proposals.jsonl
```

它们只是供人工 review 的提案，不会被 EvoMaster 自动启用。

## Analyzer 如何工作

每次运行结束后，EvoMaster 会从以下内容中收集紧凑的 `TraceDigest`：

- trajectory JSON 文件
- 日志片段
- 工具调用次数
- 工具错误次数
- 检测到的问题
- `result.json`、`metric*.json`、`grade*.json` 等可能包含分数的产物
- workspace 文件清单

如果开启 LLM analyzer，系统会调用当前 config 中的默认 LLM，让它返回结构化 JSON，包含 `skills`、`prompt_patches` 和 `tool_proposals`。如果 LLM 调用失败或返回的 JSON 无法解析，EvoMaster 会自动退回启发式候选。

Analyzer 会比较激进地把单次运行中有效或重要的经验提炼成 agent-local skill，但仍会避免写入密钥、私有 gold label 和过拟合的一次性路径。

## 配置说明

生成的 evolved overlay config 会包含：

- `evolution.base_config_dir`：让相对路径的 prompt、MCP 和 custom tool 仍然从原始 config 目录解析。
- `skills.extra_roots`：加载本轮生成的 run-local skills。
- 应用 skill 时更新 agent 的 `skills` 列表。
- 应用 prompt patch 时更新 agent 的 prompt 文件路径。

因为 overlay config 生成在 `runs/` 下，原始 config 不会被修改。

## 推荐工作流

1. 如果是新 agent 或新 config，先不用 `--evolve` 跑通一次普通命令。
2. 加上 `--evolve --evolve-iterations 1` 做一轮进化。
3. 查看 `logs/evolution.log` 和 `evolution_artifacts/iter_001/overlay_summary.json`。
4. 如果生成的 skill 或 prompt patch 有价值，再增加 evolution 轮数。
5. 对 tool proposal 做人工 review 后，再决定是否实现成正式工具。

## 常见问题

### 没有 skill 被应用

检查：

- `logs/evolution.log` 中是否有 analyzer 错误。
- `overlay_summary.json` 里的 `applied_skills`。
- analyzer 输出里的 agent name 是否和 config 中的 agent 名称一致。

如果 LLM analyzer 失败，可以改用：

```bash
python run.py ... --evolve --evolve-disable-llm-analyzer
```

### Prompt patch 没有应用

Applier 只能修改通过 `system_prompt_file` 或 `user_prompt_file` 配置、且能从原始 config/playground 目录解析到的 prompt 文件。

可以在 `logs/evolution.log` 中查看是否有 prompt 文件缺失的 warning。

### Evolved run 变差了

系统会把每一轮完整保留下来。可以对比：

```text
runs/evo_test/iterations/iter_000_baseline/
runs/evo_test/iterations/iter_001_evolved/
runs/evo_test/evolution_artifacts/iter_001/config.yaml
```

然后手动删除或修改生成的 skill/prompt overlay，再决定是否复用。

### 日志包含敏感信息

Evolution 日志会记录 LLM prompt、LLM response、trajectory 片段、工具输出和 workspace 文件清单。如果任务中包含隐私数据，不要把 `runs/` 目录提交到仓库。

