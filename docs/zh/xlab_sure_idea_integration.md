# XLab 驱动 SURE-Evolve Idea 研究闭环设计方案

## 1. 背景

SURE-Evolve 当前已经具备较完整的候选实现、并行实验、远端 VC 调度、指标计算、
successive halving、selection 和 holdout 能力。它的主要短板不在实验执行，而在
idea 产生阶段：现有 `ResearchExp` 将任务描述、当前代码和历史结果放入一次 LLM
调用，随后直接解析一批简短 JSON 字符串。这种方式可以快速产生候选，但缺少系统的
文献检索、证据关联、机制分析、重复性检查和多轮研究讨论，idea 容易宽泛、重复，或
与实际可执行约束脱节。

XLab 的定位恰好互补。它提供持久化研究 workspace、论文收集、知识图谱、研究讨论、
novelty 风险检查、artifact lineage 和可恢复 run。本文设计将 XLab 接入
SURE-Evolve，使 XLab 成为唯一的 idea 研究与产出系统，SURE-Evolve 继续负责候选
代码生成、实验调度和指标裁决。

本方案不把 XLab 的 `run-experiment` 引入 SURE 实验执行。SURE 已有成熟的领域执行
控制面；再引入一套实验控制面会造成 workspace、checkpoint、失败恢复和最终裁决的
职责冲突。

## 2. 目标与非目标

### 2.1 目标

1. XLab 成为 regular/default 和 `staged_axes` 两种搜索路径的唯一 idea 来源。
2. 每轮形成严格的“生成 idea -> 执行实验 -> 总结结果 -> 生成下一轮”闭环。
3. idea 必须有结构化假设、机制、证据、实施规格、风险和可证伪的成功标准。
4. XLab 的研究知识可以跨轮次、跨 axis、跨 SURE run 长期积累。
5. SURE 的真实实验结果必须回写 XLab，成为下一轮研究的主要证据。
6. 保留 SURE 现有的 improve/debug、VC、metric、artifact guard、selection 和
   holdout 能力。
7. 所有输入、idea、代码、实验结果和总结均通过 digest 与 artifact ID 建立可追溯
   lineage。

### 2.2 非目标

1. 不让 XLab 直接调度 SURE 训练或替代 VC executor。
2. 不让 XLab 直接决定最终 metric 胜负；SURE metric runner 仍是唯一裁决来源。
3. 不保留旧 `ResearchExp` 作为备用 idea 生成器。
4. 不在第一版中让 XLab 直接生成最终 `run_sure.py`。代码仍由 SURE
   `improve_agent` 根据冻结的候选规格生成。
5. 不设置单独的固定预算 A/B 阶段。功能正确性通过单元测试、集成测试和真实 smoke
   验证，不将旧 idea 生成器继续保留为对照路径。

## 3. 已确定的核心决策

| 问题 | 决策 |
| --- | --- |
| Idea 来源 | XLab 是唯一来源，无旧生成器 fallback |
| 搜索覆盖 | 同时覆盖 regular/default 和 `staged_axes` |
| 代码生成 | XLab 决定完整 idea/spec，SURE `improve_agent` 忠实编码 |
| 轮次闭环 | 每轮生成、执行、XLab 总结，然后进入下一轮 |
| staged 方向 | `arch`、`train`、`inference` 分别执行独立闭环 |
| staged 代码基线 | 每个单轴候选基于固定 baseline，不继承上一轮候选代码 |
| staged 预算 | 每一轮都运行完整 short/medium/final 漏斗 |
| staged 组合 | 保留三轴独立筛选和最终笛卡尔组合 |
| 知识生命周期 | 按 task/model 使用长期 XLab workspace |
| XLab 产物错误 | 在同一 durable run 内最多修复两次，仍失败则终止 SURE |
| XLab 不可用 | 整个搜索明确失败，不降级、不跳轮 |

## 4. 总体架构

```text
Task card / baseline / config / historical artifacts
                         |
                         v
              +---------------------+
              | XLab task workspace |
              |---------------------|
              | papers              |
              | knowledge graph     |
              | idea artifacts      |
              | novelty reports     |
              | round summaries     |
              +----------+----------+
                         |
                    SureIdeaBatch
                         |
                         v
              +---------------------+
              | SURE-Evolve         |
              |---------------------|
              | schema validation   |
              | improve/debug       |
              | local/VC execution  |
              | metric and ranking  |
              +----------+----------+
                         |
                    RoundResult
                         |
                         +---------------------> XLab summarize
```

职责边界如下：

- **XLab**：研究问题分解、论文与图谱检索、假设形成、同批多样性控制、novelty 风险
  分析、SURE 候选规格生成、轮次结果总结和长期研究记忆。
- **SURE-Evolve**：任务与环境约束导出、候选规格确定性校验、代码生成与调试、GPU/VC
  调度、checkpoint 管理、artifact 校验、metric 计算、晋级和最终选择。
- **接口边界**：双方只交换版本化 JSON artifact 和 content digest，不直接导入对方
  的 Python 或 TypeScript 内部模块。

## 5. XLab 能力建设

当前 XLab `product-commands.ts` 已声明 `generate-research-ideas`，运行时也接受
`xlab.research_idea.v2` 和 `xlab.research_idea.result.v2`，但当前 `main` 工作树缺少
对应的 `research_idea` skill package。旧实现存在于历史 feature 分支，manifest 和
artifact 格式早于当前 v2 运行时，不能直接复制或 cherry-pick。

需要完成两项能力：

### 5.1 补齐通用 `research_idea` v2 skill

在 XLab 中实现当前规范的 `xlab/skills/research_idea/`：

- 输入可以引用 paper collection、knowledge graph、历史 idea、实验结果和讨论 artifact；
- 支持一次联合生成多个 idea，而不是启动多个互不知情的 agent run；
- 对同批 idea 做语义去重和机制覆盖检查；
- 每个 idea 发布独立的 `xlab.research_idea.result.v2` artifact；
- 同时发布包含所有 idea 引用的 batch manifest；
- 不宣称已经证明 novelty，只记录 duplication risk、证据覆盖和不确定性；
- artifact 未通过 schema、证据或质量门禁时，XLab run 必须是 `incomplete`，不能
  发布为 `success`。

### 5.2 新增 SURE 专用研究工作流

新增 `/xlab evolve-sure-ideas --mode generate|summarize`，由一个 SURE 专用 skill
package 承载领域工作流。它依赖通用 `research_idea`、`novelty_check`、
`research_discussion`、`paper_collect` 和 `knowledge_graph`，但不执行 SURE 代码。

支持两个操作：

- `generate`：读取 `sure.idea_request.v1` 和上一轮 summary，输出
  `sure.idea_batch.v1`；
- `summarize`：读取 `sure.round_result.v1`，输出 `sure.round_summary.v1`，不生成
  下一轮 idea。

将生成和总结作为两个显式调用，可以保证持久化事件顺序确实是“生成 -> 执行 -> 总结”，
也允许在 SURE 实验完成后恢复同一个 XLab 研究链路。

## 6. 数据契约

所有 schema 使用显式版本号。未知 major version 必须拒绝；同一 major version 的新增
可选字段可以忽略。所有 JSON 文件使用 UTF-8、稳定 key 排序和结尾换行，并记录
SHA-256 digest。

### 6.1 Idea 请求：`sure.idea_request.v1`

请求是一次生成操作的冻结输入。建议结构如下：

```json
{
  "schema_version": "sure.idea_request.v1",
  "sure_run_id": "sure-run-001",
  "task_id": "asr_en_wer",
  "canonical_task": "asr",
  "search_mode": "staged_axes",
  "axis": "arch",
  "round_index": 2,
  "idea_count": 4,
  "metric": {
    "name": "wer",
    "direction": "lower"
  },
  "baseline": {
    "code_path": "/absolute/path/best_solution.py",
    "code_digest": "sha256:...",
    "score": 6.31
  },
  "task_card": {},
  "base_model_profile": {},
  "execution_contract": {
    "candidate_type": "arch",
    "allowed_wrappers": ["SURE_ASR_ZIPFORMER_WRAPPER"],
    "allowed_parameters": ["num_encoder_layers", "encoder_dim"],
    "forbidden_changes": ["evaluator", "reference_transcript"],
    "rungs": ["short", "medium", "final"]
  },
  "history": {
    "idea_artifact_ids": [],
    "round_summary_artifact_ids": [],
    "round_result_artifact_ids": []
  },
  "input_digest": "sha256:..."
}
```

regular 搜索时 `axis` 为 `null`，`execution_contract` 改为当前配置允许的候选类型及数量
分布。请求中不能包含 API key、VC credential 或其他 secret。

### 6.2 Idea 批次：`sure.idea_batch.v1`

一次成功 batch 必须恰好包含 `idea_count` 个候选：

```json
{
  "schema_version": "sure.idea_batch.v1",
  "status": "success",
  "request_digest": "sha256:...",
  "xlab_run_id": "xlab-run-001",
  "workspace": "sure-asr-en-wer-zipformer",
  "ideas": [
    {
      "idea_id": "arch-r2-i1",
      "artifact_id": "sha256:...",
      "artifact_digest": "sha256:...",
      "title": "...",
      "axis": "arch",
      "candidate_type": "arch",
      "hypothesis": "...",
      "mechanism": "...",
      "evidence_refs": [],
      "novelty": {
        "risk_level": "medium",
        "confidence": "high",
        "closest_work": []
      },
      "spec": {
        "change_set": [],
        "implementation_instructions": "...",
        "invariants": [],
        "success_criteria": [],
        "ablation": [],
        "expected_effect": "...",
        "risks": [],
        "resource_class": "training"
      }
    }
  ],
  "batch_digest": "sha256:..."
}
```

`implementation_instructions` 是给 SURE `improve_agent` 的冻结实现说明。SURE 可以添加
路径、wrapper 调用和 artifact contract 等机械性上下文，但不能删除、替换或重新解释
XLab 的 hypothesis、change set 和 invariants。

### 6.3 轮次结果：`sure.round_result.v1`

SURE 在一轮执行结束后发布：

```json
{
  "schema_version": "sure.round_result.v1",
  "sure_run_id": "sure-run-001",
  "search_mode": "staged_axes",
  "axis": "arch",
  "round_index": 2,
  "baseline_digest": "sha256:...",
  "idea_batch_digest": "sha256:...",
  "candidates": [
    {
      "idea_id": "arch-r2-i1",
      "idea_artifact_id": "sha256:...",
      "code_digest": "sha256:...",
      "rungs": [
        {
          "name": "short",
          "success": true,
          "score": 6.20,
          "runtime_seconds": 3200,
          "checkpoint_artifact": "sha256:..."
        }
      ],
      "final_status": "promoted",
      "failure_category": null,
      "reason_code": null
    }
  ],
  "ranking": [],
  "result_digest": "sha256:..."
}
```

失败必须区分：

- `candidate_failure`：idea 或实现本身失败，是有效研究证据；
- `system_failure`：调度、环境、网络或集群故障，不能被解释为 idea 无效；
- `contract_failure`：候选违反 artifact 或执行契约；
- `metric_failure`：产物存在但无法得到可信 metric。

### 6.4 轮次总结：`sure.round_summary.v1`

总结至少包含：

- 哪些机制得到当前实验支持；
- 哪些机制被否定或证据不足；
- candidate failure 与 system failure 的分离分析；
- 下一轮应探索、收缩或停止的方向；
- 已尝试 idea 的去重指纹；
- 新发现的 SURE wrapper、artifact 或资源约束；
- 使用到的 result、paper、graph 和 idea artifact ID。

总结是下一轮请求的主要历史输入，不再使用拼接的自由文本
`research_plan_and_result`。

## 7. SURE 侧集成

### 7.1 XLab RPC client

在 SURE 中新增独立的 `XlabIdeaClient`，禁止在业务逻辑中散落 subprocess 调用。它负责：

1. 使用配置中的 `XLAB_ROOT` 启动一个长驻 XLab RPC 子进程；
2. 将工作目录设置为 XLab 根目录，使 XLab 能发现 `xlab/skills/`，并将 durable state
   保存在 `XLab/.xlab/`；
3. 执行 setup check、workspace 选择、generate、summarize、resume 和 shutdown；
4. 按 JSONL LF framing 读取 RPC 响应和事件；
5. 将 XLab run ID、manifest path、artifact ID 和 digest 写入 SURE workspace；
6. RPC 断开时先按 run ID 查询并恢复，禁止静默创建重复研究 run；
7. 不在命令行、日志或 artifact 中输出 secret。

每次 SURE run 使用一个 RPC 进程；XLab workspace 则跨 SURE run 长期存在。这样既避免
每轮重复启动模型与扩展，又保留任务级知识积累。

### 7.2 SURE 候选校验

在占用 GPU 前执行确定性校验：

- batch 数量与配置完全一致；
- `axis`、`candidate_type` 和当前阶段一致；
- wrapper、参数和 action 在当前 task card 白名单内；
- `[fine_tune]` 不改变模型结构；
- `[arch]` 确实改变结构或参数量；
- `[inference]` 不触发训练；
- idea ID、artifact ID 和 digest 唯一；
- 同批及历史候选没有超过阈值的语义重复；
- spec 包含实施说明、invariants 和成功标准；
- evidence 与 novelty 字段满足 XLab 门禁。

校验失败时，将结构化诊断交回原 XLab run 修复。最多修复两次；仍不合法则整个 SURE
run 失败。不得删除坏项后用较少候选继续，也不得让旧 research agent 补齐。

### 7.3 SURE improve agent 的职责收缩

`improve_agent` 保留，但从“研究并实现 idea”收缩为“实现已冻结的 spec”：

- 可以确定具体代码组织、变量名和安全的 wrapper 调用形式；
- 可以根据 workspace 路径补充机械性参数；
- 不得将一个 idea 换成另一个 idea；
- 不得改变 axis、hypothesis、change set 或 invariants；
- 实现不可能时必须返回明确 blocker，不能自行降级为 baseline/no-op；
- 生成代码继续进入现有 debug、artifact guard 和 metric 流程。

## 8. Regular 搜索闭环

regular/default 搜索保留“当前最佳方案逐轮演化”的语义：

```text
准备 task workspace 和 evidence
        |
        v
XLab generate round 1
        |
SURE validate -> improve -> execute -> rank -> update best
        |
XLab summarize round 1
        |
XLab generate round 2 using summary + current best
        |
       ...
```

每轮具体步骤：

1. SURE 将当前 best code、best score、task card、candidate type 配额和历史 artifact
   组装为 `sure.idea_request.v1`。
2. XLab 联合生成一批多样化候选并发布 batch。
3. SURE 校验 batch，并行调用 `improve_agent` 实现候选。
4. 按现有 local/VC 规则执行、评分和排序。
5. 合法改进者更新 regular 当前 best；失败候选仍进入 round result。
6. SURE 发布 `sure.round_result.v1`。
7. XLab 执行 summarize，发布 `sure.round_summary.v1`。
8. 下一轮只读取结构化 artifact，不拼接旧 research prompt 文本。

regular 不再保留 major direction 的嵌套字典协议。研究方向、候选类型和证据均成为
idea 的显式字段。

## 9. Staged Axes 搜索闭环

### 9.1 总体顺序

```text
baseline/draft
    |
    +-> arch rounds      -> top_arch
    +-> train rounds     -> top_train
    +-> inference rounds -> top_inference
                                  |
                                  v
             arch x train x inference combinations
                                  |
                         selection -> holdout
```

三个 axis 仍相互独立地从同一 baseline 实现候选。前一轮和前一 axis 的实验知识可以被
XLab 使用，但代码不直接累积。这样可以继续把单轴结果解释为独立因素，并安全地执行最终
组合实验。

### 9.2 单个 axis 的轮次

假设 `rounds_per_axis=4`、`ideas_per_round=4`：

```text
Round 1: generate 4 -> short -> medium -> final -> summarize
Round 2: generate 4 -> short -> medium -> final -> summarize
Round 3: generate 4 -> short -> medium -> final -> summarize
Round 4: generate 4 -> short -> medium -> final -> summarize
                                      |
                         aggregate final survivors
                                      |
                              select top_axis
```

每轮都使用当前 axis 配置的完整 rung 列表：

1. short：所有本轮候选进入；
2. medium：按本轮 short 排名和 `keep` 晋级；
3. final：按本轮 medium 排名和 `keep` 晋级；
4. summary：XLab 同时看到所有 rung，而不是只看到最终分数；
5. 下一轮：XLab 避免重复失败机制，并可以针对已观察信号提出新假设。

每个 axis 结束后，仅汇总各轮 final survivor，再按照统一 metric 排序并截取
`top_arch`、`top_train` 或 `top_inference`。不能把未完成 final 的高 short 分候选与
final survivor 直接混排。

该设计比当前“先生成 16 个 idea，再统一运行一次漏斗”消耗更多 medium/final 预算。
这是为了让下一轮获得更深的真实实验反馈而明确接受的成本，不应在实现时悄悄恢复为
统一漏斗。

### 9.3 组合与最终选择

axis 顶部候选产生后，保留现有组合语义：

- 组合候选引用三个原始 idea artifact 和各自 final result；
- `arch x train` 必须按组合配置重新训练兼容 checkpoint；
- inference 设置应用在组合 checkpoint 上；
- 组合不能拼接互不兼容的单轴 checkpoint；
- combination、selection 和 holdout 结果也发布 artifact，并关联到三个上游 idea。

## 10. 知识与 Workspace 生命周期

XLab workspace 使用稳定 slug，例如：

```text
sure-asr-en-wer-zipformer
sure-tts-en-wer-f5tts-v1-base
```

workspace 按 task/model 长期复用，但每次请求记录以下 fingerprint：

- task card digest；
- base model profile digest；
- baseline code digest；
- SURE execution contract digest；
- dataset/evaluation split 标识；
- metric 定义与方向。

论文和知识图谱在 topic/model fingerprint 未变化时默认复用。baseline 或执行配置变化不会
删除旧知识，而是产生新的请求和总结版本。只有显式设置 `refresh_evidence: true` 或研究
topic fingerprint 变化时才重新收集论文。

外部文献结论和本地 SURE 结果必须分层保存：

- 论文证据表示“外部工作报告了什么”；
- SURE round result 表示“当前任务实际观察到什么”；
- summary 可以关联两者，但不能用论文结论覆盖当前失败实验。

## 11. 失败、恢复与终止语义

### 11.1 XLab 失败

以下情况直接阻断 SURE 搜索：

- XLab root、Node 环境、skill package 或必需服务不可用；
- RPC 无法启动且无法恢复；
- evidence 构建失败；
- idea batch 最终为 `failed` 或 `incomplete`；
- batch 数量、schema、digest、axis 或约束在两次修复后仍不合法；
- round summary 未能持久化或 lineage 不完整。

不存在 builtin、旧 LLM planner、跳过本轮或缩减候选数等 fallback。

### 11.2 候选实验失败

候选失败不等于 XLab 失败：

- 单个 candidate failure 被记录并进入总结；
- 单轮没有 final survivor 时仍先生成本轮总结，下一轮可以基于失败继续研究；
- 一个 axis 所有轮次结束后仍没有 final survivor，则 axis 失败，不能自动使用 neutral
  idea 掩盖研究失败；
- system failure 按现有远端结果恢复策略处理，并在 summary 中与科学失败分开；
- debug agent 可以修复实现错误，但不能改变 idea 内容。

### 11.3 Durable resume

SURE workspace 记录每次调用的：

- XLab workspace slug；
- XLab run ID；
- operation 和 round/axis；
- request、batch、result、summary artifact ID；
- manifest path 和 digest；
- 当前状态与最后确认事件。

进程重启后先查询已有 XLab run。已发布 artifact 的操作不得重复执行；运行中的操作调用
resume；终态失败按原状态返回。相同 result 重放必须幂等，digest 冲突必须 fail closed。

## 12. 配置与迁移

所有 SURE 配置新增必需的 `sure.xlab`：

```yaml
sure:
  xlab:
    root: "${XLAB_ROOT}"
    rpc_command:
      - "./pi-test.sh"
      - "--mode"
      - "rpc"
      - "--no-session"
      - "--approve"
    request_timeout_seconds: 3600
    repair_attempts: 2
    workspace_prefix: "sure"
    refresh_evidence: false
```

迁移是有意的 breaking change：

1. 从所有 SURE YAML 删除 `agents.reseach` 和 `agents.knowledge_promotion`；
2. 删除 `reseach_system_prompt.txt`、`reseach_user_prompt.txt` 和 knowledge promotion
   prompts；
3. 删除 `ResearchExp`、`KnowledgePromotionExp` 及其旧测试；
4. 从 agent declaration 删除 `reseach_agent` 和 `knowledge_promotion_agent`；
5. 用结构化 XLab artifact 引用替换 `research_plan_and_result`；
6. 保留 `max_research_rounds`、`rounds_per_axis`、`ideas_per_round` 和各 rung 配置，
   但由新闭环解释；
7. 在 `.env.template` 文档化 `XLAB_ROOT`，不提交模型 API key 或 XLab state。

## 13. 建议的代码组织

SURE-Evolve 侧建议新增：

```text
playground/sure_master/core/idea/
├── contracts.py       # Pydantic/dataclass schema 与 digest
├── xlab_client.py     # RPC 生命周期和命令调用
├── validator.py       # SURE 领域约束校验
└── prompts.py         # SureCandidateSpec 到 improve prompt 的稳定渲染
```

`playground.py` 只负责编排 regular/staged 状态机，不直接处理 RPC framing 或 schema
细节。round result 的构造应复用统一 helper，避免 regular 和 staged 产生不同反馈格式。

XLab 侧建议新增或补齐：

```text
xlab/skills/research_idea/   # 通用 v2 idea 生成
xlab/skills/sure_evolution/  # generate/summarize SURE 工作流
```

SURE-specific 规则放在 skill package 和输入 execution contract 中，不写入 XLab 共享
runtime。共享 runtime 只负责命令、run、artifact、resume 和 workspace 生命周期。

## 14. 实施顺序

### 阶段 1：XLab artifact 与 skill

1. 实现 `research_idea` v2 package 和 schema；
2. 实现 SURE evolution package 的 generate/summarize 两种操作；
3. 增加 product command、manifest validation、artifact publication 和 resume 测试；
4. 用 faux provider 完成无付费 API 的 deterministic 测试。

### 阶段 2：SURE RPC 与契约

1. 实现四类 JSON contract 和 digest；
2. 实现长驻 RPC client、setup check、run reconciliation 和 resume；
3. 实现 SURE 静态 validator 和同 run repair；
4. 用 fake RPC server 覆盖成功、超时、断连、重复消息和错误产物。

### 阶段 3：替换两类搜索循环

1. 先将 regular loop 改为 XLab generate/execute/summarize；
2. 将 staged 单轴逻辑改为每轮完整漏斗和 XLab summary；
3. 在轴末聚合各轮 final survivor；
4. 保留并接通 combination、selection、holdout lineage；
5. 删除旧 research 与 knowledge promotion 路径。

### 阶段 4：配置、文档和 smoke

1. 迁移全部 SURE YAML 和环境模板；
2. 更新 SURE 技术手册、使用指南和故障排查说明；
3. 执行 fake-XLab 的本地完整 smoke；
4. 使用最小真实 staged 配置验证一次 XLab research -> SURE experiment -> XLab
   summary -> next round；
5. 验证中断后通过同一 XLab run ID 恢复。

## 15. 测试计划

### 15.1 XLab 测试

- skill discovery 和 product command 可用；
- batch 恰好包含请求数量的候选；
- 单 idea schema、batch schema 和 digest 稳定；
- 同批重复、历史重复、缺失证据和 axis 错配被拒绝；
- novelty 低置信度得到明确风险，不被伪装成新颖性证明；
- summarize 正确区分 candidate/system/contract/metric failure；
- 两次原 run 修复后成功，以及修复耗尽后的 `incomplete`；
- resume、幂等重放、冲突 digest 和 artifact lineage；
- focused Vitest 通过，随后运行 `npm run check`。

### 15.2 SURE 单元测试

- RPC JSONL framing、事件关联、timeout、shutdown 和 reconnect；
- XLab 不可用时立即失败，确认旧 generator 从未调用；
- request 中不包含 secret；
- batch 数量、类型、axis、wrapper 和参数白名单校验；
- spec 到 improve prompt 的字段无损且禁止 idea 改写；
- round result 完整包含所有 rung、失败类型和 artifact；
- XLab schema 错误进入同 run repair，超过次数后终止；
- regular 当前 best 在轮间更新；
- staged 候选始终使用固定 baseline。

### 15.3 流程测试

- regular 严格按 generate -> execute -> summarize -> next generate 排序；
- staged 的三个 axis 各自执行配置轮数；
- staged 每轮都执行完整 short/medium/final，而不是轴末统一漏斗；
- 下一轮 request 引用上一轮 summary；
- axis 结束仅聚合 final survivor；
- top 三轴候选可以进入现有 combination；
- selection 和 holdout 的 artifact lineage 可以追溯到三个原始 XLab idea；
- 最后一轮执行后仍产生 summary-only artifact；
- 中断恢复不重复已完成 GPU 实验或 XLab 操作。

## 16. 验收标准

第一版完成必须同时满足：

1. 代码中不存在可运行的旧 SURE idea 生成路径；
2. 所有 SURE 搜索配置都要求 XLab，并在缺失时 fail fast；
3. regular 和 staged 都形成 artifact 可验证的逐轮闭环；
4. staged 每个 axis 的每轮都运行完整配置漏斗；
5. XLab summary 能看到候选的所有 rung 结果和失败分类；
6. SURE improve agent 只实现冻结 spec，无法静默替换 idea；
7. 任一最终候选都可以追溯到 XLab idea、论文/图谱证据、每轮结果和总结；
8. fake RPC 的完整测试通过；
9. 至少一个最小真实 staged smoke 完成两轮，并成功执行第二轮的反馈驱动生成；
10. 人为中断一次后，可以从同一 XLab run 和 SURE workspace 恢复。

本验收不包含旧/新 idea 生成器的固定预算 A/B。旧路径被直接移除，验收重点是新系统的
正确性、可恢复性、约束遵守和完整 lineage。

## 17. 主要风险与控制措施

### 17.1 计算成本增加

每轮完整漏斗会增加 medium/final 实验量。通过现有 `rounds_per_axis`、
`ideas_per_round` 和 rung `keep` 显式控制，不在实现层自动缩减预算。启动正式任务前必须
根据配置计算预期 job 数并记录在 run summary。

### 17.2 长期知识污染

旧 baseline 或旧 dataset 的经验可能误导新 run。所有 summary 必须绑定 task、model、
dataset、baseline 和 execution contract digest；检索时优先精确 fingerprint，再使用
同 task 的弱相关历史。

### 17.3 文献证据压过本地结果

XLab prompt 和 summary schema 必须明确：本地 SURE metric 是当前任务的主要证据；论文
只提供外部先验和解释。论文声称有效但本地重复失败的方法必须降低优先级。

### 17.4 Spec 与实际代码偏离

SURE 保存 spec digest、生成代码 digest 和候选 change artifact。执行前后检查关键参数和
wrapper，发现偏离时视为 contract failure，并交给 debug 修复，而不是照常评分。

### 17.5 双仓库版本漂移

RPC 握手必须交换 XLab runtime version、skill version 和支持的 schema version。SURE
启动时验证兼容矩阵；不兼容时 fail fast，并在 run metadata 中记录双方 git commit。

## 18. 最终状态

完成后，SURE-Evolve 不再自行“临时想几个 idea”，而是消费 XLab 产出的、带证据和
lineage 的研究候选。XLab 也不替代 SURE 的实验工程能力，而是持续读取真实实验结果，
形成可恢复、可审计、可跨 run 积累的研究闭环：

```text
外部研究证据 + 当前任务约束 + 本地历史实验
                    |
                    v
           XLab 结构化研究与 idea
                    |
                    v
           SURE 实现、执行和指标裁决
                    |
                    v
             XLab 结果总结与再研究
```

这条边界使两个系统各自保留最强能力：XLab 负责研究质量，SURE 负责实验可信度。
