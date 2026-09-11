# SURE-Master 接入 XLab 的详细实施方案

## 1. 背景与目标

当前 `/shared/chaolei.liu/SURE-Evolve` 的 `sure_master` 已经具备完整的语音模型自进化能力，包括：

- ordinary research/improve loop；
- staged arch/train/inference 三轴搜索；
- successive-halving 与多 rung 晋级；
- checkpoint promotion 与已有代码恢复；
- local、VC、remote 混合执行；
- SURE metric 评估；
- candidate ranking、selection、holdout；
- KnowledgePromotion、WisdomPromotion 与运行状态恢复。

当前每轮 idea 主要由 `ResearchExp` 生成。XLab 已经具备论文检索、知识图谱、survey、research idea、novelty、artifact lineage 和 durable run 能力。目标不是简单地把 `sure_master` 包装成一个 XLab 命令，而是建立如下闭环：

```text
用户调用 /xlab run-sure-master
        |
        v
XLab 创建 durable run 和 research workspace
        |
        v
XLab 根据任务、模型、论文/知识、历史实验反馈生成 idea batch
        |
        v
SURE 验证并执行 ideas，保留现有训练、推理、metric、排序和晋级逻辑
        |
        v
SURE 生成完整 round_result
        |
        v
XLab 总结本轮结果，形成 round_summary 和跨轮经验
        |
        v
下一轮 XLab 根据 summary、best solution 和历史 artifact 生成新 ideas
```

最终用户入口为：

```text
/xlab run-sure-master --config <sure-config> --task "<task description>"
```

这条命令启动的是完整的 `sure_master` 自进化任务，而不是只生成一个 idea、只写一份配置，或只执行一次候选评估。

## 2. 核心边界

### 2.1 XLab 负责的部分

XLab 成为 idea 和研究经验的权威来源，负责：

- `/xlab run-sure-master` public command；
- durable run、workspace、operation 和生命周期；
- 论文、survey、知识图谱和历史 idea artifact 的引用；
- 根据当前任务和历史实验结果生成结构化 idea batch；
- novelty/evidence 上下文；
- 接收每一轮完整实验结果；
- 总结有效机制、失败原因、资源限制和下一步方向；
- 持久化 request、idea、round result、round summary 及其 lineage；
- 跨 round、跨 run 的可审计经验；
- 中断后的 operation reconciliation、resume 和 publication recovery。

### 2.2 SURE-Evolve 负责的部分

SURE 保持实验控制面，负责：

- task card、base model、execution environment 和资源解析；
- candidate type 约束；
- 将 XLab idea 转换为现有 candidate 输入格式；
- candidate code 生成或实现；
- workspace isolation 和 artifact guard；
- local、VC、remote job 的启动与结果恢复；
- fine-tune、architecture、inference 的实际执行；
- SURE metric、WER/CER 等任务指标；
- candidate ranking 和 best solution 更新；
- staged axes、rungs、checkpoint promotion；
- combination、selection、holdout；
- candidate/system/contract/metric failure 的执行层分类。

### 2.3 必须避免的边界错误

- 不在 XLab 重新实现 `SureRunExp`、VC 调度、SURE metric 或 staged ranking；
- 不让独立的 XLab workflow 在外部猜测、重排或重复执行 SURE 阶段；
- 不在完整 SURE run 结束后才调用 XLab，否则 XLab idea 无法影响下一轮；
- XLab 接管 idea 后，不能在 XLab 失败时静默切回旧 `ResearchExp`，否则无法保证 XLab 是唯一 idea 来源；
- 不把 API key、VC credential、绝对宿主路径写入 public artifact；
- 不通过用户输入直接拼接任意 shell 命令；
- 不把大量候选 payload 塞进 `state.json`，应使用 artifact ID 和 digest 引用。

## 3. 现有代码接入点

### 3.1 SURE ordinary 流程

主控制器：

- `playground/sure_master/core/playground.py`
- `SureMasterPlayground.run()`

当前流程大致为：

```text
prefetch
  -> draft baseline
  -> ResearchExp.run() 生成 research_plan
  -> 按 direction 和 candidate type 筛选 idea
  -> SureRunExp.run() 执行 improve candidate
  -> 更新 direction_best_solution / best score
  -> KnowledgePromotion
  -> 下一轮 ResearchExp.run()
```

外部 provider 的最小插点是 `SureMasterPlayground.run()` 中调用 `ResearchExp.run()` 的位置，或在 `ResearchExp` 内注入 callable。启用 XLab 后，此处提交 `idea_request` 并接收 `idea_batch`，随后转换成当前 ordinary 所需的 nested plan：

```python
{
    "direction_name": {
        "idea_1": "...",
        "idea_2": "...",
    }
}
```

现有 `_filter_and_order_ideas`、candidate type limits、`SureRunExp.run()` 和 best update 逻辑继续复用。

### 3.2 SURE staged 流程

主控制器仍为：

- `playground/sure_master/core/playground.py`
- `_run_staged_axes()`
- `_generate_staged_axis_ideas()`
- `_run_axis_screening()`
- `_run_staged_records()`
- staged combination、selection、holdout helpers

每个 axis round 都应独立提交 XLab request：

```text
arch round -> XLab arch ideas -> arch rungs -> arch summary
train round -> XLab train ideas -> train rungs -> train summary
inference round -> XLab inference ideas -> inference rungs -> inference summary
```

XLab 返回的 ideas 转换为既有 axis record：

```python
{
    "idea_id": "...",
    "idea": "...",
    "axis": "arch|train|inference",
    "candidate_type": "arch|fine_tune|inference",
}
```

仍由 `_run_staged_records()` 调用：

- 首次候选执行：`SureRunExp.run()`；
- successive-halving 重跑：`SureRunExp.run_existing_code()`；
- rung 排名和 checkpoint promotion：既有 staged 逻辑；
- combination、selection、holdout：既有逻辑。

staged 的每个 rung 结果都必须进入 `round_result`，不能只把最终 survivor 反馈给 XLab。

### 3.3 候选执行边界

`playground/sure_master/core/exp/run_exp.py` 中的 `SureRunExp` 继续负责：

```text
清理输出
-> 解析 candidate code
-> candidate boundary validation
-> 写入 run_sure.py
-> 执行候选
-> required artifact validation
-> artifact guard
-> SURE metric
-> candidate status
```

XLab 只提供研究 idea 和证据上下文，不生成或修改 evaluator，不绕过 artifact guard，也不直接写外部 base model source directory。

## 4. 用户入口与 XLab package

### 4.1 Public command

修改：

- `XLab/packages/coding-agent/src/core/xlab/product-commands.ts`

加入：

```text
run-sure-master -> sure_master
```

命令示例：

```text
/xlab run-sure-master \
  --config configs/sure_master/gpt-5-example.yaml \
  --task "Improve ASR WER while preserving the required artifact contract"
```

任务文件和恢复示例：

```text
/xlab run-sure-master \
  --config <config> \
  --task-file <task-file-or-task-card> \
  --run-dir <workspace-relative-run-dir>

/xlab run-sure-master \
  --config <config> \
  --task <task> \
  --resume <xlab-run-id-or-sure-checkpoint>
```

### 4.2 Package 文件

在 `XLab/xlab/skills/sure_master/` 增加：

```text
xlab.skill.json
SKILL.md
requirements.lock
scripts/run_sure_master.py
scripts/xlab_idea_client.py
schemas/sure_idea_request.schema.json
schemas/sure_idea_batch.schema.json
schemas/sure_round_result.schema.json
schemas/sure_round_summary.schema.json
schemas/sure_master_result.schema.json
```

`xlab.skill.json` 应声明：

- schema version；
- package name/version；
- `visibility: public`；
- package-local Python entrypoint；
- UI arguments：config、task、task-file、run-dir、resume、search-mode；
- 所需 artifact 类型；
- 实际使用的 read/write/bash/subprocess 权限；
- 必要的 secret 名称，但不包含 secret 值；
- XLab research idea 依赖或调用关系。

`SKILL.md` 必须说明 generic XLab runtime 不会自动执行 `runtime.entrypoint`，需要通过 bash 显式启动 package-local launcher。launcher 是长任务，必须等待子进程、记录日志、处理 timeout/cancel，并在完成或失败时调用 XLab finalization 逻辑。

## 5. XLab idea provider 设计

### 5.1 SURE-side client

在 SURE-Evolve 增加一个独立的 `XlabIdeaClient`，避免在 `playground.py` 中散落 XLab 调用。建议职责：

- 校验 XLab deployment profile；
- 建立 XLab operation 或调用 package-local phase CLI；
- 发送 canonical JSON request；
- 接收 batch/result/summary response；
- 记录 request、operation、artifact IDs 和 digests；
- 支持 `generate`、`summarize`、`status`、`resume`、`shutdown`；
- 使用 JSONL 或等价的明确 RPC framing；
- 对断连先查询原 operation，再决定 resume 或失败；
- 检查 immutable input digest，冲突时 fail closed；
- 不自动重试未确认完成的非幂等 GPU/VC 操作。

推荐接口形状：

```python
class XlabIdeaProvider:
    def generate(self, request: dict) -> dict:
        """Return a validated sure.idea_batch.v1."""

    def summarize(self, round_result: dict) -> dict:
        """Return a validated sure.round_summary.v1."""

    def reconcile(self, operation_id: str) -> dict:
        """Read XLab durable state before retrying."""

    def close(self) -> None:
        """Release client-side resources."""
```

具体实现可以复用 XLab `research_idea` 的 `init`、`synthesize`、`resume`、`audit` 和 typed artifact contract，但不得把 XLab skill 的大段自由文本直接当作 SURE plan。必须经过 schema 校验和确定性转换。

### 5.2 Idea request

每轮生成前冻结 `sure.idea_request.v1`，至少包含：

```json
{
  "schema_version": "sure.idea_request.v1",
  "request_id": "...",
  "xlab_run_id": "...",
  "sure_run_id": "...",
  "task_id": "...",
  "task_description": "...",
  "task_card_digest": "...",
  "base_model_profile_digest": "...",
  "search_mode": "ordinary|staged_axes",
  "axis": "arch|train|inference|null",
  "round_index": 0,
  "requested_idea_count": 3,
  "metric": {"name": "WER", "direction": "lower_is_better"},
  "current_best": {"score": 0.0, "code_digest": "..."},
  "history_artifacts": [],
  "execution_contract": {},
  "input_digest": "..."
}
```

输入给 XLab 的经验包括：

- 当前 best solution 和 score；
- 当前 task card 与 base model profile；
- 上一轮 idea batch；
- 所有候选 score、status、failure category；
- 每个 staged rung 的结果；
- checkpoint 和 artifact refs；
- XLab 之前产生的 summary、survey、knowledge graph、paper 和 novelty artifact；
- 资源限制、数据限制、artifact contract 限制；
- 已尝试 idea fingerprints，避免重复探索。

不得传入 API key、VC credential 或不必要的绝对路径。

### 5.3 Idea batch

`SureMaster` 消费前必须确定性校验：

- schema major version 可识别；
- request digest 匹配；
- idea 数量等于 requested count，不能静默减少；
- 每个 idea ID 唯一；
- axis 与 candidate type 合法；
- implementation specification 非空；
- 不修改 evaluator、source directory 或 artifact contract；
- evidence/artifact refs 可解析；
- 没有重复 idea fingerprint；
- idea 能映射到现有 ordinary/staged 输入结构。

校验成功后才允许消耗 GPU/VC 资源。校验失败可在同一 XLab operation 内进行有限 repair；超过 repair 次数则终止本轮并记录 `contract_failure`。

## 6. Round result 和 XLab summary

### 6.1 Round result

每个 ordinary round、每个 staged axis round，以及必要时 combination/selection/holdout 阶段，都要构造 `sure.round_result.v1`。至少包含：

- run、task、search mode、axis、round、phase、rung；
- baseline code/score/digest；
- idea batch artifact ID/digest；
- 每个 candidate 的 idea ID、candidate type、code digest、workspace-relative refs；
- success/status/score/runtime；
- 每个 rung 的执行结果和 promotion decision；
- checkpoint artifact、remote job/result refs；
- failure category 与 reason code；
- ranking、best update、selection/holdout 结果；
- result digest 和 parent lineage。

候选失败必须保留在 result 中。`candidate_failure` 表示 idea/实现本身失败，不能被 XLab 当作系统噪声；`system_failure`、`contract_failure`、`metric_failure` 必须独立记录。

### 6.2 XLab summary

SURE 将 round result 提交给 XLab summarize。XLab 产出 `sure.round_summary.v1`，至少包含：

- 哪些机制在当前 metric 上有效；
- 哪些 idea 失败以及失败证据；
- candidate/system/contract/metric 的分离统计；
- 是否应继续某个方向、axis 或 candidate type；
- 下一轮具体探索方向；
- 已尝试 idea fingerprints；
- 数据、资源、checkpoint 和 artifact contract 限制；
- 相关 paper、survey、knowledge graph、idea、round result 引用；
- summary digest、parents 和 operation trace。

下一轮 request 只引用结构化 summary/result/idea artifact，不再依赖现有 `research_plan_and_result` 的自由文本拼接。原有 `KnowledgePromotion` 可以保留作为本地诊断和兼容输出，但启用 XLab 后不再是下一轮 idea 的权威历史。

## 7. Ordinary 和 staged 的具体行为

### 7.1 Ordinary

启用 XLab 后，每个 research round 执行：

```text
读取当前全局 best
-> 构造 idea_request
-> XLab generate idea_batch
-> 转换为 research_plan
-> 复用 _filter_and_order_ideas
-> 对每个 idea 调 SureRunExp.run
-> 按原规则更新 direction_best_solution 和 global best
-> 构造 round_result
-> XLab summarize
-> 保存 summary artifact
-> 下一轮
```

不改变：

- direction 遍历；
- candidate type 限额；
- score direction；
- best solution 更新；
- candidate workspace 和 artifact guard；
- KnowledgePromotion/WisdomPromotion 的兼容输出。

### 7.2 Staged axes

每个 axis 的每轮执行：

```text
固定 axis baseline
-> XLab 生成该 axis 的 ideas
-> 转换为 axis records
-> short rung
-> rank and keep
-> checkpoint promotion
-> medium/final rung
-> 构造包含全部 candidates/rungs 的 result
-> XLab summarize
-> 下一 axis round
```

保持：

- arch/train/inference 分轴；
- 每个 axis 的独立 history；
- successive-halving；
- `run_existing_code()` 的 checkpoint 重跑；
- late remote result recovery；
- top arch/train/inference 组合；
- selection rerank；
- holdout rerank。

XLab 不替换 staged ranking，也不把仅完成 short/medium rung 的候选误认为 final survivor。

## 8. Durable state、resume 和幂等

SURE 侧新增 operation journal 和原子 projection，至少记录：

- XLab workspace slug；
- XLab run ID、SURE run ID；
- axis、round、phase、rung；
- operation name；
- request/batch/result/summary artifact IDs；
- input、manifest、artifact digests；
- 当前状态、最后确认的 XLab event；
- repair attempt、resume count；
- timestamps、client/protocol version。

每个外部 durable boundary 后写状态：

1. request accepted；
2. batch published；
3. candidate execution started/completed；
4. round result published；
5. summary published；
6. SURE best/checkpoint committed；
7. final manifest published。

断连恢复规则：

- 先用记录的 XLab run/operation ID 查询状态；
- operation running 且可恢复时调用 resume；
- artifact 已发布时不重复执行或提交；
- 相同 digest 的重复提交视为幂等 replay；
- digest 或 immutable input 冲突时 fail closed；
- 不创建第二个 XLab run 来掩盖未知状态；
- 未确认完成的非幂等 SURE GPU/VC job 不自动重跑。

XLab publication 失败应将 run 标记为 incomplete，并允许仅重试 publication，而不是重新训练模型。

## 9. 配置与部署

在 SURE config 增加默认关闭的 `xlab` 段：

```yaml
xlab:
  enabled: false
  idea_provider:
    enabled: false
    package: sure_master
    request_schema: sure.idea_request.v1
    batch_schema: sure.idea_batch.v1
    result_schema: sure.round_result.v1
    summary_schema: sure.round_summary.v1
  workspace_slug: "..."
  launcher_timeout_seconds: 3600
  repair_attempts: 2
  resume_policy: reconcile_then_resume
  deployment_profile: "..."
```

部署 profile 通过运行时注入：

- SURE-Evolve project root；
- Python executable；
- SURE root/PYTHONPATH；
- task card root；
- base model logical mounts；
- VC runner 和 remote runner；
- checkpoint/cache 位置；
- XLab workspace/artifact store。

launcher 启动前 fail-fast 检查输入可读、workspace 可写、source 只读、VC command 可执行、候选输出只能落在 `models/`、`artifacts/`、`metric/`、`working/`。绝对路径仅用于本地进程启动，不写入 public artifacts。

## 10. 兼容性与 feature flags

必须分离两个开关：

```text
xlab.enabled=false
xlab.idea_provider.enabled=false
```

关闭时，当前 `python run.py --agent sure_master ...` 行为保持不变。

启用 XLab idea provider 时：

- XLab 是唯一 idea 来源；
- `ResearchExp` 保留但不被调用为下一轮 provider；
- XLab failure 不静默 fallback；
- SURE 的 candidate execution 和 metric 代码不改语义；
- old KnowledgePromotion/WisdomPromotion 可继续产出兼容 artifact；
- ordinary/staged 的控制逻辑仍由 `SureMasterPlayground` 负责。

这意味着不会破坏现有流程：关闭 feature flag 时逐字回归；打开时只替换“idea 生成与历史总结”两个边界，并通过 schema、digest 和 round-level tests 约束输入输出。

## 11. 关键文件

### XLab

- `xlab/skills/sure_master/xlab.skill.json`
- `xlab/skills/sure_master/SKILL.md`
- `xlab/skills/sure_master/scripts/run_sure_master.py`
- `xlab/skills/sure_master/scripts/xlab_idea_client.py`
- `xlab/skills/sure_master/schemas/*.json`
- `packages/coding-agent/src/core/xlab/product-commands.ts`
- `packages/coding-agent/src/core/xlab/run-manager.ts`
- `packages/coding-agent/src/core/xlab/artifact-store.ts`
- `packages/coding-agent/src/core/xlab/workflow.ts`
- `packages/coding-agent/src/core/xlab/experiment/lifecycle.ts`
- `xlab/skills/research_idea/SKILL.md`
- `xlab/skills/research_idea/scripts/run_idea_phase.py`

### SURE-Evolve

- `run.py`
- `playground/sure_master/core/playground.py`
- `playground/sure_master/core/exp/research_exp.py`
- `playground/sure_master/core/exp/run_exp.py`
- `playground/sure_master/core/exp/knowledge_promotion_exp.py`
- `playground/sure_master/core/exp/wisdom_promotion_exp.py`
- `playground/sure_master/core/utils/candidate_changes.py`
- `playground/sure_master/core/utils/metric.py`
- `playground/sure_master/task_cards/sure_tasks.yaml`
- `playground/sure_master/README.md`
- `configs/sure_master/*.yaml`
- `docs/zh/xlab_sure_idea_integration.md`
- 新增 `XlabIdeaClient`、contract、round-result adapter 和 operation journal 模块

## 12. 分阶段实施顺序

### Phase 1：契约与 fake provider

- 定义四类 schema；
- 实现 canonical JSON、digest、redaction、workspace-relative path 校验；
- 实现 fake XLab provider；
- 为 ordinary/staged 编写 provider adapter 单测；
- 确认旧 provider 关闭时无行为变化。

### Phase 2：SURE round adapter

- 在 ordinary `ResearchExp` 插点接入 provider；
- 在 staged `_generate_staged_axis_ideas` 插点接入 provider；
- 将候选、rung、checkpoint、remote result 和 failure 转成 round result；
- 接入 summary artifact ID/digest 到下一轮 request；
- 加入 operation journal 和 reconcile。

### Phase 3：XLab skill package

- 创建 `sure_master` package；
- 实现 package-local launcher；
- 接入 `research_idea` phase CLI 或等价 typed API；
- 加入 public product command；
- 完成 package validator、manifest 和 completion 检查。

### Phase 4：Fake end-to-end

- fake XLab 生成两轮 ordinary ideas；
- fake XLab 生成 arch/train/inference staged ideas；
- 验证每轮 feedback 会进入下一轮 request；
- 验证 candidate failure 不会被当成 system failure；
- 验证断连、重复 digest、publication retry 和 resume。

### Phase 5：低成本 smoke

- 使用现有 smoke task/config；
- 禁止真实大规模 GPU/VC 搜索；
- 验证 slash command 创建 durable run；
- 验证 XLab idea artifact、SURE result、round summary 和 final manifest；
- 验证取消、失败状态和 resume。

### Phase 6：逐配置迁移

- ordinary local；
- ordinary mixed local/VC；
- staged Icefall；
- staged F5-TTS；
- 最后再执行真实大规模 staged、selection 和 holdout。

## 13. 验收标准

### 静态与契约

- XLab 能发现 `sure_master` package；
- `run-sure-master -> sure_master` 出现在 public catalog；
- package-local 路径和 lockfile 通过校验；
- 未知 schema major、错误 digest、绝对路径和 secret 泄漏会被拒绝；
- required typed artifacts 完整提交。

### Ordinary

- 至少完成两轮 fake ordinary search；
- XLab 生成的 idea 实际进入 `ResearchExp` 替代输入；
- round 结果总结实际进入下一轮 request；
- best solution 和 score 更新规则不变；
- `SureRunExp.run()` 调用顺序和 artifact guard 不变。

### Staged

- arch/train/inference 三轴均可由 XLab 生成 ideas；
- short/medium/final rung 顺序不变；
- checkpoint promotion 和 `run_existing_code()` 不变；
- combination、selection、holdout 不变；
- 所有 rung 结果都能在 XLab summary 中追溯。

### 恢复与失败

- XLab RPC 断开后先 reconcile 原 operation；
- 相同 digest 不重复执行；
- digest 冲突 fail closed；
- candidate/system/contract/metric failure 分类正确；
- publication failure 可单独恢复；
- 中断后可继续当前 round，而不是从头创建第二次搜索。

### 兼容性

- `xlab.enabled=false` 的 ordinary/staged dry-run 与现有行为一致；
- 默认配置不要求 XLab；
- 旧 CLI 仍可直接运行；
- 不修改外部 base model source directory；
- 不需要把 GPU、VC 或 SURE metric 调度迁移到 XLab。

## 14. 可行性评估与必须修正的地方

### 14.1 总体结论

该方案可行，而且与现有代码的职责边界匹配：`SureMasterPlayground` 继续作为唯一实验控制面，XLab 只接管 idea 生成和跨轮总结。现有普通流程在 `playground.py` 的 research round 中调用 `ResearchExp.run()`，随后由 `SureRunExp.run()` 执行候选；staged 流程在 `_generate_staged_axis_ideas()` 生成 axis ideas，随后复用 `_run_staged_records()`、`run()` 和 `run_existing_code()`。因此接入点明确，不需要重写 SURE 的训练、推理、metric 或晋级算法。

### 14.2 当前计划必须明确的关键约束

1. **不能只实现外部 launcher。**

   `run.py --agent sure_master ...` 是一条长时间运行的黑盒 CLI。若 XLab launcher 只在外部启动它，SURE 运行期间没有机会把每轮结果提交给 XLab，也没有机会接收 XLab 的下一批 ideas。必须在 SURE 内增加 `XlabIdeaClient`/provider，并在普通 `ResearchExp` 插点和 staged `_generate_staged_axis_ideas` 插点调用它。

2. **XLab skill package 与 SURE round loop 是两个层次。**

   XLab package 负责创建 durable run、调用 research idea 能力、保存 typed artifacts 和处理生命周期；SURE-side client 负责把每一轮 request/response 接入正在运行的 `SureMasterPlayground`。两者不能由一个仅在任务开始和结束执行的 Python launcher 替代。

3. **需要先确认 XLab 当前 `research_idea` package 的实际状态。**

   Phase 0 必须先完成 package/runtime/schema 对齐；如果 package 尚未可用，需要先补齐通用 `research_idea` v2 skill，再实现 SURE 专用 generate/summarize adapter。当前盘点已确认 main 工作树存在 package-native `research_idea` v2，但其 Survey-oriented contract 仍需 SURE 专用适配层。

4. **不要把 SURE 专用闭环误做成普通 XLab workflow。**

   XLab workflow 可以记录 generate/validate/summarize 阶段和 durable checkpoints，但不应接管 SURE 的候选执行、VC、rung、checkpoint promotion、selection 或 holdout。SURE 的内部 loop 仍是唯一实验调度者。

5. **staged 的反馈粒度需要分层。**

   axis screening 的每个 round 应提交包含所有 candidate/rung 结果的 summary；组合、selection、holdout 也应作为独立 phase/result 追加 lineage。不能把所有阶段硬塞成一个普通 research round，也不能把未完成 final rung 的候选当作最终证据。

### 14.3 不会破坏现有流程的条件

按以下条件接入，不会破坏现有 `sure_master` 自优化流程：

- `xlab.enabled=false` 时，`ResearchExp`、原有 history、KnowledgePromotion/WisdomPromotion 和 CLI 路径完全不变；
- `xlab.enabled=true` 时，只替换 idea provider 和跨轮历史输入；`SureRunExp`、artifact guard、SURE metric、ranking、rung、checkpoint、combination、selection、holdout 保持原调用顺序；
- XLab batch 在 GPU/VC 执行前完成 schema、digest、axis、candidate type 和 artifact contract 校验；
- XLab 失败不静默 fallback，而是产生明确的 bridge/contract failure；
- 每一轮都持久化 request、batch、round result、summary 的 artifact ID 和 digest；
- resume 先 reconcile 原 XLab operation 和 SURE checkpoint，不创建第二个实验；
- ordinary 和 staged fake end-to-end 测试先通过，再启用真实资源。

## 15. 修正后的实施顺序

### Phase 0：依赖和运行时盘点

- 确认 XLab 当前 main 的 `research_idea` skill、schema、artifact hooks 和 product command 是否完整可运行；
- 确认 `research_idea` 的实际输入格式与 `sure.idea_request.v1` 的映射；
- 确认 XLab durable run/artifact store 的调用方式；
- 确认 SURE 侧可访问 XLab 的运行时方式，优先使用显式配置的 XLab root、Python/Node runtime 和 workspace profile；
- 不启动真实 GPU、VC 或外部大规模文献检索。

### Phase 1：契约和 fake provider

- 定义 `sure.idea_request.v1`、`sure.idea_batch.v1`、`sure.round_result.v1`、`sure.round_summary.v1`；
- 实现 canonical JSON、SHA-256 digest、redaction 和 workspace-relative path 校验；
- 实现 fake `XlabIdeaClient`，可返回普通和 staged 的确定性 batch/summary；
- 为非法 batch、重复 digest、candidate/system/contract/metric failure 编写测试。

### Phase 2：SURE 插点和 round adapter

- 在 ordinary `playground.py` research round 调用 provider，替代 `ResearchExp.run()` 的实际 idea 来源；
- 在 staged `_generate_staged_axis_ideas()` 调用 provider；
- 保留现有 plan/record 转换边界，保证 `_filter_and_order_ideas()` 和 `_extract_axis_ideas()` 仍能消费；
- 在候选、rung、checkpoint、remote recovery、combination、selection、holdout 后构造 result；
- 将 summary artifact 引用写入下一轮 request；
- 增加 operation journal 和 reconcile。

### Phase 3：XLab skill 和 durable adapter

- 补齐或适配当前 runtime 的 `research_idea` package；
- 创建 `sure_master` public skill 和 `run-sure-master -> sure_master` catalog 条目；
- 实现 package-local launcher，但让 launcher 启动 SURE 时使用已配置的 SURE-side XLab provider；
- 将 generate/summarize 操作与 XLab durable run/artifact lifecycle 对接；
- 完成 manifest、lockfile、schema、completion 和 package validator 检查。

### Phase 4：Fake end-to-end与兼容性回归

- 完成至少两轮 ordinary fake search；
- 完成 arch/train/inference staged fake search；
- 验证每轮 summary 会进入下一轮 request；
- 验证 candidate failure 不会被误报为 system failure；
- 验证断连、相同 digest replay、publication retry 和 resume；
- 验证 `xlab.enabled=false` 的旧 ordinary/staged dry-run 行为不变。

### Phase 5：低成本和真实 smoke

- 使用已有 smoke task/config，验证 `/xlab run-sure-master` 能创建 durable run；
- 验证 XLab idea artifact、SURE round result、round summary 和 final manifest；
- 先执行 local 或 fake VC smoke，再逐步启用 mixed local/VC；
- 最后再执行真实大规模 staged、selection 和 holdout。

## 16. 最终结论

计划是可行的，但必须按上述修正实施。核心不是“给 SURE 加一个 XLab 命令”，而是建立一个可恢复的双层闭环：

```text
XLab durable research run
  -> generate idea batch
  -> SURE round execution
  -> publish round result
  -> XLab summarize
  -> next SURE round request
```

SURE 的实验执行和裁决逻辑不会被破坏；真正变化的是 `ResearchExp` 的 idea 来源，以及下一轮使用的历史经验来源。最大的实现风险是 XLab skill/runtime contract 与 SURE contract 的适配，以及错误地把 XLab 作为任务级外部 launcher 而没有接入 SURE 的每轮 research 插点。这两项应在实现前通过 Phase 0 和 fake provider 测试解决。
