# XLab 原生研究与 SURE 候选反馈

新配置使用 `xlab.sure.native.v1`：完整本地检索、原生五模式 MCTS、原生融合和
SURE 可执行性评审。每轮精修轮初冻结的最佳方案，不启动 XLab 组件消融实验。
训练、推理和实测评分仍只由 SURE 的正常候选执行流程负责。

## 配置

ASR、TTS、SD 的配置模板已显式启用：

```yaml
xlab:
  idea_generation:
    engine: native
    max_attempts: 8
    feedback_kind: candidate_outcome
    refinement: current_best
    allow_auxiliary_experiments: false
    mcts:
      max_iterations: 64
      max_depth: 3
      branching_factor: 3
      exploration_constant: 1.2
```

每次候选生成分别运行五种模式后融合；64 是**每模式**迭代上限，不是模型调用
总数。候选槽位、训练预算和各任务的模型路由保持由原配置决定。生成尝试、调用
超时或资源问题不能触发直接生成降级；失败保留诊断与检查点。

`XLAB_SURE_FLOW_FIRST=1` 与此模式冲突，会报错。固定 evidence 摘录不能替代
完整资源包；文献模式需要 `XLAB_SURE_SURVEY_PATH` 和其声明的全部资源。

## 符号记忆与精修

`RoundResult` 发布 `parent_snapshot` 和 `evaluation_context`，`RoundSummary`
发布 `symbolic_memory`。记录使用 `xlab.candidate_outcome.v1`，包含候选、父方案、
实现与结果摘要、干预集合、评测范围、指标方向和分数。下一轮将本运行、任务与
评测条件匹配的记录交给 XLab 原生 `MemoryState`，参与节点生成、评估和重规划。

改善/下降/持平只表示完整候选在记录条件下相对父方案的观察。组合修改不产生
单组件因果结论；失败不记为机制无效，不可比较或缺失的分数不强行比较。
旧的组件移除记忆类型仍用于 XLab 原有用途，SURE 不伪造这种记录。

首次缺少结构化父方案时，根据现有实现和模型配置生成一次描述，按输入和模型
身份缓存。下一轮只有经 SURE 确认的最佳方案才能替换父方案。树中的 remove /
replace 是设计编辑，不代表额外执行移除消融。

## SD A/B 和本地资源

- A：`use_literature: true`、`use_feedback: true`。每轮重新生成查询，执行
  OutcomeRAG、图谱扩展、证据排序、关键论文处理及搜索内部组件检索。
- B：`use_literature: false`、`use_feedback: true`。只使用任务、源码接口及本组
  实验记录；不读取文献库或相关缓存，论文引用为空。五模式搜索与融合保留；
  理论迁移无证据时跳过，机制创新保留原生空证据路径。新颖性仅为模型判断，
  文献支持检查明确记为不适用。

两组使用相同 MCTS 上限，但实际调用量可能不同。新的 A/B 对照测量完整文献
增强研究能力的贡献，不能与旧的“24 条冻结证据”对照混用。

`prepare_sd_ablation` 对新部署复制完整资源包并记录所有声明文件的 SHA256，
包括模型文件、图谱、向量索引、survey 和关键论文缓存。资源复制不在线检索或
下载论文。其通用冻结入口也可用于其他任务：

```bash
"$XLAB_PYTHON" "$XLAB_ROOT/xlab/skills/sure_master/scripts/xlab_idea_client.py" \
  --freeze-survey /path/to/survey/artifacts/survey.json /new/path/literature_bundle
```

输出包含新的 `survey_path`。已有部署继续使用自己的冻结代码和协议；使用新
版本应创建新的部署目录，不能在正在执行的 A/B 轨迹中途替换生成算法。

## 产物与恢复

检查候选原生产物中的 `research_policy`、五模式 `mcts_evolution`、融合来源及
检索轨迹。搜索元数据记录 `memory_digest`、`memory_records` 和 `novelty_source`。
完整候选必须携带五模式真实搜索和融合产物，控制器拒收直接生成的伪装结果。

模式检查点绑定完整上下文、记忆、模型配置和资源身份；已发布的适配器批次
也校验请求、运行配置及资源身份。控制器启动时通过 `--check-native` 绑定
`artifacts/xlab_native_identity.json`；代码、模型或资源变化会阻止原部署恢复，
防止本地批次缓存绕过检查。未确认完成的操作保持 incomplete，
保留原有人工检查/恢复边界，不自动重复发送不确定的模型操作。

验证使用模拟模型和模拟候选执行，覆盖完整原生工作流、两轮反馈、B 组文献
隔离、资源重定位、缓存失效及零额外实验子进程，不需要启动正式训练。
