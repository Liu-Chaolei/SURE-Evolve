# TTS 部署登记与恢复

本次切换保留两个独立实验。权威索引为 `runs/tts_deployments.json`，当前运行指针为
`runs/f5tts_active_run.json`；状态以各运行的流程状态、结果文件及 Slurm step 为准。

| 名称 | 目录 | 研究方式 / 起点 |
| --- | --- | --- |
| TTS-Direct-v11 | `runs/f5tts_premium_evolution_20260914_v11_direct` | 简化生成；保留第一轮及第二轮已有结果 |
| TTS-Native-MCTS-v12 | `runs/f5tts_premium_evolution_20260916_v12_native_r12_glm` | 原生五模式 MCTS、完整本地检索与融合；重新从原始微调 baseline 起步 |

Direct 于 2026-09-16 按用户要求恢复，恢复点为第 13 个 epoch 完成、19,006 步。
第二轮第三个候选接续训练；
另外两个推理候选的评分保留。原目录未移动，源码、模型、生成语音、日志、已接受
idea 和恢复状态均保留。`pause_20260915` 保存暂停回执、依赖哈希及 Python 包版本。

## 查询与恢复旧部署

```bash
python runs/tts_deployments.py status
python runs/tts_deployments.py check-old
/shared/chaolei.liu/data/sure_asr_controller/bin/python -P runs/tts_deployments.py resume-old \
  --config runs/TTS-Direct-v11/deployment.resume_glm.yaml
```

`check-old` 只读校验，不执行训练。`resume-old` 必须在 Native 已停止、资源锁空闲时
使用；不会自动停止 Native、恢复 Qwen 或取消顶层 Slurm 分配。它校验旧依赖与断点，
归档主动暂停造成的过期调度回执，再通过旧冻结入口恢复。断点之后未提交的少量更新
会重新计算；已完成候选及已接受 idea 不重做。不要直接改旧部署的科学配置来恢复。

本次恢复使用独立的 `deployment.resume_glm.yaml` 和 `glm_profile.py`，原始配置与
冻结源码保留。科学配置与原版一致；仅将后续研究调用改为 GLM Chat Completions，
禁止 XI。切换记录见 `resume_glm_receipt.json`。Native r12 因融合来源校验失败保持
停止，修复前不要启动；当前活动部署为 Direct。

## 启动与恢复原生部署

```bash
/shared/chaolei.liu/data/sure_asr_controller/bin/python -P \
  runs/f5tts_premium_evolution_20260916_v12_native_r12_glm/launch.py --stage all
```

入口先检查冻结文件，执行八卡 8→16 步恢复验收，然后生成正式第一轮三个候选，
最后继续完整搜索。重复启动会受启动锁保护；已通过验收及已发布候选批次会复用。
也可使用 `--stage search` 恢复已开始的搜索。不得并发运行两个启动入口。

两版共享原始 50 epoch baseline（正式 search CER 2.280023586%），但不共享候选历史
和可写研究缓存。Native 独立计数 6–10 轮，每轮 3 个自由候选；训练型完整 50 epoch，
纯推理型使用轮初父模型。训练配置保持 25,600 frames/卡、AMP BF16、fused AdamW、
expandable_segments、8 卡和原始数据/评测协议。各模式 MCTS 上限 64 次迭代，并非
整轮只有 64 次模型调用。

r4 修复了分析/反馈重规划漏写 `tags`、`root_domains` 的情况：仅缺失键从成熟父方案
继承，显式不同值和错误类型仍拒绝。继承发生在根节点签名前，原始响应保留，分析
产物附带继承记录。相同修复已落到 XLab 主仓库；r4 不迁移之前的研究缓存。

r6 增加 GLM 文献排序兼容：当模型只返回 `answer` 列表时，将该列表投影到
`evidence_ids`，保留顺序并记录原始响应摘要。重复、未知或缺失 ID 仍被拒绝；
如果显式提供了错误的 `evidence_ids`，不会用 `answer` 覆盖它。

r7 同时兼容关键论文处理中的 `answer` 包装。评分字符串仅在该包装内按 JSON
数值解析，随后仍要求 0–100 的整数；普通 `score` 字段类型要求不变。摘要包装
必须满足完整字段集合。原始 ProviderResult 和请求缓存保持不变。

Native 的运行源码在自己的 `source` 和 `xlab_source` 中，完整文献包在
`literature_bundle` 中。部署后不要修改这些文件；身份变化会阻止恢复。
r5 研究及编码均使用 ZAI/glm-5.3-flash，研究通过流式 Chat Completions 调用。XI 已禁用；OPENAI_API_KEY 保留作未启用的备用配置。密钥仍从私有环境文件读取。

## 恢复与资源边界

Native 使用增量 SSE 和持久化请求缓存。单个故障操作最多 30 分钟、3 轮恢复、
每轮最多 3 次 HTTP 尝试；鉴权、格式及契约错误不盲目重试。完整搜索不受这 30 分钟
限制，单批外层时限为 24 小时。未知的进行中模型调用会阻止重发，需检查对应
`*.operation.json`；不删除操作目录、不清零预算、不切换回简化生成。

10024、10029、10052 各提供 8 卡，继续使用统一资源锁。每个分配最多一个候选；
只停止候选 step，不取消顶层分配或恢复被接管服务。分配到期后需要正常续接资源。

每小时定时监控继续保持取消。正常流程日志和有限请求恢复不等于定时监控。

## r13 融合修复（2026-09-16，已验证、未启动正式流程）

修复快照为 `runs/f5tts_premium_evolution_20260916_v12_native_r13_glm`。
当前运行指针和旧部署保持 Direct；Native 别名仍保留失败的 r12，便于追溯。

融合和局部修复提示增加按模式分组的组件名称、描述和证据 ID 清单。证据只能使用
输入中的原始 ID；嵌套搜索轨迹的标题、说明和评审文字不能充当组件或证据。
失败草稿的所有来源错误、允许值及原始草稿会传给下一次语义重试。来源校验不放宽，
不自动补配证据，重试预算仍为五次。

主仓融合测试 33 项、相关原生链路测试 27 项、r13 冻结源码融合测试 33 项通过。
真实 GLM 使用 r12 已保存的五模式输出进行隔离验证：第二份草稿通过，完成融合评分
和局部修复，总计八次调用。原始请求和响应保存在
`runs/tts_native_fusion_repair_20260916`，不会导入正式研究缓存。
这项检查的上下文由保存文件重建，是融合链路验证，不是完整首轮候选验收。

r13 资源预检通过，完整训练验收身份与原八卡验收相同，baseline 无需重训。
详见该目录的 `validation_report.json`。正式切换还需执行 r13 原生首轮候选验收，
避免与 Direct 的完整搜索控制器同时启动。全仓 npm 检查仍受已有
`xlab-experiment-materializer.test.ts:14` 未使用 `canonicalDigest` 导入阻断。

## 请求级 API 回退（r20，2026-09-16）

按用户最新要求，研究 API 调用顺序为：`XI_API_KEY` → `ZAI_API_KEY` →
`ZAI_API_KEY2` → `OPENAI_API_KEY`。这替代之前禁用 XI 的偏好。
统一调用层会在鉴权、限流、连接或接口响应失败时尝试下一项，成功后立即停止回退。
XI/OpenAI 使用 Responses，ZAI 两个 Key 使用 Chat Completions；ZAI2 默认共用
`ZAI_BASE_URL`，也支持显式 `ZAI_BASE_URL2`。

路由策略不包含密钥值，随快照保存并加入原生身份、缓存和检查点依赖。每次响应保留
实际模型、选中 Key 的名称和回退记录；切换模型时不会伪装成原模型。所有接口失败
仍受有限恢复预算约束，未知进行中调用仍要求核对，原始缓存和重试记录不清零。

真实请求已验证 `XI 401 → ZAI 429 → ZAI_API_KEY2 成功`；最后的 OpenAI 接口也已
通过实际 XLab 请求头下的 Responses 流式 JSON 测试。此前 urllib 探针的 403
不代表当前 XLab 请求方式不可用。

当前候选验收目录为 `runs/f5tts_premium_evolution_20260916_v12_native_r21_ordered`，
冻结部署相关测试 89 项通过，资源预检通过。它只运行 `ideas`，不会启动新版训练；
旧 Direct 仍是活动训练部署。进度和验收记录见该目录的状态文件及
`runs/tts_native_acceptance_status.md`。

r20 的传输回退已生效，但 HTTP 200 的评分内容不符合契约，导致生成中止。
r21 将各调用方原有内容校验放入回退链：返回格式或必填字段不合格时继续尝试
下一项，保留原始响应及原因。合法低分、停止修复和审核拒绝仍是有效结果，
不会为了获得更高分或通过审核而换接口。未完成三候选验收前，不宣称验收通过。

2026-09-16 用户随后要求跳过 XI：当前有效顺序为 `ZAI_API_KEY` →
`ZAI_API_KEY2` → `OPENAI_API_KEY`。私有 `.env` 的 XI 密钥已改名保留为
`XI_API_KEY_DISABLED_BY_USER`，调用层每次重新读取凭据时直接跳过 XI，
不会发起 XI HTTP 请求。冻结策略和搜索身份保持不变，无需重启或重做搜索。
切换时已经在途的请求可能仍持有旧凭据；详见 r21 的 `xi_disabled_receipt.json`。

## r22 可靠性修复与完整验收（2026-09-17）

新验收目录为 `runs/f5tts_premium_evolution_20260917_v12_native_r22_reliable`，
只运行 `ideas`。融合调用层和下游共用完整校验；舍弃来源按模式与组件名去重，
选中组件仍要求唯一。新颖性校验、接口冷却、缓存重验及迭代断点已接入。
断点保留搜索树、随机状态和请求出现次数；接口故障不会被记成科学搜索完成。
大型生成与融合输入去除重复提示内容，实际 API 顺序继续为 ZAI、ZAI2、OpenAI。

冻结源码 128 项测试和资源预检通过，真实融合链路及整理/审核诊断通过。
诊断不计入正式候选；正式仍需 3 个不同候选完整审核通过，不能据此提前启动训练。
原 baseline 验收身份不变，无需重训。r21 结果全部保留，但提示与可靠性身份改变，
不伪造兼容签名复用旧模式；迁移决定见新目录的 `migration_report.json`。

旧 Direct 曾自行因 LLM 调用失败停止，已在 01:58 按原配置恢复控制器。
全局活动训练指针保持 Direct，定时监控仍取消。当前进度以实际流程状态和日志为准。

02:02 复查：该 Direct 恢复尝试再次停止，原因为候选 `data_requirements` 不是
非空列表。旧部署当前阻塞，不能视为仍在训练；原有代码、配置与结果均保留。

## r22 整理阶段恢复（2026-09-17 中午）

当前操作配置是 r22 内的 `materialization_revision_2/deployment.yaml`。
这是有独立源码/配置哈希记录的公共产物整理修订，原科学快照未修改。
API 回退校验与落盘映射共用标题、贡献、假设、方法、风险及引用一致性规则。
图谱新增引用只能从带资源哈希的精确论文 ID/标题记录补齐，并写入审计引用表。

已从保存的融合产物恢复，五模式及迭代文件共 10 个哈希均未改变，融合结果未改变，
搜索及融合新增调用均为 0。只重新请求一次整理（约 56 秒），后续重放成功响应完成
映射和原生审计。首个候选已通过审核，正式接受数为 1/3，继续生成其余候选。
详见 `materialization_revision_2/verification_report.json`。未启动新版训练。
