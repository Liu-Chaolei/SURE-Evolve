# SD 部署登记、归档与恢复

当前激活版本为 **SD Native-MCTS A/B v5**，登记 ID
`sd-native-mcts-ab-bf16-b48-20260917-v5`。模型角色分工、ZAI/ZHOU/OPENAI/XCODE
路由与验证记录见 [API 角色配置](sd_api_role_policy.md)。v4 已停止并保留，
v5 复用验证基线，但不混用旧模型策略下的 MCTS 检查点。

旧部署 SD Direct-24 A/B 的完整数据集、训练设置、逐候选 Search/Holdout 结果和
失败记录见 [旧部署完整实验结果](sd_legacy_results_20260917/report.md)。

## 两个独立实验版本

| 名称 | 登记 ID | 实验定义 |
|---|---|---|
| **SD Direct-24 A/B** | `sd-direct24-ab-bf16-b48-20260915` | 旧版直接生成；A 提供 24 条冻结文献证据，B 不提供文献 |
| **SD Native-MCTS A/B v3** | `sd-native-mcts-ab-bf16-b48-20260915-v3` | 新版五模式 MCTS、完整本地检索、融合与符号记忆；B 关闭文献输入 |

登记文件为 `runs/sd_deployments.json`；当前部署指针为 `runs/current_sd_ablation.json`。
这两个版本的候选、缓存、研究历史和实验结果分别保存，不合并统计。C/D 均不运行。

## 旧版保留内容

原目录保持不动：

```text
runs/sd_ablation_20260915_bf16_b48_v1
```

完整备份：

```text
runs/archives/sd-direct24-ab-bf16-b48-20260915
runs/archives/sd-direct24-ab-bf16-b48-20260915.manifest.json
```

归档时约 8.1 GB，4,828 个文件通过 SHA256 校验。保留模型包、评分报告、预测、
源码快照、配置、生成/评审/定稿缓存、控制器状态、失败收据、调度修订和 Python
环境记录。符号链接按链接保留，`.env` 不进入备份。另有
`sd-direct24-ab-bf16-b48-20260915.dependencies.json` 记录 148 个外部文件的 SHA256，
覆盖共享音频、数据清单、参考标注、初始化及说话人嵌入权重；恢复检查会复验这些文件。

旧版公共基线已完成：BF16、每卡 batch 48、16 epoch 早停，search DER 为
12.91375%。A/B 尚未完成候选实验。旧 B 留有生成中间产物，不能当作已验证的
候选结果。两组未完成的 API 操作及 `transport_error` 诊断均原样保留。

原科学执行入口保存为 `launch.engine.sh`；`launch.sh` 已改为带归档和共享资源
检查的入口。备份里的原始启动脚本也保留，不应绕过登记工具同时启动两个版本。

### 校验和恢复旧版

在仓库根目录校验备份，此命令不调用模型或启动训练：

```bash
/shared/chaolei.liu/data/sure_asr_controller/bin/python -m \
  playground.sure_master.tools.sd_deployments check \
  --name sd-direct24-ab-bf16-b48-20260915
```

恢复前先查看各组 `last_result.json`、控制器日志、`xlab_operations.json` 和对应
生成缓存。对于已核查、决定重试的失败操作，使用其真实 operation ID：

```bash
/shared/chaolei.liu/data/sure_asr_controller/bin/python -m \
  playground.sure_master.tools.sd_deployments retry-failed \
  --name sd-direct24-ab-bf16-b48-20260915 \
  --group A --operation ACTUAL_OPERATION_ID
```

该命令只处理指定的 incomplete 收据，备份原收据并重新计算文档校验和；不会
立即发送 API 请求，也不删除已完成的模型响应缓存。不要在未核查请求状态时
使用，不要手工删除整个操作记录或清空研究历史。

核查模型服务、数据与资源分配后启动：

```bash
bash runs/sd_ablation_20260915_bf16_b48_v1/launch.sh --resume-archived
```

恢复会保留原控制器 run ID、实验历史、基线和缓存，只重置监督器中已经停止的
进程条目。若还有未核查的 API 操作或另一部署仍占用 SD 资源池，启动会拒绝。
旧 Slurm 作业失效后，应先取得并核验新的授权资源，再记录仅涉及调度的迁移；
不能假定旧作业编号永久可用。

如果原目录受损，从备份恢复到原路径后再核验；不要直接移动目录，因为旧记录
中含有绝对路径。恢复后的管理入口应重新接入登记工具，而非绕过共享资源检查。

## 新版原生部署

新目录：

```text
runs/sd_native_mcts_ab_bf16_b48_20260915_v3
```

新版通过 `prepare_sd_ablation --baseline-from OLD_BASELINE_DIR` 导入基线，模型包
和必要证据复制到新版基线目录，`baseline/baseline_import.json` 记录来源、
科学配置指纹、训练依赖及评分证据。`baseline/result.json` 明确标注 imported，
不伪装成新版重新训练的结果。

导入校验允许目录和研究引擎变化，但严格检查模型及评分源码、训练参数、
初始化权重、数据和参考文件、精度、推理配置、工作镜像及训练完成证据。
旧 A/B 的研究历史、模型响应缓存和生成中间产物不进入新版。

新版设置：

- A/B 各一次搜索，六轮、每轮四个候选。
- 原生五模式 MCTS，每模式最多 64 次迭代；深度 3、分支 3、探索常数 1.2。
- 原生融合和 `candidate_outcome` 符号记忆，无额外组件实验，不降级直接生成。
- A 使用完整冻结文献资源包，B 为 task-only，保留 MCTS 和记忆。
- BF16 训练、每卡 batch 48、四卡全局 192；主参数和优化器状态保持 FP32。
- n04/10085 与 n05/10086，每台两个独占槽位，全局最多四个候选执行。
- 每个训练候选四卡、32 CPU、120 GiB 内存；不使用 10024/10029。
- A/B 均完成选优后再打开 holdout。

完整文献包位于 `external/XLab/.xlab/runs/sd-literature/`，在原生工作目录内，
确保请求序列化使用相对路径；`bundle_snapshot.json` 记录本次冻结的
24 个文件，包括 18 个声明资源文件及 survey 等入口产物。每组的
`artifacts/xlab_native_identity.json` 绑定其代码、模型配置与资源身份。

两组先回放导入基线，核对原始 DER 差异不超过 `1e-6`；之后启动原生研究。
v2/v3 核验科学配置、模型包、评分报告和资源分配与 v1 一致后，复用 v1 已完成的
A/B 回放，分别为 `0.12913750000000002` 和 `0.1291375`。各组
`baseline_replay.json.origin` 记录来源和校验和，不把复用描述成重新评分或训练。
首轮原生候选必须通过五模式搜索、融合来源和 A/B 证据策略验证才能提交执行。
实际检索、生成和候选耗时单独记录，不沿用旧 Direct-24 的耗时估计。

启动或恢复新版：

```bash
bash runs/sd_native_mcts_ab_bf16_b48_20260915_v3/launch.sh
```

登记工具持有跨部署执行锁，并检查存活的控制器和候选进程；监督器退出后，
只要还有实验子进程存活就继续保留租约。它不会取消整个 Slurm allocation。
所有训练与 Docker 操作继续遵循 `/shared/chaolei.liu/rules.md` 和集群手册。

检查 `workflow.json`、各组 `baseline_replay.json`、控制器日志和原生产物获取
真实阶段。原生身份校验通过、生成正在进行、首轮候选验收完成，是不同状态；
不能仅因控制器启动就声称完整自进化已经跑通。

### v1 启动诊断保留

`runs/sd_native_mcts_ab_bf16_b48_20260915_v1` 保留为失败诊断版本，未完成原生候选实验。
A 因文献入口在原生工作目录之外而无法序列化请求；B 的分析输出遗漏父方案
标签/领域字段，触发成熟方案契约校验，且完整任务文本作为公开 topic 时含有本地路径。
v2 将实现任务放入私有上下文，使用简短科学主题，并在分析提示中明确要求原样保留
`tags`、`root_domains`。路径与父方案契约校验继续启用，搜索预算不变。
v1 的操作收据、模型缓存、失败产物和原源码快照均保留，不转用为 v2 研究历史。
修复验收记录见当前 v3 的 `startup_acceptance.json` 和 `activation_audit.py`。

`runs/sd_native_mcts_ab_bf16_b48_20260915_v2` 也保留为诊断版本。B 已进入 MCTS，
但未完成候选训练；A 的文献摘要响应被通用策略附加了接口不允许的 `ablation` 字段。
v3 将策略改为遵循每个操作的固定输出字段，并将公开产物里的私有任务上下文
替换为摘要指纹；完整上下文仍留在私有状态和模型输入中。路径处理也经过幂等性测试。
v2 的 `interruption_audit.json` 记录了 B 停止时的进程及未确认响应，不能把中断操作
当作完成。v3 不复用 v2 的候选搜索历史或模型缓存。

本次修复的 SURE 定向测试 11 项、XLab 原生及公开边界测试 17 项通过。
`npm run check` 仍被既有 `xlab-experiment-materializer.test.ts` 未使用导入警告阻断，
不能视为全仓检查通过。

### 2026-09-16 正式 A/B 流程启动

用户确认开始正式 A/B 训练后，发现 B 在首种搜索模式结束时因校验器错误停止。
显式空检索之后生成失败，应保存 error 与对应 blocker；旧校验却把这个组合当成
整个模式无效。修复只放行有对应 blocker 的错误记录，不把失败分支改成成功，
不改变搜索、评分和训练设置。16 项工作流测试、7 项部署管理测试通过。

A 的存活搜索继续执行。B 使用 `external/XLab-validation-r1` 的独立修复源码，
`B/search` 指向新的 `B/search_validation_r1`，以生成独立运行身份。原工作区保存在
`B/search_before_validation_r1`，原生成记录保存在 `B/xlab_ideas`；新生成记录在
`B/xlab_ideas_validation_r1`。B 已失败的搜索不算作完成的独立实验，不混入新历史。
A/B 的基础训练配置与公共基线保持一致；B 的校验代码差异、原文件哈希和新身份
见 `formal_start.json`，恢复脚本与旧配置见 `recovery/B-validation-r1/`。

已替换停止的监督器并接管存活的 A，使用原全局资源锁；接管不会忽略其他部署的
运行进程。维护命令 `launch --adopt-running` 仅用于此类同部署接管，必须先确认
旧监督器已退出并处理失败组，不能用来并发启动同一实验。

正式配置为 BF16、每卡 batch 48、10085/10086、最多四个候选并行。流程会在完整
原生候选批次通过验证后自动提交执行；启动监督器不等于模型训练已经开始。
实际阶段以 `workflow.json`、当前组工作区的训练提交记录和训练日志为准。

### 2026-09-16 切回 Direct-24

按用户要求，当前指针已切回旧版 Direct-24，新版停止并保留失败现场。
旧版备份 4,828 个文件和运行目录中 1,338 个冻结源码/配置文件校验通过；
两个已确认失败的桥接收据已备份并解除重试阻塞，原模型响应缓存未删除。
旧基线继续复用，BF16、每卡 batch 48、10085/10086 和四候选并行设置不变。

切换时控制器接口实测 HTTP 200，XI 候选生成接口实测 HTTP 401，因此旧实验
尚未实际恢复。等待进程见旧目录的 `resume_status.json`；它在 `.env` 更新后
检查接口，也每五分钟复查一次。两个原模型接口都通过后，会执行旧版
`launch.sh --resume-archived`，仍经过归档检查和跨部署资源锁。切换当前部署
指针会取消此等待进程。具体状态与诊断日志见 `resume_status.json` 和
`resume_wait.log`，不能把 `waiting_for_api` 当成训练已启动。

随后用户授权 XI 不可用时先使用 GLM。已取消 XI 等待进程，并将旧版 A/B 的
分析、生成、评审、定稿统一切换为 `glm-5.3-flash`；控制器也继续使用该模型。
通过旧版冻结传输代码的真实 JSON 响应测试后，执行原归档恢复入口。
切换记录和配置前后哈希见旧目录 `glm_fallback.json`，启动日志为 `launcher.glm.log`。

XI 缓存仍在各组 `xlab_ideas` 中；GLM 使用新的 `xlab_ideas_glm`、
`xlab_operations_glm.json` 和 `xlab_history_glm.json`，不会把 XI 的中间方案作为
GLM 候选输入。切换前没有完成的候选实验；基线与训练合同保持原值。
当前显示名称为 **SD Direct-24 A/B (GLM)**，与原 XI 版本、Native-MCTS 版本区分。

### 四台机器、八槽位扩容（镜像仓库已恢复）

用户新增授权 `10087/n06`、`10088/n13`。四个 allocation 实测均为八张 910B3，
合计 **32 张卡**；每台两个独占槽位，每个训练候选四卡、32 CPU、120G 内存。
每组仍最多同时执行四个候选，共享资源池上限为八个。精度、batch、训练预算、
数据与评分协议没有变化。

新增两台已完成 Qwen 服务 step 的交接，batch holder 停留在 STOP 状态维持
allocation；没有取消整个 allocation。`allocation_locks/handoff-10087.json`
及 `handoff-10088.json` 保存节点与 holder 身份。

控制器使用 `controlplane/parallel8_v1/sd_ablation_workflow.py`，只替换调度校验
以支持四个 allocation。训练 worker 继续使用原冻结源码，source digest 未变，
存活 srun 通过原收据重新连接，没有重新开始训练。执行合同的变化仅为
`slurm.existing_allocations` 与 `slurm.max_parallel`，迁移前状态及配置均保存。
八进程并发、第九个等待等六项测试，以及镜像就绪门控测试通过。

新增节点首次拉取镜像时，发现 n01 的 `slurm-registry.service` 因依赖失败处于
inactive，5000 端口拒绝连接；当前账号 `sudo -n systemctl start slurm-registry`
需要密码，最初未能直接恢复，当时仅原两台、四槽位可用。
三个尚未执行的镜像拉取失败请求已保存诊断并重置，未作为科学候选失败计入。

`image_readiness.json` 将新增节点暂时禁用。`prepare_images.py` 等待 Registry
恢复，随后通过 allocation 内的 `slurm-docker-run` 拉取同一不可变镜像，成功后
动态开放新节点，无需重新启动控制器。它不修改 Registry 系统服务或绕过权限。
状态见 `parallel8_migration.json` 与 `controlplane/parallel8_v1/image_readiness.json`。

另对已确认从 VBx 漂移到 AHC 的 B1 取消了该次推理 step，原请求和结果证据保留，
明确标记 `implementation_drift`、无有效分数；这不是训练失败或方法效果结论。
原始证据及调度迁移脚本位于 `recovery/parallel8_20260916/`，镜像故障恢复记录
位于 `recovery/parallel8_registry_recovery/`。

用户要求复核集群说明后，进一步读取最新 `/shared/cluster/OPERATIONS.md`，确认
n01 Docker 权限等价本机 root 是文档明确接受的信任边界。2026-09-16 13:34
使用已有本地镜像、只读容器及 systemd private bus，启动原有 Registry 与文档站
服务。没有修改服务配置、Slurm、Docker data-root 或其他用户容器。
由于挂载恢复脚本会执行 `restart shared.mount`，先确认共享盘和网络已经 active，
再以 `--job-mode=ignore-dependencies` 仅启动两个服务本体，未重启正在使用的共享盘；
共享盘的 ActiveEnterTimestamp 保持 2026-09-15 20:20:22。

手册网页恢复后已核对使用规范、Docker 权限边界及运维章节。Registry HTTPS
使用正常 CA 校验返回 200，n06/n13 均通过 `srun` 内的 `slurm-docker-run` 成功拉取
并运行同一 digest 的镜像。**四台、32 张卡、八候选槽位现已就绪**；n06 已开始
承接候选，n13 可供后续调度。实际并行任务数取决于各组当前候选数。
恢复证据见 `registry_recovery.json`；最初阻塞信息保留在镜像就绪记录中。

### API 调用优先级（历史四级配置）

此前 A/B 控制器与候选研究阶段共用四级本地 API 路由；此顺序已被下方的
“当前仅 ZAI”策略替代。原顺序为：

```text
XI_API_KEY → ZAI_API_KEY → ZAI_API_KEY2 → OPENAI_API_KEY
```

XI 默认模型为 `gpt-6-astra`；两把 ZAI key 默认使用 `glm-5.3-flash`，第二把 key
默认沿用 `ZAI_BASE_URL`；最后一级使用 `OPENAI_BASE_URL`，默认模型为
`gpt-6-astra`。可通过 `XI_MODEL`、`ZAI_MODEL`、`ZAI_MODEL2`、`OPENAI_MODEL`
和 `ZAI_BASE_URL2` 显式覆盖相应配置。缺少的 key 跳过；不会向错误地址发送另一家凭据。

401/403、429、连接故障和服务端错误触发后备接口。429 优先读取 Retry-After
或 ZAI 报告的额度重置时间；鉴权失败短暂冷却后再按原顺序尝试。普通请求参数/
上下文错误返回调用方处理，不靠切换 key 隐藏。上游流式响应完成后才交给调用方，
避免半段回答或工具调用已交付后再拼接另一家结果。所有接口都不可用时明确失败。

本地服务仅监听 `127.0.0.1:18991`，使用单独的 0600 权限令牌；真实上游 key
只由路由服务读取 `.env`，不写入运行配置、模型 worker 或调用日志。
服务位于 `runs/services/api_priority_v1`，与实验资源租约分开。
控制器使用的 `sure-priority-router` 是路由别名；实际接口及模型以
`runs/services/api_priority_v1/requests.jsonl` 为准，不能把该别名视作固定模型。

已发布批次、评分及训练结果继续复用。新生成使用各组 `xlab_ideas_priority`；
旧 GLM/XI 缓存保留，桥接收据中的已发布记录不删除。切换时 B 的第二轮未发布
请求被中断，完整记录保留；该请求按新路由策略重新生成。训练参数和存活 worker
不变。审计及配置备份见 `api_priority_revision.json` 和对应 `recovery/priority-api-*`。

八项路由单测通过，真实研究 JSON 与控制器 SDK 流式调用通过。实测 XI 返回
401 后自动切到第一把 ZAI key 成功；后两级的切换顺序已通过模拟故障测试，
不代表这两把 key 的生产可用性已得到验证。

### 当前仅 ZAI

用户随后要求停止尝试 XI，当前启用顺序为：

```text
ZAI_API_KEY → ZAI_API_KEY2
```

XI 与 OPENAI 均不在当前请求链中；两把 ZAI 都不可用时明确报不可用，不转到
其他提供方。模型均为 `glm-5.3-flash`。顺序从
`runs/services/api_priority_v1/router.json` 动态读取，后续修改顺序不必重启训练。
额度冷却记录持久化到该目录的 `cooldowns.json`，路由服务重启也不会清除冷却。

本次切换时现有请求在实际重启前已结束，没有重启 worker。十一项路由测试通过，
在线 `/health` 已确认只列出两把 ZAI key。切换证据见 `api_zai_only.json`。
第一把 key 仍有五小时额度冷却，第二把也曾触发短时限流；服务是否成功响应以
实际请求日志为准，不能仅凭路由服务存活就认定模型接口可用。

### ZAI 恢复后的并发排队修复

再次实测发现第一把 ZAI key 可完成正式请求，但并发过多时返回 1302（账号速率
限制）；第二把返回 1113（余额不足或无可用资源包），不是普通短时限流。
路由代码增加每把 key 的独立并发上限，以及冷却期间的有界等待，生产上限为
每 key 四个请求；超额请求在本地排队，单次等待窗口最多 2400 秒。窗口耗尽仍
明确返回不可用，不保证两把 key 长期不可用时实验可以继续。

此调整只影响 API 调度，不改变八个训练槽位。保留既有冷却状态，额度重置时间
从解码后的 JSON 错误中识别；1113 暂避五分钟，并在日志记录错误码，不记录 key。
路由健康接口现在报告各 key 正在处理的请求数和本地排队数。十五项测试覆盖
四个并发、第五个等待、短时冷却后继续、调用顺序及缓存冷却恢复。实际部署与
续跑状态以 `api_queue_recovery.json`（生成后）和在线 `/health` 为准。

### 2026-09-17 释放 n06、n13

按用户要求释放 10087/n06、10088/n13；先关闭候选准入，确认两台机器没有候选
步骤和槽位租约，再取消这两个 Slurm allocation。当前资源池保留 10085/n04、
10086/n05，共 16 张卡、最多四个候选并发。已有训练、评测与结果继续保留。

`image_readiness.json` 的 `released_jobs` 明确排除已释放作业，镜像预热也跳过它们。
冻结实验配置中的历史 allocation 列表保持原样以保留检查点身份；有效准入以动态
资源门控为准。释放记录由 `runs/current_sd_ablation.json` 的
`resource_release_record` 指向。

### 2026-09-17 恢复 OPENAI 兜底

用户明确当前顺序为 `ZAI_API_KEY → ZAI_API_KEY2 → OPENAI_API_KEY`，已热更新服务策略及 A/B 的 API 配置，不重启训练。XI 仍不启用。验证时第一把 ZAI 为 1310 冷却、第二把为 1113；OPENAI 探测与正式排队请求均返回 HTTP 403，尚未恢复生成。记录见运行目录 `api_openai_fallback.json`。

### OPENAI 403/1010 请求头修复

同一密钥、模型与端点的对照请求表明，urllib 默认 User-Agent 被 Cloudflare 拒绝；显式 `SURE-Evolve/1.0` 返回 200。传输层增加该默认标识并尊重调用方提供的 User-Agent，16 项相关测试通过。重启独立路由服务加载修复，只清除 OPENAI 的失效冷却记录，保留 ZAI 冷却及三路调用顺序；正式路由实测 OPENAI 返回 200。备份并协调 A 第五轮未完成操作收据后从检查点续跑，不删除已发布候选。审计见 `recovery/openai-user-agent-1789629898550326557/resume.json`。

### 2026-09-17 最终评测并行

将剩余 selection 的基线与两名候选改为三路并行；两组 selection 都完成并冻结胜者后，两组各自的基线与胜者 holdout 最多四路并行。新增任务交替优先使用 10085/n04 与 10086/n05，继续遵循原独占槽位锁与 Slurm GRES。保留原基线 srun 并通过原请求收据重连。冻结 worker 源码、配置身份、模型、评分协议与实验编号不变，控制器扩展位于 `controlplane/parallel8_v1/parallel_final.py`。

并发测试验证三项 selection 真正同时执行、实验编号及 winner 一致，以及 holdout 必须等待两组 selection 完成；跨组 holdout 调度亦并行。当前每项推理为单卡，剩余独立任务最多四项，因此没有宣称占满 16 卡；会话级拆分需要另行验证 RTTM 合并和指标一致性。本次未增加重复评测或修改推理 batch。审计见 `parallel_final_revision.json`。

### 2026-09-17 全部候选 holdout

按用户要求停止新增 selection，评测 A/B 全部已保留有效模型的候选。两组原 selection 此前已完成，保留历史结果。本次清点为 40 个有效候选加 1 个基线；34 项复用已完成快照评测（核验模型清单哈希），A-r6-i2 接续正在执行的正式 holdout，其余 6 项新增推理。6 个没有通过验证的候选列入 excluded.json。

独立运行目录 `runs/sd_all_candidates_holdout_20260917_200910`，入口 `evaluate.py`，当前指针 `runs/current_sd_all_candidates_holdout.json`。只使用 10085/n04、10086/n05，每任务 1 NPU、16 CPU、48G，两个原父槽位各容纳两个推理子槽位，每机最多四项。与原评测通过父锁互斥，已启动任务不取消；释放资源后自动填充。新增六项可全部并行，其余结果直接复用，不重复计算凑满卡数。

清单、结果指纹及父/子槽位互斥检查通过。结果持续写入该目录的 results.csv、results.md、status.json，不向搜索回写 holdout，也不训练或调试新的候选。

### 2026-09-17 Native-MCTS v4 启动

当前新部署为 `sd-native-mcts-ab-bf16-b48-20260917-v4`，目录 `runs/sd_native_mcts_ab_bf16_b48_20260917_v4`。v1/v2/v3 诊断版本保留，v4 使用当前 SURE/XLab 冻结快照、已验证基线及独立原生研究目录。A 文献资源预检通过，B 仍为 task-only；保持五模式 MCTS、六轮四候选、BF16、每卡 batch 48、4 卡训练。

从真实 v3 失败记录定位延后评估分支的 applied/error 双终态：v4 冻结 XLab 中更新既有暂定终态为失败并移除失败的可选边，保留诊断节点和 blocker，严格重复记录校验不放宽。同时补齐分析/重规划 prompt 的父方案身份契约。68 项原生搜索、恢复、适配和工作流测试通过；测试依赖单独放在 validation_deps，未修改共享运行环境。两项修复及外部代码清单随运行保存。

A/B 通过统一路由 `ZAI_API_KEY → ZAI_API_KEY2 → OPENAI_API_KEY`，不使用 XI。只用 n04/10085、n05/10086，并与旧部署全候选 holdout 共用 allocation_locks，保留旧任务。启动检查阶段 A/B 控制器已运行，首轮候选及实际训练仍待产出验证，不能把控制器存活视为首轮成功。参见 startup_revision.json、workflow.json。
