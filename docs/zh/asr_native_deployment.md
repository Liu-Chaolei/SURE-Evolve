# ASR 原生研究部署与旧部署保存

2026-09-17 新部署名为 **ASR-Native-MCTS-v2**，目录
`runs/asr_native_mcts_v2_20260917`。旧正式部署为 **ASR-Direct-v1**：
`runs/asr_formal_6x4_npu4_v2_recovery_20260914_121324`。
失败的 `asr_native_mcts_v1_20260915_r4` 已停止交接等待器并归档，保留诊断产物，
不迁移其搜索缓存。名称记录在 `runs/asr_deployments.json`；活动训练指针为
`runs/current_asr_evolution.json`。

## 科学与执行边界

- 导入原始 baseline-only 运行，regular200 WER 为 5.6946006749%，不重训基线。
- 独立六轮、每轮四个结构候选；不导入 Direct 候选历史、最优方案或权重。
- 五模式原生 MCTS，每模式 8 次迭代、深度 3、分支 3、探索常数 1.2。
- 全 TEDLIUM3，30 epoch，FP32，seed42，MUSAN，官方 Unigram500，每卡 duration900。
- 每候选独占 4 NPU、32 CPU、128G，两台机器各两个候选。
- 只修改六阶段 `num-encoder-layers`、`encoder-dim`、`feedforward-dim`、
  `encoder-unmasked-dim`；总层数≤20、encoder≤512、FFN≤1536、宽度32对齐。
- SURE 负责执行与评测；不调用 XLab run_experiment，不做额外消融或短训。
- 保留正确 PackedSequence 解码与 `(20,30]` 原生累计平均导出、greedy context2/max1。
- regular200 用于搜索；selection196 比较基线和 regular 前两名；test1155
  只评基线和 selection 胜者，不重训，不将 test 反馈给研究。

## 最新可靠性实现

冻结当前 XLab 完整原生研究实现，包括树/RNG/请求计数断点、资源身份校验、
基于实测候选的符号记忆。`asr_native_support` 从已验证的 TTS 包装修复提取通用
引用校验与审计逻辑；不包含 TTS 保存的响应、实验分支或训练配置。
`asr_native_reliability.py` 提供 ASR 入口：

- 原生研究直接逐请求路由：ZAI_API_KEY → ZAI_API_KEY2 → OPENAI_API_KEY。
  ZAI 为 glm-5.3-flash/Chat Completions，OPENAI 为 gpt-6-astra/Responses。
  冻结策略只有这三个槽，明确排除 XI；key2 默认复用 ZAI_BASE_URL。
- 保存实际模型、槽位、内容校验与冷却回执；有效负面判断不触发换供应商找好评分。
- 持久化每操作30分钟/3恢复周期/每周期3次HTTP预算，重启不清空预算。
- 最终 materialization 与融合结果的科学字段一致；校验后才能发布。
- 只允许已检索证据中的论文 ID 延后补齐书目；保留原始 ID 并标记 deferred，
  审计明确显示引用未核实。未知证据、捏造标题或结构漂移仍阻断。
- 只在精确 summary schema 中移除额外 `ablation: []`，不补造缺失字段，
  不删非空消融；记录转换并保留原始模型响应。
- 移除对合法文本中嵌套大括号的误判，仍拒绝明确模板占位符。

SURE 控制器使用独立 ASR loopback 路由；native 不再被其二次包装。
凭据只从 `.env`/私有 token 注入，不写入配置或文档。

## 启动与交接

```bash
python -m playground.sure_master.tools.prepare_asr_native --old-run OLD_RUN --output NEW_RUN
python -m playground.sure_master.tools.asr_deployment_ops prepare-ideas --root NEW_RUN
python -m playground.sure_master.tools.asr_deployment_transition --root NEW_RUN
```

`prepare-ideas` 是正式首轮研究，导入基线但不提交训练、不切换活动指针。
交接需要四个通过完整评审的候选，以及旧 Direct 的最终 selection/test 完成。
v2 对已完成旧运行不执行 checkpoint-pause；保留旧路径、配置、模型、评测和恢复入口。
只有交接完成才写 `TRAINING_ENABLED`；四个不同槽位出现首批有限 loss 后，
写 `training_verification.json`。研究失败时等待器只记录问题，不清缓存或重试预算。
`TRANSITION_STOP` 停止尚未交接的等待器。

旧 Direct 的24个候选已完成 regular 评测。最后一轮非必要的 LLM 文字总结失败，
运行目录中的 `controller_entry.py` 仅对 `exp_29_knowledge_promotion` 使用确定性结果汇总；
备份为 `controller_entry.before_finalization.py`，记录为 `finalization_amendment.json`。
冻结科学源代码未修改，后续仍通过原流程做真实 selection/test，不能将 regular 当 test。

## 资源、恢复与监控

保留10023/n09与10028/n03父allocation，不取消、不唤醒原Qwen holder。
两台机器共用原槽位锁目录。保护n11/n13及其他所有服务。

10023到期为2026-09-20 17:04:37，10028为当日23:52:05（北京时间）。
现有Slurm执行器在父allocation到期后转普通sbatch队列，沿同一候选请求从checkpoint
恢复；队列不保证立即获得16卡。忙碌但未到期时不额外抢占服务。
完成epoch后的checkpoint可恢复模型、优化器、调度器和进度；未保存工作可能补跑。

独立小时监控仍位于 `/shared/chaolei.liu/asr-monitor/formal-v2`，不写入训练代码。
任务说明已更新为同时监控旧最终评测和v2研究，交接后监控v2。
不要因旧Direct完成而停止整个监控，也不要自动重启归档的v1。

离线验证覆盖结构编译、部署恢复、API槽位/协议/冷却/实际模型、有限恢复、MCTS断点、
materialization契约、引用延后和原生流水线。原XLab一个写死XI优先的测试不适用于ASR
三槽策略，使用专门的 `test_asr_native_routes.py` 验证；未运行训练smoke。
