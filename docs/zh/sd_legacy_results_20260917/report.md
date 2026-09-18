# SD 旧部署完整实验结果汇总

## 1. 范围与结论

- 实验：**SD Direct-24 A/B**，登记 ID `sd-direct24-ab-bf16-b48-20260915`。
- 本报告仅涵盖旧 Direct-24 流程，不包含 Native-MCTS 新部署。
- 两组六轮搜索及原定最终评测已完成；追加全候选 holdout 于 **2026-09-17 21:40:58（北京时间）**完成。
- 共 40 个有效候选（A/B 各 20）及 1 个基线，41 项 holdout 全部成功；6 个失败候选未纳入有效模型评测，2 个预算槽位未生成。
- 按 selection 选定模型，A 的 holdout 比基线差，B 略优于基线；本次结果**不能支持文献输入显著改善泛化性能**。
- 全候选 holdout 最低值是事后探索性结果，不等价于未经测试集选择的最终模型成绩。

## 2. 数据集

任务为 AMI 会议语音说话人分离（speaker diarization）。清单使用 AMI `Array1-01.wav` 单通道远场音频，抽查训练与 holdout 音频为 16 kHz 单声道。训练只使用下面列出的 AMI 清单；原配方目录名含 AliMeeting/AISHELL4，不表示本次混入这两个数据集。

| 划分 | 会话数 | 会议组数 | 音频时长（小时） | UEM 时长（小时） | 用途 |
|---|---:|---:|---:|---:|---|
| train | 109 | 28 | 57.8071 | 57.8063 | 训练模型参数 |
| train_validation | 3 | 1 | 1.2172 | 1.2172 | 早停与最优检查点选择 |
| search | 8 | 2 | 3.3426 | 3.3425 | 反馈自进化、比较候选 |
| selection | 4 | 1 | 2.5867 | 2.5867 | 原定流程的候选复评与选优 |
| holdout | 12 | 3 | 6.5313 | 6.5312 | 最终评测及追加的全候选分析 |

总计 **136 个会话、71.4849 小时音频**。数据划分记录 seed=42；训练随机种子为 3407。按清单的会议组 `group_id` 检查，五个划分两两无交叉；这不等价于已核实所有说话人身份完全不交叉。

评测会话：

- **train_validation**：`IS1007a, IS1007b, IS1007c`。
- **search**：`ES2011a, ES2011b, ES2011c, ES2011d, IS1008a, IS1008b, IS1008c, IS1008d`。
- **selection**：`TS3004a, TS3004b, TS3004c, TS3004d`。
- **holdout**：`ES2004a, ES2004b, ES2004c, ES2004d, IS1009a, IS1009b, IS1009c, IS1009d, TS3003a, TS3003b, TS3003c, TS3003d`。

完整会话列表、清单路径和 SHA-256 见 [results.json](results.json)。已核验当前清单哈希与旧部署冻结指纹一致。

## 3. 实验设计与训练设置

| 项目 | A 组 | B 组 |
|---|---|---|
| 文献输入 | 24 条冻结证据、16 条参考文献记录 | 关闭文献输入 |
| 实测反馈 | 保留 | 保留 |
| 研究流程 | Direct-24：分析、生成、审核、定稿 | 同左 |
| 计划搜索预算 | 6 轮，每轮最多 4 个候选 | 同左 |
| 实际发布/执行槽位 | 24 | 22 |
| 有效候选 / 执行失败 | 20 / 4 | 20 / 2 |
| 控制器记录的成功训练候选 | 9 | 7 |
| 独立重复实验 | 1 次 | 1 次 |

旧部署不是五模式 MCTS，也不应被描述为新部署的符号记忆消融。C/D 组取消，没有实验结果。B 第六轮的生成收据明确记录 `Generation attempt budget exhausted`，只发布两项；不能写成两组均完成了 24 个候选。成功训练候选数不含基线，也不等同于实际训练尝试次数。

基线与训练契约：

- DiariZen，WavLM Base+ 初始化，Conformer 分割网络，WeSpeaker ResNet34-LM 说话人嵌入，AHC 聚类。
- BF16 autocast，FP32 主参数；每卡 batch=48，4 卡/global batch=192，梯度累积=1。
- 训练种子和 shuffle seed=3407；最多 100 epochs，每 epoch 验证，patience=10。
- 按验证 loss 选最优 5 个检查点平均；训练从 SSL backbone 加新分离头初始化，不继承父候选优化器状态。
- 基线 WavLM/网络学习率分别为 2e-5/1e-3；训练片段 8 秒、shift 6 秒。候选可在允许边界内修改架构、训练机制和推理设置。
- 基线推理：8 秒窗口，segmentation_step=0.1，batch=32，AHC threshold=0.7，min_cluster_size=30，median filtering 开启。
- 使用 Ascend 910B3，主要在 n04/n05 执行，运行中也曾扩展 n06/n13。并发度随资源调整，不是固定的 8 路。
- 基线训练 16 epochs 早停；验证 loss 从 0.5322 降至 epoch 6 的最佳 0.4099。

## 4. 统一评分协议

- 指标：DER，越低越好。表中结果均为百分数，原始产物保存 0–1 比例；极端候选可超过基线很多。
- 冻结 SURE pipeline：`sd.any.der.meeteval_v1`，meeteval 0.4.3，collar=0.25 秒。
- 参考和预测 RTTM 按固定 manifest UEM 区域裁剪，regions=`all`。
- **逐会话 DER 的算术平均**（`session_mean_error_rate`），不是按语音时长加权的 corpus DER。
- 同一划分下基线和候选使用相同协议；search、selection、holdout 的数值不能跨划分当作同一测试集比较。
- 绝对下降（百分点）=基线 DER−候选 DER；相对下降=（基线−候选）/基线×100%。正值表示改善。
- 此实验未执行“统一协议 vs 非统一协议”的回放对照，因此仅凭 A/B 结果不能证明统一评测机制的因果有效性。

## 5. 原定选优流程结果

下表是原定 selection 流程选出的模型；后来用户要求不再新增 selection，但已完成的 selection 结果仍应如实保留。

| 模型 | Search DER (%) | Selection DER (%) | Holdout DER (%) | Holdout 绝对下降（百分点） | Holdout 相对下降 (%) |
|---|---:|---:|---:|---:|---:|
| 基线 | 12.9138 | 25.6050 | 9.2483 | 0.0000 | 0.0000 |
| A-r6-i2 | 8.6675 | 9.5675 | 9.9433 | -0.6950 | -7.5149 |
| B-r3-i3 | 8.5238 | 9.7400 | 9.2050 | 0.0433 | 0.4686 |

A 选中 A-r6-i2：在继承模型上将 AHC `min_cluster_size` 调至 60（threshold=0.65、step=0.05）。B 选中 B-r3-i3：实际执行的是 AHC threshold=0.65、step=0.05；不能把其结果归因于候选原始标题中的 VBx。

Selection 全部记录：

| 组别 | 候选 | Selection DER (%) | 选中 |
|---|---|---:|---|
| A | A-r6-i2 | 9.5675 | 是 |
| A | A-r5-i1 | 10.0300 | 否 |
| A | baseline | 25.6050 | 否 |
| B | B-r3-i3 | 9.7400 | 是 |
| B | B-r6-i1 | 9.7400 | 否 |
| B | baseline | 25.6050 | 否 |

## 6. 全候选 Holdout 汇总

这部分先后复用了 34 项已完成快照评测、1 项正式 holdout，并补跑 6 项，共 41 项。对所有有效模型做冻结推理，不重训、不用 holdout 分数回写自动搜索。

| 组别 | 有效候选数 | Holdout 最低值候选 | 最低 DER (%) | 相对基线下降 (%) | 中位 DER (%) | 平均 DER (%) | 优于基线的候选数 |
|---|---:|---|---:|---:|---:|---:|---:|
| A | 20 | A-r2-i3 | 9.1217 | 1.3696 | 9.9288 | 10.4582 | 1 |
| B | 20 | B-r2-i3 | 9.0875 | 1.7391 | 9.5521 | 11.3647 | 6 |

A-r2-i3 的改动记录以 Conformer 层数为 6 为主要结构改动；B-r2-i3 的改动记录以 dropout=0.2 为主要改动。完整声明字段与模型路径见 JSON/CSV；继承的父方案也属于实际系统，不能把成绩全部归因于单个参数。

平均值和中位数仅描述本次候选集合，候选之间有继承关系和重复配置，不是独立重复实验。Holdout 最低候选没有按独立验证集预先选定，只能作为探索性上界。

## 7. 全部候选明细

编号 `A-r2-i3` 表示 A 组第 2 轮第 3 个槽位。`--` 表示无有效结果，绝不表示 DER=0。类型及失败原因来自最终候选状态产物。

### A 组

| 候选 | 类型 | Search DER (%) | Holdout DER (%) | 相对基线下降 (%) | 状态 |
|---|---|---:|---:|---:|---|
| A-r1-i1 | arch | 12.8688 | 9.4333 | -2.0004 | success |
| A-r1-i2 | inference | -- | -- | -- | artifact_guard_failed |
| A-r1-i3 | arch | 13.4600 | 9.2742 | -0.2793 | success |
| A-r1-i4 | fine_tune | -- | -- | -- | artifact_guard_failed |
| A-r2-i1 | fine_tune | 13.0988 | 9.4600 | -2.2887 | success |
| A-r2-i2 | fine_tune | 13.4000 | 10.3525 | -11.9391 | success |
| A-r2-i3 | arch | 12.9025 | 9.1217 | 1.3696 | success |
| A-r2-i4 | inference | 15.0663 | 10.2333 | -10.6506 | success |
| A-r3-i1 | inference | 8.9337 | 11.1608 | -20.6794 | success |
| A-r3-i2 | fine_tune | 13.7225 | 10.5208 | -13.7592 | success |
| A-r3-i3 | arch | 13.1263 | 9.6867 | -4.7396 | success |
| A-r3-i4 | inference | 8.9025 | 9.8942 | -6.9832 | success |
| A-r4-i1 | inference | 8.9337 | 11.1608 | -20.6794 | success |
| A-r4-i2 | inference | 8.8212 | 9.5967 | -3.7664 | success |
| A-r4-i3 | inference | -- | -- | -- | boundary_error |
| A-r4-i4 | arch | 8.8125 | 10.0158 | -8.2988 | success |
| A-r5-i1 | inference | 8.6675 | 9.8183 | -6.1633 | success |
| A-r5-i2 | inference | -- | -- | -- | execution_failed |
| A-r5-i3 | arch | 9.5913 | 9.9700 | -7.8032 | success |
| A-r5-i4 | inference | 8.8913 | 9.9142 | -7.1995 | success |
| A-r6-i1 | inference | 8.6913 | 9.8400 | -6.3975 | success |
| A-r6-i2 | inference | 9.5675 | 9.9433 | -7.5149 | success |
| A-r6-i3 | inference | 9.0800 | 10.0983 | -9.1908 | success |
| A-r6-i4 | inference | 24.6687 | 19.6683 | -112.6689 | success |

### B 组

| 候选 | 类型 | Search DER (%) | Holdout DER (%) | 相对基线下降 (%) | 状态 |
|---|---|---:|---:|---:|---|
| B-r1-i1 | inference | -- | -- | -- | implementation_drift |
| B-r1-i2 | fine_tune | 13.5062 | 9.4375 | -2.0454 | success |
| B-r1-i3 | arch | 13.0087 | 9.4425 | -2.0995 | success |
| B-r1-i4 | fine_tune | 13.5337 | 10.1758 | -10.0288 | success |
| B-r2-i1 | inference | -- | -- | -- | execution_failed |
| B-r2-i2 | inference | 12.9650 | 9.3550 | -1.1534 | success |
| B-r2-i3 | arch | 12.6263 | 9.0875 | 1.7391 | success |
| B-r2-i4 | inference | 15.1363 | 10.0408 | -8.5691 | success |
| B-r3-i1 | fine_tune | 13.6500 | 10.8475 | -17.2914 | success |
| B-r3-i2 | inference | 16.9950 | 16.3092 | -76.3471 | success |
| B-r3-i3 | inference | 9.7400 | 9.2050 | 0.4686 | success |
| B-r3-i4 | fine_tune | 13.1612 | 10.2400 | -10.7227 | success |
| B-r4-i1 | inference | 9.9075 | 9.6617 | -4.4693 | success |
| B-r4-i2 | inference | 9.9075 | 9.6617 | -4.4693 | success |
| B-r4-i3 | inference | 12.5088 | 9.1458 | 1.1083 | success |
| B-r4-i4 | arch | 12.7112 | 9.3325 | -0.9101 | success |
| B-r5-i1 | inference | 25.4312 | 19.3675 | -109.4161 | success |
| B-r5-i2 | inference | 25.6925 | 19.0300 | -105.7668 | success |
| B-r5-i3 | inference | 8.5625 | 9.2358 | 0.1352 | success |
| B-r5-i4 | inference | 25.4312 | 19.3675 | -109.4161 | success |
| B-r6-i1 | inference | 9.7400 | 9.2050 | 0.4686 | success |
| B-r6-i2 | inference | 12.5087 | 9.1458 | 1.1083 | success |
| B-r6-i3 | -- | -- | -- | -- | not_generated |
| B-r6-i4 | -- | -- | -- | -- | not_generated |

失败分类：`artifact_guard_failed` 为产物契约校验失败；`boundary_error` 为修改边界校验失败；`implementation_drift` 为实际实现偏离审核方案；`execution_failed` 为执行失败；`not_generated` 为生成预算耗尽未发布。没有用父模型替换失败候选制造 holdout 成绩。

## 8. 解释限制

1. 每组只有一次独立搜索、一个训练种子，没有多次重复或置信区间。B 相对基线约 0.0433 个百分点的优势不能宣称统计显著。
2. 实际预算不完全相同：A 执行 24 项，B 执行 22 项；成功训练候选也分别为 9/7。
3. 旧部署期间模型接口经过多次恢复和路由切换，包括 ZAI/GLM 与后期 OPENAI；A 后续轮次与 B 的执行时间不同，不能假定两组全程使用完全一致的 LLM 提供方。新 v5 的角色策略不属于本次旧部署结果。
4. 部分候选调试中改变了原假设，例如 VBx 意图变为 AHC；必须以实际模型和改动产物解释结果。已明确拦截的失败候选单列，但不能声称每个成功候选都严格复现原计划。
5. 用户在 A 搜索尚未结束时要求提前查看全部候选 holdout，快照评测独立执行且不自动回写搜索，但测试集已提前暴露，不能把整个过程描述为从未查看过 holdout 的严格盲测。
6. 基线 selection DER=25.6050%，明显高于其他划分，提示划分之间难度差异；不能据此认定评分异常，也不能跨划分计算改善率。
7. Search 上 A/B 均明显改善，但 A 被选模型在 holdout 退化，提示搜索/选优收益未稳定泛化。当前数据不支持“A 加文献稳定优于 B”这一结论。

## 9. 产物与复核

- [48 个计划槽位完整 CSV](all_candidates.csv)：含成功、失败、未生成及产物路径。
- [结构化结果与数据集清单指纹](results.json)：含 40 个候选有效结果、原定 selection/holdout、会话列表及来源哈希。
- 原始运行目录：`/shared/chaolei.liu/SURE-Evolve/runs/sd_ablation_20260915_bf16_b48_v1`。
- 全候选 holdout 目录：`/shared/chaolei.liu/SURE-Evolve/runs/sd_all_candidates_holdout_20260917_200910`。
- 可复核生成器：`runs/build_sd_legacy_report.py`。本次汇总不启动训练或修改任何原始评分。
- 旧部署已完成，SD 计算资源已按用户要求释放。
