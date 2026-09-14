# ASR 修正版：六轮、每轮四候选

配置为 `configs/sure_master/asr-formal-6x4-npu4-v2.yaml`。沿用旧 SURE-Evolve 的
ordinary / architecture_only / full_during_search 流程，XLab 只负责想法生成与
研究反馈；不调用其 run_experiment，不增加组件消融。基线一次，六轮各四候选，
正常共 25 次完整训练；最多四候选同时运行，每个训练条件独占四张 NPU。

统一配置：全量 TEDLIUM3、30 epoch、FP32、seed 42、每卡 max-duration 900、
SpecAugment 与 MUSAN（p=0.5、SNR 10–20），固定官方 Unigram-500。结构候选继承
父方案的设计而非权重，训练/解码方法不可修改。SURE 环节使用 ZAI glm-5.3-flash，
XLab X1–X4 使用 XI gpt-6-astra，凭据来自现有 .env，不进入工作节点请求。

## 数据与推理修正

`prepare_asr_corrected` 在独立数据目录重建小写监督文本并保留 `<unk>`，复用
TEDLIUM 特征，另外准备 MUSAN 特征。数据指纹包含原监督、tokenizer、特征分片和
MUSAN。训练节点的数据 staging 同时包含 MUSAN；推理阶段不加载训练噪声。

所有阶段使用 (epoch-20, epoch-30] 的最后十轮平均，语义等同于
`--epoch 30 --avg 10 --use-averaged-model true`。CPU 平均后导出带哈希的
`averaged-model-epoch-30-avg-10.pt`；解码复用该权重，模型前向仍在 NPU。
保留已验证的 PackedSequence 修复，避免短语音批量推理的帧错位。

模型包保留 epoch-20、epoch-30 和实际平均推理模型，分别记录续训 checkpoint、
inference_checkpoint、tokenizer、平均窗口和源权重哈希；清理后可独立回放。
完整训练已经完成而仅解码失败时跳过重新训练。更换数据或 tokenizer 后不能续接旧权重。

regular 200 条用于搜索；六轮后 selection 196 条只比较新基线与 regular 前两名；
test 1155 条只比较基线与最终胜者，不重训。旧基线 7.41% 和官方本地 5.17% 仅为参考。

## 启动与 n09 交接

先用 CPU Slurm 作业准备新数据，运行离线协议检查。通过以下入口等待资源就绪、
冻结 SURE 源码及 Icefall 源码，然后启动原来的 `tedlium_workflow --stage search`：

```bash
/shared/chaolei.liu/data/sure_asr_controller/bin/python -P -m playground.sure_master.tools.launch_asr_corrected \
  --config configs/sure_master/asr-formal-6x4-npu4-v2.yaml \
  --output /shared/chaolei.liu/SURE-Evolve/runs/asr_formal_6x4_npu4_v2_20260913_151000 \
  --preparation-job 10014 \
  --checks-receipt /shared/chaolei.liu/data/asr_v2_preparation/check_result.json \
  --handoff-job 9980
```

当前仅授权替换 n09 的 9980，9991/n13 与 9992/n11 为保护对象。新基线先在现有
9980 allocation 内请求独占训练 step；确认请求存在后，停止原 batch srun client
以维持 allocation，再向原服务 step 9980.0 显式发送 TERM 并确认其退出，让新训练
step 获取资源。不能仅使用默认 scancel：它可能等待已暂停的 srun client 处理请求。
不取消整个 9980 作业，不改写 Slurm 配置的 NPU 可见设备编号。

接管期间实际限额沿用原 allocation 的 64 CPU、256G 内存，训练使用四张 NPU。
同一 allocation 只承载一个候选，其他候选可转入正常 Slurm 队列；正常新分配沿用
80 CPU、500G、1T 临时盘。allocation 到期后使用现有 checkpoint 断点恢复。
不提供 --handoff-job 时直接通过普通 Slurm 提交。

新请求就绪后按运行记录停止/退役旧 ASR 流程，保留旧模型与报告。
`runs/current_asr_evolution.json` 指向唯一新流程，文件锁防止重复控制器。

## 监控与验证

- `activation.json` / `activation.log`：资源准备、控制器启动和交接状态。
- `workflow_state.json` / `controller.log`：正式搜索状态。
- 实验 `metric/slurm/*/allocation.json` 或 `job.json`：现有分配或普通作业归属。
- 实验 `metric/resource_usage.jsonl`：实际容器 RSS、CPU 亲和性和 cgroup 内存限额。
- 训练日志、每轮 WER、平均模型收据及最终 selection/test 报告用于验收。

执行针对性的离线单元测试，不运行独立 smoke、短训、benchmark 或 API 测试。
MUSAN 资源准备完成后直接开始正式基线。

当前容器 cgroup 命名空间不能直接显示宿主 Slurm 内存组，可能出现空组的零用量与
无限限额；判断内存应读取宿主 `/sys/fs/cgroup/memory/slurm/uid_22637/job_9980/`
或运行目录中的 `resource_samples.json`。容器 RSS 汇总仍可用，但会重复计算共享页。

恢复已经交接的运行时，直接从已有源码快照执行 `tedlium_workflow --stage search`
并传入原 deployment.yaml 和原输出目录；不要重新冻结配置或再次交接服务。
