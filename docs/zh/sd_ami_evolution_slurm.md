# AMI DiariZen 四卡结构自进化

采用 `ordinary-sd-npu.yaml` 的官方 WavLM-updated 协议：WavLM-Base+ SSL 初始化，
其余 SD 网络随机初始化；最多 100 epochs、验证 Loss patience 10、最优五个 checkpoint 平均。
每个训练候选使用 4 NPU、每卡 batch 16，候选串行；两轮、每轮四个结构候选。
DER 使用 SURE 的 collar=0.25 秒、会话平均和 UEM 裁剪。

遵循 `/shared/chaolei.liu/rules.md` 及 <http://127.0.0.1:18088/>。
控制器在 n01，模型执行通过 Slurm 和 `slurm-docker-run`，使用内部 Registry digest。
单作业时间片为 24 小时，到期自动恢复完整 checkpoint；取消作业不会自动重提。
当前部署按用户要求关闭预计训练耗时的 24 小时暂停门槛。

## 准备

1. 用 `prepare_task_data sd` 生成五个 AMI 划分及 `preparation.json`。
   适配器在堆叠 batch 前取第 0 声道，并把二值标签转换成 FP32，兼容 ES2010d 双声道及 Ascend MSE。
2. 用 `prepare_wavlm_initialization` 转换官方 `microsoft/wavlm-base-plus`。
   转换工具产生 `.provenance.json`；成品 DiariZen v2 checkpoint 不可替代它。
3. 准备 `pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin`，检查下载大小及官方 SHA256。
4. 按 `docker/sure-master-diarizen/README.md` 构建、推送镜像。
   评分使用独立 `/opt/sure-metrics/bin/python`，镜像包含 md-eval-22.pl，不在运行时下载评分程序。
5. SD survey 使用已校验的 `XLab/.xlab/runs/20260910-survey-sd-v2/artifacts/survey.json`。
   LLM 凭据通过现有 `.pi/agent` 私有配置加载，不写入镜像、部署 YAML 或模型 worker 请求。

## 验证和启动

`probe_sd_training --config ... --workspace ...` 必须在分配了四卡的容器中运行：
预热 20 次、测量 100 次更新并做完整训练验证，保存状态，再启动新进程恢复到第 121 次更新。
它产生 `acceptance.json`，不会产生正式 `training_completion.json`。

`probe_sd_inference --config ... --training-workspace ... --workspace ...` 在单卡容器中
使用上述组件 checkpoint，推理最短的完整 search 会话并调用独立 SURE 评分进程。
组件 DER 仅用于链路验收，不进入正式候选排名。

在仓库根目录，用协调器解释器生成冻结部署：

```bash
python -m playground.sure_master.tools.prepare_sd_deployment \
  --image registry.cluster.local:5000/users/chaolei-liu/sure-diarizen@sha256:a414f32bf5be92d773639555403573ee4da1d87458c2103e581ad85b66c0a8de \
  --output runs/sd_ami_official
bash runs/sd_ami_official/launch.sh
```

首次会训练基线，随后自动运行两轮、selection 和 holdout。
相同部署再次启动复用已保存的控制器/候选状态；修改初始化、数据、训练协议或代码后使用新目录。
训练候选不能继承上一候选的优化器或额外训练时长。

## 结果与恢复

- `search/workspaces/task_0/metric/controller_state.json`：轮次状态。
- 各实验 `metric/slurm/*/job.json`：Slurm job ID 与提交状态。
- 各实验 `models/diarizen_training/training.log`：真实训练日志。
- `models/diarizen_training/state/checkpoint-*`：最近两份事务式 checkpoint，含双优化器和各 rank RNG。
- `models/diarizen_training/evidence/training_completion.json`：官方结束条件及权重证据。
- `search/workspaces/task_0/metric/final_evaluation.json`：selection/holdout 对照。

可选的 `SURE_TRAIN_BUDGET_SECONDS` 大于零时启用预计预算暂停；零表示关闭。
暂停会保存状态并阻止后续提交及 debug。不要删除暂停标记来悄悄改变实验预算，应明确记录新的运行决定。
基础设施失败、短测或未完成训练不会作为有效 DER 结果。
