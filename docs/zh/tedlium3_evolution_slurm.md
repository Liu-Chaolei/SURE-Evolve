# TEDLIUM3 Zipformer：Slurm 自进化运行

本配置采用 ordinary + XLab：100 小时固定子集、10 epoch 搜索，默认六轮、每轮四个候选；六轮后最近三轮存在改善时继续，连续三轮无改善停止，最多十轮。架构、训练和推理候选没有数量配额。

搜索基线只训练一次并供所有轮次复用。每个训练候选从随机初始化开始；推理候选使用本轮开始时冻结的父模型。搜索结束后，对基线和 regular 排名前两名的完整方案，在全量数据上分别从头训练 30 epoch。两阶段的数据量不同，因此不复用全量基线的第十个 epoch 充当子集基线。若入选方案只是在同一个训练方案上采用不同推理设置，全量阶段共享一次训练，再分别解码评分。

## 环境与资源

遵循 `/shared/chaolei.liu/rules.md` 及 <http://127.0.0.1:18088/>。计算节点必须在 Slurm step 中使用 `sudo -n slurm-docker-run`；不可直接 Docker 或手工覆盖 Ascend 可见卡变量。正式作业使用内部 Registry digest。

控制器环境与 XLab 模型环境分离：

```bash
/shared/chaolei.liu/data/sure_asr_controller/bin/python -m pip install \
  -r requirements.txt -r playground/sure_master/requirements.txt
```

配置入口为 `configs/sure_master/xlab-tedlium3-npu-evolution.yaml`。XLab 使用已有 `20260910-survey-asr-v2/artifacts/survey.json` 及其 resource manifest，控制器通过 `with_xlab_environment` 读取现有个人 API-key 配置，不把凭据写入 worker 请求。

训练候选：单节点、8 NPU、160 CPU、1000G 内存、1T 临时盘；纯推理：1 NPU、20 CPU、128G 内存。最多四个候选并行。代码、数据和持久模型在 `/shared`，选定的数据视图在每次作业内缓存到 `/local/job`。

## 执行顺序

在仓库根目录运行：

```bash
/shared/chaolei.liu/data/sure_asr_controller/bin/python \
  -m playground.sure_master.tools.tedlium_workflow --stage all
```

也可分别执行 `--stage prepare`、`--stage probe`、`--stage benchmark`、`--stage search`。

1. `prepare`：CPU Slurm 作业验证 STM/音频，准备全量特征及同一份 BPE，并生成共享特征的 `search_100h` 视图。输出位于 `/shared/chaolei.liu/data/sure_tedlium3_full`。已有 1 小时 smoke 数据不覆盖。
2. `probe`：依次完成真实单卡 CPU/NPU loss 与梯度对比、2 卡和 8 卡 HCCL 更新/同步/checkpoint 重载。
3. `benchmark`：用真实训练批次测试每卡 duration 30/60/120/240，预热 20 step，测量 200 step。选择内存占比不超过 90% 的最快稳定配置，写入 `benchmark/benchmark.json`。
4. `search`：冻结测得的 batch 设置，执行基线、XLab 搜索、全量训练、selection 和 test。没有通过 8 卡探测和测速不能启动。

regular 200 条、selection 196 条来自按演讲隔离的 dev；test 1155 条来自官方 test。训练过程的 validation 也使用 regular。test 不参与候选生成和选择。最终 WER 按 SURE 英文规范化口径报告，不自动视作官方 benchmark 复现。

## 恢复与结果

默认运行目录为 `runs/tedlium3_evolution`。同一运行目录保存阶段 job ID、生成响应、已发布 XLab 批次、候选作业请求与结果、控制器轮次 checkpoint。重启同一入口复用这些状态，不重新训练已完成基线、不重复生成已发布批次。

Slurm accounting 可能未启用，状态查询优先使用 `squeue` 与 `scontrol`。TIMEOUT 作业可继续使用完整的 epoch/step checkpoint；取消作业不会自动重提。提交结果不明确时停止并要求核对已有 job，避免重复计费。

候选工作区中 `metric/slurm/<request-digest>/job.json` 保存 job ID；可以用 `scancel JOBID` 取消指定作业。取消整个运行时也应停止控制器，以免继续启动尚未提交的实验。

主要输出：

- `search/workspace/metric/controller_state.json`：搜索恢复状态。
- `search/workspace/metric/search_progress.json`：轮次、停滞计数及最佳 regular WER。
- `search/workspace/metric/full_training.json`：全量基线与入选方案的训练产物。
- `search/workspace/metric/final_evaluation.json`：selection/test 结果。
- `search/workspace/best_solution/model_artifact.json`：最终模型引用。
- `result.json`：整次执行状态。只有真实训练和评测完成后才会标记 completed。

搜索与模型效果分别验收。基础设施失败不能当作没有改善；测试未改善时如实报告，不继续用 test 调参。单种子结果不代表多种子稳健性。

## 固定代码快照

长期运行前可执行 `python -m playground.sure_master.tools.freeze_tedlium_run`。
它将控制器和 worker 代码复制到运行目录的内容寻址快照，生成保留凭据占位符的
`deployment.yaml`。后台运行时使用该快照作为 `PYTHONPATH`，并给工作流传入
`--config /绝对运行目录/deployment.yaml --output /绝对运行目录`。
`workflow_state.json` 保存当前阶段、进程 PID 和阻断原因。
8 卡探测记录的 backend digest 必须与后续使用的运行版本一致。
