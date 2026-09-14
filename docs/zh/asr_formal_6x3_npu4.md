# ASR 正式六轮结构搜索

入口配置：`configs/sure_master/asr-formal-6x3-npu4.yaml`。

- 全量 TEDLIUM3 train、原生 Zipformer RNN-T、FP32、seed 42、每卡 max-duration 900。
- 基线从头训练一次 30 epoch；之后固定六轮，每轮三个结构候选。
- 每个训练作业单节点独占四张 NPU、80 CPU、500G 内存，最多三个候选并行。
- S1–S5 和备用 draft/reseach：ZAI `glm-5.3-flash`；X1–X4：XI `gpt-6-astra`。
- `startup_mode: direct_formal` 只允许 `--stage search`，不运行 smoke、probe、benchmark、API 小请求或样例想法生成，也不要求这些报告文件存在。
- 正式训练、数据和候选范围的运行时检查仍保留。基线/候选全部完成 30 epoch 后才进入比较。
- regular 评分；六轮后 selection 比较基线与前两名；test 仅评估最终选择和基线，不重训。

通过 `freeze_tedlium_run` 创建新的代码快照和 deployment.yaml，再从快照启动：

```bash
python -m playground.sure_master.tools.tedlium_workflow \
  --stage search --config /absolute/run/deployment.yaml --output /absolute/run
```

代码快照与执行配置使用新目录；不导入旧的 100 小时/10 epoch 基线。
通信端口由每个 Slurm 作业独立租用，同节点的两个四卡作业不会共享进程组。
Slurm 到时或节点故障利用同一候选 checkpoint 恢复；取消作业不会自动重新提交。

进度见 `workflow_state.json`、`search/workspace/metric/controller_state.json` 和每个实验的
`metric/slurm/*/job.json`。出现底层 API 错误时保留脱敏诊断和请求 ID，不清除旧记录。
