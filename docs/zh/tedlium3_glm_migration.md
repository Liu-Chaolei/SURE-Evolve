# TEDLIUM3 自进化：GLM API 配置

`configs/sure_master/glm-tedlium3-npu-evolution.yaml` 为独立的 GLM 运行配置。
它通过 `api_profile.env_file` 指定仓库根目录 `.env`，读取 `ZAI_API_KEY` 和
`ZAI_BASE_URL`；不会使用 `.pi/agent` 中旧服务的密钥或模型。

按照智谱 Coding Plan 文档，Chat Completions 客户端使用
`https://open.bigmodel.cn/api/coding/paas/v4`；`/api/v1` 是 Codex 专用入口。
参考：https://docs.bigmodel.cn/cn/coding-plan/latest-model

| 角色 | 模型 |
| --- | --- |
| XLab 背景分析、检索规划、排序 | `glm-5.3` |
| XLab 想法生成与反馈重规划 | `glm-5.3` |
| XLab 想法评估、审查、融合 | `glm-5.3-flash` |
| 所有 SURE agent | `glm-5.3-flash` |

SURE 使用 `llm.zai_flash`，仍以 `provider: openai` 表示兼容协议。
XLab 子进程的 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 从 ZAI 变量成对注入；
四个 `XLAB_RESEARCH_IDEA_*_MODEL` 均显式设置。持久配置只保存变量占位符。
启动时对两个模型验证 JSON 输出，并验证 Flash 的流式工具调用。预检失败停止，
不会改用其他模型或回退旧 API。

当前正式配置直接在全量训练数据上完成 30 epoch，已移除默认的 `initial_baseline_run`。
旧的 100 小时/10 epoch 基线不符合新训练协议，因此新运行先训练一次全预算基线。
若显式提供 `initial_baseline_run`，仍须通过数据、训练预算、BPE、参考文本和代码的一致性验证；
不匹配会拒绝导入，不会把小规模模型当作完整训练结果。

原 XLab 预检将 `meta` 数组误判为空。修复后预检使用与检索一致的
`component_metadata_records`，支持数组和字典，不修改已签名资源文件。

准备独立代码快照：

```bash
/shared/chaolei.liu/data/sure_asr_controller/bin/python \
  -m playground.sure_master.tools.freeze_tedlium_run \
  --config configs/sure_master/glm-tedlium3-npu-evolution.yaml \
  --output /shared/chaolei.liu/SURE-Evolve/runs/tedlium3_glm
```

设备探测记录只有在 backend digest 一致时可复用。测速报告还必须匹配当前全量训练数据
与 30 epoch 预算；旧报告不含该签名时，在新目录重新执行 benchmark。之后运行
`tedlium_workflow --stage search`，指定新运行的 `deployment.yaml` 与 `--output`。
搜索阶段直接完成基线及每个结构候选的完整训练，选优后只做冻结模型复评。
API 检查结果写入 `api_preflight.json`，运行状态写入 `workflow_state.json`；
仅在显式导入匹配的完整基线时才生成 `baseline_import.json`。
