# SD Native-MCTS API 角色配置

2026-09-17 起部署为 `sd-native-mcts-ab-bf16-b48-20260917-v5`。
A/B 使用完全相同的角色分工。

| 环节 | 首选模型 |
|---|---|
| 检索查询、文献摘要、证据整理、父实现描述 | glm-5.3-flash |
| 初始研究方案、核心问题分析 | gpt-6-astra |
| 五模式 MCTS 分支生成 | gpt-6-astra |
| MCTS 评估、诊断、新颖性判断、执行可行性审核 | glm-5.3-flash |
| 融合生成、融合评审、融合修复 | gpt-6-astra |
| 候选代码实现、调试 | glm-5.3-flash |
| 影响后续研究的知识更新、重规划 | gpt-6-astra |
| 训练、推理、SURE 指标计算 | 无语言模型 API |

普通结果摘要没有独立的额外 API 任务；已有文献摘要走 GLM。
`knowledge_promotion` 和 `wisdom_promotion` 会影响后续搜索，因此保留 OPENAI。

GLM 角色使用 `sure-glm-router`：

1. `ZAI_API_KEY` / `ZAI_BASE_URL`，`glm-5.3-flash`。
2. `ZHOU_API_KEY` / `ZHOU_API_BASE_URL`，`glm-5.3-flash`。
3. `OPENAI_API_KEY` / `OPENAI_BASE_URL`，`gpt-6-astra` 兜底。
4. `XCODE_API_KEY` / `XCODE_API_BASE_URL`，OPENAI 不可用时继续兜底；默认
   `gpt-6-astra`，可用 `XCODE_MODEL` 指定该接口支持的模型。

OPENAI 角色使用 `sure-openai-router`，按 OPENAI → XCODE 调用，不降级至 GLM。
这两类请求都不使用 XI 或 ZAI_API_KEY2。

研究调用按显式 operation ID 分流，在响应缓存查找之前绑定模型。
融合评审与 MCTS 评估虽然原先共用 evaluation_model，但现在按操作区分。
SURE 的 draft/improve/debug agent 指向 GLM，知识更新指向 OPENAI。
实际提供方与模型记录在 `runs/services/api_roles_v1/requests.jsonl`，不记录密钥。
独立服务监听 127.0.0.1:18993；不更改其他实验的路由服务。

v4 原始记录和缓存全部保留。因为原生恢复身份包含代码与模型，v5 使用独立
原生工作目录，不把不同模型策略下的 MCTS 检查点混在一起；首轮重新生成，
不复用 v4 研究响应。已验证基线与回放继续复用，不重训。
训练仍为 BF16、每卡 batch 48、4 卡；只使用 n04/10085 和 n05/10086。

验证：19 项路由及角色单测、68 项原生回归、2 项部署角色集成检查通过，
A/B 原生身份预检通过。上线时 GLM 实测由 ZHOU 返回 200；OPENAI 直连与路由
请求均返回上游临时 503，保留等待重试，不据此宣称 OPENAI 可用。

后续增加 XCODE 兜底，21 项路由及角色测试通过，包括两类请求逐级失败后转
XCODE，以及 XCODE 未配置时明确保持不可用。配置时 `.env` 尚未包含 XCODE
密钥和地址，因此没有宣称该接口可用；路由每次请求重新读取 `.env`，补齐后
自动生效。此前 v4 研究记录保持原状；v5 仅新增路由末级，不改变模型角色别名。
