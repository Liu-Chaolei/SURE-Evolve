# 语音知识图谱处理

本轮分别处理 ASR、TTS、SD 的后一批论文集合，保留来源和统一论文 ID。没有 PDF 的论文保留元数据节点；不使用前一批采集结果，不生成 survey 或研究 idea。

## 输入和正式运行

所有 run 位于 `.xlab/runs/`。

| 方向 | 输入 run | 知识图谱 run | 论文记录 | PDF |
| --- | --- | --- | ---: | ---: |
| SD | `20260819-083505-89fc6345` | `20260908-144923-kg-sd` | 907 | 579 |
| TTS | `20260819-083508-b555a9ad` | `20260908-144923-kg-tts` | 1,288 | 879 |
| ASR | `20260819-083241-c8191d5e` | `20260908-144923-kg-asr` | 2,000 | 1,145 |

2,603 份 PDF 对应 2,460 个不同文件哈希。相同 PDF 共享已校验的解析缓存；引用上下文和知识抽取仍按各自集合处理。

## 执行方式

持久化控制脚本和状态在 `.xlab/operations/speech-kg-20260908/`：

- `workflow.py`：准备输入、NPU 预检、解析、抽取、构建、审计和汇总。
- `parse-npu.sbatch`：Slurm 上分配 4 张 Ascend NPU、64 CPU 和 192 GiB 内存。
- `parse-cpu.sbatch`：CPU 后备方案，4 个解析进程、每进程 16 线程。
- `supervisor.json`：控制进程、解析作业、抽取进程和最近心跳。
- `summary.json`、`README.md`：三个方向的当前进度和产物入口。

NPU 预检的 24 篇论文已全部生成完整解析包。正式解析顺序为 SD → TTS → ASR；前一个方向完成解析后，抽取进程接续执行，解析进程继续下一个方向。

MinerU 使用固定的 3.4.4 版本，每个工作进程每批最多处理 24 份 PDF。共享缓存位于 `.xlab/cache/mineru/`，按 PDF 哈希、解析器、模型配置和解析参数校验。完整 Markdown、content list、middle JSON 缺一不可。

Qwen 服务模型为 `Qwen3.8-27B-W8A8`。抽取在当前控制主机运行，访问 n20 服务的转发入口 `http://127.0.0.1:28091/v1`；计算节点自己的 localhost 不是该入口。

2026-09-09 按用户要求从 `28090` 切换到 `28091`。各 run 的 `artifacts/endpoint-migration.json` 记录已完成结果的哈希和原始服务地址；历史来源保持不变，后续请求使用新端口。未完成的 LLM 轮次在新服务重新执行。

2026-09-09 在用户确认独占 `28091` 服务后，并发从 8 提高到 64；继续提速时，SD 当前这轮重试维持 **64 并发**，后续 TTS、ASR 配置为 **96 并发**，在阶段边界自动生效。`llm_rpm=0`，以并发数控制负载。仍使用 600 秒请求超时、120,000 字符输入上限和 6,000 tokens 输出上限。`--llm-disable-thinking` 显式关闭 thinking，并参与缓存校验。服务中断会保存检查点，将未完成论文保留为 pending，恢复后继续。变更记录见 `.xlab/operations/speech-kg-20260908/concurrency-change.json`。

## 查看与恢复

从仓库根目录查看：

```bash
cat .xlab/operations/speech-kg-20260908/supervisor.json
cat .xlab/operations/speech-kg-20260908/summary.json
squeue --me
```

各 run 的 `artifacts/logs/operator-*.log` 保存阶段日志。`paper_documents.json` 和 `extractions.manifest.json` 保存处理状态。控制进程仍运行时，不要重复启动。

控制进程退出后，可使用知识图谱环境的 Python 重新执行：

```bash
python .xlab/operations/speech-kg-20260908/workflow.py supervise --validation-job 8989
```

也可使用技能的 `scripts/build_graph.py`，指定原 `--run-dir` 重跑失败阶段。已有合格检查点会被复用；设备资源在解析阶段重新探测。

## 验收和产物

三个方向分别要求解析覆盖率 ≥90%、结构化成功率 ≥98%、抽取成功率 ≥90%、Core 覆盖率 ≥90%、语义边引文覆盖率 ≥95%。pending 项阻止成功验收。缺失全文和处理失败的数量分别报告。

最终产物包括逐篇解析与抽取记录、`nodes.jsonl`、`edges.jsonl`、实体别名、`graph.db`、图谱预览、`method_graph.json` 和 `graph_report.json`。只有三个方向审计均通过，汇总中的 `complete` 才为 true。
