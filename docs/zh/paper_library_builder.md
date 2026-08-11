# ASR/TTS 论文资料库构建器使用说明

本文说明仓库中已经实现的 `paper_library/` 构建器。建设动机和完整实现方案见 [ASR/TTS Paper 资料库：建设动机与实现方案](./paper_library_motivation_and_implementation.md)，概念设计及资料库未来如何辅助自进化 Agent，请参阅 [ASR/TTS 研究资料库设计讨论](./asr_tts_research_library_design.md)。当前实现只负责资料库构建和只读查询，尚未接入 `sure_master`。

## 1. 范围与安全边界

- 输入是已经放入 `data/papers/` 的 PDF；构建器不选择或下载论文。
- PDF 是只读输入。所有状态、缓存、快照和索引均写入 `data/paper_library/`。
- 默认配置不启用外部抽取服务，因此可以安全地完成配置检查、扫描、解析、状态查看和结构化候选发现。
- Evidence 和 Research Design Card 的完整抽取需要显式配置 OpenAI 或 Anthropic provider、model 和 API key。论文内容会作为不可信数据传给 provider；启用 provider 意味着论文内容会发送到对应外部服务。
- 资料库中的论文证据与 SURE 本地实验结果保持分层；本构建器不会修改外部 SURE 源码或实验结果。

## 2. 安装与入口

从仓库根目录安装项目依赖：

```bash
uv sync
# 或
pip install -e .
```

两种入口等价：

```bash
python -m paper_library --help
paper-library --help
```

下文统一使用 `python -m paper_library`。默认配置位于：

```text
configs/paper_library/default.yaml
```

其中相对路径以配置文件所在目录为基准，因此默认读取 `data/papers/`，写入 `data/paper_library/`。

## 3. 不调用外部模型的安全工作流

```bash
CONFIG=configs/paper_library/default.yaml

python -m paper_library validate-config --config "$CONFIG"
python -m paper_library scan --config "$CONFIG"
python -m paper_library parse --config "$CONFIG"
python -m paper_library extract --config "$CONFIG" candidates
python -m paper_library status --config "$CONFIG"
python -m paper_library validate --config "$CONFIG"
```

默认 `extraction.provider: disabled` 时：

- `extract candidates` 使用本地结构规则生成候选设计单元；
- `extract evidence`、`extract cards` 和 `build` 会以配置错误退出，不会偷偷调用模型；
- `validate` 对现有对象进行只读检查，不创建缺失的 build 目录、不修改 stage 状态，也不覆盖 `reports/validation.json`；
- 完整 `build` 内部执行的全局验证会显式更新 `reports/validation.json`。

单篇调试可以使用稳定的 `document_id`：

```bash
python -m paper_library parse --config "$CONFIG" --document-id doc_...
python -m paper_library extract --config "$CONFIG" candidates --document-id doc_...
python -m paper_library validate --config "$CONFIG" --document-id doc_...
```

`--force` 会重做所选阶段；上游 cache key 或产物完整性变化会只向下游传播 stale 状态。

## 4. 启用结构化抽取

不要把密钥直接提交到 YAML。复制默认配置到本地且不提交的配置文件，并显式设置 provider。例如 OpenAI：

```yaml
extraction:
  provider: openai
  model: YOUR_STRUCTURED_OUTPUT_MODEL
  api_key: ${OPENAI_API_KEY}
  base_url: null
  prompt_version: 1.0.0
  temperature: 0
  max_retries: 2
```

Anthropic 配置只需将 provider、model 和环境变量改为对应值：

```yaml
extraction:
  provider: anthropic
  model: YOUR_ANTHROPIC_MODEL
  api_key: ${ANTHROPIC_API_KEY}
  base_url: null
  prompt_version: 1.0.0
  temperature: 0
  max_retries: 2
```

配置加载器会读取配置文件向上最近的 `.env`，并严格展开 `${VAR}`；变量未定义会直接报错。API key 不进入语义配置 hash。

完整流水线：

```bash
python -m paper_library build --config configs/paper_library/local.yaml
```

也可以逐层运行以便检查中间产物：

```bash
python -m paper_library parse --config configs/paper_library/local.yaml
python -m paper_library extract --config configs/paper_library/local.yaml candidates
python -m paper_library extract --config configs/paper_library/local.yaml evidence
python -m paper_library extract --config configs/paper_library/local.yaml cards
python -m paper_library validate --config configs/paper_library/local.yaml
python -m paper_library publish --config configs/paper_library/local.yaml
```

抽取固定分三次进行：候选设计发现、原子 Evidence 抽取、Research Design Card 构建。模型输出只被视为候选数据，必须通过 Pydantic schema、引用和 grounding 检查。

## 5. 产物目录与契约

默认 build root 的实际布局为：

```text
data/paper_library/
├── state/build.sqlite3
├── objects/
│   ├── parsed/*.json
│   ├── candidates/*.jsonl
│   ├── evidence/*.jsonl
│   └── cards/*.jsonl
├── records/<corpus-version>/
│   ├── parsed_documents.jsonl
│   ├── evidence_records.jsonl
│   └── research_design_cards.jsonl
├── indexes/<corpus-version>/lexical.sqlite3
├── manifests/
│   ├── current.json
│   └── <corpus-version>.json
├── reviews/
│   ├── evidence_candidates.jsonl
│   └── card_candidates.jsonl
├── failures/
├── reports/validation.json
└── staging/
```

`objects/` 是逐文档增量缓存；消费者不应把它当作稳定语料接口。`records/<corpus-version>/` 是排序稳定、经过验证的规范快照。`manifests/current.json` 原子地指向当前完整版本，manifest 绑定每个 JSONL 的路径、record count 和 SHA-256。

SQLite stage 状态保存 cache key、输入 hash、attempt、输出 URI、输出 SHA-256、record count 和失败分类。缓存命中不仅检查 key，还会检查物理产物 hash 和数量；损坏对象不会被下游抽取继续消费。

## 6. Grounding、复核与发布

溯源链为：

```text
ResearchDesignCard field
  → EvidenceRecord
  → ParsedDocument source object
  → PDF page + bbox + verbatim text
  → original PDF
```

Evidence 引用会校验 document、object type、object ID、页码、bbox 和原文。Card 中 reported 事实字段必须通过 `field_evidence` 指向 Evidence ID。

需要人工复核的 Evidence/Card 会追加为不可变候选记录。当前实现能够保存复核决定的数据模型和辅助函数，但 CLI 尚未提供复核命令，复核决定也尚未自动应用到发布。因此不要把 `reviews/` 中出现候选误解为已完成人工审批。

默认配置中的：

```yaml
validation:
  publish_review_required: false
```

表示 **不允许** 发布 `quality.review_required=true` 的记录。设置为 `true` 才会允许这些记录进入快照；该选项不是“要求先有人复核”的开关。

## 7. 词法索引与查询

发布后，为指定或当前 corpus 构建 SQLite FTS5/BM25 索引：

```bash
python -m paper_library index --config "$CONFIG"
python -m paper_library index --config "$CONFIG" --corpus-version corpus_...
```

Evidence 与 Design Card 使用不同 FTS 表和独立 BM25 分数，不应跨类型比较分数。查询示例：

```bash
python -m paper_library query --config "$CONFIG" \
  --kind evidence "word error rate"

python -m paper_library query --config "$CONFIG" \
  --kind design "streaming transducer ablation" \
  --limit 5 --record-type ablation
```

还可使用 `--paper-id`、`--record-type` 和 `--corpus-version`。`--limit` 范围为 1–100。查询只接受与当前 corpus 内容、配置及完整 FTS 行 hash 一致的索引；缺失、损坏或 stale 索引会被拒绝。

## 8. 命令与退出码

| 命令 | 作用 |
|---|---|
| `validate-config` | 严格加载配置并输出语义 hash |
| `scan` | 扫描、hash、按内容去重并同步 source 状态 |
| `parse` | 生成或复用 `ParsedDocument` |
| `extract candidates` | 发现研究设计候选；disabled 模式可本地运行 |
| `extract evidence` | 使用 provider 抽取并 grounding Evidence |
| `extract cards` | 使用 grounded Evidence 构建设计卡 |
| `validate` | 只读验证规范对象，可按 document 过滤 |
| `publish` | 验证并发布确定性 corpus 快照 |
| `build` | scan → parse → candidates → evidence → cards → 全局验证 → publish |
| `index` | 构建当前实现的词法索引 |
| `query` | 只读查询 Evidence 或 Design 索引 |
| `status` | 汇总 stage/status 数量 |
| `retry` | 已注册但当前版本尚未实现 |

退出码：

- `0`：成功；
- `2`：构建阶段部分失败；
- `3`：配置错误；
- `4`：验证或资料库错误；
- `5`：缺少可选依赖；
- `10`：未预期内部错误。

命令的 JSON 结果写 stdout，错误写 stderr，便于脚本消费。

## 9. 当前限制

- PyMuPDF V1 已保留页面、block/span、bbox 和基础 table/figure 信息，但 section hierarchy、复杂 merged table、equation 和 figure 恢复仍较基础；OCR/GROBID/视觉解析器尚未实现。
- Prompt 文件名仍绑定 V1，prompt 文件内容尚未进入 cache identity。
- Provider 返回的 Candidate ID 尚未统一重算和去重。
- 复核决定尚未接入发布流程。
- `retry` CLI、完整版本化 build/failure report 和中断发布恢复仍待实现。
- 向量/FAISS 配置及 optional extra 是预留接口；当前只有 FTS5 词法索引可用。
- 不同 PDF 修订版之间的逻辑 paper identity 尚未自动合并。
- 本阶段不包含 `sure_master` 消费接口，也不改变现有基于本地实验结果的研究探索主路径。

## 10. 测试与质量检查

```bash
python -m pytest tests/paper_library -q
ruff check paper_library tests/paper_library
```

如果当前 Python 环境尚未安装测试依赖，可使用临时 uv 环境：

```bash
uv run --no-project \
  --with pytest --with pydantic --with PyYAML --with python-dotenv \
  --with PyMuPDF --with openai --with anthropic \
  python -m pytest tests/paper_library -q
```

生成目录 `data/paper_library/` 和本地 PDF 语料默认由 `.gitignore` 排除。代码、schema、测试、默认配置和文档仍应正常纳入版本控制。源 PDF 如需共享，应另行制定版权许可和 Git LFS/对象存储策略，而不是直接提交到普通 Git 历史。
