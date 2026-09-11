# Scholar Profile Pipeline — 学者科研画像流水线

从公开学术信息自动构建学者的**科研主线图谱**、筛选代表性论文、生成高保真度的 **AI Agent System Prompt**，最终用于**多智能体学术讨论模拟**。

## 流水线概览

```
names.txt                ──→ [0] 生成 CSV ──→ scholars.csv
                              (build_scholars_csv.py)

scholars.csv             ──→ [0.5] 自动发现资源 ──→ sources.txt
                              (build_scholar_sources.py)

sources.txt + DBLP URL   ──→ [1] 主线抽取 ──→ mainline_graph.json
                              (scholar_mainline_builder.py)

DBLP URL + mainline      ──→ [2] 论文筛选评分 ──→ paper_digest_scored.json + Word
                              (scholar_author_digest.py)

paper_digest + mainline  ──→ [3] System Prompt 生成 ──→ system_prompt.docx
                              (professor_system_prompt_builder.py)
```

`run_batch_parallel.py` 一键编排上述全流程，支持并行处理多位学者。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置环境变量

```bash
# 必需
export OPENAI_API_KEY="<你的 LLM API Key>"

# 推荐
export LLM_BASE_URL="https://token-plan-cn.xiaomimimo.com/v1"
export LLM_MODEL="mimo-v2.5-pro"

# 资源自动发现（可选但推荐）
export WEB_SEARCH_MCP_URL="http://127.0.0.1:17890/mcp"

# 代理（可选）
export HTTPS_PROXY="http://127.0.0.1:7897"
```

### 3. 从名字列表开始（推荐）

```bash
# 准备名字列表，每行一个名字
echo "Yann LeCun" > names.txt
echo "Geoffrey Hinton" >> names.txt

# 自动生成 CSV（通过 DBLP 搜索 API 查找准确链接）
python build_scholars_csv.py --input names.txt --output scholars.csv --sources-dir sources

# 一键跑全流程
python run_batch_parallel.py --csv scholars.csv --max-workers 3
```

Pipeline 会自动对每个学者：
1. 检查 sources 文件 → 缺失则自动发现资源
2. 构建主线图谱
3. 筛选评分论文
4. 生成 System Prompt

### 4. 手动逐步运行（可选）

```bash
# 发现资源
python build_scholar_sources.py \
  --author "Yann LeCun" \
  --dblp-url https://dblp.org/pid/l/YannLeCun \
  --output sources/Yann_LeCun_sources.txt

# 构建主线图谱
python scholar_mainline_builder.py \
  --author "Yann LeCun" \
  --source-file sources/Yann_LeCun_sources.txt \
  --dblp-url https://dblp.org/pid/l/YannLeCun \
  --output runs/Yann_LeCun/mainline_graph.json

# 论文评分
python scholar_author_digest.py \
  --dblp-url https://dblp.org/pid/l/YannLeCun \
  --mainline-graph runs/Yann_LeCun/mainline_graph.json \
  --output runs/Yann_LeCun/scholar_report.docx \
  --json-output runs/Yann_LeCun/paper_digest_scored.json

# 生成 System Prompt
python professor_system_prompt_builder.py \
  --input-json runs/Yann_LeCun/paper_digest_scored.json \
  --mainline-graph runs/Yann_LeCun/mainline_graph.json \
  --sources-file sources/Yann_LeCun_sources.txt \
  --output runs/Yann_LeCun/system_prompt.docx
```

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `build_scholars_csv.py` | 名字列表 → CSV（自动查 DBLP 链接） |
| `build_scholar_sources.py` | 自动发现学者资源（DBLP + OpenAlex + Scholar + Wikipedia + HTTP MCP web search） |
| `scholar_mainline_builder.py` | 爬取主页/访谈 → LLM 提取研究主题图谱 |
| `scholar_author_digest.py` | DBLP 论文 → 多源摘要补全 → 5 维度评分 |
| `professor_system_prompt_builder.py` | 论文 + 图谱 + 来源 → 6 板块 System Prompt |
| `run_pipeline.py` | 单学者流水线入口（带状态上报和 manifest 生成） |
| `run_batch_parallel.py` | 批量并行编排全流程 |
| `utils.py` | 公共工具函数 |
| `config.py` | 配置管理 |
| `exceptions.py` | 自定义异常 |

## 项目结构

```
├── build_scholars_csv.py              # 名字 → CSV
├── build_scholar_sources.py           # 自动发现资源
├── scholar_mainline_builder.py        # 主线图谱
├── scholar_author_digest.py           # 论文评分
├── professor_system_prompt_builder.py # System Prompt
├── run_batch_parallel.py              # 批量流水线
├── utils.py / config.py / exceptions.py
│
├── scholars.csv                       # 学者列表
├── sources/                           # 学者资源文件（手写或自动生成）
├── profiles/                          # 学者画像（本地，不上传）
│   ├── pinn/                          #   PINN 领域
│   ├── other/                         #   其他领域
│   └── high-quality/                  #   精选高质量
├── runs/                              # Pipeline 运行输出（本地，不上传）
├── docs/                              # 文档
└── requirements.txt
```

## 环境变量

| 变量 | 说明 | 必需 |
|------|------|------|
| `OPENAI_API_KEY` | LLM API Key | ✅ |
| `LLM_BASE_URL` | LLM API 地址 | 否（有默认值） |
| `LLM_MODEL` | 模型名称 | 否（默认 `mimo-v2-flash`） |
| `WEB_SEARCH_MCP_URL` | HTTP MCP 搜索端点 | 否（无则跳过资源搜索） |
| `HTTPS_PROXY` | HTTPS 代理 | 否 |
| `HTTP_PROXY` | HTTP 代理 | 否 |
| `PROXY_MODE` | `on` / `off` / `auto` | 否（默认 `auto`） |

## 评分公式

```
score = 0.25 × 署名权重
      + 0.45 × 主线匹配度
      + 0.15 × 主题持续性
      + 0.05 × 引文影响
      + 0.10 × 外部证据

分桶: core (≥0.50) / peripheral (≥0.30) / student-led (<0.30)
```

署名权重：第一作者+通讯=0.80，通讯或第一=0.45，前三=0.25，其他=0.10

引文影响：最低值 0.4（新论文或小众领域论文不会因引用少而被过度惩罚）

详见 [`docs/PIPELINE_WORKFLOW_DETAIL.md`](docs/PIPELINE_WORKFLOW_DETAIL.md)。

## 状态上报

`run_pipeline.py` 在执行过程中会自动写入状态文件：

- `state.json` — 当前阶段、计数器和诊断信息
- `events.jsonl` — 带时间戳的事件日志
- `manifest.json` — 最终 manifest，包含验证结果

这些文件可用于 XLab harness 监控 pipeline 运行状态。

## 注意事项

- DBLP 搜索 API 有限速，批量查链接时每次间隔 2 秒
- 自动发现的资源会验证 URL 有效性，过滤社交媒体和泛化主页
- 新闻/报道类 URL 自动截断以避免 LLM 内容审核
- 代理推荐通过环境变量设置（`HTTPS_PROXY`），而非命令行参数
- `profiles/` 和 `runs/` 包含大量数据，已在 `.gitignore` 中排除
- `.env` 文件含 API Key，切勿提交

claude --resume 7ab5a11c-0654-4df1-8c4f-35a6dce8e589

$env:OPENAI_API_KEY="tp-c7d38141pdzquz7fnrh90vxn1uhcs7duxz8ds1htccm13m5d" 
$env:LLM_BASE_URL="https://token-plan-cn.xiaomimimo.com/v1"
$env:LLM_MODEL="mimo-v2.5-pro"
$env:WEB_SEARCH_MCP_URL="http://127.0.0.1:17890/mcp"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
$env:HTTP_PROXY="http://127.0.0.1:7897" 