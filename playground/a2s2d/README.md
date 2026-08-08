# A2S2D Playground

A2S2D (Auditable Agent-Assisted Social Science Discovery) — 社会科学研究线索发现系统，作为 EvoMaster playground 实现。

基于五篇方法论文构成的分析框架，使用 LLM 进行变量理解，由固定程序执行统计估计和审计。

## 八阶段流水线

| 阶段 | 名称 | 类型 | 描述 |
|------|------|------|------|
| 1 | Input & Initial Metadata | 纯程序 | 读取数据、编码表，构建初始变量元数据 |
| 2 | Variable Identification | LLM Agent | 识别变量角色、无回答值、测量处理建议 |
| 3 | Candidate Relations | 纯程序 | 生成候选 X-Y 关系，排除机械关系 |
| 4 | Model Design | LLM Agent | 选择模型族、控制变量、依赖结构、方向裁决 |
| 5 | Model Execution | 纯程序 | 执行固定模型注册表，自动诊断 |
| 6 | Objective Audit | 纯程序 | FDR 校正、置换检验、Bootstrap、群体稳定性 |
| 7 | Robustness & Follow-up | 纯程序 | 稳健性检查、交互、分组、中介分析 |
| 8 | Literature Search | LLM Agent | OpenAlex 检索、LLM 相关性判断 |

## 五篇论文分析框架

- **Hoffman & Walters (2022)**: 多层结构、组内/组间效应、中心化、随机截距和随机斜率
- **Imbens (2024)**: 研究设计优先于估计；因果分析必须明确识别策略和假设
- **Cinelli, Forney & Pearl (2024)**: 依据因果结构区分好控制、坏控制、中介、碰撞点和工具变量
- **Theobald et al. (2019)**: 依据结果变量支持集和分布选择线性或广义线性模型
- **Solon, Haider & Wooldridge (2015)**: 加权必须对应明确目的，并比较加权与未加权结果

## 关键约束

1. **Agent 不看预回归 P 值**: 模型选择 Agent 不会看到任何预回归 P 值，不能根据显著性选择变量或模型
2. **Agent 信心不进入统计评分**: 综合信心只由 FDR、效应量、Bootstrap 等客观指标构成
3. **方向裁决用固定规则**: 不是 LLM 决定，而是预定义的得分规则（时间顺序 +3, 构念方向 +2）
4. **候选上限按 Y 分层**: 避免全局截断导致结果变量覆盖不均
5. **扩展分析不回写主发现**: 不允许按扩展结果重新挑选主模型

## 目录结构

```
playground/a2s2d/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── playground.py          # A2S2DPlayground 主编排器
│   └── exp/
│       ├── __init__.py        # A2S2DBaseExp 基类
│       ├── input_exp.py       # 阶段1
│       ├── identification_exp.py  # 阶段2
│       ├── candidate_exp.py   # 阶段3
│       ├── model_design_exp.py    # 阶段4
│       ├── model_exec_exp.py  # 阶段5
│       ├── audit_exp.py       # 阶段6
│       ├── robustness_exp.py  # 阶段7
│       └── literature_exp.py  # 阶段8
├── lib/                       # A2S2D 核心库（vendored）
│   ├── config.py              # ProjectConfig（支持 from_dict）
│   ├── types.py               # 数据类型
│   ├── modeling.py            # 模型注册表
│   ├── audit.py               # 审计程序
│   ├── analysis_framework.py  # 五篇论文规则
│   └── ...
└── prompts/
    ├── variable_identification_system.txt
    ├── variable_identification_user.txt
    ├── model_selection_system.txt
    ├── model_selection_user.txt
    ├── literature_system.txt
    └── literature_user.txt
```

## 配置

配置文件: `configs/a2s2d/config.yaml`

包含：
- **EvoMaster 标准**: `llm`、`agents`、`session`、`logging`
- **A2S2D 专用**: `inputs`、`columns`、`modeling`、`audit`、`framework`、`literature`
- **流水线**: `pipeline.stages` 定义各阶段

## 使用方式

### 运行完整流水线

```bash
python run.py --agent a2s2d --config configs/a2s2d/config.yaml --task /home/liuchaolei/Agent/EvoMaster/playground/a2s2d/description.md
```

### 从断点恢复

```bash
python run.py --agent a2s2d --config configs/a2s2d/config.yaml --task "..." --resume runs/a2s2d_<timestamp>/
```

### 环境变量

需要在 `.env` 中设置：

```bash
# LLM
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_ENDPOINT=https://api.openai.com/v1/responses

# 文献检索
OPENALEX_API_KEY=your_key
```

## 输出

运行结果保存在 `runs/a2s2d_<timestamp>/`，包含：

```
runs/a2s2d_<timestamp>/
├── config.yaml                    # 配置快照
├── run_manifest.json              # 运行清单 + 断点
├── logs/
│   ├── evomaster.log
│   ├── variable_metadata_before_identification.csv  # 阶段1
│   ├── variable_role_table.csv                      # 阶段2
│   ├── variable_identification_value_plan.csv
│   ├── candidate_relation_table.csv                 # 阶段3
│   ├── proposed_model_selection_plan.json           # 阶段4
│   ├── model_result_table.csv                       # 阶段5
│   ├── discovery_audit_table.csv                    # 阶段6
│   ├── permutation_table.csv
│   ├── bootstrap_table.csv
│   ├── robustness_table.csv                         # 阶段7
│   ├── followup_table.csv
│   ├── literature_assessment_table.csv              # 阶段8
│   └── literature_matched_works.csv
├── trajectories/
│   └── trajectory.json
└── workspace/
    └── data/                      # 数据文件 symlinks
```

## 数据要求

支持的数据格式：
- **Stata**: `.dta`（推荐，保留变量标签和值标签）
- **CSV**: `.csv`
- **Excel**: `.xlsx`、`.xls`
- **Parquet**: `.parquet`

编码表格式：
- **Excel**: `.xlsx`（自动检测表头）
- **CSV**: `.csv`
- **JSON**: `.json`

## 设计决策

1. **代码迁移（vendored）**: A2S2D 核心库完整复制到 `lib/`，保持统计正确性，便于独立演进
2. **配置统一**: 原 TOML 配置转换为 YAML，与 EvoMaster 生态一致，支持 `${VAR}` 环境变量替换
3. **直接 API 调用**: 阶段 2/4/8 使用 A2S2D 的 `build_provider()` + structured output，而非 EvoMaster agent 工具循环
4. **状态序列化**: 使用 JSON checkpoint（非 pickle），便于调试和跨版本兼容
5. **schema 版本控制**: 断点恢复时校验 `pipeline_schema_version`，避免不兼容恢复
