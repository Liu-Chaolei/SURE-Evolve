# A2S2D Task Description

## Overview

**A2S2D** (Auditable Agent-Assisted Social Science Discovery) 是一个社会科学研究线索发现系统，基于五篇方法论文构成的分析框架，使用 LLM 进行变量理解，由固定程序执行统计估计和审计。

## Task

从中国综合社会调查（CGSS）2023 年数据中发现具有统计显著性和实质意义的社会科学研究线索。

**输入**：
- CGSS2023.dta（Stata 数据文件，约 12,000 个样本，800+ 变量）
- CGSS2023编码表.xlsx（变量编码表，包含变量名、标签、问题文本、取值标签等）

**输出**：
- research_leads.csv（发现的研究线索）
- literature_assessment_table.csv（文献相关性判断）
- 完整审计报告（FDR 校正、Bootstrap、置换检验等）

## Pipeline Stages

| 阶段 | 名称 | 类型 | 描述 |
|------|------|------|------|
| 1 | Input & Initial Metadata | 程序 | 加载数据和编码表，构建初始变量元数据 |
| 2 | Variable Identification | LLM Agent | 识别变量角色（outcome/explanatory/control）、无回答值、测量处理建议 |
| 3 | Candidate Relations | 程序 | 生成候选 X-Y 关系，排除机械关系 |
| 4 | Model Design | LLM Agent | 选择模型族、控制变量、依赖结构、方向裁决 |
| 5 | Model Execution | 稯序 | 执行固定模型注册表，自动诊断 |
| 6 | Objective Audit | 程序 | FDR 校正、置换检验、Bootstrap、群体稳定性 |
| 7 | Robustness & Follow-up | 程序 | 稳健性检查、交互、分组、中介分析 |
| 8 | Literature Search | LLM Agent | OpenAlex 检索、LLM 相关性判断 |

## Key Constraints

### LLM Agent Constraints
1. **Agent 不看预回归 P 值**: 模型选择 Agent 不会看到任何预回归 P 值，不能根据显著性选择变量或模型
2. **Agent 信心不进入统计评分**: 综合信心只由 FDR、效应量、Bootstrap 等客观指标构成
3. **方向裁决用固定规则**: 不是 LLM 决定，而是预定义的得分规则（时间顺序 +3, 构念方向 +2）

### Statistical Constraints
1. **候选上限按 Y 分层**: 避免全局截断导致结果变量覆盖不均
2. **扩展分析不回写主发现**: 不允许按扩展结果重新挑选主模型
3. **五篇论文约束**: Hoffman & Walters (多层结构)、Imbens (因果识别)、Cinelli et al. (控制变量)、Theobald et al. (模型族)、Solon et al. (权重)

## Evaluation Criteria

### Research Lead Classification
- **lead**: 通过严格 FDR (< 0.05)、效应量评级、Bootstrap 方向率 (> 0.8)、群体稳定性
- **exploratory_lead**: 通过宽松 FDR (< 0.10)，但稳健性证据不充分
- **weak_lead**: 仅有统计显著性，但效应量或稳健性不足
- **rejected**: 未通过 FDR 或存在诊断警报

### Objective Confidence Score
综合信心由以下指标构成（权重）：
- FDR Q 值 (30%)
- 效应量评级 (20%)
- Bootstrap 方向一致性 (20%)
- 置换检验显著性 (15%)
- 群体稳定性 (10%)
- 文献新颖性 (5%)

## Timeline

| 时间 | 阶段 |
|------|------|
| 0-5 min | Stage 1: 数据加载 |
| 10-30 min | Stage 2: 变量识别（LLM API 调用） |
| 1-2 min | Stage 3: 候选关系生成 |
| 30-60 min | Stage 4: 模型设计（LLM API 调用） |
| 5-30 min | Stage 5: 模型执行 |
| 10-20 min | Stage 6: 客观审计 |
| 10-30 min | Stage 7: 稳健性检查 |
| 20-40 min | Stage 8: 文献检索 |

**总运行时间**: 约 2-3 小时（取决于候选数量和 API 响应速度）

## Requirements

### Environment
- Python 3.10+
- 必要包: pandas, statsmodels, scipy, numpy, openpyxl
- LLM API: OpenAI 兼容 endpoint（支持 structured outputs）

### Configuration
```yaml
inputs:
  data: "CGSS2023.dta"
  codebook: "CGSS2023编码表.xlsx"
  
llm_a2s2d:
  provider: "openai"
  endpoint: "${OPENAI_ENDPOINT}"
  model: "${OPENAI_MODEL}"
  api_key_env: "OPENAI_API_KEY"
```

### Data Files
- CGSS2023.dta: Stata 数据文件
- CGSS2023编码表.xlsx: Excel 编码表（必须包含 variable 列）

## Citation

A2S2D 分析框架基于以下五篇论文：

1. Hoffman L, Walters RW. 2022. *Catching Up on Multilevel Modeling*. Annual Review of Psychology.
2. Imbens GW. 2024. *Causal Inference in the Social Sciences*. Annual Review of Statistics.
3. Cinelli C, Forney A, Pearl J. 2024. *A Crash Course in Good and Bad Controls*. Sociological Methods & Research.
4. Theobald EJ, et al. 2019. *Beyond Linear Regression*. Physical Review Physics Education Research.
5. Solon G, Haider SJ, Wooldridge JM. 2015. *What Are We Weighting For?* Journal of Human Resources.