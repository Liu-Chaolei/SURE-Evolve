# SURE-Evolve 中文论文初稿

主稿为 `sure_evolve_zh.tex`，采用本目录现有的 `spconf.sty` 双栏模板，参考文献使用 `IEEEbib.bst` 和独立的 `sure_evolve_refs.bib`。原始 `Template.tex` 及示例文献保留。

已编译的阅读版为 `sure_evolve_zh.pdf`，当前共五页：四页正文及一页参考文献。结果占位替换成实际内容后需重新检查版面。

这是一份供内容讨论的中文初稿。正文以面向语音场景的自动研究框架为定位，围绕“大规模语音文献—开放研究方案生成—控制变量实验—SURE Eval 反馈”展开，以 ASR、TTS 和 SD 为代表性验证实例。不设置模块消融实验，不主张文献或反馈分别带来性能或效率增益。研究方向可以涉及模型结构、训练方法或推理策略；“受控”指比较候选时保持非干预条件可比，而非限制只能搜索模型结构。

## 本次修订与章节结构

根据作者最新注释，方法从“结构候选”调整为一般的研究方案。公式以候选方案 `c` 表示模型及其训练、推理方法；涉及训练的候选开展相应训练实验，仅修改推理的候选复用对应权重。正文保留必要的实验条件，将操作步骤式的说明改为研究动机、方法关系和实验逻辑。

章节安排为：引言、相关工作、SURE-Evolve 框架、实验与结果、案例分析、结论。“统一评测与实验可比性”归入第三章；案例分析独立讨论代表性候选的研究假设、语音理论与观测结果，当前仅预留位置，不虚构候选或有效机制。

原有 `architecture_only` 配置对应此前的受限结构搜索版本，不代表本稿最新确认的研究范围。正式实验需核对实际采用的方案生成与执行配置。正文中列出的 30/100 epoch 等数值描述基线配方，候选相对基线的干预应单独报告，不能将所有训练和推理设置同时写成固定不变。

语音特色落在领域文献、模型实验及不同预测产物如何构成研究闭环。案例中的理论解释需由真实改动与文献支持，并与实测发现区分；不以解释本身证明某个机制已经得到因果验证。

## 编译

需要 XeLaTeX（含 ctex、fontspec、TikZ 等常用宏包）和 BibTeX，或可联网下载宏包的 Tectonic。中文字体使用 Noto Serif CJK SC、Noto Sans CJK SC；西文字体使用 TeX Gyre Termes、TeX Gyre Heros，可在导言区替换成机器已有字体。

```bash
cd /shared/chaolei.liu/SURE-Evolve/ICASSP
xelatex -synctex=1 -interaction=nonstopmode -halt-on-error sure_evolve_zh.tex
bibtex sure_evolve_zh
xelatex -synctex=1 -interaction=nonstopmode -halt-on-error sure_evolve_zh.tex
xelatex -synctex=1 -interaction=nonstopmode -halt-on-error sure_evolve_zh.tex
```

或者：

```bash
tectonic --synctex --keep-intermediates sure_evolve_zh.tex
```

## 正式结果填充

正文蓝色“待填”是显式占位，不是实验事实。主结果表中的“待填”不能替换成估计值，摘要和结论的结果句也需由最终实验支持。需要补入：

- 作者与单位。
- 各数据划分的样本数、时长、清单版本，以及 Seed-zh 的正式来源引用。
- 正式基线与最终候选的模型、初始化、训练和推理配置，明确干预内容和控制条件；主结果应与这些配置对应。
- 研究与执行环境、SURE Eval 的版本及必要的软件引用；LLM 型号和版本、生成参数、实际轮数、调试预算、硬件及独立运行次数。
- SD 重叠语音处理和评分后端版本。目前稿件明确写出 collar 0.25 秒、UEM 裁剪、会话误差率平均，不能直接与其他协议的公开 DER 比较。
- 三个任务在保留测试集上的基线与最终模型结果、参数量、推理耗时和资源消耗。
- 一个具有真实论文依据、实际改动和观测结果的候选案例，并补入相应语音理论分析。

ASR、TTS 和 SD 各自超过一万篇论文按作者提供的事实写入，不再查询外部存储；三个集合之和不表述为跨领域去重论文总量。正文不使用已弃用的论文库模块。

现阶段 TTS 以 CER 报告可懂度，不将其扩展为自然度或说话人相似度结论；若之后安排这些评价，再补入相应协议和结果。不要以完整系统的单一基线对照推断某个模块的独立作用。

## 引文依据

- Zipformer：作者 arXiv 页面 `https://arxiv.org/abs/2310.11230`，标明 ICLR 2024。
- F5-TTS：本机 F5-TTS README 的作者推荐 BibTeX，使用 arXiv:2410.06885。
- DiariZen：本机 DiariZen README 的作者推荐 ICASSP 2025 引文。
- The AI Scientist：作者 GitHub 项目的推荐引文，arXiv:2408.06292。
- RAG：Meta Research 的 NeurIPS 2020 论文页面。
- TED-LIUM 3：数据集推荐引文与 arXiv:1805.04699。
- WenetSpeech4TTS：ISCA Archive 的 Interspeech 2024 原文。
- AMI：Springer 章节 `10.1007/11677482_3`，会议为 MLMI 2005，出版年份为 2006。

这是中文内容讨论稿；英文投稿版需重新核对摘要字数和最终页数，不能直接按中文字符数换算。
