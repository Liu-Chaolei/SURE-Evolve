# SD 文献与反馈四组消融

> 本文保留原四组设计。当前 C/D 已取消，只开展 A/B；旧版对照设计见
> [SD Direct-24 A/B 文献消融实验](sd_ab_ablation.md)，当前原生部署与恢复入口见
> [SD 部署登记](sd_deployments.md)。

四组各进行一次六轮搜索，每轮四个候选。A 使用文献和反馈，B 去文献，
C 去反馈，D 同时去掉两者；公共基线训练一次。结果是单次受控观察，
不报告跨运行标准差，也不将候选数量当作独立重复次数。

## 研究和执行协议

采用 `diarizen.evolution.v1`、`search_scope: all`，允许结构、训练方法、
推理和组合改动。初始化、AMI 五个划分、seed 3407、四张 NPU、FP32、
100 epoch 上限、固定验证、patience 10 和最佳五个 checkpoint 平均不变。
推理候选复用本轮冻结父模型；训练候选继承设计，重新初始化训练。

`sure.ablation.use_literature` 与 `use_feedback` 是两个布尔开关，缺省均开启。
四组共用 XLab 的直接分析、生成、评审、定稿调用路径和模型路由。
A/C 使用冻结的 SD survey 摘录（按领域相关性选取 24 条 evidence 和 16 条
references，并记录 SHA256）；B/D 不读取这些文件、不调用检索，也不要求
虚构文献支持。共同提供的模型源码和接口说明属于所有组的实现上下文。

C/D 每轮只得到初始方案，不得到历史分数、总结、最佳设计、权重或 lineage。
控制器另行保存完整结果用于离线选优。执行代理没有文件读取或网络检索工具；
生成的候选必须遵守任务卡中的数据和历史访问边界。

SD 支持 `SURE_TASK_WRAPPER --action prepare_source --candidate-source ...`，以及
`--action candidate --candidate-source ... --parameters-json ...`。后者必须声明
`requires_training`，可传递 architecture/training/inference 改动。
`diarizen/sure_candidate.py` 的扩展接口见
`playground/sure_master/data/sd_ami_free_ablation.md`。框架保护数据遍历和训练、
验证循环；自定义优化器和调度器完整纳入恢复状态。

每轮最多八次生成尝试、四个执行槽位；不足部分计为缺额，执行失败不补抽。
每个候选最多三次调试，次数跨恢复保存。节点中断恢复同一候选，不重新抽样。
搜索前两名与基线在 selection 上选优；四组均冻结后才能打开 holdout。

## 资源和 Docker

必须遵循 `/shared/chaolei.liu/rules.md` 和 `http://127.0.0.1:18088/`。
授权作业为 **10085–10088、10094–10097**。10089 已取消，不在白名单中。
这些是 Slurm allocation 编号，不是节点编号；节点在作业运行时确认。

每个 allocation 请求 8×Ascend 910B3、64 CPU、256 GiB 内存、100 GB 临时盘。
每个节点同一时间只运行一个四卡训练候选（推理使用一张卡），四组并行、
组内串行。白名单内的节点可以用于恢复；不得退回普通 sbatch 或使用其他作业。
所有作业排队或失效时，控制器等待并更新 resources.json。

作业运行后，交接程序核验所有者、配额、服务 step、holder PID 和启动标识，
再保存 allocation 并停止原 Qwen 服务 step。不会按作业名称批量停止服务。
训练经真实 Slurm step 中的 `sudo -n slurm-docker-run` 启动；不覆盖设备映射。
镜像在 n01 构建，使用个人 Registry 的不可变 digest；正式训练不在 n01 进行。
检查点与最终产物保存在 `/shared`，临时盘内容不作为唯一恢复副本。

## 准备与运行

使用 `docker/sure-master-diarizen/Dockerfile.ablation` 创建仅包含 Dockerfile 的
构建上下文，通过 `cluster-docker-build` 发布；不打包密钥、数据或模型。

```bash
/shared/chaolei.liu/data/sure_asr_controller/bin/python -m \
  playground.sure_master.tools.prepare_sd_ablation \
  --image registry.cluster.local:5000/users/chaolei-liu/sure-diarizen-ablation@sha256:IMAGE_DIGEST \
  --output runs/sd_ablation_RUN_ID
bash runs/sd_ablation_RUN_ID/launch.sh
```

准备器冻结公共源码、DiariZen、XLab、评分源码和文献上下文，并生成 baseline、
A/B/C/D 五份 deployment.yaml。恢复时继续使用已有源码快照和 launch.sh，
不重新运行准备器，不修改已运行的协议或配置。

`workflow.json` 记录控制器 PID、身份和阶段；`baseline/resources.json` 记录排队状态。
各组 `controller.log`、`search/workspace/metric/controller_state.json`、
`ablation_round_N.json` 和 `metric/slurm/*/allocation.json` 保留实验与作业关系。
`result.json` 表示搜索和 selection 完成，`holdout.json` 表示最终测试完成。

阶段退出但未产生结果时，监督器标记失败并保留现场；查看该阶段日志后，
从冻结源码调用 `sd_ablation_workflow --root RUN_DIR --stage baseline|A|B|C|D|holdout`
恢复对应阶段。待其完成后再次启动 launch.sh，监督器会识别结果继续后续步骤。
未确认完成的 API 操作保留请求收据，不盲目重复调用。

## 验证与分析

离线回归包含 `test_sd_ablation`、`test_free_idea_generation`、`test_architecture_scope`、
`test_slurm_allocations`、`test_official_training`、`test_sd_launch`、`test_multitask`。
Torch 运行时的 CPU 检查为 `runtime.test_sd_evolution` 和 `test_training_state`。
在 n01 的普通 CPU Docker 中运行这些小模型测试时设置
`TORCH_DEVICE_BACKEND_AUTOLOAD=0`，避免 torch-npu 的自动后端探测尝试初始化未挂载的 NPU。

最终生成 analysis/results.csv、candidates.csv、search_curves.csv 及 SVG/PDF 曲线。
未完成的结果保留为空，不填入估计分数。DER 错误分量按会话分别归一化后平均，
与主 DER 的会话平均协议对应。计算时间和 LLM 用量单独报告；相同候选机会
与训练上限并不意味着实际 FLOPs 或费用完全相同。
