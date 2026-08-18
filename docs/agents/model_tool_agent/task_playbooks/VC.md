# VC Model Onboarding Playbook

本文是 `docs/agents/model_tool_agent/AGENTS.md` 的 VC 任务补充，基于
`Plachta__Seed-VC` 的接入经验。

## 1. 任务边界

Voice Conversion 是“源音频 + 参考音频 -> 转换后音频”。当前最小契约：

```json
{
  "source_audio_path": "fixture/zh/source.mp3",
  "reference_audio_path": "fixture/zh/ref.mp3"
}
```

输出：

```json
{
  "audio_path": "artifacts/outputs/seed_vc_v2_smoke.wav",
  "source_audio_path": "...",
  "reference_audio_path": "...",
  "task": "VC"
}
```

`model.spec.yaml` 推荐：

```yaml
task_type: "vc"
io_contract:
  input_type: "audio_pair"
  output_type: "audio_path"
  primary_field: "audio_path"
  required_fields: ["audio_path", "source_audio_path", "reference_audio_path"]
  nonempty_fields: ["audio_path"]
  json_serializable: true
```

VC 第一阶段不做音色相似度指标，只验证可加载、可推理、输出音频契约正确。

## 2. 目录与权重

VC 模型通常依赖上游源码、多个 checkpoint、HF cache 和配置文件。推荐结构：

```text
src/sure_eval/models/{model}/
├── .runtime/
│   ├── source/<upstream-repo>/
│   ├── huggingface/
│   ├── cache/
│   └── matplotlib/
├── checkpoints/
├── fixture/zh/
├── artifacts/outputs/
├── model.py
├── validate.py
├── local_uv_setup.sh
├── local_uv_validate.sh
├── Dockerfile
├── docker_build.sh
└── docker_validate.sh
```

Seed-VC 经验：

- 上游源码放在 `.runtime/source/seed-vc`。
- V2 推理入口是 `inference_v2.py`。
- 上游默认使用相对路径 `./checkpoints`，wrapper 需要在调用时 `cwd` 到上游源码目录，
  使 checkpoint 解析留在 model-local 范围内。
- 如果首次推理会自动下载权重，要通过 `HF_ENDPOINT`、`HF_HOME`、`HF_HUB_CACHE`
  指向 `.runtime`。

## 2.5 Fixture

共享 fixture 库中的 VC 代表样例位于：

```text
fixtures/tasks/vc/seed_vc_zh_smoke/
```

索引见 `fixtures/tasks/vc/README.md`。接入新模型时，优先从该目录选择样例复制到
模型目录；当前共享 VC 样例只有音频，模型目录下仍需根据 wrapper contract 创建
`gt.jsonl`。

VC metric namespace:

```text
src/sure_eval/evaluation/tasks/vc/
```

正式 VC 指标必须走该 namespace 的统一入口。`validate.py` / `docker_validate.sh`
只负责生成或验证 converted audio 和 wrapper contract；已有 converted audio 后，
不要为了刷新指标重新跑模型推理。

推荐入口是 evaluation CLI：

```bash
sure-eval metric describe vc \
  --language zh \
  --metrics vc_cer,sim/wavlm-large \
  --output /tmp/vc_pipeline.json \
  --json

sure-eval metric run \
  --pipeline /tmp/vc_pipeline.json \
  --samples-jsonl /tmp/vc_samples.jsonl \
  --output-dir /tmp/sure_eval/vc_eval \
  --device cuda \
  --cache-dir /hpc_stor03/sjtu_home/junhao.du/.cache/sure-eval/tts-metrics \
  --validate-env \
  --json
```

`samples_jsonl` 每行必须显式区分音频角色：

```json
{"sample_id":"vc_smoke","converted_audio":"outputs/vc.wav","source_audio":"source.wav","reference_audio":"speaker.wav","reference_text":"源音频文本","language":"zh"}
```

历史 wrapper：

```text
scripts/run_vc_metric_pipeline.py
scripts/run_vc_metric_pipeline_docker.py
scripts/run_vc_metric_pipeline_docker.sh
```

这些 wrapper 仍可用于特定已有镜像，但新接入和 agent 调用优先使用
`sure-eval metric describe/run`。

VC 指标输入必须显式区分：

- `converted_audio`: VC 模型输出音频。
- `source_audio`: 提供内容的源音频。
- `reference_audio`: 提供音色的参考音频。
- `reference_text`: 源音频对应文本；用于 `vc_wer` / `vc_cer`。

若 TTS metric cache 已经存在，VC 可以复用同一 provider cache，因为 VC metric pipeline
复用 TTS 的 semantic、speaker similarity 和 MOS provider 栈。复用 cache 时必须在
artifact 或 run report 中记录 `cache_dir`。

Seed-VC re-onboarding 已验证：本地 uv 生成的 `artifacts/outputs/seed_vc_v2_smoke.wav`
可以直接交给 `scripts/run_vc_metric_pipeline_docker.py` 做正式 VC 评估，不需要重新
跑 Seed-VC 推理。命令必须显式传入 converted/source/reference audio 和 source text。

注意不要随意换成空的独立 metric cache。曾经使用
`/hpc_stor03/sjtu_home/junhao.du/.cache/sure-eval/vc-metrics` 时，semantic/speaker
provider 会重新下载资源，且 MOS provider 缺 `dnsmos`、EmergentTTS-Eval 和
UTMOS-demo 资源，导致 `ok: false`。当前通过路径复用：

```text
/hpc_stor03/sjtu_home/junhao.du/.cache/sure-eval/tts-metrics
```

如果必须使用新的 VC cache，必须先完整准备 semantic、speaker、DNSSMOS、WV-MOS 和
UTMOS provider 资源，不能把 provider 缺失误判为 VC 模型失败。

Seed-VC 已验证命令：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
.venv.hostbak/bin/python scripts/run_vc_metric_pipeline_docker.py \
  --converted-audio /hpc_stor03/sjtu_home/junhao.du/sure-eval-sandbox/src/sure_eval/models/Plachta__Seed-VC/artifacts/outputs/seed_vc_v2_smoke.wav \
  --reference-audio /hpc_stor03/sjtu_home/junhao.du/sure-eval-sandbox/src/sure_eval/models/Plachta__Seed-VC/fixture/zh/ZH_B00001_S00000_W000000.mp3 \
  --source-audio /hpc_stor03/sjtu_home/junhao.du/sure-eval-sandbox/src/sure_eval/models/Plachta__Seed-VC/fixture/zh/ZH_B00000_S00000_W000002.mp3 \
  --reference-text '空投认为继续往下跌的空间并不大啊，这个可以参考前期的低位一万四千七百一十元每吨。' \
  --language zh \
  --sample-id seed_vc_v2_smoke \
  --gpu 0 \
  --device cuda:0 \
  --cache-dir /hpc_stor03/sjtu_home/junhao.du/.cache/sure-eval/tts-metrics \
  --work-dir /hpc_stor03/sjtu_home/junhao.du/sure-eval-sandbox/src/sure_eval/models/Plachta__Seed-VC/artifacts/vc_metric_parts \
  --output /hpc_stor03/sjtu_home/junhao.du/sure-eval-sandbox/src/sure_eval/models/Plachta__Seed-VC/artifacts/vc_metric_report_local_pipeline.json
```

已验证输出：

```text
artifacts/vc_metric_report_local_pipeline.json
```

该报告必须包含 `ok: true`、`errors: []`，并至少覆盖 `vc_cer`、`sim/*`、`dnsmos`、
`wv-mos`、`utmos` 中实际启用的指标。

## 3. 下载与网络

VC 上游常依赖 Hugging Face。规则：

- 使用 `hf-mirror.com` 时不要开代理。
- 先关闭代理：

```bash
. /hpc_stor03/sjtu_home/junhao.du/.local/bin/ssr-off
```

- 再执行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 <download-or-validate-command>
```

Seed-VC 经验：

- `facebook/hubert-large-ll60k/pytorch_model.bin` 可通过 hf-mirror 下载。
- 如果 Hugging Face Xet 下载很慢或留下 incomplete blob，要清理不完整文件并改用
  no-proxy mirror。
- 下载完成后用 `torch.load` 或模型 load 阶段验证，不只看文件存在。

## 4. Fixture

VC fixture 至少两条音频：

```text
fixture/zh/
├── source.mp3
├── reference.mp3
└── gt.jsonl
```

`gt.jsonl` 建议：

```json
{
  "key": "vc_smoke_1",
  "source_audio": "source.mp3",
  "reference_audio": "reference.mp3",
  "task": "VC"
}
```

可以复用 TTS fixture，但必须明确哪条是 source、哪条是 reference。VC 不依赖
target text；如果 fixture 来自带文本的数据集，文本只能作为说明，不作为主要输入。

## 5. Backend 选择

VC 优先 GPU。接入顺序：

1. 先尝试 model-local uv，快速验证上游 V2 推理链路。
2. 如果 uv 通过，再固化 Docker。
3. 如果上游依赖复杂或需要系统包，Docker 是对外运行的主路径。

Seed-VC local uv 经验：

- `resemblyzer` 会引入 `webrtcvad==2.0.10`，在缺 Python.h 时构建失败。
- 如果 V2 推理路径不 import `resemblyzer`，可从 local requirements 中去掉，并记录原因。
- 本地验证命令可用：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda DIFFUSION_STEPS=2 SURE_XFORGE_STATIC_ONLY=0 \
src/sure_eval/models/Plachta__Seed-VC/.venv/bin/python \
src/sure_eval/models/Plachta__Seed-VC/validate.py
```

## 6. Wrapper 要求

`model.py` 必须：

- import 阶段不加载大模型。
- `load()` 阶段加载所有 checkpoint。
- `predict()` 接收 `source_audio_path` 和 `reference_audio_path`。
- 输出写到 `artifacts/outputs`。
- 记录 sample rate、num samples、device、source_dir。
- 对上游 `cwd`、cache、checkpoint 路径做 model-local 重定向。

输出示例：

```json
{
  "audio_path": "artifacts/outputs/seed_vc_v2_smoke.wav",
  "source_audio_path": "fixture/zh/source.mp3",
  "reference_audio_path": "fixture/zh/reference.mp3",
  "task": "VC",
  "raw": {"sample_rate": 22050, "num_samples": 165632, "device": "cuda"}
}
```

## 7. Docker 验证

Docker 规则：

- 镜像内放 Python 环境和依赖。
- `.runtime/source`、`.runtime/huggingface`、`fixture`、`artifacts` 通过 volume 挂载。
- 权重和 HF cache 不 bake 进镜像。
- 对上游退出慢要用 timeout 包裹，但不能把真正失败吞掉。

Seed-VC 经验：

- 推理和 `VALIDATE_CONTRACT` 已通过后，上游进程可能不及时退出并继续占 GPU。
- `docker_validate.sh` 可用 `timeout ${VALIDATE_TIMEOUT_SECONDS}s python validate.py`。
- 如果 stdout 有 `exit status 124`，必须检查 `artifacts/validation.log` 是否已经写入
  `VALIDATE_CONTRACT passed`。只有 contract 已通过，才能记录为
  `passed_with_manual_container_stop_after_success` 或等价状态。
- re-onboarding 时不能把 `.runtime/source/seed-vc` 仅作为只读旧目录软链接挂载。
  上游 `hf_utils.py` 会在 source cwd 下写 `./checkpoints`，若 source 挂载只读会报
  `Read-only file system: './checkpoints/...'`。处理方式是使用 model-local 可写
  runtime copy，或把 `checkpoints` 显式重定向到 re-onboard 可写目录，并记录原因。
- HF cache 软链接必须用 `readlink -f` 验证。坏链接会导致 Transformers 报
  `There was a problem when trying to write in your cache folder` 或
  `FileNotFoundError: .../.runtime/huggingface/hub/models--...`。
- VC metric 必须调用 `src/sure_eval/evaluation/tasks/vc`。如果真实 provider 依赖缺失，
  不能把 stub 报告当成真实指标；应同时保存：
  - provider 失败报告，记录缺失依赖，例如 `funasr`、`torch`、`librosa`、
    `transformers`、`pytorch_lightning`、`addict`。
  - stub 报告，只用于证明 pipeline 入参、产物路径和 JSON contract 打通。
- 正式 VC metric 应优先使用与 TTS 对齐的分段 Docker runner：
  `scripts/run_vc_metric_pipeline_docker.py` 或
  `scripts/run_vc_metric_pipeline_docker.sh`。不要只用
  `scripts/run_vc_metric_pipeline.py` 加 `.venv.hostbak` 判断正式失败；单环境 runner
  可能缺少 FunASR、speaker verification、MOS provider 等重依赖，而 Docker runner 会按
  semantic / speaker / MOS 分段进入已验证指标镜像。
- 若 Docker validate 的输出音频挂载到了 `docker_artifacts/outputs/`，而本地 metric
  wrapper 默认查 `artifacts/outputs/`，需要显式传入 `CONVERTED_AUDIO` 或让 wrapper
  自动 fallback 到 `docker_artifacts/outputs/`。不要因此误判“模型没有生成音频”。
- `docker_validate.sh` 使用 `docker run ... | tee log` 时，必须启用 `set -o pipefail`
  或以 `validation.log` 的阶段记录为准；否则可能被 `tee` 管道掩盖真实退出码。

已验证镜像：

```text
docker.v2.aispeech.com/sjtu/sjtu_yukai-dujunhao-sure_plachtaa__seed-vc:v1.0
```

## 8. Verdict 标准

`verdict.json` 至少记录：

- `status`
- `task: "VC"`
- configured backends: `uv`, `docker`
- Docker image、image id、registry digest
- output audio path、sample rate、num samples
- timeout 或手动 stop 说明
- HF mirror / proxy 规则

不能只看脚本 exit code。Seed-VC 的 `docker_validate.sh` 可能在 timeout 后仍 exit 0，
所以必须以 `validation.log` 的阶段记录为准。

## 9. 新 VC 模型接入检查表

- [ ] 已读 `AGENTS.md` 和本文。
- [ ] 明确 source audio 与 reference audio。
- [ ] 上游源码、checkpoint、HF cache 都在 model-local `.runtime`。
- [ ] `HF_ENDPOINT`、`HF_HOME`、`HF_HUB_CACHE` 指向可复现路径。
- [ ] wrapper 对上游相对 checkpoint 路径做 model-local 处理。
- [ ] local uv GPU 通过或失败原因结构化记录。
- [ ] Docker GPU 通过，timeout 行为已核验。
- [ ] `validation.log` 有 `VALIDATE_CONTRACT passed`。
- [ ] 正式 VC metric 走 `scripts/run_vc_metric_pipeline*.py` 或等价 wrapper，
      并引用 `src/sure_eval/evaluation/tasks/vc`。
- [ ] `verdict.json` 或 run report 记录 metric report 路径、runner、cache_dir 和核心指标。
- [ ] 镜像 push/pull digest 已记录。
