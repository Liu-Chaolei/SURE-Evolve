# UV 环境策略 Playbook

## 何时使用 UV

| 条件 | 使用 UV |
|------|---------|
| 纯 Python 依赖 | ✅ 推荐 |
| 主要依赖为 PyPI 包 | ✅ 推荐 |
| 无复杂 C++ 扩展 | ✅ 推荐 |
| 无系统库依赖 | ✅ 推荐 |
| 无 CUDA 编译需求 | ✅ 推荐 |

## 环境创建方式

```bash
cd src/sure_eval/models/{model_dir_slug}

# 创建环境
uv venv --python=python3.10

# 激活环境
source .venv/bin/activate

# 安装依赖
uv pip install -r requirements.txt
# 或
uv pip install -e .
```

`model_dir_slug` 必须由 `model.spec.yaml` 的 `weights.repo_id` 推导：
`weights.repo_id.replace("/", "__")`。不要由 `model_name`、任务前缀或人工简称推导
模型目录。

Re-onboarding 例外规则：

- 在 `src/sure_eval/models_reonboard/runs/<model>/` 复现模型接入时，必须创建
  run-local 的新 `.venv` 目录。
- 不允许把 re-onboarding `.venv` 软链到
  `src/sure_eval/models/<model>/.venv`，否则无法证明 tool-agent 重新接管了环境配置。
- 可以复用 uv package cache、模型权重、HF/ModelScope cache 或上游源码，但必须在
  `artifact_manifest.json` / `verdict.json` 中明确记录哪些是复用资源。

## Lock/Sync 约定

```bash
# 导出精确依赖
uv pip freeze > requirements.lock

# 从 lock 恢复
uv pip install -r requirements.lock
```

## 常见失败和修复

### 1. 系统 PyTorch 与虚拟环境冲突

**症状**: `ModuleNotFoundError: No module named 'torch._utils'`

**原因**: UV 环境隔离导致无法访问系统已安装的 PyTorch

**修复**:
```bash
# 在虚拟环境中重新安装 PyTorch
uv pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. NumPy 版本冲突

**症状**: `AttributeError: np.sctypes was removed in the NumPy 2.0`

**原因**: NeMo 等库不支持 NumPy 2.0

**修复**:
```bash
uv pip install numpy==1.26.4
```

### 3. Torchvision 版本不匹配

**症状**: `RuntimeError: operator torchvision::nms does not exist`

**修复**:
```bash
# 安装与 PyTorch 匹配的 torchvision
uv pip install torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
```

### 4. S2TT metric 缺少 sacrebleu

**症状**: speech-understanding 或 S2TT metric runner 报：

```text
No module named 'sacrebleu'
```

**原因**: `sure_eval.evaluation.tasks.s2tt.metrics.BLEUMetric` 依赖 `sacrebleu`。

**修复**: 使用 uv 和清华源安装到当前 SURE 环境，不要改模型推理代码：

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv pip install \
  -p .venv.hostbak/bin/python \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  sacrebleu
```

安装后重跑原 metric runner。

### 5. CUDA torch wheel 下载超时

**症状**: 本地 uv 环境安装 CUDA 版 PyTorch 报：

```text
Failed to fetch https://download-r2.pytorch.org/...torch-*.whl
operation timed out
```

**原因**: `download.pytorch.org` / `download-r2.pytorch.org` 在当前网络下可能不稳定。

**修复**:

- 优先使用已有本地 uv cache 或模型本地缓存。
- 若必须访问 PyTorch 官方 CUDA wheel，按用户约定临时开启代理：

```bash
bash -lc '. /hpc_stor03/sjtu_home/junhao.du/.local/bin/ssr-on && \
  UV_CACHE_DIR=/tmp/uv-cache uv pip install \
    --python <model_dir>/.venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1+cu124 torchaudio==2.5.1+cu124; \
  status=$?; \
  . /hpc_stor03/sjtu_home/junhao.du/.local/bin/ssr-off; \
  exit $status'
```

代理只用于 GitHub/HuggingFace/PyTorch 官方源等外网受阻场景，用完必须关闭。
如果 uv 报 hardlink fallback：

```text
Failed to hardlink files; falling back to full copy
```

这是 cache 与目标目录跨文件系统导致的性能警告；可设置
`UV_LINK_MODE=copy` 消除噪声，不应把它当成安装失败。

### 6. TTS 本地 uv 依赖 pinning

F5-TTS / IndexTTS-2 re-onboarding 已验证：

- 对当前 CUDA 12.8 driver，TTS 本地 uv 优先 pin：
  `torch==2.8.0+cu128`、`torchaudio==2.8.0+cu128`。
- 不要让 uv 无约束安装 PyTorch；曾出现 `torch 2.12.1+cu130` 被安装后
  `torch.cuda.is_available() == False` 的失败。
- F5-TTS 的 `datasets==2.14.x` 与新 `pyarrow` 不兼容时，会报
  `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'`，
  应 pin `pyarrow<21`。
- IndexTTS-2 需要 `transformers==4.52.1` 才能识别 `qwen3` model_type。
- IndexTTS-2 在当前 index 下不要写 `modelscope[audio]==1.27.0`；该 extra
  可能要求不可解的 `mindaec==0.0.2`。已验证组合是
  `modelscope==1.27.0` 加显式音频依赖。
- IndexTTS-2 的 `indextts.s2mel.dac` 会 import `audiotools`，本地 uv 需要
  `descript-audiotools==0.7.2`。

## 集群网络限制

### 常见环境约束

在部分集群环境中，以下网络限制可能影响模型 onboarding：

- **HuggingFace 被封锁**：无法访问 `https://huggingface.co`，必须使用 **ModelScope** 作为模型/数据下载源。
- **PyTorch CDN 被封锁**：`https://download.pytorch.org` 可能无法访问，安装 torch 时需使用清华镜像：
  ```bash
  uv pip install torch==2.4.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- **当前超算环境默认优先清华源**：如果没有明确证据要求访问官方 PyPI，
  uv 安装 PyPI 依赖应优先使用清华源，避免默认 `https://pypi.org/simple`
  DNS/连接失败或长时间卡住：
  ```bash
  UV_CACHE_DIR="$PWD/.runtime/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/.runtime/uv-python" \
  uv pip install --python .venv/bin/python \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt
  ```
- **uv 默认目录可能不可写**：不要让 uv 写入只读的
  `~/.cache/uv` 或 `~/.local/share/uv/python`。每个模型 onboarding
  必须设置 run-local 路径：
  ```bash
  export UV_CACHE_DIR="$PWD/.runtime/uv-cache"
  export UV_PYTHON_INSTALL_DIR="$PWD/.runtime/uv-python"
  ```
- **matplotlib 默认配置目录可能不可写**：NeMo、Whisper、TTS 等依赖可能在
  import 阶段触发 matplotlib cache。每个模型本地验证脚本应设置：
  ```bash
  export MPLCONFIGDIR="$PWD/.runtime/matplotlib"
  mkdir -p "$MPLCONFIGDIR"
  ```
- **不要用管道掩盖安装失败**：如果需要保存日志，脚本必须设置
  `set -o pipefail`，否则 `uv pip install ... | tee build.log` 可能把
  uv 的失败退出码掩盖成成功。
- **torchvision 版本兼容性**：安装依赖后必须验证 torchvision 与 torch 版本匹配：
  ```bash
  .venv/bin/python -c "import torch, torchvision; print(f'torch {torch.__version__} + torchvision {torchvision.__version__}')"
  ```
  若 torchvision 主版本号与 torch 不匹配（例如 torch 2.4.0 + torchvision 0.26.0），必须降级 torchvision 到匹配版本。参考对应关系：
  | torch | torchvision |
  |-------|-------------|
  | 2.4.0 | 0.19.0 |
  | 2.5.0 | 0.20.0 |
  | 2.6.0 | 0.21.0 |

### 验证清单（onboarding 后必做）

```bash
# 1. 验证 torchvision 兼容性
.venv/bin/python -c "import torch, torchvision; tv_major = int(torchvision.__version__.split('.')[1]); torch_major = int(torch.__version__.split('.')[1]); assert tv_major == torch_major - 5, f'torchvision {torchvision.__version__} incompatible with torch {torch.__version__}'"

# 2. 验证 ModelScope 可达
.venv/bin/python -c "from modelscope.hub.api import HubApi; api = HubApi(); print('ModelScope OK')"

# 3. 验证模型可加载（不触发下载）
.venv/bin/python -c "from model import ModelWrapper; m = ModelWrapper(); m.load(); print('Model load OK')"
```

## 验证命令

```bash
# 验证 Python 版本
.venv/bin/python --version

# 验证关键包
.venv/bin/python -c "import torch; print(torch.__version__)"
.venv/bin/python -c "import numpy; print(numpy.__version__)"
```
