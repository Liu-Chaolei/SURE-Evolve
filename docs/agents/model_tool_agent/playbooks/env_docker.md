# Docker 环境策略 Playbook

## 何时必须容器化

| 条件 | 使用 Docker |
|------|-------------|
| 宿主机污染风险高 | ✅ 必须 |
| 依赖特定 OS 版本 | ✅ 必须 |
| 复杂系统库依赖 | ✅ 必须 |
| 多模型环境冲突 | ✅ 必须 |
| 需要可复现的完整环境 | ✅ 推荐 |
| `deployment_type: local` 且 Phase-1/local validation 已通过 | ✅ 必须 |

说明：Phase-1 backend 可以是 `uv`、`pixi`、`conda` 或 `docker`。这只影响本地最小
验证路径，不影响最终部署闭环。只要 `deployment_type: local`，本地验证通过后就必须
按本 playbook 完成 Docker build、Docker validate、registry push/pull 验证和 VC
验证；不能因为 chosen_backend 不是 docker 而跳过。

## Docker 基本要求

### 本地基础镜像优先

在当前超算环境中，构建模型镜像前必须先查本地已有镜像：

```bash
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'
```

不要默认从 registry 拉取基础镜像。若 `FROM <image>` 指向的镜像本地不存在，
BuildKit 会先访问 registry metadata；registry 不通时会在 Dockerfile 执行前失败。
处理方式：

1. 选择本地已存在且同任务/同依赖族的基础镜像。
2. 在 `backend_choice.json` 或 `failure_classification.json` 记录原 base 不存在和切换原因。
3. 不要把旧模型权重 bake 进新镜像；权重通过运行时挂载或 model-local `.runtime` 提供。

### Dockerfile 模板

```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# 基础依赖
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    libsndfile1 ffmpeg \
    git wget \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 安装 PyTorch
RUN pip install torch==2.4.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 工作目录
WORKDIR /workspace

# 复制代码
COPY . .

# 安装模型依赖
RUN pip install -e .

# 入口
CMD ["python", "server.py"]
```

### 构建和运行

```bash
# 构建
docker build -t sure-eval-model:{model_name} .

# 运行（CPU）
docker run -v $(pwd)/data:/data sure-eval-model:{model_name}

# 运行（GPU）
docker run --gpus all -v $(pwd)/data:/data sure-eval-model:{model_name}
```

## DevContainer 配置

### .devcontainer/devcontainer.json

```json
{
  "name": "SURE-EVAL Model Environment",
  "image": "nvidia/cuda:12.1-devel-ubuntu22.04",
  "features": {
    "ghcr.io/devcontainers/features/python:1": {
      "version": "3.10"
    }
  },
  "runArgs": ["--gpus=all"],
  "postCreateCommand": "pip install -e .",
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python"]
    }
  }
}
```

## 高风险仓库隔离原则

1. **不修改宿主机**：所有操作在容器内完成
2. **只读挂载**：代码以只读方式挂载
3. **输出分离**：工件输出到独立卷
4. **网络隔离**：按需开启外网访问

## 常见失败和修复

### 1. GPU 不可用

**症状**: `RuntimeError: No CUDA GPUs are available`

**修复**:
```bash
# 检查 nvidia-docker
docker run --gpus all nvidia/cuda:12.1-base nvidia-smi

# 重新安装 nvidia-container-toolkit
```

### 1.1 本地 GPU OOM 不能作为闭环终点

**症状**: 本地 Docker 或 local uv 能看到 CUDA，但验证失败为
`torch.OutOfMemoryError` / `CUDA out of memory`，尤其是本地 2080Ti/11GB
这类小显存调试卡。

**要求**:

- 不能直接把模型标记为最终失败或只保留 CPU fallback。
- 必须继续提交到超算/VC 的更大显存 GPU 队列验证，例如 `pdgpu-a10`,
  `pdgpu-3090`, `pdgpu-4090`，按当前账号 `vc info -u` 可见队列选择。
- VC 验证使用已经构建好的 Docker 镜像；镜像必须能从 registry 拉取，或明确
  记录该任务只在本机可运行且不能算 cluster-ready。
- `verdict.json` 必须区分：
  - `local_gpu_oom`
  - `docker_local_gpu_oom`
  - `vc_gpu_passed`
  - `vc_gpu_failed`

**推荐流程**:

```bash
vc submit \
  -p <gpu_partition> \
  -i <image_tag> \
  -j <job_name> \
  -n 1 -c 8 -m 32G -g 1 \
  -pj <project> \
  -d <absolute_model_dir> \
  -e PYTHONPATH=<absolute_repo_src> \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --cmd '/opt/<model>_venv/bin/python validate.py'
```

注意：VC 作业本身已经运行在 `-i <image_tag>` 指定的容器中，**不要在 VC
容器内部再调用 `docker run` 或本地 `docker_validate.sh`**。`docker_validate.sh`
用于本地 Docker 验证；VC 验证应直接执行镜像内 Python 和模型 `validate.py`。

`vc submit -e/--env` 是可变长参数，必须放在不会吞掉其它位置参数的位置；不要在
`-e` 后追加日志路径。本环境单 GPU 提交可能限制为 CPU<=8、MEM<=32G，若提交返回
“单GPU申请的CPU核数不能超过8个，且单GPU申请的MEM大小不能超过32G”，必须按
`-c 8 -m 32G -g 1` 重提并记录原因。

只有 VC 也失败，或者没有可用 GPU 队列/镜像无法拉取并已记录原因，才允许将
GPU 验证标记为 blocked。

### 2. 权限问题

**症状**: `Permission denied`

**修复**:
```dockerfile
# Dockerfile 中添加
RUN useradd -m -u 1000 modeluser
USER modeluser
```

### 3. 模型下载超时

**症状**: 容器内下载 HuggingFace 模型超时

**修复**:
```bash
# 挂载 host cache
docker run -v ~/.cache/huggingface:/root/.cache/huggingface ...
```
