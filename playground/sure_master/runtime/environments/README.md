# Isolated model environments

Each task has two launch profiles: `configs/sure_master/ordinary-{asr,tts,sd}-{cuda,npu}.yaml`.
Set `SURE_WORKER_PYTHON` to that task/backend's interpreter. Model libraries do not
belong in the coordinator environment. CPU SURE scoring uses `SURE_METRIC_PYTHON`.

| Task | CUDA base | NPU base | Model installation |
|---|---|---|---|
| ASR | Existing Icefall CUDA image | `docker/sure-master-ascend/Dockerfile` | Preserve the existing k2/torchaudio build pair |
| F5-TTS | Torch/torchaudio 2.4.0 + CUDA 12.1, Python 3.10/3.11 | Same pinned Ascend base image as ASR, with matching Torch/torch_npu/torchaudio | `f5tts.txt`; editable F5 source with `--no-deps` |
| DiariZen | Torch/torchaudio 2.1.1 + CUDA 12.1, Python 3.10/3.11 | Same pinned Ascend base image as ASR, with matching Torch/torch_npu/torchaudio | `diarizen.txt`; editable DiariZen and its custom pyannote-audio |

NPU images must match the host driver/CANN version. Keep the image-provided Torch,
Torch NPU and torchaudio versions pinned while installing the overlays; do not
install upstream CUDA-only torch constraints or `onnxruntime-gpu` in an NPU image.
The DiariZen custom pyannote package has its own Python dependencies: install it
with constraints matching the image's Torch/torchaudio, then run `pip check`.

F5's training/inference wrappers do not use bitsandbytes, FlashAttention or the
Gradio UI. Their CUDA-only or UI dependencies are intentionally excluded from the
worker overlay. The workspace source adapter disables fused AdamW and uses the
Torch attention backend. NPU mel FFT and vocoder synthesis explicitly use CPU;
the F5 backbone's parameters and gradients remain on NPU.

DiariZen segmentation and speaker embedding run on the selected device. Clustering
and the permutation assignment use the upstream CPU/SciPy implementations.

Use `probe_task.py --component imports` first, then a bounded synthetic
`--component train` or `--component arch`. A launch profile or passing CPU contract
test is **not** a claim of successful CUDA/NPU hardware validation. Preserve the
probe JSON and `pip freeze` alongside the exact container image digest.

F5 and DiariZen workspace adapters use SoundFile for WAV loading and audio metadata,
so the model path does not depend on torchaudio 2.10's TorchCodec/FFmpeg ABI. These
are CPU I/O operations; neural tensors are explicitly moved to the selected device.
