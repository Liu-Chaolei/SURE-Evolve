from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .common import JsonObject, atomic_write_json, text, utc_now


GPU_BACKENDS = {"vlm-engine", "hybrid-engine"}
SUPPORTED_BACKENDS = {
    "auto",
    "pipeline",
    "vlm-engine",
    "hybrid-engine",
}


def require_llm_environment() -> tuple[str, str, str]:
    api_url = text(os.environ.get("KNOWLEDGE_GRAPH_LLM_API_URL"))
    api_key = text(os.environ.get("KNOWLEDGE_GRAPH_LLM_API_KEY"))
    model = text(os.environ.get("KNOWLEDGE_GRAPH_LLM_MODEL"))
    missing = [
        name
        for name, value in (
            ("KNOWLEDGE_GRAPH_LLM_API_URL", api_url),
            ("KNOWLEDGE_GRAPH_LLM_API_KEY", api_key),
            ("KNOWLEDGE_GRAPH_LLM_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing required LLM environment variables: " + ", ".join(missing)
        )
    parsed = urlsplit(api_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "KNOWLEDGE_GRAPH_LLM_API_URL must be an absolute HTTP(S) URL."
        )
    return api_url, api_key, model


def public_api_url(api_url: str) -> str:
    parsed = urlsplit(api_url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def resolve_mineru_command() -> Path:
    sibling = Path(sys.executable).resolve().parent / "mineru"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return sibling
    located = shutil.which("mineru")
    if located:
        return Path(located).resolve()
    raise ValueError(
        "MinerU executable is unavailable. Recreate the skill environment from "
        "requirements.lock."
    )


def _visible_gpu_filter() -> set[str] | None:
    raw = text(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not raw:
        return None
    values = {value.strip() for value in raw.split(",") if value.strip()}
    return values or None


def probe_gpus() -> tuple[list[JsonObject], str | None]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return [], "nvidia-smi is unavailable"
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return [], f"GPU probe failed: {error}"
    visible = _visible_gpu_filter()
    gpus: list[JsonObject] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, total_raw, free_raw, utilization_raw = parts
        if visible is not None and index not in visible:
            continue
        try:
            total_mib = int(total_raw)
            free_mib = int(free_raw)
            utilization = int(utilization_raw)
        except ValueError:
            continue
        gpus.append(
            {
                "index": index,
                "name": name,
                "memory_total_mib": total_mib,
                "memory_free_mib": free_mib,
                "utilization_percent": utilization,
            }
        )
    return gpus, None


def build_resource_plan(
    *,
    output_path: Path,
    requested_backend: str,
    requested_gpus: list[str],
    max_mineru_workers: int,
    llm_workers: int,
    min_gpu_free_gib: float,
    max_gpu_utilization: int,
    requested_device: str = "auto",
) -> JsonObject:
    if requested_backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported MinerU backend: {requested_backend}")
    mineru_command = resolve_mineru_command()
    if requested_device not in {"auto", "cpu", "cuda", "npu"}:
        raise ValueError(f"Unsupported MinerU device: {requested_device}")
    if requested_device in {"cpu", "npu"} and requested_gpus:
        raise ValueError("--gpu requires a CUDA MinerU device.")
    gpus, probe_warning = probe_gpus() if requested_device in {"auto", "cuda"} else ([], None)
    available = {text(gpu.get("index")): gpu for gpu in gpus}
    if requested_gpus:
        unknown = [index for index in requested_gpus if index not in available]
        if unknown:
            raise ValueError(
                "Requested GPU indexes are unavailable or hidden by "
                f"CUDA_VISIBLE_DEVICES: {', '.join(unknown)}"
            )
        selected = [available[index] for index in requested_gpus]
    else:
        selected = [
            gpu
            for gpu in gpus
            if int(gpu.get("memory_free_mib") or 0) >= min_gpu_free_gib * 1024
            and int(gpu.get("utilization_percent", 100)) <= max_gpu_utilization
        ]
        selected.sort(
            key=lambda gpu: (
                -int(gpu.get("memory_free_mib") or 0),
                int(text(gpu.get("index")) or 0),
            )
        )

    device = requested_device
    if device == "auto":
        device = "cuda" if selected and requested_backend != "pipeline" else "cpu"
    npus: list[str] = []
    if device == "npu":
        visible = text(os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
        npus = [value.strip() for value in visible.split(",") if re.fullmatch(r"\d+", value.strip())]
        if not npus or not shutil.which("npu-smi"):
            raise ValueError("NPU parsing requires an Ascend allocation with ASCEND_RT_VISIBLE_DEVICES and npu-smi.")
        if requested_backend not in {"auto", "pipeline"}:
            raise ValueError("Ascend parsing currently supports --mineru-backend pipeline.")
    backend = requested_backend
    if backend == "auto":
        backend = "vlm-engine" if device == "cuda" else "pipeline"
    if backend in GPU_BACKENDS and device != "cuda":
        raise ValueError(f"MinerU backend {backend} requires a CUDA device.")
    if device == "cuda" and not selected:
        raise ValueError(
            f"MinerU backend {backend} requires an eligible GPU. Use --gpu INDEX, "
            "lower the resource thresholds, or choose --mineru-backend pipeline."
        )
    if device != "cuda":
        selected = []
    if device == "cuda":
        mineru_workers = min(max_mineru_workers, len(selected))
    elif device == "npu":
        mineru_workers = min(max_mineru_workers, len(npus))
    else:
        allocated_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        cpu_threads = max(int(os.environ.get("OMP_NUM_THREADS", "16")), 1)
        mineru_workers = min(max_mineru_workers, max(allocated_cpus // cpu_threads, 1))
    plan: JsonObject = {
        "schema_version": "xlab.knowledge_graph_resources.v1",
        "generated_at": utc_now(),
        "mineru": {
            "command": str(mineru_command),
            "requested_backend": requested_backend,
            "backend": backend,
            "device": device,
            "workers": max(mineru_workers, 1),
            "selected_gpus": [text(gpu.get("index")) for gpu in selected],
            "selected_npus": npus[:mineru_workers],
            "all_visible_gpus": gpus,
            "minimum_free_gib": min_gpu_free_gib,
            "maximum_utilization_percent": max_gpu_utilization,
            "probe_warning": probe_warning,
        },
        "llm": {
            "workers": llm_workers,
        },
    }
    atomic_write_json(output_path, plan)
    return plan
