"""Materialize external sources before applying explicit backend compatibility edits."""

from __future__ import annotations

import shutil
import ast
from pathlib import Path


def snapshot_source(source: Path, target: Path) -> Path:
    source, target = source.resolve(), target.absolute()
    if target.is_symlink():
        raise ValueError("Source snapshot cannot be a symlink")
    if source == target.resolve():
        return target
    target.resolve().relative_to(Path.cwd().resolve())
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        symlinks=False,
        ignore=shutil.ignore_patterns(
            ".git",
            ".cache",
            ".env",
            ".env.*",
            "__pycache__",
            "ckpts",
            "checkpoints",
            "data",
            "runs",
            "results",
            "exp",
            "*.log",
        ),
    )
    return target


def prepare_f5_source(root: Path) -> None:
    """Patch only a workspace copy. CPU FFT is explicit and differentiable."""
    root.resolve().relative_to(Path.cwd().resolve())
    marker = root / ".sure_backend_patch"
    if marker.exists():
        return
    utils = root / "src/f5_tts/infer/utils_infer.py"
    source = utils.read_text()
    source = source.replace(
        "load_file(ckpt_path, device=device)", 'load_file(ckpt_path, device="cpu")'
    )
    source = source.replace(
        "torch.load(ckpt_path, map_location=device,",
        'torch.load(ckpt_path, map_location="cpu",',
    )
    source = source.replace(
        "torch.cuda.empty_cache()", "None  # cleanup belongs to RuntimeBackend"
    )
    source = source.replace(
        'dtype = torch.float32 if mel_spec_type == "bigvgan" else None',
        'dtype = torch.float32 if os.environ.get("SURE_PRECISION", "fp32") == "fp32" else None',
    )
    utils.write_text(source)
    trainer = root / "src/f5_tts/model/trainer.py"
    trainer.write_text(trainer.read_text().replace("fused=True", "fused=False"))
    modules = root / "src/f5_tts/model/modules.py"
    source = modules.read_text()
    source += """\n# SURE NPU: spectral preprocessing runs on CPU, attention remains on NPU.
_sure_original_mel_forward = MelSpec.forward
def _sure_mel_forward(self, waveform):
    device = waveform.device
    if device.type == "npu":
        self.to("cpu")
        return _sure_original_mel_forward(self, waveform.to("cpu")).to(device)
    return _sure_original_mel_forward(self, waveform)
MelSpec.forward = _sure_mel_forward
"""
    modules.write_text(source)
    package = root / "src/f5_tts/__init__.py"
    package.write_text(
        (package.read_text() if package.exists() else "")
        + "\nimport os as _sure_os\nif _sure_os.environ.get('SURE_ACCELERATOR') == 'npu':\n    import torch_npu\n"
    )
    for relative in (
        "infer/utils_infer.py",
        "model/dataset.py",
        "train/datasets/prepare_csv_wavs.py",
    ):
        source_file = root / "src/f5_tts" / relative
        if source_file.exists():
            prepare_audio_io(source_file)
    marker.write_text("portable-v1")


def prepare_audio_io(path: Path) -> None:
    """Patch only the simple load/info calls used by these model sources."""
    path.resolve().relative_to(Path.cwd().resolve())
    source = path.read_text()
    if "torchaudio.load(" not in source and "torchaudio.info(" not in source:
        return
    tree = ast.parse(source)
    insertion = 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        insertion = tree.body[0].end_lineno
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insertion = max(insertion, node.end_lineno)
    lines = source.splitlines(keepends=True)
    lines.insert(
        insertion,
        "from playground.sure_master.runtime.audio_io import load_audio as _sure_load_audio, audio_info as _sure_audio_info\n",
    )
    source = (
        "".join(lines)
        .replace("torchaudio.load(", "_sure_load_audio(")
        .replace("torchaudio.info(", "_sure_audio_info(")
    )
    ast.parse(source)
    path.write_text(source)
