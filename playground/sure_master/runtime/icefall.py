"""Icefall-specific device compatibility; never used by other tasks."""
from __future__ import annotations
import os
import shutil
from pathlib import Path

def prepare_npu_recipe(source: Path, destination: Path) -> Path:
    """Materialize links before patching; never write through Icefall symlinks."""
    if os.environ.get("SURE_ASR_RECIPE_PROFILE") not in {"tedlium3_zipformer", "tedlium3_zipformer_native"}:
        raise ValueError("The NPU adapter currently supports tedlium3_zipformer")
    if os.environ.get("SURE_USE_FP16", "0") != "0":
        raise ValueError("NPU compatibility validation currently requires FP32")
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_file():
            if target.is_symlink():
                target.unlink()
            shutil.copy2(item, target)
        elif item.is_dir() and not target.exists():
            target.symlink_to(item.resolve(), target_is_directory=True)
    for name in ("train.py", "decode.py", "model.py"):
        path = destination / name
        text = path.read_text()
        if name == "train.py":
            text = text.replace("from icefall.dist import cleanup_dist, setup_dist",
                                "from playground.sure_master.runtime.distributed import cleanup_dist, setup_dist")
            threads = max(1, min(32, int(os.environ.get("SURE_CPU_THREADS", "4"))))
            text = text.replace("torch.set_num_threads(1)", f"torch.set_num_threads({threads})")
            text = text.replace("from torch.cuda.amp import GradScaler", "from torch_npu.npu.amp import GradScaler")
            text = text.replace("k2.RaggedTensor(y).to(device)", "k2.RaggedTensor(y)")
            marker = "        if params.print_diagnostics and batch_idx == 5:"
            text = text.replace(marker, "        from playground.sure_master.runtime.profiling import profile_step\n"
                                "        profile_step(batch, params.batch_idx_train, rank, world_size)\n\n" + marker)
        text = text.replace("torch.cuda", "torch.npu")
        text = text.replace('torch.device("cuda",', 'torch.device("npu",')
        text = "import torch_npu\n" + text
        if name == "model.py":
            text = text.replace("import k2\n", "import k2\nfrom playground.sure_master.runtime import npu_k2\n")
            for function in ("rnnt_loss_smoothed", "rnnt_loss_pruned", "get_rnnt_prune_ranges"):
                if f"k2.{function}(" not in text:
                    raise ValueError(f"Unsupported Icefall recipe: missing {function}")
                text = text.replace(f"k2.{function}(", f"npu_k2.{function}(")
            text = text.replace("self.decoder(sos_y_padded)", "self.decoder(sos_y_padded.to(encoder_out.device))")
        if name == "decode.py" and os.environ.get("SURE_ASR_CPU_AVERAGING") == "1":
            marker = 'if __name__ == "__main__":'
            if marker not in text:
                raise ValueError("Missing decode entrypoint for CPU averaging")
            override = "from playground.sure_master.runtime.asr_averaging import average_checkpoints_with_averaged_model"
            if override not in text:
                text = text.replace(marker, override + "\n\n" + marker)
        path.write_text(text)
    scaling = destination / "scaling.py"
    scaling.write_text(scaling.read_text().replace("import k2\n", "from playground.sure_master.runtime import npu_k2 as k2\n"))
    # Pack variable-length encoder outputs without the Ascend padded-row bug.
    beam_search = destination / "beam_search.py"
    beam_text = beam_search.read_text()
    packing_call = "torch.nn.utils.rnn.pack_padded_sequence("
    if packing_call not in beam_text and "from npu_rnn import pack_padded_sequence" not in beam_text:
        raise ValueError("Unsupported Icefall recipe: missing packed sequence decoding")
    if "from npu_rnn import pack_padded_sequence" not in beam_text:
        beam_text = beam_text.replace("import torch\n", "import torch\nfrom npu_rnn import pack_padded_sequence\n")
    beam_search.write_text(beam_text.replace(packing_call, "pack_padded_sequence("))
    helper = destination / "npu_rnn.py"
    if helper.is_symlink():
        helper.unlink()
    shutil.copy2(Path(__file__).with_name("npu_rnn.py"), helper)
    return destination


def prepare_cpu_recipe(source: Path, destination: Path) -> Path:
    """Keep upstream CPU code but honor the controller's CPU thread budget."""
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_file():
            if target.is_symlink():
                target.unlink()
            shutil.copy2(item, target)
        elif item.is_dir() and not target.exists():
            target.symlink_to(item.resolve(), target_is_directory=True)
    train = destination / "train.py"
    threads = max(1, min(32, int(os.environ.get("SURE_CPU_THREADS", "4"))))
    train.write_text(train.read_text().replace("torch.set_num_threads(1)", f"torch.set_num_threads({threads})"))
    return destination


def recipe_changes(recipe: Path | None = None) -> dict[str, str]:
    """Record executable source differences from the read-only TEDLIUM recipe."""
    import ast
    import hashlib
    recipe = recipe or Path('base_model/recipe')
    original = Path('base_model/root/egs/tedlium3/ASR/zipformer')
    if not original.is_dir() or not recipe.is_dir():
        return {}
    changes = {}
    for path in sorted(recipe.glob('*.py')):
        source = path.read_text()
        upstream = original / path.name
        if upstream.exists() and ast.dump(ast.parse(source)) == ast.dump(ast.parse(upstream.read_text())):
            continue
        changes[path.name] = hashlib.sha256(source.encode()).hexdigest()
    return changes
