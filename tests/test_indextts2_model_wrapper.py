from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


def _load_indextts2_model_module():
    module_path = Path("src/sure_eval/models/IndexTeam__IndexTTS-2/model.py")
    spec = importlib.util.spec_from_file_location("indextts2_model_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_model_wrapper_uses_transformers_loader_for_qwen_emotion(monkeypatch, tmp_path):
    model_module = _load_indextts2_model_module()

    model_root = tmp_path / "model"
    source_root = tmp_path / "source"
    qwen_root = model_root / "qwen0.6bemo4-merge"
    qwen_root.mkdir(parents=True)
    source_root.mkdir()
    for relative_path in [
        "config.yaml",
        "gpt.pth",
        "s2mel.pth",
        "bpe.model",
        "wav2vec2bert_stats.pt",
        "feat1.pt",
        "feat2.pt",
        "qwen0.6bemo4-merge/config.json",
    ]:
        (model_root / relative_path).touch()

    transformers_loader = object()
    modelscope_loader = object()
    observed: dict[str, object] = {}

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = transformers_loader
    fake_modelscope = types.ModuleType("modelscope")
    fake_modelscope.AutoModelForCausalLM = modelscope_loader

    fake_indextts = types.ModuleType("indextts")
    fake_infer_v2 = types.ModuleType("indextts.infer_v2")
    fake_infer_v2.AutoModelForCausalLM = modelscope_loader

    class FakeIndexTTS2:
        def __init__(self, **kwargs):
            observed["loader"] = fake_infer_v2.AutoModelForCausalLM
            observed["kwargs"] = kwargs
            observed["world_size_during_init"] = os.environ.get("WORLD_SIZE")
            observed["rank_during_init"] = os.environ.get("RANK")

    fake_infer_v2.IndexTTS2 = FakeIndexTTS2

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)
    monkeypatch.setitem(sys.modules, "indextts", fake_indextts)
    monkeypatch.setitem(sys.modules, "indextts.infer_v2", fake_infer_v2)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")

    wrapper = model_module.ModelWrapper(
        {
            "model_path": str(model_root),
            "source_path": str(source_root),
            "device": "cuda:0",
        }
    )

    wrapper.load()

    assert observed["loader"] is transformers_loader
    assert observed["world_size_during_init"] is None
    assert observed["rank_during_init"] is None
    assert os.environ["WORLD_SIZE"] == "1"
    assert os.environ["RANK"] == "0"
    assert observed["kwargs"]["model_dir"] == str(model_root)
