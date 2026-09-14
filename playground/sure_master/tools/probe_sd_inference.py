"""Replay a component checkpoint on one complete search session and score it."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

from playground.sure_master.core.artifacts import file_digest
from playground.sure_master.core.utils.metric import SureMetricRunner
from playground.sure_master.core.utils.task_cards import resolve_task_card
from playground.sure_master.runtime.accelerator import RuntimeBackend
from playground.sure_master.runtime.model_source import snapshot_source, prepare_audio_io, prepare_diarizen_inference_source
from playground.sure_master.runtime.training_state import atomic_json
from playground.sure_master.tasks.diarization import clip_rttm, validate_rttm_outputs, write_session_rttm
from playground.sure_master.tools.probe_task import expand


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-workspace", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    sure = expand(yaml.safe_load(args.config.read_text())["sure"])
    training = args.training_workspace.resolve()
    if json.loads((training / "acceptance.json").read_text())["status"] != "passed":
        raise ValueError("Four-rank training and resume probe must pass first")
    work = args.workspace.resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    runtime = RuntimeBackend("npu")
    torch = runtime.torch
    state_root = training / "state"
    checkpoint = state_root / json.loads((state_root / "latest.json").read_text())["directory"]
    metadata = json.loads((checkpoint / "state.json").read_text())
    if file_digest(checkpoint / "training.pt") != metadata["files"]["training.pt"]:
        raise ValueError("Corrupt probe checkpoint")
    source = snapshot_source(Path(sure["task"]["resources"]["source"]), work / "source")
    prepare_diarizen_inference_source(source)
    pipeline_file = source / "diarizen/pipelines/inference.py"
    text = pipeline_file.read_text().replace(
        'torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")',
        'torch.device(os.environ["SURE_MODEL_DEVICE"])')
    pipeline_file.write_text(text)
    prepare_audio_io(pipeline_file)
    os.environ["SURE_MODEL_DEVICE"] = str(runtime.device)
    sys.path[:0] = [str(source), str(source / "pyannote-audio")]
    import toml
    from diarizen.pipelines.inference import DiariZenPipeline
    from pyannote.audio.core.task import Specifications, Problem, Resolution

    job = json.loads((training / "job.json").read_text())
    resources = sure["task"]["resources"]
    model_dir = work / "model"
    model_dir.mkdir(exist_ok=True)
    state = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=False)
    torch.save(state["model"], model_dir / "pytorch_model.bin")
    del state
    config = {
        "model": {"path": "diarizen.models.eend.model_wavlm_conformer.Model",
                  "args": {**job["architecture"], "wavlm_src": resources["wavlm"]}},
        "inference": {"args": {"seg_duration": 8, "segmentation_step": .1,
                               "batch_size": 32, "apply_median_filtering": True}},
        "clustering": {"args": {"method": "AgglomerativeClustering", "min_speakers": 1,
                                "max_speakers": 20, "min_cluster_size": 30, "ahc_threshold": .7}},
    }
    (model_dir / "config.toml").write_text(toml.dumps(config))
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion, Specifications, Problem, Resolution]):
        pipeline = DiariZenPipeline(model_dir, resources["embedding"])
    split = sure["datasets"]["search"]
    row = min((json.loads(line) for line in Path(split["manifest"]).read_text().splitlines()),
              key=lambda r: r["duration"])
    manifest = work / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    start = time.monotonic()
    annotation = pipeline(row["audio"], sess_name=row["session_id"])
    elapsed = time.monotonic() - start
    with (work / "hyp.rttm").open("w") as stream:
        write_session_rttm(annotation, row, stream)
    (work / "processed_sessions.json").write_text(json.dumps([row["session_id"]]))
    validate_rttm_outputs(work / "hyp.rttm", manifest)
    clip_rttm(work / "hyp.rttm", manifest, work / "scoring_hyp.rttm")
    clip_rttm(Path(split["roles"]["ref"]), manifest, work / "scoring_ref.rttm")
    score = SureMetricRunner(sure["root"], sure["pythonpath"], device="cpu",
                            python=sure["metric_runtime"]["python"]).run(
        resolve_task_card(sure["task_cards_path"], "sd_der"), work, work / "metric",
        {"ref": str(work / "scoring_ref.rttm"), "hyp": str(work / "scoring_hyp.rttm")})
    if not score.success:
        raise RuntimeError(f"SURE probe scoring failed: {score.error}")
    atomic_json(work / "acceptance.json", {"status": "passed", "component_test": True,
                "session_id": row["session_id"], "duration": row["duration"],
                "inference_seconds": elapsed, "score": score.score, "metric": score.metric})


if __name__ == "__main__":
    main()
