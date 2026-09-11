"""Run the real ASR baseline and metric independently of LLM availability."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from playground.sure_master.core.utils.metric import SureMetricRunner
from playground.sure_master.core.utils.model_artifact import retain_model_artifact
from playground.sure_master.core.utils.task_cards import resolve_task_card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icefall", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sure", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "npu"), required=True)
    parser.add_argument("--model-artifact", type=Path, help="Decode and score a frozen artifact without training")
    args = parser.parse_args()
    from playground.sure_master.runtime.accelerator import check_accelerator
    check_accelerator(args.backend)
    project = Path(__file__).resolve().parents[3]
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "base_model").mkdir()
    for name, target in {"root": args.icefall, "recipe": args.icefall / "egs/tedlium3/ASR/zipformer", "data": args.data}.items():
        (workspace / "base_model" / name).symlink_to(target.resolve(), target_is_directory=True)
    (workspace / "input").mkdir()
    shutil.copy2(args.data / "refs/asr_tedlium3_smoke_ref.txt", workspace / "input/ref.txt")
    shutil.copy2(project / "playground/sure_master/baselines/zipformer_large_cr_ctc_rnnt_baseline.py", workspace / "run_sure.py")
    if args.model_artifact:
        command = [sys.executable, str(project / "playground/sure_master/tools/run_icefall_zipformer_candidate.py"),
                   "--action", "decode_only", "--candidate-type", "inference",
                   "--model-artifact", str(args.model_artifact.resolve())]
        (workspace / "run_sure.py").write_text("import subprocess\nif __name__ == '__main__':\n    subprocess.run(" + repr(command) + ", check=True)\n")
    env = {**os.environ, "PYTHONPATH": str(project), "SURE_ASR_DATASET": "tedlium3",
           "SURE_ASR_RECIPE_PROFILE": "tedlium3_zipformer", "SURE_ACCELERATOR": args.backend,
           "SURE_USE_FP16": "0", "SURE_MAX_TRAIN_EPOCHS": "1", "SURE_BASELINE_WORLD_SIZE": "1",
           "SURE_BASELINE_EPOCH": "1", "SURE_BASELINE_AVG": "1", "SURE_BASELINE_USE_AVERAGED_MODEL": "0",
           "SURE_BASELINE_USE_PRETRAINED": "0", "SURE_MAX_DURATION": "30", "SURE_TRAIN_DURATION_MIN": "10",
           "SURE_DURATION_AUTOTUNE_MIN": "10", "SURE_ASR_EVAL_SPLITS": "dev", "SURE_DECODE_METHOD": "greedy_search",
           "SURE_BASELINE_DECODE_MAX_DURATION": "30", "SURE_DECODE_DURATION_MIN": "30",
           "SURE_DECODE_DURATION_MAX": "30", "SURE_DECODE_DURATION_SAFETY_STEPS": "0", "SURE_ENABLE_MUSAN": "0",
           "SURE_ICEFALL_PYTHON": sys.executable}
    env.update(SURE_ASR_EVAL_REF_ONLY="1", SURE_CPU_THREADS="4")
    subprocess.run([sys.executable, "run_sure.py"], cwd=workspace, env=env, check=True)
    card = resolve_task_card(project / "playground/sure_master/task_cards/sure_tasks.yaml", "asr_en_wer")
    result = SureMetricRunner(args.sure, device="cpu").run(card, workspace, workspace / "metric", card.artifact_contract)
    if not result.success:
        raise RuntimeError(result.error)
    artifact = retain_model_artifact(workspace)
    report = {"status": "passed", "backend": args.backend, "score": result.score,
              "metric": result.metric, "model_artifact": artifact}
    (workspace / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
