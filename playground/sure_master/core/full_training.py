"""Retrain selected ASR recipes on full data; never fine-tune subset weights."""

from __future__ import annotations
import json
import hashlib
from functools import partial
from pathlib import Path

from .artifacts import load_bundle
from .utils.slurm import atomic_json


def training_plan(artifact: str) -> dict:
    current = Path(artifact).resolve()
    seen = set()
    inference = None
    while current not in seen:
        seen.add(current)
        load_bundle(current)
        changes = current.parent / "artifacts/candidate_changes.json"
        if not changes.exists():
            raise ValueError(
                "Full training requires wrapper-recorded training provenance"
            )
        data = json.loads(changes.read_text())
        if inference is None:
            inference = data["inference_config"]
        if data.get("training_config", {}).get("actual_train_epoch"):
            return {
                "recipe": str(current.parent / "recipe"),
                "train_args": data["diff_from_defaults"]["train_extra_args"],
                "inference": inference,
                "training_artifact": str(current),
                "decode_args": json.loads(
                    (
                        Path(artifact).parent / "artifacts/candidate_changes.json"
                    ).read_text()
                )["diff_from_defaults"]["decode_extra_args"],
            }
        parent = data.get("parent_model_artifact")
        if not parent:
            raise ValueError("Inference candidate has no traceable training parent")
        current = Path(parent).resolve()
    raise ValueError("Cycle in ASR training lineage")


def retrain_selected(
    playground, baseline: dict, candidates: list[dict]
) -> tuple[dict, list[dict]]:
    config = playground.sure_config.get("full_training") or {}
    if not config.get("enabled"):
        return baseline, candidates
    eligible = [
        r
        for r in candidates
        if r.get("model_artifact") and playground._is_valid_score(r.get("score"))
    ]
    eligible.sort(
        key=lambda r: r["score"], reverse=not playground.task_card.is_lower_better
    )
    selected = [baseline, *eligible[:2]]
    plans = {r["idea_id"]: training_plan(r["model_artifact"]) for r in selected}
    unique, aliases, owners = [], [], {}
    for record in selected:
        plan = plans[record["idea_id"]]
        signature = hashlib.sha256(json.dumps(plan["train_args"]).encode())
        for source in sorted(Path(plan["recipe"]).glob("*.py")):
            signature.update(source.name.encode())
            signature.update(source.read_bytes())
        key = signature.hexdigest()
        if key in owners:
            aliases.append((record, owners[key]))
        else:
            owners[key] = record["idea_id"]
            unique.append(record)
    jobs, experiments = [], []
    for record in unique:
        plan = plans[record["idea_id"]]
        exp = playground._create_run_exp("improve", playground.exp_index)
        playground.exp_index += 1
        exp.candidate_phase = "full_training"
        exp.candidate_idea_id = record["idea_id"]
        exp.execution_env.update(
            SURE_MAX_TRAIN_EPOCHS=str(config.get("epochs", 30)),
            SURE_BASELINE_EPOCH=str(config.get("epochs", 30)),
            SURE_BASELINE_USE_PRETRAINED="0",
            SURE_DECODE_ONLY="0",
        )
        for name in (
            "SURE_PARENT_MODEL_ARTIFACT",
            "SURE_ASR_INITIAL_MODEL_ARTIFACT",
            "SURE_FROZEN_MODEL_ARTIFACT",
        ):
            exp.execution_env.pop(name, None)
        kind = (
            "arch"
            if any(
                x in plan["train_args"]
                for x in (
                    "--encoder-dim",
                    "--num-encoder-layers",
                    "--feedforward-dim",
                    "--encoder-unmasked-dim",
                )
            )
            else "fine_tune"
        )
        infer = plan["inference"]
        argv = [
            "--action",
            "train_decode",
            "--candidate-type",
            kind,
            "--train-args-json",
            json.dumps(plan["train_args"]),
            "--decode-args-json",
            json.dumps(plan["decode_args"]),
            "--decode-method",
            infer["decoding_method"],
            "--decode-avg",
            str(infer["decode_avg"]),
            "--use-averaged-model",
            str(infer["use_averaged_model"]),
        ]
        code = (
            "import os, subprocess\nfrom pathlib import Path\n"
            "def main():\n"
            "    assert Path('base_model/recipe/train.py').is_file()\n"
            "    subprocess.run([os.environ['SURE_ICEFALL_PYTHON'], os.environ['SURE_ASR_ZIPFORMER_WRAPPER'], "
            + ", ".join(repr(v) for v in argv)
            + "], check=True)\n"
            "    assert Path('artifacts/hyp.txt').is_file()\n"
            "if __name__ == '__main__':\n    main()\n"
        )
        jobs.append(
            partial(
                exp.run_existing_code,
                code=code,
                role_paths=playground._role_paths(),
                candidate_type_hint=kind,
                base_model_source_overrides={
                    "data": config["data"],
                    "recipe": plan["recipe"],
                },
            )
        )
        experiments.append(exp)
    results = playground.execute_parallel_tasks(
        jobs,
        max_workers=min(4, len(jobs)),
        workspace_names=[e.exp_name for e in experiments],
    )
    records = []
    for original, result in zip(unique, results):
        if isinstance(result, Exception):
            raise result
        success, score, _, code, details = result
        if not success or not details.get("produced_artifacts", {}).get(
            "model_artifact"
        ):
            raise RuntimeError(
                f"Full training failed for {original['idea_id']}; cannot substitute the subset checkpoint"
            )
        records.append(
            {
                **original,
                "search_score": original["score"],
                "score": score,
                "code": code,
                "model_artifact": details["produced_artifacts"]["model_artifact"],
            }
        )
    by_id = {record["idea_id"]: record for record in records}
    replay_jobs, replay_exps = [], []
    for original, owner in aliases:
        plan = plans[original["idea_id"]]
        exp = playground._create_run_exp("improve", playground.exp_index)
        playground.exp_index += 1
        exp.candidate_phase = "full_training_inference"
        exp.candidate_idea_id = original["idea_id"]
        exp.execution_env["SURE_PARENT_MODEL_ARTIFACT"] = by_id[owner]["model_artifact"]
        inference = plan["inference"]
        parameters = {
            "decode_args_json": plan["decode_args"],
            "decode_method": inference["decoding_method"],
            "decode_avg": inference["decode_avg"],
            "use_averaged_model": inference["use_averaged_model"],
        }
        code = (
            "import os, subprocess\nfrom pathlib import Path\n"
            "def main():\n"
            "    subprocess.run([os.environ['SURE_WORKER_PYTHON'], os.environ['SURE_TASK_WRAPPER'], "
            "'--action', 'infer', '--parameters-json', "
            + repr(json.dumps(parameters))
            + "], check=True)\n"
            "    assert Path('artifacts/hyp.txt').is_file()\n"
            "if __name__ == '__main__':\n    main()\n"
        )
        replay_jobs.append(
            partial(
                exp.run_existing_code,
                code=code,
                role_paths=playground._role_paths(),
                candidate_type_hint="inference",
                base_model_source_overrides={"data": config["data"]},
            )
        )
        replay_exps.append(exp)
    if replay_jobs:
        replay_results = playground.execute_parallel_tasks(
            replay_jobs,
            max_workers=len(replay_jobs),
            workspace_names=[e.exp_name for e in replay_exps],
        )
        for (original, owner), result in zip(aliases, replay_results):
            if isinstance(result, Exception):
                raise result
            success, score, _, code, details = result
            artifact = details.get("produced_artifacts", {}).get("model_artifact")
            if not success or not artifact:
                raise RuntimeError(
                    f"Full-data inference failed for {original['idea_id']}"
                )
            by_id[original["idea_id"]] = {
                **original,
                "search_score": original["score"],
                "score": score,
                "code": code,
                "model_artifact": artifact,
                "shared_training_with": owner,
            }
    records = [by_id[record["idea_id"]] for record in selected]
    atomic_json(
        Path(playground.session.config.workspace_path) / "metric/full_training.json",
        records,
    )
    return records[0], records[1:]
