#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sure_eval.evaluation.scripts import run_task


DEFAULT_TASKS = ("ASR", "S2TT", "SER", "SLU", "GR")


def read_key_text(path: Path) -> tuple[list[str], list[str]]:
    keys: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"{path} contains a non key-tab-text row: {line!r}")
            key, text = line.split("\t", 1)
            keys.append(key)
            texts.append(text)
    return keys, texts


def _pipeline_trace_to_dict(report: Any) -> list[dict[str, Any]]:
    return [
        {
            "stage": node.stage,
            "node_id": node.node_id,
            "version": node.version,
            "internal_stages": list(node.internal_stages),
        }
        for node in getattr(report, "pipeline_trace", ())
    ]


def _route_config_path(route_task: str) -> str:
    normalized = route_task.lower()
    if normalized in {"ser", "gr"}:
        normalized = "classification"
    return f"src/sure_eval/evaluation/tasks/{normalized}/routes.yaml"


def _route_output_dir(artifacts_dir: Path, task: str) -> Path:
    return artifacts_dir / ".sure_eval_metrics" / task.lower()


def _metric_payload_from_report(report: Any, *, route_task: str) -> dict[str, Any]:
    details = dict(getattr(report, "details", {}) or {})
    details.setdefault("route_config_path", _route_config_path(route_task))
    details.setdefault("pipeline_id", report.pipeline_id)
    details.setdefault("pipeline_kind", getattr(report, "pipeline_kind", "atomic"))
    details.setdefault("member_pipeline_ids", list(getattr(report, "member_pipeline_ids", ())))
    details.setdefault("computation_node_ids", list(getattr(report, "computation_node_ids", ())))
    details.setdefault("pipeline_trace", _pipeline_trace_to_dict(report))
    return {
        "metric_name": report.metric,
        "score": report.score,
        "pipeline_id": report.pipeline_id,
        "pipeline_kind": getattr(report, "pipeline_kind", "atomic"),
        "member_pipeline_ids": list(getattr(report, "member_pipeline_ids", ())),
        "computation_node_ids": list(getattr(report, "computation_node_ids", ())),
        "details": details,
        "backend": "sure_eval.evaluation.scripts.run_task",
    }


def evaluate_asr(ref_path: Path, hyp_path: Path, language: str, output_dir: Path) -> dict[str, Any]:
    ref_keys, references = read_key_text(ref_path)
    hyp_keys, predictions = read_key_text(hyp_path)
    ensure_aligned(ref_path, hyp_path, ref_keys, hyp_keys)
    metric = "cer" if language == "zh" else "wer"
    report = run_task(
        "asr",
        ref_file=str(ref_path),
        hyp_file=str(hyp_path),
        language=language,
        metric=metric,
        output_dir=str(output_dir),
    )
    return _metric_payload_from_report(report, route_task="asr")


def evaluate_s2tt(ref_path: Path, hyp_path: Path, language: str, output_dir: Path) -> dict[str, Any]:
    ref_keys, references = read_key_text(ref_path)
    hyp_keys, predictions = read_key_text(hyp_path)
    ensure_aligned(ref_path, hyp_path, ref_keys, hyp_keys)
    report = run_task(
        "s2tt",
        ref_file=str(ref_path),
        hyp_file=str(hyp_path),
        language=language,
        metric="bleu",
        output_dir=str(output_dir),
    )
    return _metric_payload_from_report(report, route_task="s2tt")


def evaluate_classification(
    ref_path: Path,
    hyp_path: Path,
    task: str,
    output_dir: Path,
    prompt_path: Path | None = None,
) -> dict[str, Any]:
    ref_keys, references = read_key_text(ref_path)
    hyp_keys, predictions = read_key_text(hyp_path)
    ensure_aligned(ref_path, hyp_path, ref_keys, hyp_keys)
    if task == "SLU":
        if prompt_path is None or not prompt_path.exists():
            raise ValueError(f"SLU requires prompt_jsonl: {prompt_path}")
        report = run_task(
            "slu",
            ref_file=str(ref_path),
            hyp_file=str(hyp_path),
            prompt_jsonl=str(prompt_path),
            output_dir=str(output_dir),
        )
        route_task = "slu"
    else:
        report = run_task(
            task.lower(),
            ref_file=str(ref_path),
            hyp_file=str(hyp_path),
            output_dir=str(output_dir),
        )
        route_task = task.lower()
    return _metric_payload_from_report(report, route_task=route_task)


def ensure_aligned(ref_path: Path, hyp_path: Path, ref_keys: list[str], hyp_keys: list[str]) -> None:
    if ref_keys != hyp_keys:
        raise ValueError(
            f"ref/hyp key mismatch for {ref_path.name} and {hyp_path.name}: "
            f"ref_keys={ref_keys}, hyp_keys={hyp_keys}"
        )


def evaluate_task(task: str, artifacts_dir: Path, asr_language: str, s2tt_language: str) -> dict[str, Any]:
    task_lower = task.lower()
    ref_path = artifacts_dir / f"ref_{task_lower}.txt"
    hyp_path = artifacts_dir / f"hyp_{task_lower}.txt"
    prompt_path = artifacts_dir / f"prompt_{task_lower}.jsonl"
    if not ref_path.exists() or not hyp_path.exists():
        missing = [str(path) for path in (ref_path, hyp_path) if not path.exists()]
        return {
            "status": "missing",
            "task": task,
            "missing": missing,
        }

    output_dir = _route_output_dir(artifacts_dir, task_lower)
    if task == "ASR":
        metric = evaluate_asr(ref_path, hyp_path, asr_language, output_dir)
    elif task == "S2TT":
        metric = evaluate_s2tt(ref_path, hyp_path, s2tt_language, output_dir)
    elif task in {"SER", "SLU", "GR"}:
        metric = evaluate_classification(
            ref_path,
            hyp_path,
            task=task,
            output_dir=output_dir,
            prompt_path=prompt_path if task == "SLU" else None,
        )
    else:
        raise ValueError(f"Unsupported speech understanding task: {task}")

    ref_keys, _ = read_key_text(ref_path)
    return {
        "status": "complete",
        "task": task,
        "num_samples": len(ref_keys),
        "ref_file": str(ref_path),
            "hyp_file": str(hyp_path),
            "route_output_dir": str(output_dir),
            "metric": metric,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SURE speech-understanding metrics from existing ref/hyp files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory containing ref_<task>.txt and hyp_<task>.txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the combined metric report JSON.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated tasks to evaluate. Defaults to ASR,S2TT,SER,SLU,GR.",
    )
    parser.add_argument("--asr-language", default="zh")
    parser.add_argument("--s2tt-language", default="zh")
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = [task.strip().upper() for task in args.tasks.split(",") if task.strip()]
    errors: list[dict[str, Any]] = []
    task_results: dict[str, Any] = {}

    for task in tasks:
        try:
            task_results[task] = evaluate_task(task, args.artifacts_dir, args.asr_language, args.s2tt_language)
        except Exception as exc:  # noqa: BLE001 - report all task failures in one artifact.
            errors.append({"task": task, "error": str(exc)})
            task_results[task] = {"status": "error", "task": task, "error": str(exc)}

    complete = [task for task, result in task_results.items() if result.get("status") == "complete"]
    report = {
        "ok": not errors and len(complete) == len(tasks),
        "model": args.model,
        "artifacts_dir": str(args.artifacts_dir),
        "tasks": tasks,
        "complete_tasks": complete,
        "metric_namespaces": {
            "ASR": "sure_eval.evaluation.scripts.run_task:asr",
            "S2TT": "sure_eval.evaluation.scripts.run_task:s2tt",
            "SER": "sure_eval.evaluation.scripts.run_task:ser",
            "SLU": "sure_eval.evaluation.scripts.run_task:slu",
            "GR": "sure_eval.evaluation.scripts.run_task:gr",
        },
        "task_results": task_results,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
