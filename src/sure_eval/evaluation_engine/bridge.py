"""Execute the pinned standalone SURE evaluation engine through its CLI contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import yaml

from sure_eval.results import file_sha256
from sure_eval.results.schema import validate_dataset_identity


ENGINE_LOCK_SCHEMA = "sure.eval.evaluation_engine_lock.v1"
EVALUATION_INPUT_SCHEMA = "sure.eval.evaluation_input_manifest.v1"
BRIDGE_RESULT_SCHEMA = "sure.eval.evaluation_bridge_result.v1"
SUPPORTED_TASKS = {
    "asr",
    "s2tt",
    "kws",
    "vad",
    "classification",
    "ser",
    "gr",
    "slu",
    "sd",
    "sa_asr",
    "se",
    "speech_enhancement",
    "tts",
    "vc",
    "tse",
}
ROLE_FLAGS = {
    "ref_file": "--ref-file",
    "hyp_file": "--hyp-file",
    "src_file": "--src-file",
    "prompt_jsonl": "--prompt-jsonl",
    "label_spec": "--label-spec",
    "reference_jsonl": "--reference-jsonl",
    "sample_output": "--sample-output",
    "wekws_label_file": "--wekws-label-file",
    "wekws_score_file": "--wekws-score-file",
    "wekws_frame_score_file": "--wekws-frame-score-file",
    "samples_jsonl": "--samples-jsonl",
}
ROLE_CONTRACT_NAMES = {
    "ref_file": "ref",
    "hyp_file": "hyp",
    "src_file": "src",
    "prompt_jsonl": "prompt_jsonl",
    "label_spec": "label_spec",
    "reference_jsonl": "reference_jsonl",
    "sample_output": "sample_output",
    "wekws_label_file": "wekws_label_file",
    "wekws_score_file": "wekws_score_file",
    "wekws_frame_score_file": "wekws_frame_score_file",
    "samples_jsonl": "samples_jsonl",
}


class EvaluationEngineError(RuntimeError):
    """Raised when the pinned engine or one of its deterministic stages fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationEngineError(f"expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class EvaluationInput:
    task: str
    language: str | None
    dataset: dict[str, Any]
    roles: dict[str, Path]
    role_sha256: dict[str, str]
    metric: str | None = None
    metrics: tuple[str, ...] = ()
    pipeline_id: str | None = None
    keyword: str | None = None
    macro_recall_false_alarms: int = 0
    device: str = "cuda"
    cache_dir: str | None = None


def load_evaluation_input(path: str | Path) -> EvaluationInput:
    manifest_path = Path(path).resolve(strict=True)
    payload = _read_json(manifest_path)
    unknown_root = set(payload) - {
        "schema",
        "task",
        "language",
        "dataset",
        "selection",
        "roles",
        "options",
    }
    if unknown_root:
        raise EvaluationEngineError(
            f"unsupported evaluation input fields: {', '.join(sorted(unknown_root))}"
        )
    if payload.get("schema") != EVALUATION_INPUT_SCHEMA:
        raise EvaluationEngineError("evaluation input manifest schema is not v1")
    task = str(payload.get("task", "")).strip().lower().replace("-", "_")
    if task not in SUPPORTED_TASKS:
        raise EvaluationEngineError(f"unsupported evaluation task: {task!r}")
    dataset = validate_dataset_identity(payload.get("dataset"), "evaluation_input.dataset")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise EvaluationEngineError("evaluation_input.selection must be a mapping")
    unknown_selection = set(selection) - {"metric", "metrics", "pipeline_id"}
    if unknown_selection:
        raise EvaluationEngineError(
            f"unsupported evaluation selection fields: {', '.join(sorted(unknown_selection))}"
        )
    metric = selection.get("metric")
    raw_metrics = selection.get("metrics") or ()
    if not isinstance(raw_metrics, (list, tuple)):
        raise EvaluationEngineError("evaluation selection.metrics must be a list")
    metrics = tuple(str(item) for item in raw_metrics)
    pipeline_id = selection.get("pipeline_id")
    selected_count = int(bool(metric)) + int(bool(metrics)) + int(bool(pipeline_id))
    if selected_count != 1:
        raise EvaluationEngineError(
            "evaluation selection requires exactly one of metric, metrics, or pipeline_id"
        )

    raw_roles = payload.get("roles")
    if not isinstance(raw_roles, dict) or not raw_roles:
        raise EvaluationEngineError("evaluation_input.roles must be a non-empty mapping")
    roles: dict[str, Path] = {}
    role_sha256: dict[str, str] = {}
    for role, raw in raw_roles.items():
        if role not in ROLE_FLAGS:
            raise EvaluationEngineError(f"unsupported evaluation input role: {role}")
        if not isinstance(raw, dict):
            raise EvaluationEngineError(f"roles.{role} must be a mapping")
        if set(raw) != {"path", "sha256"}:
            raise EvaluationEngineError(
                f"roles.{role} requires exactly path and sha256 fields"
            )
        role_path = Path(str(raw.get("path", ""))).expanduser().resolve(strict=True)
        expected = str(raw.get("sha256", ""))
        actual = file_sha256(role_path)
        if actual != expected:
            raise EvaluationEngineError(
                f"roles.{role} checksum mismatch: expected {expected}, got {actual}"
            )
        roles[role] = role_path
        role_sha256[role] = actual

    options = payload.get("options") or {}
    if not isinstance(options, dict):
        raise EvaluationEngineError("evaluation_input.options must be a mapping")
    unknown_options = set(options) - {
        "keyword",
        "macro_recall_false_alarms",
        "device",
        "cache_dir",
    }
    if unknown_options:
        raise EvaluationEngineError(
            f"unsupported evaluation options: {', '.join(sorted(unknown_options))}"
        )
    return EvaluationInput(
        task=task,
        language=str(payload["language"]) if payload.get("language") else None,
        dataset=dataset,
        roles=roles,
        role_sha256=role_sha256,
        metric=str(metric) if metric else None,
        metrics=metrics,
        pipeline_id=str(pipeline_id) if pipeline_id else None,
        keyword=str(options["keyword"]) if options.get("keyword") else None,
        macro_recall_false_alarms=int(options.get("macro_recall_false_alarms", 0)),
        device=str(options.get("device", "cuda")),
        cache_dir=str(options["cache_dir"]) if options.get("cache_dir") else None,
    )


class EvaluationEngine:
    def __init__(self, lock_path: str | Path | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.lock_path = Path(lock_path) if lock_path else repo_root / "config/evaluation_engine.lock.yaml"
        payload = yaml.safe_load(self.lock_path.read_text(encoding="utf-8")) or {}
        if payload.get("schema") != ENGINE_LOCK_SCHEMA:
            raise EvaluationEngineError("unsupported evaluation engine lock schema")
        self.lock = payload
        self.source_checkout = Path(str(payload["source_checkout"])).resolve(strict=True)
        self.python_executable = Path(str(payload["python_executable"])).resolve(strict=True)
        self.source_root = self.source_checkout / "src"
        self.entrypoint = str(payload["entrypoint"])

    @property
    def identity(self) -> dict[str, str]:
        return {
            "repository": str(self.lock["repository"]),
            "commit": str(self.lock["commit"]),
            "tree_sha256": str(self.lock["tree_sha256"]),
        }

    def verify_pin(self) -> dict[str, Any]:
        actual_commit = self._git("rev-parse", "HEAD").strip()
        actual_remote = self._git("remote", "get-url", "origin").strip()
        paths = [str(item) for item in self.lock.get("tree_paths") or ()]
        dirty = self._git("status", "--porcelain", "--", *paths).strip()
        archive = subprocess.run(
            ["git", "archive", "--format=tar", "HEAD", *paths],
            cwd=self.source_checkout,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        actual_tree_sha256 = hashlib.sha256(archive).hexdigest()
        checks = {
            "repository": actual_remote == self.lock["repository"],
            "commit": actual_commit == self.lock["commit"],
            "tree_clean": dirty == "",
            "tree_sha256": actual_tree_sha256 == self.lock["tree_sha256"],
            "python_executable": self.python_executable.is_file(),
        }
        if not all(checks.values()):
            raise EvaluationEngineError(
                "evaluation engine pin verification failed: "
                + json.dumps(
                    {
                        "checks": checks,
                        "actual_commit": actual_commit,
                        "actual_remote": actual_remote,
                        "dirty": dirty,
                        "actual_tree_sha256": actual_tree_sha256,
                    },
                    sort_keys=True,
                )
            )
        return {
            "schema": "sure.eval.evaluation_engine_verification.v1",
            "status": "verified",
            "identity": self.identity,
            "checks": checks,
            "source_checkout": str(self.source_checkout),
            "python_executable": str(self.python_executable),
            "verified_at": _utc_now(),
        }

    def evaluate(
        self,
        evaluation_input: EvaluationInput,
        *,
        output_dir: str | Path,
        trace_dir: str | Path,
    ) -> dict[str, Any]:
        output = Path(output_dir).resolve(strict=False)
        traces = Path(trace_dir).resolve(strict=False)
        if output.exists():
            raise EvaluationEngineError(f"evaluation output directory already exists: {output}")
        output.mkdir(parents=True)
        traces.mkdir(parents=True, exist_ok=True)
        verification = self.verify_pin()
        (traces / "00_engine_verification.json").write_text(
            json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        plan_path = output / "agent_plan.json"
        pipeline_path = output / "pipeline.json"
        engine_output = output / "engine_output"
        selection_args = self._selection_args(evaluation_input)
        language_args = ["--language", evaluation_input.language] if evaluation_input.language else []
        self._run_stage(
            "agent_plan",
            [
                "agent",
                "plan",
                evaluation_input.task,
                *language_args,
                *selection_args,
                "--output",
                str(plan_path),
                "--json",
            ],
            traces / "01_agent_plan.json",
        )
        plan = _read_json(plan_path)
        if plan.get("can_run_now") is not True:
            raise EvaluationEngineError(
                f"selected evaluation pipeline is not ready: {plan.get('blocking_issues')}"
            )

        self._run_stage(
            "metric_describe",
            [
                "metric",
                "describe",
                evaluation_input.task,
                *language_args,
                *selection_args,
                "--output",
                str(pipeline_path),
                "--json",
            ],
            traces / "02_metric_describe.json",
        )
        pipeline = _read_json(pipeline_path)
        required_roles = set(pipeline.get("required_roles") or ())
        provided_roles = {ROLE_CONTRACT_NAMES[role] for role in evaluation_input.roles}
        missing_roles = required_roles - provided_roles
        if missing_roles:
            raise EvaluationEngineError(
                f"evaluation input is missing pipeline roles: {', '.join(sorted(missing_roles))}"
            )

        run_args = [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--output-dir",
            str(engine_output),
            "--validate-env",
            "--json",
        ]
        for role, role_path in evaluation_input.roles.items():
            run_args.extend((ROLE_FLAGS[role], str(role_path)))
        if evaluation_input.keyword:
            run_args.extend(("--keyword", evaluation_input.keyword))
        if evaluation_input.macro_recall_false_alarms:
            run_args.extend(
                ("--macro-recall-false-alarms", str(evaluation_input.macro_recall_false_alarms))
            )
        run_args.extend(("--device", evaluation_input.device))
        if evaluation_input.cache_dir:
            run_args.extend(("--cache-dir", evaluation_input.cache_dir))
        stage = self._run_stage(
            "metric_run",
            run_args,
            traces / "03_metric_run.json",
        )

        report_path = engine_output / "report.json"
        description_path = engine_output / "pipeline_description.json"
        report = _read_json(report_path)
        description = _read_json(description_path)
        if report.get("pipeline_id") != pipeline.get("pipeline_id"):
            raise EvaluationEngineError("engine report pipeline_id differs from described pipeline")
        if description.get("pipeline_id") != pipeline.get("pipeline_id"):
            raise EvaluationEngineError("engine pipeline description differs from selected pipeline")
        result = {
            "schema": BRIDGE_RESULT_SCHEMA,
            "status": "success",
            "engine": self.identity,
            "task": evaluation_input.task,
            "language": evaluation_input.language,
            "dataset": evaluation_input.dataset,
            "pipeline_id": pipeline["pipeline_id"],
            "metric": report.get("metric"),
            "score": report.get("score"),
            "inputs": {
                role: {"path": str(path), "sha256": evaluation_input.role_sha256[role]}
                for role, path in evaluation_input.roles.items()
            },
            "agent_plan_path": str(plan_path),
            "pipeline_path": str(pipeline_path),
            "engine_report_path": str(report_path),
            "pipeline_description_path": str(description_path),
            "trace_paths": [
                str(traces / "00_engine_verification.json"),
                str(traces / "01_agent_plan.json"),
                str(traces / "02_metric_describe.json"),
                str(traces / "03_metric_run.json"),
            ],
            "metric_run_summary": stage["stdout_json"],
            "completed_at": _utc_now(),
        }
        result_path = output / "evaluation_bridge_result.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    def _selection_args(self, value: EvaluationInput) -> list[str]:
        if value.pipeline_id:
            return ["--pipeline-id", value.pipeline_id]
        if value.metrics:
            return ["--metrics", ",".join(value.metrics)]
        return ["--metric", str(value.metric)]

    def _run_stage(self, stage: str, args: list[str], trace_path: Path) -> dict[str, Any]:
        command = [
            str(self.python_executable),
            "-c",
            self.entrypoint,
            *args,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.source_root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        started_at = _utc_now()
        start = perf_counter()
        completed = subprocess.run(
            command,
            cwd=self.source_checkout,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        duration = perf_counter() - start
        try:
            stdout_json = json.loads(completed.stdout) if completed.stdout.strip() else None
        except json.JSONDecodeError:
            stdout_json = None
        trace = {
            "schema": "sure.eval.evaluation_stage_trace.v1",
            "stage": stage,
            "command": command,
            "cwd": str(self.source_checkout),
            "engine": self.identity,
            "started_at": started_at,
            "duration_seconds": duration,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "stdout_json": stdout_json,
        }
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise EvaluationEngineError(
                f"evaluation stage {stage!r} failed with code {completed.returncode}: "
                f"{completed.stderr or completed.stdout}"
            )
        if stdout_json is None:
            raise EvaluationEngineError(f"evaluation stage {stage!r} did not emit JSON")
        return trace

    def _git(self, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=self.source_checkout,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise EvaluationEngineError(
                f"git {' '.join(args)} failed: {completed.stderr.strip()}"
            )
        return completed.stdout
