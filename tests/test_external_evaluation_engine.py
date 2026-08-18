from __future__ import annotations

import json
from pathlib import Path

from sure_eval.evaluation_engine import EvaluationEngine, load_evaluation_input
from sure_eval.evaluation_engine.bridge import SUPPORTED_TASKS
from sure_eval.evaluation_engine import EvaluationRunOrchestrator
from sure_eval.models.registry import model_artifact_digest
from sure_eval.results import file_sha256
from sure_eval.storage import StorageConfig


def test_pinned_external_engine_verifies_real_source_checkout() -> None:
    verification = EvaluationEngine().verify_pin()

    assert verification["status"] == "verified"
    assert all(verification["checks"].values())


def test_bridge_task_catalog_covers_every_pinned_engine_task() -> None:
    task_root = (
        EvaluationEngine().source_checkout / "src/sure_eval/evaluation/tasks"
    )
    source_tasks = {
        path.name
        for path in task_root.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }

    assert source_tasks == {
        "asr",
        "classification",
        "kws",
        "s2tt",
        "sa_asr",
        "sd",
        "se",
        "slu",
        "tse",
        "tts",
        "vad",
        "vc",
    }
    assert source_tasks <= SUPPORTED_TASKS
    assert {"ser", "gr", "speech_enhancement"} <= SUPPORTED_TASKS


def test_real_asr_evaluation_writes_three_stage_trace(tmp_path: Path) -> None:
    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    ref.write_text("utt-1\thello world\nutt-2\treproducible evaluation\n", encoding="utf-8")
    hyp.write_text("utt-1\thello world\nutt-2\treproducible evaluation\n", encoding="utf-8")
    manifest_path = tmp_path / "evaluation_input.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "sure.eval.evaluation_input_manifest.v1",
                "task": "asr",
                "language": "en",
                "dataset": {
                    "id": "trace_fixture__v1.0.0",
                    "name": "trace_fixture",
                    "version": "v1.0.0",
                    "split": "test",
                    "manifest_sha256": "a" * 64,
                    "sample_count": 2,
                },
                "selection": {"metric": "wer"},
                "roles": {
                    "ref_file": {"path": str(ref), "sha256": file_sha256(ref)},
                    "hyp_file": {"path": str(hyp), "sha256": file_sha256(hyp)},
                },
            }
        ),
        encoding="utf-8",
    )

    result = EvaluationEngine().evaluate(
        load_evaluation_input(manifest_path),
        output_dir=tmp_path / "evaluation",
        trace_dir=tmp_path / "trace",
    )

    assert result["status"] == "success"
    assert result["pipeline_id"].startswith("asr.en.wer.")
    assert result["score"] == 0.0
    for name in (
        "00_engine_verification.json",
        "01_agent_plan.json",
        "02_metric_describe.json",
        "03_metric_run.json",
    ):
        assert (tmp_path / "trace" / name).is_file()


def test_real_reevaluation_from_verified_registry_stages_merge_delta(tmp_path: Path) -> None:
    storage = StorageConfig(
        repo_root=tmp_path / "repo",
        models_read_root=tmp_path / "nfs/models",
        results_read_root=tmp_path / "nfs/results",
        models_write_root=tmp_path / "repo/src/sure_eval/models",
        results_write_root=tmp_path / "repo/results",
    )
    model_id = "Org__TraceModel"
    protocol = {
        "id": "standard_system",
        "version": "2.0",
        "effective_params_sha256": "b" * 64,
    }
    dataset = {
        "id": "trace_fixture__v1.0.0",
        "name": "trace_fixture",
        "version": "v1.0.0",
        "split": "test",
        "manifest_sha256": "c" * 64,
        "sample_count": 2,
    }
    model_root = storage.models_read_root / model_id
    model_root.mkdir(parents=True)
    (model_root / "config.yaml").write_text(
        f"name: {model_id}\ntask: ASR\n",
        encoding="utf-8",
    )
    artifact_files = [
        {
            "path": "config.yaml",
            "sha256": file_sha256(model_root / "config.yaml"),
            "size_bytes": (model_root / "config.yaml").stat().st_size,
        }
    ]
    model_sha = model_artifact_digest(artifact_files)
    artifact_manifest = model_root / "publication_artifacts.json"
    artifact_manifest.write_text(
        json.dumps(
            {
                "schema": "sure.eval.model_artifact_manifest.v1",
                "model_id": model_id,
                "files": artifact_files,
            }
        ),
        encoding="utf-8",
    )
    (model_root / "publication.json").write_text(
        json.dumps(
            {
                "schema": "sure.eval.model_publication.v1",
                "status": "verified",
                "model_id": model_id,
                "artifact_sha256": model_sha,
                "artifact_manifest_path": "publication_artifacts.json",
                "artifact_manifest_sha256": file_sha256(artifact_manifest),
                "verified_by": "integration-test",
                "verified_at": "2026-08-08T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    result_root = storage.results_read_root / model_id
    prediction = result_root / "inference_runs/inf-001/predictions/predictions.jsonl"
    prediction.parent.mkdir(parents=True)
    prediction.write_text('{"id":"utt-1","text":"hello world"}\n', encoding="utf-8")
    prediction_manifest = prediction.parent.parent / "prediction_manifest.json"
    prediction_manifest.write_text(
        json.dumps(
            {
                "schema": "sure.eval.prediction_manifest.v2",
                "model": {"id": model_id, "artifact_sha256": model_sha},
                "dataset": dataset,
                "protocol": protocol,
                "sample_ids_sha256": "d" * 64,
                "sample_count": 2,
                "files": [
                    {
                        "role": "structured_predictions",
                        "path": str(prediction.relative_to(result_root)),
                        "sha256": file_sha256(prediction),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    published_index = result_root / "result_index.json"
    published_index.write_text(
        json.dumps(
            {
                "schema": "sure.eval.result_index.v2",
                "publication": {
                    "status": "verified",
                    "verified_by": "integration-test",
                    "verified_at": "2026-08-08T00:00:00Z",
                },
                "model": {"id": model_id, "artifact_sha256": model_sha},
                "inference_runs": [
                    {
                        "id": "inf-001",
                        "created_at": "2026-08-08T00:00:00Z",
                        "protocol": protocol,
                        "datasets": [
                            {
                                "dataset": dataset,
                                "prediction": {
                                    "manifest_path": str(
                                        prediction_manifest.relative_to(result_root)
                                    ),
                                    "manifest_sha256": file_sha256(prediction_manifest),
                                },
                            }
                        ],
                    }
                ],
                "evaluation_runs": [],
            }
        ),
        encoding="utf-8",
    )
    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    ref.write_text("utt-1\thello world\nutt-2\treproducible evaluation\n", encoding="utf-8")
    hyp.write_text("utt-1\thello world\nutt-2\treproducible evaluation\n", encoding="utf-8")
    evaluation_input = tmp_path / "evaluation_input.json"
    evaluation_input.write_text(
        json.dumps(
            {
                "schema": "sure.eval.evaluation_input_manifest.v1",
                "task": "asr",
                "language": "en",
                "dataset": dataset,
                "selection": {"metric": "wer"},
                "roles": {
                    "ref_file": {"path": str(ref), "sha256": file_sha256(ref)},
                    "hyp_file": {"path": str(hyp), "sha256": file_sha256(hyp)},
                },
            }
        ),
        encoding="utf-8",
    )

    result = EvaluationRunOrchestrator(storage=storage).run(
        model_id=model_id,
        evaluation_id="eval-002",
        inference_id="inf-001",
        inference_source="published",
        input_manifests=[evaluation_input],
    )

    local_root = storage.results_write_root / model_id
    delta = json.loads((local_root / "result_delta.json").read_text())
    report = json.loads(next(local_root.glob("evaluation_runs/eval-002/*/report.json")).read_text())
    report_rows = [
        json.loads(line)
        for line in (local_root / "report.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    publication = json.loads((local_root / "publication_manifest.json").read_text())
    assert result["status"] == "staged_for_human_review"
    assert result["automatic_publish_allowed"] is False
    assert delta["base_published_index"]["sha256"] == file_sha256(published_index)
    assert delta["additions"]["evaluation_runs"][0]["inference_id"] == "inf-001"
    assert report["protocol"] == protocol
    assert report["metric"] == {"name": "wer", "score": 0.0}
    assert len(report_rows) == 1
    assert report_rows[0]["schema"] == "sure.eval.report_row.v2"
    assert report_rows[0]["evaluation"]["id"] == "eval-002"
    assert report_rows[0]["dataset"] == dataset
    assert report_rows[0]["protocol"] == protocol
    assert publication["schema"] == "sure.eval.publication_manifest.v2"
    assert publication["derived_reports"]["report_jsonl"]["row_count"] == 1
    assert (local_root / "report_snapshot.md").is_file()
