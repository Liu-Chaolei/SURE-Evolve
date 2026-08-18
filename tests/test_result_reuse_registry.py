from __future__ import annotations

import json
from pathlib import Path

import pytest

from sure_eval.results import (
    AppendOnlyResultStore,
    DerivedReportError,
    ReuseRequest,
    ResultRegistry,
    file_sha256,
    validate_result_index,
)
from sure_eval.storage import PathPolicyError, StorageConfig


MODEL_ID = "Org__Model"
MODEL_SHA = "a" * 64
DATASET = {
    "id": "aishell1__v1.0.2",
    "name": "aishell1",
    "version": "v1.0.2",
    "split": "test",
    "manifest_sha256": "b" * 64,
    "sample_count": 2,
}
PROTOCOL = {
    "id": "standard_system",
    "version": "2.0",
    "effective_params_sha256": "c" * 64,
}
ENGINE = {
    "repository": "git@github.com:PigeonDan1/sure-evaluation.git",
    "commit": "87b6bc44e3b557ec69ce7db785594a2a549cc1bd",
    "tree_sha256": "e" * 64,
}
PIPELINE_ID = "asr.zh.cer.wetext_norm_zh_tn_v1.wenet_cer_v1"


def _storage(tmp_path: Path) -> StorageConfig:
    return StorageConfig(
        repo_root=tmp_path / "repo",
        models_read_root=tmp_path / "nfs" / "models",
        results_read_root=tmp_path / "nfs" / "results",
        models_write_root=tmp_path / "repo" / "src" / "sure_eval" / "models",
        results_write_root=tmp_path / "repo" / "results",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _published_fixture(tmp_path: Path) -> tuple[StorageConfig, Path]:
    storage = _storage(tmp_path)
    root = storage.results_read_root / MODEL_ID
    inference_root = root / "inference_runs" / "inf-001"
    prediction_file = inference_root / "predictions" / "aishell1.jsonl"
    prediction_file.parent.mkdir(parents=True)
    prediction_file.write_text('{"id":"a","text":"hello"}\n', encoding="utf-8")
    prediction_manifest = {
        "schema": "sure.eval.prediction_manifest.v2",
        "model": {"id": MODEL_ID, "artifact_sha256": MODEL_SHA},
        "dataset": DATASET,
        "protocol": PROTOCOL,
        "sample_ids_sha256": "d" * 64,
        "sample_count": 2,
        "files": [
            {
                "role": "structured_predictions",
                "path": str(prediction_file.relative_to(root)),
                "sha256": file_sha256(prediction_file),
            }
        ],
    }
    prediction_manifest_path = inference_root / "prediction_manifest.json"
    _write_json(prediction_manifest_path, prediction_manifest)

    pipeline_root = root / "evaluation_runs" / "eval-001" / "aishell1" / "cer"
    engine_report_path = pipeline_root / "engine_report.json"
    description_path = pipeline_root / "pipeline_description.json"
    report_path = pipeline_root / "report.json"
    evaluation_input_path = pipeline_root / "evaluation_input.json"
    _write_json(evaluation_input_path, {"fixture": True})
    _write_json(engine_report_path, {"pipeline_id": PIPELINE_ID, "score": 0.1})
    _write_json(description_path, {"pipeline_id": PIPELINE_ID, "nodes": []})
    _write_json(
        report_path,
        {
            "schema": "sure.eval.published_report.v2",
            "status": "success",
            "model": {"id": MODEL_ID, "artifact_sha256": MODEL_SHA},
            "dataset": DATASET,
            "protocol": PROTOCOL,
            "evaluation": {
                "pipeline_id": PIPELINE_ID,
                "engine_commit": ENGINE["commit"],
                "engine_tree_sha256": ENGINE["tree_sha256"],
            },
            "metric": {"name": "cer", "score": 0.1},
            "lineage": {
                "inference_id": "inf-001",
                "inference_source": "published",
                "evaluation_input_path": str(evaluation_input_path.relative_to(root)),
                "evaluation_input_sha256": file_sha256(evaluation_input_path),
            },
            "completed_at": "2026-08-08T00:01:00Z",
        },
    )
    index = {
        "schema": "sure.eval.result_index.v2",
        "publication": {
            "status": "verified",
            "verified_by": "human-reviewer",
            "verified_at": "2026-08-08T00:00:00Z",
        },
        "model": {"id": MODEL_ID, "artifact_sha256": MODEL_SHA},
        "inference_runs": [
            {
                "id": "inf-001",
                "created_at": "2026-08-08T00:00:00Z",
                "protocol": PROTOCOL,
                "datasets": [
                    {
                        "dataset": DATASET,
                        "prediction": {
                            "manifest_path": str(prediction_manifest_path.relative_to(root)),
                            "manifest_sha256": file_sha256(prediction_manifest_path),
                        },
                    }
                ],
            }
        ],
        "evaluation_runs": [
            {
                "id": "eval-001",
                "created_at": "2026-08-08T00:01:00Z",
                "inference_id": "inf-001",
                "engine": ENGINE,
                "datasets": [
                    {
                        "dataset": DATASET,
                        "protocol": PROTOCOL,
                        "pipelines": [
                            {
                                "pipeline_id": PIPELINE_ID,
                                "report_path": str(report_path.relative_to(root)),
                                "report_sha256": file_sha256(report_path),
                                "engine_report_path": str(engine_report_path.relative_to(root)),
                                "engine_report_sha256": file_sha256(engine_report_path),
                                "pipeline_description_path": str(
                                    description_path.relative_to(root)
                                ),
                                "pipeline_description_sha256": file_sha256(description_path),
                            }
                        ],
                    }
                ],
            }
        ],
    }
    _write_json(root / "result_index.json", index)
    return storage, root


def _request(**overrides) -> ReuseRequest:
    values = {
        "model_id": MODEL_ID,
        "model_artifact_sha256": MODEL_SHA,
        "protocol_id": PROTOCOL["id"],
        "protocol_version": PROTOCOL["version"],
        "effective_params_sha256": PROTOCOL["effective_params_sha256"],
        "datasets": (DATASET,),
        "mode": "auto",
        "engine_commit": ENGINE["commit"],
        "engine_tree_sha256": ENGINE["tree_sha256"],
        "pipeline_ids": {f"{DATASET['id']}@test": (PIPELINE_ID,)},
    }
    values.update(overrides)
    return ReuseRequest(**values)


def _stage_published_inference_evaluation(
    store: AppendOnlyResultStore,
    published_root: Path,
    *,
    evaluation_id: str,
    score: float,
    created_at: str,
) -> Path:
    run_root = store.reserve_evaluation_run(evaluation_id)
    item_root = run_root / "item-001"
    item_root.mkdir()
    evaluation_input = item_root / "evaluation_input.json"
    engine_report = item_root / "engine_report.json"
    pipeline_description = item_root / "pipeline_description.json"
    report_path = item_root / "report.json"
    _write_json(evaluation_input, {"fixture": evaluation_id})
    _write_json(engine_report, {"pipeline_id": PIPELINE_ID, "score": score})
    _write_json(pipeline_description, {"pipeline_id": PIPELINE_ID, "nodes": []})
    _write_json(
        report_path,
        {
            "schema": "sure.eval.published_report.v2",
            "status": "success",
            "model": {"id": MODEL_ID, "artifact_sha256": MODEL_SHA},
            "dataset": DATASET,
            "protocol": PROTOCOL,
            "evaluation": {
                "pipeline_id": PIPELINE_ID,
                "engine_commit": ENGINE["commit"],
                "engine_tree_sha256": ENGINE["tree_sha256"],
            },
            "metric": {"name": "cer", "score": score},
            "lineage": {
                "inference_id": "inf-001",
                "inference_source": "published",
                "evaluation_input_path": str(
                    evaluation_input.relative_to(store.model_root)
                ),
                "evaluation_input_sha256": file_sha256(evaluation_input),
            },
            "completed_at": created_at,
        },
    )
    entry = {
        "id": evaluation_id,
        "created_at": created_at,
        "inference_id": "inf-001",
        "engine": ENGINE,
        "datasets": [
            {
                "dataset": DATASET,
                "protocol": PROTOCOL,
                "pipelines": [
                    {
                        "pipeline_id": PIPELINE_ID,
                        "report_path": str(report_path.relative_to(store.model_root)),
                        "report_sha256": file_sha256(report_path),
                        "engine_report_path": str(
                            engine_report.relative_to(store.model_root)
                        ),
                        "engine_report_sha256": file_sha256(engine_report),
                        "pipeline_description_path": str(
                            pipeline_description.relative_to(store.model_root)
                        ),
                        "pipeline_description_sha256": file_sha256(
                            pipeline_description
                        ),
                    }
                ],
            }
        ],
    }
    store.append_published_inference_evaluation(
        entry,
        published_index_path=published_root / "result_index.json",
    )
    return report_path


def test_exact_verified_result_is_reused(tmp_path: Path) -> None:
    storage, root = _published_fixture(tmp_path)

    validate_result_index(root, expected_model_id=MODEL_ID)
    decision = ResultRegistry(storage=storage).decide(_request())

    assert decision.status == "ready"
    assert decision.action == "reuse_result"
    assert decision.inference_id == "inf-001"
    assert decision.evaluation_id == "eval-001"


def test_changed_evaluation_engine_reuses_only_predictions(tmp_path: Path) -> None:
    storage, _ = _published_fixture(tmp_path)

    decision = ResultRegistry(storage=storage).decide(
        _request(engine_commit="new-commit")
    )

    assert decision.action == "reevaluate"
    assert decision.inference_id == "inf-001"
    assert decision.evaluation_id is None


def test_protocol_or_dataset_mismatch_requires_new_inference(tmp_path: Path) -> None:
    storage, _ = _published_fixture(tmp_path)
    registry = ResultRegistry(storage=storage)

    protocol_decision = registry.decide(
        _request(protocol_id="strict_core", effective_params_sha256="f" * 64)
    )
    changed_dataset = dict(DATASET)
    changed_dataset.update(
        id="aishell1__v1.0.3", version="v1.0.3", manifest_sha256="1" * 64
    )
    dataset_decision = registry.decide(
        _request(
            datasets=(changed_dataset,),
            pipeline_ids={f"{changed_dataset['id']}@test": (PIPELINE_ID,)},
        )
    )

    assert protocol_decision.action == "infer_and_evaluate"
    assert dataset_decision.action == "infer_and_evaluate"


def test_strict_reuse_modes_block_when_candidate_is_missing(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    registry = ResultRegistry(storage=storage)

    assert registry.decide(_request(mode="reevaluate")).action == "blocked"
    assert registry.decide(_request(mode="reuse_result")).action == "blocked"


def test_retest_does_not_consult_published_results(tmp_path: Path) -> None:
    decision = ResultRegistry(storage=_storage(tmp_path)).decide(_request(mode="retest"))

    assert decision.action == "infer_and_evaluate"
    assert decision.reason_codes == ("forced_retest",)


def test_corrupt_published_artifact_blocks_instead_of_falling_back(tmp_path: Path) -> None:
    storage, root = _published_fixture(tmp_path)
    prediction = root / "inference_runs/inf-001/predictions/aishell1.jsonl"
    prediction.write_text("tampered\n", encoding="utf-8")

    decision = ResultRegistry(storage=storage).decide(_request())

    assert decision.status == "blocked"
    assert decision.reason_codes == ("published_index_invalid",)


def test_append_only_store_uses_one_model_root_and_refuses_overwrite(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    store = AppendOnlyResultStore(MODEL_ID, MODEL_SHA, storage=storage)

    first = store.reserve_inference_run("inf-001")
    second = store.reserve_evaluation_run("eval-001")
    publication = store.write_publication_manifest()
    store.append_inference_entry({"id": "inf-001", "protocol": PROTOCOL, "datasets": []})
    store.append_evaluation_entry(
        {"id": "eval-001", "inference_id": "inf-001", "engine": ENGINE, "datasets": []}
    )

    assert first.parent.parent == store.model_root
    assert second.parent.parent == store.model_root
    assert publication.parent == store.model_root
    assert (
        json.loads(store.index_path.read_text())["publication"]["status"]
        == "pending_human_review"
    )
    with pytest.raises(FileExistsError):
        store.reserve_inference_run("inf-001")
    with pytest.raises(PathPolicyError):
        store.write_json_once(storage.results_read_root / MODEL_ID / "forbidden.json", {})


def test_existing_published_model_uses_checksum_bound_merge_delta(tmp_path: Path) -> None:
    storage, published_root = _published_fixture(tmp_path)
    store = AppendOnlyResultStore(MODEL_ID, MODEL_SHA, storage=storage)
    _stage_published_inference_evaluation(
        store,
        published_root,
        evaluation_id="eval-002",
        score=0.2,
        created_at="2026-08-08T00:02:00Z",
    )

    manifest_path = store.write_publication_manifest()
    manifest = json.loads(manifest_path.read_text())
    delta = json.loads((store.model_root / manifest["result_delta"]).read_text())

    assert manifest["publication_strategy"] == "merge_delta"
    assert manifest["automatic_publish_allowed"] is False
    assert delta["base_published_index"]["sha256"] == file_sha256(
        published_root / "result_index.json"
    )
    assert [item["id"] for item in delta["additions"]["evaluation_runs"]] == ["eval-002"]
    assert manifest["schema"] == "sure.eval.publication_manifest.v2"
    assert manifest["derived_reports"]["report_jsonl"]["row_count"] == 2


def test_reevaluations_append_to_deterministic_model_report_views(tmp_path: Path) -> None:
    storage, published_root = _published_fixture(tmp_path)
    published_before = {
        path.relative_to(published_root): file_sha256(path)
        for path in published_root.rglob("*")
        if path.is_file()
    }
    store = AppendOnlyResultStore(MODEL_ID, MODEL_SHA, storage=storage)
    _stage_published_inference_evaluation(
        store,
        published_root,
        evaluation_id="eval-002",
        score=0.2,
        created_at="2026-08-08T00:02:00Z",
    )
    first_manifest = json.loads(store.write_publication_manifest().read_text())
    first_report = (store.model_root / "report.jsonl").read_bytes()

    _stage_published_inference_evaluation(
        store,
        published_root,
        evaluation_id="eval-003",
        score=0.3,
        created_at="2026-08-08T00:03:00Z",
    )
    second_manifest = json.loads(store.write_publication_manifest().read_text())
    second_report = (store.model_root / "report.jsonl").read_bytes()
    second_snapshot = (store.model_root / "report_snapshot.md").read_bytes()
    second_delta = (store.model_root / "result_delta.json").read_bytes()
    rows = [json.loads(line) for line in second_report.splitlines()]

    assert [row["evaluation"]["id"] for row in rows] == [
        "eval-001",
        "eval-002",
        "eval-003",
    ]
    assert second_report.startswith(first_report)
    assert first_manifest["derived_reports"]["report_jsonl"]["row_count"] == 2
    assert second_manifest["derived_reports"]["report_jsonl"]["row_count"] == 3
    assert second_manifest["derived_reports"]["report_jsonl"]["sha256"] == file_sha256(
        store.model_root / "report.jsonl"
    )
    assert second_manifest["derived_reports"]["report_snapshot"]["sha256"] == file_sha256(
        store.model_root / "report_snapshot.md"
    )

    store.write_publication_manifest()
    assert (store.model_root / "report.jsonl").read_bytes() == second_report
    assert (store.model_root / "report_snapshot.md").read_bytes() == second_snapshot
    assert (store.model_root / "result_delta.json").read_bytes() == second_delta
    assert {
        path.relative_to(published_root): file_sha256(path)
        for path in published_root.rglob("*")
        if path.is_file()
    } == published_before


def test_tampered_evaluation_artifact_blocks_report_refresh(tmp_path: Path) -> None:
    storage, published_root = _published_fixture(tmp_path)
    store = AppendOnlyResultStore(MODEL_ID, MODEL_SHA, storage=storage)
    report_path = _stage_published_inference_evaluation(
        store,
        published_root,
        evaluation_id="eval-002",
        score=0.2,
        created_at="2026-08-08T00:02:00Z",
    )
    report_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(DerivedReportError, match="checksum mismatch"):
        store.write_publication_manifest()


def test_delta_protocol_must_match_published_inference(tmp_path: Path) -> None:
    storage, published_root = _published_fixture(tmp_path)
    store = AppendOnlyResultStore(MODEL_ID, MODEL_SHA, storage=storage)
    _stage_published_inference_evaluation(
        store,
        published_root,
        evaluation_id="eval-002",
        score=0.2,
        created_at="2026-08-08T00:02:00Z",
    )
    delta_path = store.model_root / "result_delta.json"
    delta = json.loads(delta_path.read_text())
    delta["additions"]["evaluation_runs"][0]["datasets"][0]["protocol"] = {
        "id": "strict_core",
        "version": "2.0",
        "effective_params_sha256": "f" * 64,
    }
    _write_json(delta_path, delta)

    with pytest.raises(DerivedReportError, match="protocol differs"):
        store.write_publication_manifest()


def test_changed_published_index_blocks_delta_refresh(tmp_path: Path) -> None:
    storage, published_root = _published_fixture(tmp_path)
    store = AppendOnlyResultStore(MODEL_ID, MODEL_SHA, storage=storage)
    _stage_published_inference_evaluation(
        store,
        published_root,
        evaluation_id="eval-002",
        score=0.2,
        created_at="2026-08-08T00:02:00Z",
    )
    published_index_path = published_root / "result_index.json"
    published_index = json.loads(published_index_path.read_text())
    published_index["publication"]["verified_at"] = "2026-08-08T01:00:00Z"
    _write_json(published_index_path, published_index)

    with pytest.raises(ValueError, match="published result index changed"):
        store.write_publication_manifest()
