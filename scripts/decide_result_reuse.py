#!/usr/bin/env python3
"""Select exact verified NFS results and write a local reuse decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sure_eval.models.registry import ModelRegistry
from sure_eval.results import AppendOnlyResultStore, ReuseRequest, ResultRegistry
from sure_eval.storage import load_storage_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="result_reuse_request.v1 JSON")
    parser.add_argument("--decision-id", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if payload.get("schema") != "sure.eval.result_reuse_request.v1":
        raise ValueError("reuse request schema is not v1")
    if set(payload) != {"schema", "model_id", "protocol", "datasets", "mode", "evaluation"}:
        raise ValueError("reuse request contains missing or unsupported root fields")
    storage = load_storage_config()
    model = ModelRegistry(storage=storage).require_verified(str(payload["model_id"]))
    protocol = payload["protocol"]
    evaluation = payload["evaluation"]
    request = ReuseRequest(
        model_id=model.name,
        model_artifact_sha256=model.artifact_sha256,
        protocol_id=str(protocol["id"]),
        protocol_version=str(protocol["version"]),
        effective_params_sha256=str(protocol["effective_params_sha256"]),
        datasets=tuple(payload["datasets"]),
        mode=str(payload["mode"]),
        engine_commit=str(evaluation["engine_commit"]),
        engine_tree_sha256=str(evaluation["engine_tree_sha256"]),
        pipeline_ids={
            str(key): tuple(str(item) for item in values)
            for key, values in evaluation["pipeline_ids"].items()
        },
    )
    decision = ResultRegistry(storage=storage).decide(request).to_dict()
    store = AppendOnlyResultStore(model.name, model.artifact_sha256, storage=storage)
    output = store.write_reuse_decision(args.decision_id, decision)
    print(json.dumps({**decision, "decision_path": str(output)}, ensure_ascii=False, indent=2))
    return 2 if decision["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
