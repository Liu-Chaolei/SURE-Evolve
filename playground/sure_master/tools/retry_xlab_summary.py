"""Retry a saved XLab round publication without rerunning model experiments."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from playground.sure_master.core.contracts import CandidateResult, RoundResult, RungResult
from playground.sure_master.core.xlab_client import XlabIdeaClient
from playground.sure_master.tools.preflight import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    request = args.request.resolve()
    if workspace not in request.parents or not request.name.endswith(".request.json"):
        raise ValueError("Request must be a saved round request inside the original workspace")
    config = load_config(args.config)
    provider = config["xlab"]["idea_provider"]
    raw = json.loads(request.read_text())
    candidates = [CandidateResult(**{**item, "rungs": [RungResult(**r) for r in item.get("rungs", [])]})
                  for item in raw.get("candidates", [])]
    result = RoundResult(**{**raw, "candidates": candidates})
    client = XlabIdeaClient(provider["command"], workspace_root=workspace,
                            receipt_path=provider.get("receipt_path", "artifacts/xlab_operations.json"),
                            timeout_seconds=provider.get("timeout_seconds", 3600),
                            environment=provider.get("environment"))
    try:
        summary = client.retry_summary(result)
        output = request.with_name(request.name.replace(".request.json", ".summary.json"))
        output.write_text(json.dumps(asdict(summary), indent=2) + "\n")
        print(json.dumps({"status": "published", "summary": str(output)}))
    finally:
        client.close()


if __name__ == "__main__":
    main()
