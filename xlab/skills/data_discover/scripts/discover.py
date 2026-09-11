#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import Request, urlopen


def probe(url: str) -> dict[str, object]:
    try:
        request = Request(url, headers={"User-Agent": "XLab/1.0 data-discover"})
        with urlopen(request, timeout=15) as response:
            return {"reachable": True, "status": response.status}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    candidates = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(candidates, list):
        raise ValueError("candidate input must be a JSON array")
    sources = []
    for raw in candidates:
        if not isinstance(raw, dict) or not raw.get("source_id") or not raw.get("url"):
            continue
        item = dict(raw)
        item["access_probe"] = (
            {"reachable": None, "note": "offline"}
            if args.offline
            else probe(str(item["url"]))
        )
        item["priority_score"] = (
            (3 if item.get("priority") == "must_collect" else 1)
            + (2 if item.get("collection_ready") else 0)
            + (1 if item.get("official", True) else 0)
        )
        sources.append(item)
    sources.sort(key=lambda item: (-int(item["priority_score"]), str(item["source_id"])))
    output = {
        "schema_version": "xlab.data_source_plan.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "collect_now": [
            item["source_id"]
            for item in sources
            if item.get("collection_ready") and item["access_probe"].get("reachable") is not False
        ],
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
