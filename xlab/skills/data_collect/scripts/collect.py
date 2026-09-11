#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-bytes", type=int, default=200_000_000)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    selected = set(plan.get("collect_now") or [])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for source in plan.get("sources") or []:
        source_id = str(source.get("source_id") or "")
        result = {"source_id": source_id, "url": source.get("url"), "status": "skipped", "files": [], "blocker": None}
        if source_id not in selected:
            result["blocker"] = "not selected for collection"
        else:
            try:
                request = Request(str(source["url"]), headers={"User-Agent": "XLab/1.0 data-collect"})
                with urlopen(request, timeout=60) as response:
                    payload = response.read(args.max_bytes + 1)
                    content_type = response.headers.get("content-type", "")
                if len(payload) > args.max_bytes:
                    raise ValueError("payload exceeds max-bytes")
                if "text/html" in content_type:
                    raise ValueError("source returned HTML instead of a data payload")
                source_dir = output_dir / source_id
                source_dir.mkdir(parents=True, exist_ok=True)
                target = source_dir / "payload"
                target.write_bytes(payload)
                result.update({
                    "status": "collected",
                    "files": [{"path": str(target), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}],
                })
            except Exception as exc:
                result.update({"status": "failed", "blocker": str(exc)})
        results.append(result)
    manifest = {"schema_version": "xlab.raw_data_manifest.v1", "plan": args.plan, "sources": results}
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
