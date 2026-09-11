#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rows(path: Path, format_name: str) -> list[dict[str, object]]:
    if format_name == "csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if format_name == "jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    raise ValueError("JSON input must be an array")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--format", required=True, choices=["csv", "json", "jsonl"])
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("mapping must be a non-empty object")
    raw = rows(Path(args.input), args.format)
    normalized = []
    invalid = 0
    for index, row in enumerate(raw):
        item = {target: row.get(source) for target, source in mapping.items()}
        if any(value in (None, "") for value in item.values()):
            invalid += 1
            continue
        item.update({"_source": args.input, "_source_row": index})
        normalized.append(item)
    Path(args.output).write_text("".join(json.dumps(item) + "\n" for item in normalized), encoding="utf-8")
    report = {
        "schema_version": "xlab.processing_report.v1",
        "input_rows": len(raw),
        "output_rows": len(normalized),
        "invalid_rows": invalid,
        "mapping": mapping,
        "dataset_path": args.output,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
