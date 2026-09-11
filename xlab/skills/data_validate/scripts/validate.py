#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--required-fields", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.input)
    required = [field.strip() for field in args.required_fields.split(",") if field.strip()]
    records = []
    errors = []
    seen = set()
    duplicates = 0
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("row is not an object")
            missing = [field for field in required if record.get(field) in (None, "")]
            if missing:
                raise ValueError(f"missing fields: {missing}")
            key = json.dumps(record, sort_keys=True)
            if key in seen:
                duplicates += 1
            seen.add(key)
            records.append(record)
        except Exception as exc:
            errors.append({"line": index, "error": str(exc)})
    contract = {
        "schema_version": "xlab.dataset.v1",
        "path": args.input,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(records),
        "duplicates": duplicates,
        "errors": errors,
        "required_fields": required,
        "valid": not errors and duplicates == 0 and bool(records),
    }
    Path(args.output).write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return 0 if contract["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
