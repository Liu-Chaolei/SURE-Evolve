#!/usr/bin/env python3
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_model(identifier: str) -> str:
    path = Path(identifier)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode())
            digest.update(sha256_file(child).encode())
        return digest.hexdigest()
    return hashlib.sha256(identifier.encode()).hexdigest()


def validate_dataset(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not all(key in value for key in ("id", "input", "expected")):
                raise ValueError(f"{path}:{line_number} must contain id, input, and expected")
            count += 1
    if count == 0:
        raise ValueError("dataset contains no cases")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--metrics", default="exact_match,token_f1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    if not dataset.is_file():
        raise FileNotFoundError(f"dataset does not exist: {dataset}")
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    supported = {"exact_match", "token_f1"}
    unsupported = set(metrics) - supported
    if unsupported:
        raise ValueError(f"unsupported metrics: {', '.join(sorted(unsupported))}")

    plan = {
        "schema_version": "xlab.benchmark_plan.v1",
        "model": {
            "identifier": args.model,
            "fingerprint": fingerprint_model(args.model),
        },
        "dataset": {
            "path": str(dataset),
            "sha256": sha256_file(dataset),
            "case_count": validate_dataset(dataset),
        },
        "task_type": args.task_type,
        "metrics": metrics,
        "seed": args.seed,
        "baseline": args.baseline,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "case_count": plan["dataset"]["case_count"]}))


if __name__ == "__main__":
    main()
