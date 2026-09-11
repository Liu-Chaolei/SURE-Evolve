#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    same_benchmark = (
        candidate.get("benchmark_fingerprint") == baseline.get("benchmark_fingerprint")
        and candidate.get("case_count") == baseline.get("case_count")
    )
    if not same_benchmark:
        raise ValueError("candidate and baseline do not cover the same benchmark plan")
    shared = sorted(set(candidate["metrics"]) & set(baseline["metrics"]))
    if not shared:
        raise ValueError("reports have no shared metrics")
    metrics = {
        name: {
            "candidate": candidate["metrics"][name],
            "baseline": baseline["metrics"][name],
            "delta": candidate["metrics"][name] - baseline["metrics"][name],
        }
        for name in shared
    }
    result = {
        "schema_version": "xlab.evaluation_comparison.v1",
        "same_benchmark": True,
        "threshold": args.threshold,
        "metrics": metrics,
        "passed": all(value["delta"] >= args.threshold for value in metrics.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": result["passed"]}))


if __name__ == "__main__":
    main()
