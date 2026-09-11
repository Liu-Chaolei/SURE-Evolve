#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-metadata", required=True)
    parser.add_argument("--code-metadata", required=True)
    parser.add_argument("--target-experiment", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--reported-metrics", required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--stage-artifact", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paper = json.loads(Path(args.paper_metadata).read_text(encoding="utf-8"))
    code = json.loads(Path(args.code_metadata).read_text(encoding="utf-8"))
    environment = json.loads(Path(args.environment).read_text(encoding="utf-8"))
    execution = json.loads(Path(args.execution).read_text(encoding="utf-8"))
    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    reported = json.loads(Path(args.reported_metrics).read_text(encoding="utf-8"))
    measured = evaluation.get("metrics", {})
    shared = sorted(set(reported) & set(measured))
    comparison = {}
    for name in shared:
        expected = reported[name]
        actual = measured[name]
        delta = actual - expected
        comparison[name] = {
            "reported": expected,
            "measured": actual,
            "absolute_delta": abs(delta),
            "within_tolerance": math.isfinite(delta) and abs(delta) <= args.tolerance,
        }
    status = (
        "success"
        if execution.get("status") == "success"
        and shared
        and all(value["within_tolerance"] for value in comparison.values())
        else "incomplete"
    )
    limitations = []
    if not shared:
        limitations.append("No measured metric could be aligned with a reported paper metric.")
    if any(not value["within_tolerance"] for value in comparison.values()):
        limitations.append("At least one aligned metric falls outside the declared tolerance.")
    report = {
        "schema_version": "xlab.reproduction_report.v1",
        "paper": paper,
        "code": code,
        "target_experiment": args.target_experiment,
        "environment": environment,
        "commands": [execution.get("command")],
        "reported_metrics": reported,
        "measured_metrics": measured,
        "comparison": comparison,
        "stage_artifacts": args.stage_artifact,
        "status": status,
        "limitations": limitations,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": status, "aligned_metrics": len(shared)}))


if __name__ == "__main__":
    main()
