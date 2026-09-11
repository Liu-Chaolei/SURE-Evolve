#!/usr/bin/env python3
import argparse
import hashlib
import json
import shlex
import subprocess
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            values.append(value)
    return values


def normalize(value: object) -> str:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def token_f1(expected: object, output: object) -> float:
    expected_tokens = normalize(expected).split()
    output_tokens = normalize(output).split()
    if not expected_tokens and not output_tokens:
        return 1.0
    overlap = sum((Counter(expected_tokens) & Counter(output_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(output_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def run_command(command: list[str], case: dict, timeout: float) -> object:
    completed = subprocess.run(
        command,
        input=json.dumps({"id": case["id"], "input": case["input"]}) + "\n",
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"runner exited {completed.returncode}")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict) or "output" not in value:
        raise ValueError("runner output must be one JSON object containing output")
    return value["output"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions")
    source.add_argument("--runner-command")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    dataset_path = Path(plan["dataset"]["path"])
    actual_digest = sha256_file(dataset_path)
    if actual_digest != plan["dataset"]["sha256"]:
        raise ValueError("dataset checksum changed after benchmark preparation")
    cases = read_jsonl(dataset_path)
    predictions = {}
    command = None
    if args.predictions:
        for value in read_jsonl(Path(args.predictions)):
            if "id" not in value or "output" not in value:
                raise ValueError("prediction records must contain id and output")
            predictions[str(value["id"])] = value["output"]
    else:
        command = shlex.split(args.runner_command)
        if not command:
            raise ValueError("runner command is empty")

    results = []
    failures = []
    for case in cases:
        case_id = str(case["id"])
        try:
            output = predictions[case_id] if command is None else run_command(command, case, args.timeout)
            result = {
                "id": case_id,
                "output": output,
                "exact_match": float(normalize(output) == normalize(case["expected"])),
                "token_f1": token_f1(case["expected"], output),
            }
            results.append(result)
        except (KeyError, OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            failures.append({"id": case_id, "error": str(error)})

    if failures:
        raise RuntimeError(f"{len(failures)} benchmark cases failed: {failures[:3]}")
    metrics = {
        name: sum(item[name] for item in results) / len(results)
        for name in plan["metrics"]
    }
    benchmark_contract = {
        "dataset_sha256": plan["dataset"]["sha256"],
        "task_type": plan["task_type"],
        "metrics": plan["metrics"],
        "seed": plan["seed"],
    }
    benchmark_fingerprint = hashlib.sha256(
        json.dumps(benchmark_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "schema_version": "xlab.evaluation_report.v1",
        "plan_sha256": sha256_file(plan_path),
        "benchmark_fingerprint": benchmark_fingerprint,
        "case_count": len(results),
        "metrics": metrics,
        "failures": failures,
        "cases": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "metrics": metrics}))


if __name__ == "__main__":
    main()
