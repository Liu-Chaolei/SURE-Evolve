"""Export observed SD results; never fill missing experiments with estimates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from playground.sure_master.core.utils.slurm import atomic_json


def summarize(root: Path):
    rows, candidates, curves = [], [], []
    control = root / "control.json"
    active = json.loads(control.read_text())["active_groups"] if control.exists() else list("ABCD")
    baseline_file = root / "baseline/result.json"
    baseline = json.loads(baseline_file.read_text())["baseline"] if baseline_file.exists() else None
    for group in "ABCD":
        directory = root / group
        result_path, holdout_path = directory / "result.json", directory / "holdout.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        final = json.loads(holdout_path.read_text()) if holdout_path.exists() else {}
        winner = next((item for item in final.get("records", []) if item["idea_id"] == final.get("winner")), {})
        base_test = next((item for item in final.get("records", []) if item["idea_id"] == "baseline"), {})
        actual, best, elapsed = 0, baseline["score"] if baseline else None, 0.0
        valid = improved = 0
        for number in range(1, 7):
            path = directory / f"search/workspace/metric/ablation_round_{number}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            for entry in record["candidates"]:
                actual += 1
                success = entry.get("success") and type(entry.get("score")) in (float, int)
                valid += int(bool(success))
                improved += int(bool(success and baseline and entry["score"] < baseline["score"]))
                duration = entry.get("runtime_seconds")
                cards = 1 if entry.get("candidate_type") == "inference" else 4
                if duration is not None:
                    elapsed += float(duration) * cards / 3600
                if success:
                    best = entry["score"] if best is None else min(best, entry["score"])
                candidates.append({"group": group, "round": number, "attempt": actual,
                    "idea_id": entry.get("idea_id"), "candidate_type": entry.get("candidate_type"),
                    "success": bool(success), "search_der": entry.get("score"),
                    "failure_category": entry.get("failure_category"), "reason": entry.get("reason_code"),
                    "runtime_seconds": duration, "workspace": entry.get("workspace")})
                curves.append({"group": group, "attempt": actual, "best_search_der": best,
                               "recorded_accelerator_hours": elapsed})
        usage = {}
        for path in (directory / "xlab_ideas").rglob("*.json"):
            value = json.loads(path.read_text()).get("usage", {})
            for key, count in value.items():
                if type(count) in (int, float):
                    usage[key] = usage.get(key, 0) + count
        atomic_json(directory / "research_usage.json", usage)
        relative = None
        if winner and base_test and base_test["score"]:
            relative = (base_test["score"] - winner["score"]) / base_test["score"]
        components = {}
        if winner.get("workspace"):
            report_path = Path(winner["workspace"]) / "metric/report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                sessions = report.get("details", {}).get("scoring_result", {}).get("per_session", {})
                for key in ("missed_speaker_time", "falarm_speaker_time", "speaker_error_time"):
                    values = []
                    for item in sessions.values():
                        names = ("missed_speaker_time", "falarm_speaker_time", "speaker_error_time")
                        if any(item.get(name) is None for name in names):
                            continue
                        total = sum(item[name] for name in names)
                        values.append(item[key] * item["error_rate"] / total if total else 0.0)
                    components[key + "_rate"] = sum(values) / len(values) if values else None
        rows.append({"group": group, "status": "cancelled" if group not in active else "completed" if winner else "pending",
            "selection_der": result.get("best_score"), "holdout_der": winner.get("score"),
            "baseline_holdout_der": base_test.get("score"), "relative_error_reduction": relative,
            "attempted_candidates": actual, "valid_candidates": valid,
            "candidates_better_than_baseline": improved, "recorded_accelerator_hours": elapsed,
            "winner": final.get("winner"),
            **{key + "_rate": components.get(key + "_rate") for key in
               ("missed_speaker_time", "falarm_speaker_time", "speaker_error_time")}})
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    for name, values in (("results", rows), ("candidates", candidates), ("search_curves", curves)):
        atomic_json(output / f"{name}.json", values)
        if values:
            with (output / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    measured = [point for point in curves if point["best_search_der"] is not None]
    if measured:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for group in "ABCD":
            selected = [point for point in measured if point["group"] == group]
            for axis, key in zip(axes, ("attempt", "recorded_accelerator_hours")):
                axis.step([point[key] for point in selected],
                          [100 * point["best_search_der"] for point in selected], where="post", label=group)
        axes[0].set_xlabel("Candidate attempts")
        axes[0].set_xlim(0, 24)
        axes[1].set_xlabel("Recorded accelerator hours")
        for axis in axes:
            axis.set_ylabel("Best search DER (%)")
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(output / "search_der.svg")
        figure.savefig(output / "search_der.pdf")
        plt.close(figure)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    summarize(parser.parse_args().root.resolve())
