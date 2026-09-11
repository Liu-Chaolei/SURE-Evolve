#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    turns = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            turn = json.loads(line)
            if not isinstance(turn, dict):
                raise ValueError(f"turn {line_number} must be an object")
            turns.append(turn)
    return turns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--roles", required=True)
    parser.add_argument("--turns", required=True)
    parser.add_argument("--source-artifact", action="append", required=True)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    roles = [value.strip() for value in args.roles.split(",") if value.strip()]
    if len(set(roles)) < 2:
        raise ValueError("discussion requires at least two distinct roles")
    turns = read_jsonl(Path(args.turns))
    if len(turns) < 2:
        raise ValueError("discussion requires at least two turns")
    for index, turn in enumerate(turns, 1):
        required = {"round", "role", "statement", "stance", "citations", "action"}
        missing = required - set(turn)
        if missing:
            raise ValueError(f"turn {index} is missing: {', '.join(sorted(missing))}")
        if turn["role"] not in roles:
            raise ValueError(f"turn {index} uses undeclared role: {turn['role']}")
        if not 1 <= turn["round"] <= args.max_rounds:
            raise ValueError(f"turn {index} exceeds round budget")
        if not turn["citations"]:
            raise ValueError(f"turn {index} has no source citation")
    actions = sorted({turn["action"] for turn in turns if turn["action"]})
    if not actions:
        raise ValueError("discussion produced no actionable result")
    result = {
        "schema_version": "xlab.research_discussion.v1",
        "topic": args.topic,
        "goal": args.goal,
        "roles": roles,
        "round_count": max(turn["round"] for turn in turns),
        "turns": turns,
        "key_points": [turn["statement"] for turn in turns],
        "consensus": [turn["statement"] for turn in turns if turn["stance"] == "support"],
        "disputes": [turn["statement"] for turn in turns if turn["stance"] in ("oppose", "uncertain")],
        "actions": actions,
        "open_questions": [turn["statement"] for turn in turns if turn["stance"] == "uncertain"],
        "source_artifacts": args.source_artifact,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rounds": result["round_count"], "actions": len(actions)}))


if __name__ == "__main__":
    main()
