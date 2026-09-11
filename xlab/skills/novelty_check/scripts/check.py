#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def tokens(value: object) -> set[str]:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    normalized = "".join(character.casefold() if character.isalnum() else " " for character in text)
    return {token for token in normalized.split() if len(token) > 2}


def similarity(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea", required=True)
    parser.add_argument("--papers", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    idea = json.loads(Path(args.idea).read_text(encoding="utf-8"))
    papers = json.loads(Path(args.papers).read_text(encoding="utf-8")).get("papers", [])
    claim_fields = ("research_question", "hypothesis", "method", "expected_contribution")
    claims = [{"field": field, "text": idea[field]} for field in claim_fields if idea.get(field)]
    if not claims:
        raise ValueError("idea contains no checkable claims")
    idea_tokens = tokens([claim["text"] for claim in claims])
    ranked = []
    for paper in papers:
        paper_tokens = tokens({"title": paper.get("title"), "abstract": paper.get("abstract")})
        ranked.append({
            "paper_id": str(paper["id"]),
            "title": paper["title"],
            "similarity": round(similarity(idea_tokens, paper_tokens), 6),
        })
    ranked.sort(key=lambda item: (-item["similarity"], item["paper_id"]))
    closest = ranked[:10]
    maximum = closest[0]["similarity"] if closest else 0
    risk = "high" if maximum >= 0.55 else "medium" if maximum >= 0.3 else "low"
    confidence = "high" if len(papers) >= 20 else "medium" if len(papers) >= 5 else "low"
    gaps = [] if confidence != "low" else ["Paper coverage is below five records."]
    report = {
        "schema_version": "xlab.novelty_report.v1",
        "idea_claims": claims,
        "closest_work": closest,
        "overlap": [
            {"paper_id": item["paper_id"], "similarity": item["similarity"]}
            for item in closest if item["similarity"] >= 0.3
        ],
        "differences": [
            "Token-level evidence does not establish method equivalence; inspect full-text method and experiment relations."
        ],
        "evidence_gaps": gaps,
        "risk_level": risk if closest else "unknown",
        "confidence": confidence,
        "recommendations": [
            "Review the highest-ranked full texts and graph relations before experiment investment.",
            *("Expand paper search coverage." for _ in gaps),
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "risk": report["risk_level"], "confidence": confidence}))


if __name__ == "__main__":
    main()
