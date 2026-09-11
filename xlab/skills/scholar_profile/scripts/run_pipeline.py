#!/usr/bin/env python3
"""
Single-scholar pipeline entry point for XLab skill.
Runs the full 5-stage pipeline for one scholar.
"""

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path

# Ensure scripts directory is on the path
sys.path.insert(0, str(Path(__file__).parent))


def run_pipeline(scholar_name: str, output_dir: str, native_name: str = None,
                 seed_url: str = None, language: str = "en",
                 run_id: str = "", skill_version: str = "1.0.0") -> dict:
    """Run full 5-stage pipeline for a single scholar.

    Returns dict with paths to all generated artifacts.
    """
    # Validate environment
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Set it via: pi auth, environment variable, or auth.json"
        )

    base_url = os.environ.get("LLM_BASE_URL", "http://58.210.177.113:8888/v1")
    model = os.environ.get("LLM_MODEL", "mimo-v2-flash")

    search_endpoint = os.environ.get("WEB_SEARCH_MCP_URL", "http://127.0.0.1:17890/mcp")
    http_proxy = os.environ.get("HTTP_PROXY", "")
    https_proxy = os.environ.get("HTTPS_PROXY", "")

    print(f"[config] OPENAI_API_KEY: {'set' if api_key else 'NOT SET'}")
    print(f"[config] LLM_BASE_URL: {base_url}")
    print(f"[config] LLM_MODEL: {model}")
    print(f"[config] WEB_SEARCH_MCP_URL: {search_endpoint}")
    print(f"[config] HTTP_PROXY: {http_proxy or 'not set'}")
    print(f"[config] HTTPS_PROXY: {https_proxy or 'not set'}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Stage 1: Name → CSV
    print(f"\n{'='*60}")
    print(f"[Stage 1/5] DBLP Name → CSV")
    print(f"{'='*60}")
    from build_scholars_csv import run_single as build_csv
    csv_path = build_csv(scholar_name, output_dir=str(out), native_name=native_name)

    # Stage 2: Source Discovery
    print(f"\n{'='*60}")
    print(f"[Stage 2/5] Source Discovery")
    print(f"{'='*60}")
    from build_scholar_sources import run_single as build_sources
    source_file = build_sources(
        scholar_name, output_dir=str(out), csv_path=csv_path,
        seed_urls=[seed_url] if seed_url else None,
    )

    # Stage 3: Mainline Extraction
    print(f"\n{'='*60}")
    print(f"[Stage 3/5] Mainline Extraction (LLM)")
    print(f"{'='*60}")
    from scholar_mainline_builder import run_single as build_mainline
    mainline_file = build_mainline(
        scholar_name, output_dir=str(out), source_file=source_file,
        api_key=api_key, base_url=base_url, model=model,
    )

    # Stage 4: Paper Scoring
    print(f"\n{'='*60}")
    print(f"[Stage 4/5] Paper Scoring")
    print(f"{'='*60}")
    from scholar_author_digest import run_single as build_digest
    digest_file = build_digest(
        scholar_name, output_dir=str(out), mainline_file=mainline_file,
    )

    # Stage 5: Prompt Generation
    print(f"\n{'='*60}")
    print(f"[Stage 5/5] System Prompt Generation (LLM)")
    print(f"{'='*60}")
    from professor_system_prompt_builder import run_single as build_prompt
    prompt_path = build_prompt(
        scholar_name, output_dir=str(out), digest_file=digest_file,
        mainline_file=mainline_file, sources_file=source_file,
        language=language, api_key=api_key, base_url=base_url, model=model,
    )

    results = {
        "scholar_csv": csv_path,
        "source_urls": source_file,
        "mainline_json": mainline_file,
        "author_digest": digest_file,
        "system_prompt": prompt_path,
        "output_dir": str(out),
    }
    mainline = json.loads(Path(mainline_file).read_text(encoding="utf-8"))
    digest = json.loads(Path(digest_file).read_text(encoding="utf-8"))
    source_urls = [
        line.strip()
        for line in Path(source_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    papers = digest.get("papers") if isinstance(digest, dict) else []
    profile = {
        "schema_version": "xlab.scholar_profile.v1",
        "scholar_name": scholar_name,
        "sources": source_urls,
        "mainline_graph": mainline,
        "representative_papers": papers if isinstance(papers, list) else [],
        "evidence_warnings": [],
        "system_prompt_path": str(out / "system_prompt.md"),
    }
    profile_path = out / "scholar_profile.json"
    profile_path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    results["scholar_profile"] = str(profile_path)

    if run_id:
        relative_root = f".xlab/runs/{run_id}"
        manifest = {
            "schema_version": "2",
            "run_id": run_id,
            "skill_name": "scholar_profile",
            "skill_version": skill_version,
            "status": "success",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "scholar_name": scholar_name,
                "native_name": native_name,
                "seed_url": seed_url,
                "language": language,
            },
            "outputs": results,
            "validation": {
                "sources": len(source_urls),
                "representative_papers": len(profile["representative_papers"]),
                "system_prompt_chars": len(
                    (out / "system_prompt.md").read_text(encoding="utf-8")
                ),
            },
            "artifacts": [
                {
                    "type": "scholar_profile",
                    "schema_version": "1",
                    "path": f"{relative_root}/scholar_profile.json",
                },
                {
                    "type": "system_prompt",
                    "schema_version": "1",
                    "path": f"{relative_root}/system_prompt.md",
                },
            ],
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run scholar profile pipeline")
    parser.add_argument("--scholar-name", required=True, help="Full name of the scholar")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--native-name", default=None, help="Native-language name for Chinese scholars")
    parser.add_argument("--seed-url", default=None, help="Optional seed URL")
    parser.add_argument("--language", default="en", choices=["en", "zh"], help="Output language")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--skill-version", default="1.0.0")
    args = parser.parse_args()

    results = run_pipeline(
        scholar_name=args.scholar_name,
        output_dir=args.output_dir,
        native_name=args.native_name,
        seed_url=args.seed_url,
        language=args.language,
        run_id=args.run_id,
        skill_version=args.skill_version,
    )
    return results


if __name__ == "__main__":
    main()
