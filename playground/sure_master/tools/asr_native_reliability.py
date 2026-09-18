"""Fresh ASR native provider: TTS reliability rules without TTS response replay."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import runpy
import sys
import time


def project_empty_ablation(result, fields):
    value = result.json_value
    if (fields in ({"summary", "insight"}, {"summary"}) and isinstance(value, dict)
            and set(value) == fields | {"ablation"} and value["ablation"] == []):
        return replace(result, json_value={key: value[key] for key in fields})
    return result


def install(root):
    from research_idea_lib.algorithm import keynote_pipeline
    from research_idea_lib import audit, manifest, pipeline
    from research_idea_lib.asr_deferred_reference_audit import deferred_audit_report
    original_exact = keynote_pipeline._exact_output

    def exact(result, fields):
        projected = project_empty_ablation(result, fields)
        if projected is not result:
            with (root / "keynote_projections.jsonl").open("a") as handle:
                handle.write(json.dumps({"time": time.time(), "input_digest": result.trace.input_digest,
                    "normalization": "discard_empty_ablation_annotation", "raw_response_retained": True}) + "\n")
        return original_exact(projected, fields)

    keynote_pipeline._exact_output = exact
    original_audit = audit.audit_artifacts

    def checked_audit(run_dir):
        from research_idea_lib.algorithm.runtime_adapters import run_paths
        report = original_audit(run_dir)
        paths = run_paths(run_dir)
        if paths["research_idea_json"].exists() and paths["idea_result_json"].exists():
            return deferred_audit_report(report, json.loads(paths["research_idea_json"].read_text()),
                                         json.loads(paths["idea_result_json"].read_text()))
        return report

    audit.audit_artifacts = checked_audit
    manifest.audit_artifacts = checked_audit
    original_merge = pipeline.merge_workflow_retrieval_context

    def merge(context, artifact):
        result = original_merge(context, artifact)
        result["reference_validation"] = deepcopy(artifact.get("persistence", {}).get("idea_result", {}).get("reference_validation", {}))
        return result

    pipeline.merge_workflow_retrieval_context = merge


def main():
    root = Path(os.environ["XLAB_ROOT"])
    sys.path[:0] = [str(root / "xlab/skills/research_idea/scripts"),
                    str(root / "xlab/skills/sure_master/scripts")]
    from dotenv import dotenv_values
    from research_idea_lib.providers.routing import routing_policy_from_environment, routing_credentials
    policy = routing_policy_from_environment()
    credentials = routing_credentials(policy)
    route = next(r for r in policy["routes"] if credentials.get(r["slot"]))
    # Native calls route directly, never through the controller's loopback proxy.
    for key in list(os.environ):
        if key.startswith(("XI_", "ZAI_", "OPENAI_", "LLM_", "XLAB_RESEARCH_IDEA_")):
            del os.environ[key]
    os.environ.update(OPENAI_API_KEY=credentials[route["slot"]], OPENAI_BASE_URL=route["base_url"],
                      XLAB_RESEARCH_IDEA_STREAM="1", XLAB_SURE_RESPONSES="0")
    for phase in ("AGENT", "GENERATION", "EVALUATION", "FUSION"):
        os.environ[f"XLAB_RESEARCH_IDEA_{phase}_MODEL"] = "glm-5.3-flash"
    install(Path(os.environ["XLAB_SURE_RUN_ROOT"]))
    entry = root / "xlab/skills/sure_master/scripts/xlab_idea_client.py"
    sys.argv[0] = str(entry)
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
