from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import RuntimeConfig
from .inputs import SurveyRequest


AGENT_SCHEMA_VERSION = "xlab.literature_survey.agent.v1"
AGENT_EXPORT_VERSION = 4


def survey_agent_requested(runtime: RuntimeConfig) -> bool:
    return bool(runtime.llm_refinement_enabled and runtime.llm_api_key_set)


def run_survey_agent(
    *,
    run_dir: str | Path,
    request: SurveyRequest,
    runtime: RuntimeConfig,
    papers: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    graph_context: dict[str, Any],
) -> dict[str, Any]:
    from .xcientist.api import generate_survey_from_xlab_context

    return generate_survey_from_xlab_context(
        run_dir=run_dir,
        request=request,
        runtime=runtime,
        papers=papers,
        clusters=clusters,
        graph_context=graph_context,
    )
