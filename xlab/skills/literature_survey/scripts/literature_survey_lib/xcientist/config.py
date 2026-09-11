from __future__ import annotations

import os
import shutil
import json
import hashlib
import re
import unicodedata
import pypdfium2 as pdfium
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .utils.config_utils import merge_with_default_survey_config
from ..common import atomic_write_json, utc_now

_PACKAGE_ROOT = Path(__file__).resolve().parent
_CONFIG_DIR = _PACKAGE_ROOT / "config"


def bundled_config_path(config_name: str = "deep_survey_fast") -> Path:
    name = Path(config_name).name
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    path = _CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Bundled SurveyAgent config not found: {path}")
    return path


def is_secret_override_key(key: str) -> bool:
    normalized = key.lower()
    if any(part in normalized for part in ("api_key", "apikey", "secret", "password")):
        return True
    return "token" in normalized.replace(".", "_").replace("-", "_").split("_")


def apply_overrides(config: DictConfig, overrides: dict[str, Any] | None) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("SurveyAgent override keys must be non-empty strings.")
        if is_secret_override_key(key):
            raise ValueError(f"Refusing to persist secret-like SurveyAgent override: {key}")
        OmegaConf.update(config, key, value, merge=True)


def first_env(*names: str, default: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default



def load_survey_agent_config(
    *,
    run_dir: str | Path,
    topic: str,
    graph_db: str | Path | None = None,
    config_name: str = "deep_survey_fast",
    overrides: dict[str, Any] | None = None,
) -> DictConfig:
    """Load the integrated Xcientist SurveyAgent config with XLab run-local paths."""
    run_dir = Path(run_dir).expanduser().resolve()
    if not os.environ.get("OPENAI_API_KEY", "").strip() and os.environ.get("LLM_API_KEY", "").strip():
        os.environ["OPENAI_API_KEY"] = os.environ["LLM_API_KEY"].strip()
    if not topic.strip():
        raise ValueError("Survey topic must be non-empty.")

    artifacts_dir = run_dir / "artifacts"
    state_dir = run_dir / "state" / "xcientist"
    cache_dir = state_dir / "cache"
    for directory in (artifacts_dir, state_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if graph_db is not None:
        source_graph_db_path = Path(graph_db).expanduser().resolve()
        if not source_graph_db_path.exists():
            raise FileNotFoundError(f"Knowledge graph database does not exist: {source_graph_db_path}")
        graph_db_path = state_dir / "graph.db"
        if source_graph_db_path != graph_db_path.resolve() and not graph_db_path.exists():
            shutil.copy2(source_graph_db_path, graph_db_path)
    else:
        graph_db_path = state_dir / "graph.db"

    alias_input = os.environ.get("XLAB_LITERATURE_SURVEY_CITATION_ALIASES")
    if alias_input:
        aliases = json.loads(Path(alias_input).read_text())
        if aliases.get("schema_version") != "xlab.citation_aliases.v1" or not isinstance(aliases.get("aliases"), list):
            raise ValueError("Invalid citation alias input")
        if graph_db is None or aliases.get("graph_run_id") != Path(graph_db).resolve().parent.parent.name:
            raise ValueError("Citation aliases belong to a different input graph")
        for reference in aliases.get("verified_references", []):
            manifest = json.loads(Path(reference["fetch_manifest"]).read_text())
            matches = [paper for paper in manifest.get("papers", []) if paper.get("id") == reference["paper_id"] and paper.get("download_status") == "downloaded"]
            if manifest.get("schema_version") != "xlab.paper_fetch_manifest.v1" or len(matches) != 1:
                raise ValueError("Supplemental citation has no verified paper-fetch record")
            fetched = matches[0]
            pdf_path = Path(fetched["pdf_path"]).resolve()
            digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            if digest != fetched["sha256"] or digest != reference["pdf_sha256"] or reference["title"] != fetched["title"]:
                raise ValueError("Supplemental citation PDF or title does not match its fetch manifest")
            if any(reference.get(field) != fetched.get(field) for field in ("authors", "year", "venue", "url")):
                raise ValueError("Supplemental citation metadata differs from its fetch manifest")
            document = pdfium.PdfDocument(pdf_path)
            text = []
            try:
                for index in range(len(document)):
                    page = document[index]
                    text_page = page.get_textpage()
                    text.append(text_page.get_text_range())
                    text_page.close()
                    page.close()
            finally:
                document.close()
            title_key = re.sub(r"\W", "", unicodedata.normalize("NFKC", reference["title"]).casefold())
            heading_key = re.sub(r"\W", "", unicodedata.normalize("NFKC", text[0][:6000]).casefold())
            if title_key not in heading_key:
                raise ValueError("Supplemental citation title was not found on its PDF first page")
            reference["verified_reference_text"] = "\n\n".join(text)
            reference["source"] = "verified_reference"
        alias_path = state_dir / "citation_aliases.json"
        previous = json.loads(alias_path.read_text()) if alias_path.exists() else None
        if aliases != previous:
            archive = state_dir / ("superseded-citations-" + utc_now().replace(":", "-"))
            for name in ("refined.raw.md", "references.raw.json", "evaluation.json", "engine_result.json"):
                path = state_dir / name
                if path.exists():
                    archive.mkdir(parents=True, exist_ok=True)
                    path.replace(archive / name)
            atomic_write_json(alias_path, aliases)

    os.environ["XLAB_SURVEY_AGENT_BASE_DIR"] = str(run_dir)
    os.environ["XLAB_SURVEY_AGENT_GRAPH_DB"] = str(graph_db_path)

    config = OmegaConf.load(bundled_config_path(config_name))
    config = merge_with_default_survey_config(config)

    xlab_overrides: dict[str, Any] = {
        "BasicInfo.topic": topic.strip(),
        "BasicInfo.base_dir": str(run_dir),
        "BasicInfo.cache_path": str(cache_dir),
        "BasicInfo.save_path": str(artifacts_dir / "survey.md"),
        "BasicInfo.save_json_path": str(artifacts_dir / "survey.xcientist.json"),
        "BasicInfo.evaluation_save_path": str(artifacts_dir / "survey.xcientist.evaluation.txt"),
        "BasicInfo.output_base_dir": str(artifacts_dir),
        "APIInfo.llm_api_base_url": first_env("LLM_BASE_URL", "LLM_API_BASE", default="https://api.minimaxi.com/v1"),
        "APIInfo.llm_model_name": first_env("LLM_MODEL", default="MiniMax-M3"),
        "APIInfo.llm_max_context_length": int(os.environ.get("LLM_CONTEXT_WINDOW", "512000")),
        "APIInfo.chat_timeout": int(os.environ.get("XLAB_LITERATURE_SURVEY_REQUEST_TIMEOUT_SECONDS", "300")),
        "APIInfo.semantic_scholar_api_max_retry": int(os.environ.get("XLAB_LITERATURE_SURVEY_MAX_RETRIES", "3")),
        "APIInfo.arxiv_api_max_retry": int(os.environ.get("XLAB_LITERATURE_SURVEY_MAX_RETRIES", "3")),
        "ModuleInfo.PaperGraphRetriever.db_path": str(graph_db_path),
    }
    stream = os.environ.get("XLAB_LITERATURE_SURVEY_USE_STREAM")
    if stream is not None:
        xlab_overrides["APIInfo.use_stream_mode"] = stream.strip().lower() in {"1", "true", "yes", "on"}
    workers = os.environ.get("XLAB_LITERATURE_SURVEY_BATCH_WORKERS")
    if workers is not None:
        worker_count = int(workers)
        if not 1 <= worker_count <= 128:
            raise ValueError("XLAB_LITERATURE_SURVEY_BATCH_WORKERS must be between 1 and 128.")
        xlab_overrides["APIInfo.batch_chat_agent_worker"] = worker_count
    thinking = os.environ.get("XLAB_LITERATURE_SURVEY_ENABLE_THINKING")
    if thinking is not None:
        xlab_overrides["APIInfo.enable_thinking"] = thinking.strip().lower() in {"1", "true", "yes", "on"}
    stream_timeout = os.environ.get("XLAB_LITERATURE_SURVEY_STREAM_TIMEOUT_SECONDS")
    if stream_timeout is not None:
        seconds = int(stream_timeout)
        if not 30 <= seconds <= 3600:
            raise ValueError("XLAB_LITERATURE_SURVEY_STREAM_TIMEOUT_SECONDS must be between 30 and 3600.")
        xlab_overrides["APIInfo.stream_read_timeout"] = seconds
    apply_overrides(config, xlab_overrides)
    apply_overrides(config, overrides)

    snapshot_path = state_dir / "config.snapshot.yaml"
    snapshot_path.write_text(OmegaConf.to_yaml(config, resolve=False), encoding="utf-8")
    return config
