from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

from .common import (
    JsonObject,
    append_jsonl,
    artifact_paths,
    as_list,
    as_mapping,
    atomic_write_json,
    compact_key,
    normalize_space,
    read_json,
    sha256_file,
    text,
    utc_now,
)
from .llm import LlmClient, LlmServiceUnavailable, completion_url
from .resources import public_api_url


EXTRACTOR_VERSION = "papergraph-step2-llm-v2"
PROMPT_VERSION = "papergraph-high-recall-v2"
GENERIC_NAMES = {
    "approach",
    "architecture",
    "baseline",
    "dataset",
    "framework",
    "method",
    "model",
    "network",
    "our approach",
    "our method",
    "proposed method",
    "system",
}
DATASET_SUFFIX = re.compile(r"\b(?:dataset|benchmark|corpus|suite|leaderboard)\b", re.IGNORECASE)
METHOD_NAME = re.compile(
    r"\b([A-Z][A-Za-z0-9]*(?:[-+][A-Za-z0-9]+)*(?:\s+[A-Z][A-Za-z0-9]*(?:[-+][A-Za-z0-9]+)*){0,4})\b"
)
COMPARE_SENTENCE = re.compile(
    r"([^.!?]*(?:compare(?:d|s)?\s+(?:against|with|to)|outperform(?:s|ed)?|baseline(?:s)?|than)\s+[^.!?]*[.!?])",
    re.IGNORECASE,
)
DATASET_SENTENCE = re.compile(
    r"([^.!?]*(?:evaluat(?:e|ed|ion)|train(?:ed|ing)?|test(?:ed|ing)?|benchmark(?:ed)?|dataset(?:s)?)\s+[^.!?]*[.!?])",
    re.IGNORECASE,
)
SECTION = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*$")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


IMPORTANT_SECTION = re.compile(
    r"abstract|introduction|method|approach|architecture|experiment|result|"
    r"evaluation|ablation|limitation|discussion|conclusion|future|reference",
    re.IGNORECASE,
)


MAIN_SYSTEM = """You extract a paper's ideation graph for PaperGraph Step 2.
Use only the supplied paper text and metadata. Return one JSON object, without
Markdown fences. Never invent names, evidence, results, citations, or code URLs.
Every extracted claim must include a short exact quote copied from PAPER_TEXT.
Distinguish top-level contributions from their internal components."""

GRAPH_SYSTEM = """You perform high-recall baseline and dataset extraction for
PaperGraph Step 2. Use only PAPER_TEXT. Return one JSON object without Markdown
fences. A baseline is an explicitly compared predecessor or competing method,
not the paper's own contribution. A dataset is an explicitly used evaluation or
training resource, not a metric. Every item needs a short exact quote."""

CALIBRATION_SYSTEM = """You calibrate PaperGraph Step 2 entities and align them
to supplied citation metadata. Return one JSON object without Markdown fences.
You may remove false positives and may add missing concrete baselines or
datasets, but every kept or added item must have an exact quote from PAPER_TEXT.
citation_paper_id must be null or exactly one ID from LOCAL_REFERENCES."""


def _checkpoint_path(directory: Path, paper_id: str) -> Path:
    digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:24]
    return directory / f"{digest}.json"


def _normalized_grounding(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", text(value)).replace("‐", "-")
    return normalize_space(normalized).lower()


def _quote_grounded(quote: object, source: str) -> bool:
    candidate = _normalized_grounding(quote)
    if len(candidate) < 8:
        return False
    grounded_source = _normalized_grounding(source)
    parts = [
        part.strip(" .…")
        for part in re.split(r"(?:\.{3}|…)", candidate)
        if len(part.strip(" .…")) >= 6
    ]
    return bool(parts) and all(part in grounded_source for part in parts)


def _name_grounded(name: object, acronym: object, source: str) -> bool:
    normalized_source = _normalized_grounding(source)
    return any(
        len(candidate) >= 2 and candidate in normalized_source
        for candidate in (
            _normalized_grounding(name),
            _normalized_grounding(acronym),
        )
        if candidate
    )


def _keywords(value: object) -> list[str]:
    result: list[str] = []
    for item in as_list(value):
        normalized = normalize_space(item)
        if normalized and normalized.lower() not in {entry.lower() for entry in result}:
            result.append(normalized)
        if len(result) == 12:
            break
    return result


def _sentences(markdown: str) -> list[str]:
    cleaned = normalize_space(re.sub(r"(?m)^#{1,6}\s+", "", markdown))
    return [
        normalize_space(value)
        for value in re.split(r"(?<=[.!?])\s+", cleaned)
        if len(normalize_space(value)) >= 20
    ]


def _valid_candidate_name(name: str) -> bool:
    key = compact_key(name)
    return not (
        len(name) < 2
        or key in {compact_key(value) for value in GENERIC_NAMES}
        or name.lower() in {"we", "our", "the", "this", "table", "figure"}
    )


def _candidate_names(sentence: str) -> list[str]:
    names: list[str] = []
    for match in METHOD_NAME.finditer(sentence):
        name = normalize_space(match.group(1)).strip(" ,;:()[]")
        if not _valid_candidate_name(name):
            continue
        if compact_key(name) not in {compact_key(value) for value in names}:
            names.append(name)
        if len(names) >= 8:
            break
    return names


def _baseline_candidate_names(sentence: str) -> list[str]:
    match = re.search(
        r"(?:compare(?:d|s)?\s+(?:against|with|to)|baseline(?:s)?(?:\s+(?:include|are))?|outperform(?:s|ed)?)\s+(.+)",
        sentence,
        re.IGNORECASE,
    )
    segment = match.group(1) if match else sentence
    segment = re.split(
        r"\b(?:and\s+)?(?:evaluat(?:e|ed|ion)|train(?:ed|ing)?|test(?:ed|ing)?|on\s+\S+\s+(?:dataset|benchmark|corpus))\b",
        segment,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return _candidate_names(segment)


def _dataset_candidate_names(sentence: str) -> list[str]:
    names: list[str] = []
    for match in METHOD_NAME.finditer(sentence):
        name = normalize_space(match.group(1)).strip(" ,;:()[]")
        if not _valid_candidate_name(name):
            continue
        before = sentence[max(0, match.start() - 20) : match.start()].lower()
        after = sentence[match.end() : match.end() + 20]
        if not (
            DATASET_SUFFIX.search(name)
            or DATASET_SUFFIX.search(after)
            or re.search(r"(?:on|using|with|from)\s+$", before)
        ):
            continue
        if compact_key(name) not in {compact_key(value) for value in names}:
            names.append(name)
        if len(names) >= 8:
            break
    return names


def _citation_index(references: list[JsonObject]) -> dict[str, JsonObject]:
    index: dict[str, JsonObject] = {}
    for reference in references:
        item = as_mapping(reference)
        for key in (item.get("paperId"), item.get("title")):
            normalized = compact_key(key)
            if normalized:
                index[normalized] = item
    return index


def _candidate_hints(markdown: str, references: list[JsonObject]) -> JsonObject:
    reference_index = _citation_index(references)
    baselines: list[JsonObject] = []
    datasets: list[JsonObject] = []
    seen_baselines: set[str] = set()
    seen_datasets: set[str] = set()
    for sentence in _sentences(markdown):
        if COMPARE_SENTENCE.search(sentence):
            for name in _baseline_candidate_names(sentence):
                key = compact_key(name)
                if key in seen_baselines:
                    continue
                seen_baselines.add(key)
                reference = reference_index.get(key, {})
                baselines.append(
                    {
                        "name": name,
                        "quote": sentence[:500],
                        "hint_source": "regex_compare_sentence",
                        "citation_paper_id": as_mapping(reference).get("paperId"),
                        "citation_title": as_mapping(reference).get("title"),
                    }
                )
        if DATASET_SENTENCE.search(sentence):
            for name in _dataset_candidate_names(sentence):
                key = compact_key(name)
                if key in seen_datasets:
                    continue
                seen_datasets.add(key)
                datasets.append(
                    {
                        "name": name,
                        "quote": sentence[:500],
                        "hint_source": "regex_dataset_sentence",
                    }
                )
    return {"baselines": baselines[:80], "datasets": datasets[:80]}


def _local_contexts(markdown: str, names: list[str]) -> list[JsonObject]:
    contexts: list[JsonObject] = []
    normalized_names = [normalize_space(name) for name in names if normalize_space(name)]
    for name in normalized_names[:80]:
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        for match in pattern.finditer(markdown):
            start = max(match.start() - 220, 0)
            end = min(match.end() + 220, len(markdown))
            contexts.append(
                {
                    "name": name,
                    "context": normalize_space(markdown[start:end]),
                }
            )
            break
    return contexts[:120]


def _compact_references(references: list[JsonObject]) -> list[JsonObject]:
    result: list[JsonObject] = []
    for reference in references[:300]:
        item = as_mapping(reference)
        result.append(
            {
                "paperId": item.get("paperId"),
                "title": item.get("title"),
                "venue": item.get("venue"),
                "year": item.get("year"),
                "externalIds": as_mapping(item.get("externalIds")),
                "source": item.get("source") or item.get("reference_source"),
            }
        )
    return result


def _paper_text(markdown: str, maximum: int) -> tuple[str, bool]:
    if len(markdown) <= maximum:
        return markdown, False
    matches = list(SECTION.finditer(markdown))
    selected: list[str] = [markdown[: min(maximum // 3, 30_000)]]
    for index, match in enumerate(matches):
        if not IMPORTANT_SECTION.search(match.group(1)):
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        selected.append(markdown[match.start() : min(end, match.start() + 15_000)])
    selected.append(markdown[-min(maximum // 6, 20_000) :])
    result: list[str] = []
    used = 0
    seen: set[str] = set()
    for chunk in selected:
        digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        if digest in seen or used >= maximum:
            continue
        seen.add(digest)
        accepted = chunk[: maximum - used]
        result.append(accepted)
        used += len(accepted)
    return "\n\n[...SECTION BOUNDARY...]\n\n".join(result), True


def _metadata_prompt(structure: JsonObject) -> str:
    metadata = as_mapping(structure.get("metadata"))
    return json.dumps(
        {
            "title": metadata.get("title"),
            "authors": as_list(metadata.get("authors")),
            "year": metadata.get("year"),
            "venue": metadata.get("venue"),
            "abstract": metadata.get("abstract"),
            "tldr": metadata.get("tldr"),
            "urls": as_list(metadata.get("urls")),
        },
        ensure_ascii=False,
    )


def _main_prompt(structure: JsonObject, paper_text: str) -> str:
    return f"""PASS: MAIN
PAPER_METADATA:
{_metadata_prompt(structure)}

PAPER_TEXT:
{paper_text}

Return exactly this shape:
{{
  "metadata": {{
    "domain": ["specific research domain"],
    "paper_type": ["method|dataset|benchmark|system|theory|survey|other"],
    "structured_summary": {{
      "problem": "grounded concise summary or null",
      "method": "grounded concise summary or null",
      "results": "grounded concise summary or null"
    }},
    "code_url": "explicit repository URL or null"
  }},
  "ideation_resource": {{
    "problems": [{{"keywords":[],"summary":"","insight":null,"quote":"","related_to_core":""}}],
    "core_contributions": [{{"name":"","acronym":null,"type":"Method|Architecture|Algorithm|System|Dataset|Theory","keywords":[],"summary":"","insight":null,"quote":""}}],
    "core_relations": [{{"source":"","target":"","relation":"","keywords":[],"summary":"","insight":null,"quote":""}}],
    "components": [{{"name":null,"keywords":[],"summary":"","insight":null,"quote":"","related_to_core":""}}],
    "innovations": [{{"keywords":[],"summary":"","insight":null,"quote":"","related_to_core":""}}],
    "limitations": [{{"keywords":[],"summary":"","insight":null,"quote":"","related_to_core":""}}],
    "future_work": [{{"keywords":[],"summary":"","insight":null,"quote":"","related_to_core":""}}]
  }}
}}
core_contributions must contain at least one explicitly introduced top-level
contribution. If the paper has none, return an empty list; never use its title
as a fallback."""


def _graph_prompt(
    structure: JsonObject,
    paper_text: str,
    cores: list[JsonObject],
    hints: JsonObject,
) -> str:
    core_names = [text(core.get("name")) for core in cores]
    return f"""PASS: GRAPH
PAPER_METADATA:
{_metadata_prompt(structure)}
CORE_CONTRIBUTIONS:
{json.dumps(core_names, ensure_ascii=False)}
REGEX_CANDIDATE_HINTS:
{json.dumps(hints, ensure_ascii=False)}

PAPER_TEXT:
{paper_text}

Return exactly:
{{
  "graph_data": {{
    "baselines": [{{"name":"","acronym":null,"related_to_core":"","keywords":[],"summary":"","metrics":[],"insight":null,"quote":""}}],
    "datasets": [{{"name":"","acronym":null,"related_to_core":"","keywords":[],"summary":"","metrics":[],"insight":null,"quote":""}}]
  }}
}}
Use the hints as recall aids, not as facts. Include compared predecessors,
related-work methods, backbones/decoders, named ablations, benchmark datasets,
training datasets, and appendix/supplement evaluation resources when the text
grounds them. Use null rather than an empty metrics array when no metric is stated."""


def _calibration_prompt(
    structure: JsonObject,
    cores: list[JsonObject],
    graph_data: JsonObject,
    paper_text: str,
    hints: JsonObject,
) -> str:
    references = [as_mapping(value) for value in as_list(structure.get("semantic_references"))]
    entities = []
    names: list[str] = []
    for kind in ("baselines", "datasets"):
        for value in as_list(graph_data.get(kind)):
            item = as_mapping(value)
            name = normalize_space(item.get("name"))
            if name:
                names.append(name)
            entities.append(
                {
                    "kind": kind[:-1],
                    "name": item.get("name"),
                    "acronym": item.get("acronym"),
                    "related_to_core": item.get("related_to_core"),
                    "quote": item.get("quote"),
                }
            )
    for kind in ("baselines", "datasets"):
        for value in as_list(hints.get(kind)):
            name = normalize_space(as_mapping(value).get("name"))
            if name:
                names.append(name)
    return f"""PASS: CALIBRATE
CORE_CONTRIBUTIONS:
{json.dumps([text(core.get("name")) for core in cores], ensure_ascii=False)}
EXTRACTED_ENTITIES:
{json.dumps(entities, ensure_ascii=False)}
REGEX_CANDIDATE_HINTS:
{json.dumps(hints, ensure_ascii=False)}
LOCAL_CONTEXTS:
{json.dumps(_local_contexts(paper_text, names), ensure_ascii=False)}
LOCAL_REFERENCES:
{json.dumps(_compact_references(references), ensure_ascii=False)}

Return exactly:
{{
  "calibrations": [{{
    "kind": "baseline|dataset",
    "name": "exact extracted entity name or exact new grounded entity name",
    "keep": true,
    "canonical_name": "grounded canonical name or null",
    "acronym": "grounded acronym or null",
    "related_to_core": "exact core name",
    "citation_paper_id": "ID from LOCAL_REFERENCES or null",
    "summary": "grounded concise summary for added items or null",
    "metrics": [],
    "quote": "exact quote for added items or null",
    "reason": "short calibration reason"
  }}]
}}
Include one calibration for every EXTRACTED_ENTITIES item. You may include
additional calibrations for missing concrete baselines/datasets found in the
hints or local contexts. Added items must use keep=true and include a grounded
quote copied exactly from PAPER_TEXT."""


def _valid_core(item: JsonObject, source: str) -> JsonObject | None:
    name = normalize_space(item.get("name"))
    acronym = normalize_space(item.get("acronym")) or None
    quote = normalize_space(item.get("quote"))
    if (
        not name
        or compact_key(name) in {compact_key(value) for value in GENERIC_NAMES}
        or not _name_grounded(name, acronym, source)
        or not _quote_grounded(quote, source)
    ):
        return None
    core_type = text(item.get("type"))
    if core_type not in {
        "Method",
        "Architecture",
        "Algorithm",
        "System",
        "Dataset",
        "Theory",
    }:
        core_type = "Method"
    return {
        "name": name,
        "acronym": acronym,
        "type": core_type,
        "keywords": _keywords(item.get("keywords")),
        "summary": normalize_space(item.get("summary")) or quote,
        "insight": normalize_space(item.get("insight")) or None,
        "quote": quote,
        "fallback_from_title": False,
    }


def _related_core(value: object, cores: list[JsonObject]) -> str | None:
    aliases: dict[str, str] = {}
    for core in cores:
        name = text(core.get("name"))
        aliases[compact_key(name)] = name
        acronym = text(core.get("acronym"))
        if acronym:
            aliases[compact_key(acronym)] = name
    matched = aliases.get(compact_key(value))
    if matched:
        return matched
    return text(cores[0].get("name")) if len(cores) == 1 else None


def _resource_items(
    value: object,
    *,
    source: str,
    cores: list[JsonObject],
    component: bool = False,
) -> tuple[list[JsonObject], int]:
    result: list[JsonObject] = []
    rejected = 0
    for raw in as_list(value):
        item = as_mapping(raw)
        quote = normalize_space(item.get("quote"))
        related = _related_core(item.get("related_to_core"), cores)
        if not related or not _quote_grounded(quote, source):
            rejected += 1
            continue
        record: JsonObject = {
            "keywords": _keywords(item.get("keywords")),
            "summary": normalize_space(item.get("summary")) or quote,
            "insight": normalize_space(item.get("insight")) or None,
            "quote": quote,
            "related_to_core": related,
        }
        if component:
            name = normalize_space(item.get("name"))
            record["name"] = name or None
        result.append(record)
    return result, rejected


def _validate_main(value: JsonObject, source: str) -> tuple[JsonObject, int]:
    ideation = as_mapping(value.get("ideation_resource"))
    cores: list[JsonObject] = []
    core_keys: set[str] = set()
    for raw in as_list(ideation.get("core_contributions")):
        core = _valid_core(as_mapping(raw), source)
        key = compact_key(core.get("name")) if core else ""
        if core is not None and key not in core_keys:
            core_keys.add(key)
            cores.append(core)
    if not cores:
        raise ValueError(
            "No grounded top-level core contribution survived validation"
        )
    rejected = len(as_list(ideation.get("core_contributions"))) - len(cores)
    resources: dict[str, object] = {"core_contributions": cores}
    for key in ("problems", "innovations", "limitations", "future_work"):
        items, count = _resource_items(
            ideation.get(key), source=source, cores=cores
        )
        resources[key] = items
        rejected += count
    components, count = _resource_items(
        ideation.get("components"), source=source, cores=cores, component=True
    )
    resources["components"] = components
    rejected += count
    aliases = {
        compact_key(core.get("name")): text(core.get("name")) for core in cores
    }
    relations: list[JsonObject] = []
    for raw in as_list(ideation.get("core_relations")):
        relation = as_mapping(raw)
        source_core = aliases.get(compact_key(relation.get("source")))
        target_core = aliases.get(compact_key(relation.get("target")))
        quote = normalize_space(relation.get("quote"))
        if (
            not source_core
            or not target_core
            or source_core == target_core
            or not _quote_grounded(quote, source)
        ):
            rejected += 1
            continue
        relations.append(
            {
                "source": source_core,
                "target": target_core,
                "relation": normalize_space(relation.get("relation")) or "related",
                "keywords": _keywords(relation.get("keywords")),
                "summary": normalize_space(relation.get("summary")) or quote,
                "insight": normalize_space(relation.get("insight")) or None,
                "quote": quote,
            }
        )
    resources["core_relations"] = relations
    metadata = as_mapping(value.get("metadata"))
    summary = as_mapping(metadata.get("structured_summary"))
    code_url = text(metadata.get("code_url"))
    if (
        not code_url.startswith(("http://", "https://"))
        or code_url.lower() not in source.lower()
    ):
        code_url = ""
    validated: JsonObject = {
        "metadata": {
            "domain": [
                normalize_space(item)
                for item in as_list(metadata.get("domain"))
                if normalize_space(item)
            ][:12],
            "paper_type": [
                normalize_space(item)
                for item in as_list(metadata.get("paper_type"))
                if normalize_space(item)
            ][:8],
            "structured_summary": {
                key: normalize_space(summary.get(key)) or None
                for key in ("problem", "method", "results")
            },
            "code_url": code_url or None,
        },
        "ideation_resource": resources,
    }
    return validated, rejected


def _hint_entities(hints: JsonObject, source: str, cores: list[JsonObject]) -> dict[str, list[JsonObject]]:
    result: dict[str, list[JsonObject]] = {"baselines": [], "datasets": []}
    for kind in ("baselines", "datasets"):
        seen: set[str] = set()
        for raw in as_list(hints.get(kind)):
            hint = as_mapping(raw)
            name = normalize_space(hint.get("name"))
            quote = normalize_space(hint.get("quote"))
            key = compact_key(name)
            related = text(cores[0].get("name")) if len(cores) == 1 else None
            if (
                not name
                or key in seen
                or key in {compact_key(value) for value in GENERIC_NAMES}
                or not related
                or not _name_grounded(name, None, source)
                or not _quote_grounded(quote, source)
            ):
                continue
            seen.add(key)
            result[kind].append(
                {
                    "name": name,
                    "acronym": None,
                    "related_to_core": related,
                    "keywords": [],
                    "summary": quote,
                    "metrics": None,
                    "insight": None,
                    "quote": quote,
                    "hint_source": hint.get("hint_source"),
                    "citation_paper_id": hint.get("citation_paper_id"),
                    "citation_title": hint.get("citation_title"),
                }
            )
    return result


def _merge_hint_entities(graph_data: JsonObject, hints: JsonObject, source: str, cores: list[JsonObject]) -> JsonObject:
    hint_entities = _hint_entities(hints, source, cores)
    merged: JsonObject = {}
    for kind in ("baselines", "datasets"):
        values = [as_mapping(value) for value in as_list(graph_data.get(kind))]
        seen = {compact_key(value.get("name")) for value in values if text(value.get("name"))}
        for hint in hint_entities[kind]:
            if compact_key(hint.get("name")) not in seen:
                values.append(hint)
                seen.add(compact_key(hint.get("name")))
        merged[kind] = values
    return merged


def _validate_graph(
    value: JsonObject, source: str, cores: list[JsonObject], hints: JsonObject | None = None
) -> tuple[JsonObject, int]:
    graph_data = as_mapping(value.get("graph_data"))
    graph_data = _merge_hint_entities(graph_data, hints or {}, source, cores)
    core_keys = {
        compact_key(core.get("name")) for core in cores
    } | {
        compact_key(core.get("acronym"))
        for core in cores
        if text(core.get("acronym"))
    }
    result: dict[str, object] = {}
    rejected = 0
    for kind in ("baselines", "datasets"):
        accepted: list[JsonObject] = []
        seen: set[str] = set()
        for raw in as_list(graph_data.get(kind)):
            item = as_mapping(raw)
            name = normalize_space(item.get("name"))
            acronym = normalize_space(item.get("acronym")) or None
            quote = normalize_space(item.get("quote"))
            key = compact_key(name)
            related = _related_core(item.get("related_to_core"), cores)
            if (
                not name
                or key in seen
                or key in {compact_key(value) for value in GENERIC_NAMES}
                or (kind == "baselines" and key in core_keys)
                or not related
                or not _name_grounded(name, acronym, source)
                or not _quote_grounded(quote, source)
            ):
                rejected += 1
                continue
            seen.add(key)
            metrics = [
                normalize_space(metric)
                for metric in as_list(item.get("metrics"))
                if normalize_space(metric)
            ]
            accepted.append(
                {
                    "name": name,
                    "acronym": acronym,
                    "related_to_core": related,
                    "keywords": _keywords(item.get("keywords")),
                    "summary": normalize_space(item.get("summary")) or quote,
                    "metrics": metrics or None,
                    "insight": normalize_space(item.get("insight")) or None,
                    "quote": quote,
                    "hint_source": item.get("hint_source"),
                    "citation_paper_id": item.get("citation_paper_id"),
                    "citation_title": item.get("citation_title"),
                }
            )
        result[kind] = accepted
    return {"graph_data": result}, rejected


def _calibration_entity(calibration: JsonObject, source: str, cores: list[JsonObject]) -> JsonObject | None:
    name = normalize_space(calibration.get("canonical_name") or calibration.get("name"))
    acronym = normalize_space(calibration.get("acronym")) or None
    quote = normalize_space(calibration.get("quote"))
    related = _related_core(calibration.get("related_to_core"), cores)
    if (
        not name
        or compact_key(name) in {compact_key(value) for value in GENERIC_NAMES}
        or not related
        or not _name_grounded(name, acronym, source)
        or not _quote_grounded(quote, source)
    ):
        return None
    metrics = [normalize_space(metric) for metric in as_list(calibration.get("metrics")) if normalize_space(metric)]
    return {
        "name": name,
        "acronym": acronym,
        "related_to_core": related,
        "keywords": _keywords(calibration.get("keywords")),
        "summary": normalize_space(calibration.get("summary")) or quote,
        "metrics": metrics or None,
        "insight": normalize_space(calibration.get("insight")) or None,
        "quote": quote,
        "calibration_added": True,
    }


def _calibrate(
    value: JsonObject,
    graph_data: JsonObject,
    cores: list[JsonObject],
    references: list[JsonObject],
    source: str,
) -> tuple[JsonObject, int]:
    reference_by_id = {
        text(reference.get("paperId")): reference
        for reference in references
        if text(reference.get("paperId"))
    }
    calibrations: dict[tuple[str, str], JsonObject] = {}
    additions: dict[str, list[JsonObject]] = {"baselines": [], "datasets": []}
    for raw in as_list(value.get("calibrations")):
        item = as_mapping(raw)
        kind = text(item.get("kind"))
        name = normalize_space(item.get("name") or item.get("canonical_name"))
        if kind not in {"baseline", "dataset"} or not name:
            continue
        plural = "baselines" if kind == "baseline" else "datasets"
        if (kind, compact_key(name)) not in calibrations:
            added = _calibration_entity(item, source, cores)
            if added is not None:
                additions[plural].append(added)
        calibrations[(kind, compact_key(name))] = item
    result: dict[str, object] = {}
    rejected = 0
    for plural, singular in (("baselines", "baseline"), ("datasets", "dataset")):
        accepted: list[JsonObject] = []
        seen: set[str] = set()
        for raw in as_list(graph_data.get(plural)):
            entity = as_mapping(raw)
            calibration = calibrations.get(
                (singular, compact_key(entity.get("name"))), {}
            )
            if calibration and calibration.get("keep") is False:
                rejected += 1
                continue
            canonical = normalize_space(calibration.get("canonical_name"))
            acronym = normalize_space(calibration.get("acronym"))
            if canonical and _name_grounded(canonical, acronym, source):
                entity["name"] = canonical
            if acronym and _name_grounded(entity.get("name"), acronym, source):
                entity["acronym"] = acronym
            related = _related_core(calibration.get("related_to_core"), cores)
            if related:
                entity["related_to_core"] = related
            citation_id = text(calibration.get("citation_paper_id") or entity.get("citation_paper_id"))
            reference = reference_by_id.get(citation_id)
            entity.update(
                {
                    "citation_paper_id": citation_id if reference else None,
                    "citation_paperId": citation_id if reference else None,
                    "citation_title": reference.get("title") if reference else entity.get("citation_title"),
                    "citation_venue": reference.get("venue") if reference else None,
                    "citation_year": reference.get("year") if reference else None,
                    "s2_metadata": reference if reference else None,
                    "urls": (
                        [reference.get("url")]
                        if reference and text(reference.get("url"))
                        else as_list(entity.get("urls"))
                    ),
                }
            )
            accepted.append(entity)
            seen.add(compact_key(entity.get("name")))
        for addition in additions[plural]:
            key = compact_key(addition.get("name"))
            if key in seen:
                continue
            calibration = calibrations.get((singular, key), {})
            citation_id = text(calibration.get("citation_paper_id"))
            reference = reference_by_id.get(citation_id)
            addition.update(
                {
                    "citation_paper_id": citation_id if reference else None,
                    "citation_paperId": citation_id if reference else None,
                    "citation_title": reference.get("title") if reference else None,
                    "citation_venue": reference.get("venue") if reference else None,
                    "citation_year": reference.get("year") if reference else None,
                    "s2_metadata": reference if reference else None,
                    "urls": [reference.get("url")] if reference and text(reference.get("url")) else [],
                }
            )
            accepted.append(addition)
            seen.add(key)
        result[plural] = accepted
    return result, rejected


def _pass_checkpoint(
    directory: Path,
    pass_name: str,
    *,
    input_sha256: str,
    value: JsonObject,
    response: JsonObject,
) -> None:
    atomic_write_json(
        directory / f"{pass_name}.json",
        {
            "schema_version": "xlab.llm_pass.v1",
            "prompt_version": PROMPT_VERSION,
            "input_sha256": input_sha256,
            "validated": True,
            "value": value,
            "response": response,
            "generated_at": utc_now(),
        },
    )


def _cached_pass(
    directory: Path, pass_name: str, input_sha256: str
) -> JsonObject | None:
    cached = as_mapping(read_json(directory / f"{pass_name}.json"))
    if (
        cached.get("schema_version") == "xlab.llm_pass.v1"
        and cached.get("prompt_version") == PROMPT_VERSION
        and cached.get("input_sha256") == input_sha256
        and cached.get("validated") is True
    ):
        return as_mapping(cached.get("value"))
    return None


def _call_validated(
    client: LlmClient,
    *,
    paper_id: str,
    pass_name: str,
    system: str,
    prompt: str,
    validator: object,
    validation_retries: int,
    max_tokens: int,
) -> tuple[JsonObject, JsonObject, int]:
    previous = ""
    last_error = "validation did not run"
    for attempt in range(validation_retries + 1):
        user = prompt
        if attempt:
            user += (
                "\n\nYour previous JSON failed deterministic validation:\n"
                + last_error
                + "\nReturn a corrected complete JSON object. Previous JSON:\n"
                + previous[:20_000]
            )
        value, response = client.chat(
            paper_id=paper_id,
            pass_name=f"{pass_name}-{attempt + 1}",
            system=system,
            user=user,
            max_tokens=max_tokens,
        )
        previous = json.dumps(value, ensure_ascii=False)
        try:
            validated, rejected = validator(value)
            return validated, response, rejected
        except ValueError as error:
            last_error = str(error)
    raise ValueError(
        f"{pass_name} failed deterministic validation after "
        f"{validation_retries + 1} attempts: {last_error}"
    )


def _extract_one(
    structure_path: Path,
    *,
    output_path: Path,
    llm_dir: Path,
    client: LlmClient,
    max_input_chars: int,
    validation_retries: int,
    max_tokens: int,
) -> JsonObject:
    structure = as_mapping(read_json(structure_path))
    paper_id = text(structure.get("paper_id"))
    mineru = as_mapping(structure.get("mineru_output"))
    markdown_path = Path(text(mineru.get("markdown")))
    full_text = markdown_path.read_text(encoding="utf-8", errors="replace")
    selected_text, truncated = _paper_text(full_text, max_input_chars)
    structure_sha256 = sha256_file(structure_path)
    selected_text_sha256 = hashlib.sha256(selected_text.encode("utf-8")).hexdigest()
    semantic_references_sha256 = _json_sha256(
        as_list(structure.get("semantic_references"))
    )
    references = [as_mapping(item) for item in as_list(structure.get("semantic_references"))]
    candidate_hints = _candidate_hints(full_text, references)
    candidate_hints_sha256 = _json_sha256(candidate_hints)
    structure_provenance = as_mapping(structure.get("provenance"))
    input_sha256 = hashlib.sha256(
        (
            PROMPT_VERSION
            + "\x1f"
            + EXTRACTOR_VERSION
            + "\x1f"
            + client.model
            + "\x1f"
            + client.endpoint
            + "\x1f"
            + str(client.enable_thinking)
            + "\x1f"
            + str(max_input_chars)
            + "\x1f"
            + text(structure_provenance.get("markdown_sha256"))
            + "\x1f"
            + structure_sha256
            + "\x1f"
            + selected_text_sha256
            + "\x1f"
            + semantic_references_sha256
            + "\x1f"
            + candidate_hints_sha256
            + "\x1f"
            + text(structure_provenance.get("paper_edges_sha256"))
            + "\x1f"
            + text(structure_provenance.get("source_manifest_sha256"))
            + "\x1f"
            + text(structure_provenance.get("paper_collect_manifest_sha256"))
        ).encode("utf-8")
    ).hexdigest()
    llm_dir.mkdir(parents=True, exist_ok=True)
    rejected = 0

    main = _cached_pass(llm_dir, "main", input_sha256)
    if main is None:
        main, response, count = _call_validated(
            client,
            paper_id=paper_id,
            pass_name="main",
            system=MAIN_SYSTEM,
            prompt=_main_prompt(structure, selected_text),
            validator=lambda value: _validate_main(value, full_text),
            validation_retries=validation_retries,
            max_tokens=max_tokens,
        )
        rejected += count
        _pass_checkpoint(
            llm_dir,
            "main",
            input_sha256=input_sha256,
            value=main,
            response=response,
        )
    cores = [
        as_mapping(value)
        for value in as_list(
            as_mapping(main.get("ideation_resource")).get("core_contributions")
        )
    ]

    graph = _cached_pass(llm_dir, "graph", input_sha256)
    if graph is None:
        graph, response, count = _call_validated(
            client,
            paper_id=paper_id,
            pass_name="graph",
            system=GRAPH_SYSTEM,
            prompt=_graph_prompt(structure, selected_text, cores, candidate_hints),
            validator=lambda value: _validate_graph(value, full_text, cores, candidate_hints),
            validation_retries=validation_retries,
            max_tokens=max_tokens,
        )
        rejected += count
        _pass_checkpoint(
            llm_dir,
            "graph",
            input_sha256=input_sha256,
            value=graph,
            response=response,
        )
    graph_data = as_mapping(graph.get("graph_data"))

    calibration = _cached_pass(llm_dir, "calibration", input_sha256)
    if calibration is None:
        calibration, response, count = _call_validated(
            client,
            paper_id=paper_id,
            pass_name="calibration",
            system=CALIBRATION_SYSTEM,
            prompt=_calibration_prompt(structure, cores, graph_data, selected_text, candidate_hints),
            validator=lambda value: (
                {
                    "graph_data": _calibrate(
                        value,
                        graph_data,
                        cores,
                        references,
                        full_text,
                    )[0]
                },
                _calibrate(
                    value,
                    graph_data,
                    cores,
                    references,
                    full_text,
                )[1],
            ),
            validation_retries=validation_retries,
            max_tokens=max_tokens,
        )
        rejected += count
        _pass_checkpoint(
            llm_dir,
            "calibration",
            input_sha256=input_sha256,
            value=calibration,
            response=response,
        )

    source_metadata = as_mapping(structure.get("metadata"))
    extracted_metadata = as_mapping(main.get("metadata"))
    result: JsonObject = {
        "schema_version": "xlab.paper_extraction.v2",
        "extractor": EXTRACTOR_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now(),
        "paper_id": paper_id,
        "source_venue": source_metadata.get("venue"),
        "pub_year": source_metadata.get("year"),
        "metadata": {
            **extracted_metadata,
            "title": source_metadata.get("title"),
            "authors": as_list(source_metadata.get("authors")),
            "abstract": source_metadata.get("abstract"),
            "tldr": source_metadata.get("tldr"),
            "external_ids": as_mapping(source_metadata.get("external_ids")),
            "urls": as_list(source_metadata.get("urls")),
        },
        "ideation_resource": as_mapping(main.get("ideation_resource")),
        "graph_data": as_mapping(calibration.get("graph_data")),
        "candidate_hints": candidate_hints,
        "provenance": {
            **structure_provenance,
            "structure_path": str(structure_path),
            "structure_sha256": structure_sha256,
            "selected_text_sha256": selected_text_sha256,
            "semantic_references_sha256": semantic_references_sha256,
            "candidate_hints_sha256": candidate_hints_sha256,
            "llm_max_input_chars": max_input_chars,
            "markdown_path": str(markdown_path),
            "content_list_path": mineru.get("content_list_json"),
            "middle_json_path": mineru.get("middle_json"),
        },
        "quality": {
            "llm_validated": True,
            "grounded": True,
            "model": client.model,
            "enable_thinking": client.enable_thinking,
            "endpoint": client.endpoint,
            "input_truncated": truncated,
            "selected_input_chars": len(selected_text),
            "full_markdown_chars": len(full_text),
            "rejected_items": rejected,
            "passes": ["main", "graph", "calibration"],
        },
    }
    atomic_write_json(output_path, result)
    return result


def _cache_valid(
    checkpoint: Path,
    structure_path: Path,
    structure: JsonObject,
    model: str,
    endpoint: str,
    max_input_chars: int,
    enable_thinking: bool | None = None,
) -> JsonObject | None:
    cached = as_mapping(read_json(checkpoint))
    provenance = as_mapping(cached.get("provenance"))
    source_provenance = as_mapping(structure.get("provenance"))
    quality = as_mapping(cached.get("quality"))
    current_structure_sha256 = sha256_file(structure_path) if structure_path.is_file() else ""
    if (
        cached.get("schema_version") == "xlab.paper_extraction.v2"
        and cached.get("extractor") == EXTRACTOR_VERSION
        and cached.get("prompt_version") == PROMPT_VERSION
        and quality.get("llm_validated") is True
        and text(provenance.get("markdown_sha256"))
        == text(source_provenance.get("markdown_sha256"))
        and text(provenance.get("source_manifest_sha256"))
        == text(source_provenance.get("source_manifest_sha256"))
        and text(provenance.get("paper_collect_manifest_sha256"))
        == text(source_provenance.get("paper_collect_manifest_sha256"))
        and text(provenance.get("paper_edges_sha256"))
        == text(source_provenance.get("paper_edges_sha256"))
        and text(provenance.get("semantic_references_sha256"))
        == text(source_provenance.get("semantic_references_sha256"))
        and text(provenance.get("candidate_hints_sha256"))
        == _json_sha256(
            _candidate_hints(
                Path(text(as_mapping(structure.get("mineru_output")).get("markdown"))).read_text(
                    encoding="utf-8", errors="replace"
                ),
                [as_mapping(item) for item in as_list(structure.get("semantic_references"))],
            )
        )
        and text(provenance.get("structure_sha256")) == current_structure_sha256
        and int(provenance.get("llm_max_input_chars") or 0) == max_input_chars
        and text(quality.get("model")) == model
        and text(quality.get("endpoint")) == endpoint
        and quality.get("enable_thinking") == enable_thinking
    ):
        return cached
    return None


def extract_documents(
    run_dir: Path,
    structure_manifest: JsonObject,
    *,
    client: LlmClient,
    workers: int,
    max_input_chars: int,
    validation_retries: int,
    max_tokens: int,
) -> JsonObject:
    paths = artifact_paths(run_dir)
    paths["extractions"].mkdir(parents=True, exist_ok=True)
    records = [
        as_mapping(value)
        for value in as_list(structure_manifest.get("structures"))
        if as_mapping(value).get("status") == "structured"
    ]
    successes: dict[str, Path] = {}
    failures: list[JsonObject] = []
    jobs = {}
    reused = 0
    service_paused = False
    migration_path = paths["artifacts"] / "endpoint-migration.json"
    request = as_mapping(read_json(paths["request"]))
    accepted: JsonObject = {}
    if migration_path.is_file() and request.get("endpoint_migration_sha256") == sha256_file(migration_path):
        migration = as_mapping(read_json(migration_path))
        if migration.get("target_endpoint") == client.endpoint and migration.get("model") == client.model:
            accepted = as_mapping(migration.get("completed_records"))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for record in records:
            paper_id = text(record.get("paper_id"))
            structure_path = Path(text(record.get("path")))
            structure = as_mapping(read_json(structure_path))
            checkpoint = _checkpoint_path(paths["extractions"], paper_id)
            producer_endpoint = client.endpoint
            retained = as_mapping(accepted.get(paper_id))
            if checkpoint.is_file() and retained.get("sha256") == sha256_file(checkpoint):
                producer_endpoint = text(retained.get("producer_endpoint"))
            if _cache_valid(
                checkpoint,
                structure_path,
                structure,
                client.model,
                producer_endpoint,
                max_input_chars,
                client.enable_thinking,
            ):
                successes[paper_id] = checkpoint
                reused += 1
                continue
            future = executor.submit(
                _extract_one,
                structure_path,
                output_path=checkpoint,
                llm_dir=paths["llm"] / checkpoint.stem,
                client=client,
                max_input_chars=max_input_chars,
                validation_retries=validation_retries,
                max_tokens=max_tokens,
            )
            jobs[future] = (paper_id, checkpoint)
        for future in as_completed(jobs):
            paper_id, checkpoint = jobs[future]
            try:
                future.result()
                successes[paper_id] = checkpoint
            except CancelledError:
                continue
            except LlmServiceUnavailable:
                service_paused = True
                for queued in jobs:
                    queued.cancel()
            except Exception as error:
                failure = {
                    "phase": "extract",
                    "paper_id": paper_id,
                    "error": str(error)[:2000],
                    "recorded_at": utc_now(),
                }
                failures.append(failure)
                append_jsonl(paths["failures"], failure)
    extractions: list[JsonObject] = [
        {"paper_id": paper_id, "path": str(successes[paper_id]), "status": "extracted"}
        for paper_id in sorted(successes)
    ]
    extractions.extend(
        {
            "paper_id": text(failure.get("paper_id")),
            "path": None,
            "status": "failed",
            "error": failure.get("error"),
        }
        for failure in sorted(failures, key=lambda item: text(item.get("paper_id")))
    )
    completed_ids = {text(record.get("paper_id")) for record in extractions}
    pending_ids = sorted(text(record.get("paper_id")) for record in records if text(record.get("paper_id")) not in completed_ids)
    extractions.extend({"paper_id": paper_id, "path": None, "status": "pending"} for paper_id in pending_ids)
    manifest: JsonObject = {
        "schema_version": "xlab.paper_extractions.v2",
        "extractor": EXTRACTOR_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now(),
        "structures_path": str(paths["structure_manifest"]),
        "model": client.model,
        "extractions": extractions,
        "summary": {
            "eligible": len(records),
            "extracted": len(successes),
            "failed": len(failures),
            "reused": reused,
            "pending": len(pending_ids),
        },
    }
    atomic_write_json(paths["extraction_manifest"], manifest)
    if service_paused:
        raise LlmServiceUnavailable("LLM service unavailable; validated checkpoints saved and remaining papers left pending. Resume extraction after service recovery.")
    return manifest


def migrate_endpoint(run_dir: Path, api_url: str, model: str) -> JsonObject:
    """Retain completed records under their original provenance during an explicit handoff."""
    paths = artifact_paths(run_dir)
    request = as_mapping(read_json(paths["request"]))
    llm = as_mapping(request.get("llm"))
    if not request or model != llm.get("model"):
        raise ValueError("Endpoint migration requires the same configured model.")
    parsed_url = urlsplit(api_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("The destination must be an absolute HTTP(S) API URL.")
    target = completion_url(api_url)
    source = completion_url(text(llm.get("api_url")))
    migration_path = paths["artifacts"] / "endpoint-migration.json"
    previous: JsonObject = {}
    if migration_path.is_file() and request.get("endpoint_migration_sha256") == sha256_file(migration_path):
        previous_migration = as_mapping(read_json(migration_path))
        if previous_migration.get("target_endpoint") == source and previous_migration.get("model") == model:
            previous = as_mapping(previous_migration.get("completed_records"))
    completed: JsonObject = {}
    structures = as_mapping(read_json(paths["structure_manifest"]))
    for value in as_list(structures.get("structures")):
        record = as_mapping(value)
        if record.get("status") != "structured":
            continue
        paper_id = text(record.get("paper_id"))
        checkpoint = _checkpoint_path(paths["extractions"], paper_id)
        if not checkpoint.is_file():
            continue
        digest = sha256_file(checkpoint)
        quality = as_mapping(as_mapping(read_json(checkpoint)).get("quality"))
        producer = text(quality.get("endpoint"))
        previous_record = as_mapping(previous.get(paper_id))
        if producer != source and not (
            previous_record.get("sha256") == digest
            and previous_record.get("producer_endpoint") == producer
        ):
            continue
        structure_path = Path(text(record.get("path")))
        if quality.get("grounded") is not True or quality.get("passes") != ["main", "graph", "calibration"]:
            continue
        if _cache_valid(checkpoint, structure_path, as_mapping(read_json(structure_path)), model, producer,
                        int(request["llm_max_input_chars"]), False if request.get("llm_disable_thinking") else None):
            completed[paper_id] = {"path": str(checkpoint), "sha256": digest, "producer_endpoint": producer}
    migration: JsonObject = {
        "schema_version": "xlab.endpoint_migration.v1", "created_at": utc_now(),
        "source_endpoint": source, "target_endpoint": target, "model": model,
        "source_request": request, "completed_records": completed,
    }
    history = paths["artifacts"] / "endpoint-migrations" / f"{_json_sha256(migration)}.json"
    atomic_write_json(history, migration)
    atomic_write_json(migration_path, migration)
    llm["api_url"] = public_api_url(api_url)
    request["llm"] = llm
    request["endpoint_migration_sha256"] = sha256_file(migration_path)
    atomic_write_json(paths["request"], request)
    return {"source_endpoint": source, "target_endpoint": target, "retained_completed": len(completed), "history_path": str(history)}
