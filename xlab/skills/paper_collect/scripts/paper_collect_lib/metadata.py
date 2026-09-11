from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from difflib import SequenceMatcher
from pathlib import Path

from .common import (
    JsonObject,
    artifact_paths,
    as_list,
    as_mapping,
    atomic_write_json,
    atomic_write_jsonl,
    extract_external_ids,
    extract_ids_from_text,
    find_named_list,
    find_paper_mapping,
    identity_keys,
    integer,
    is_mapping,
    normalize_arxiv,
    normalize_doi,
    normalize_space,
    normalize_title,
    number,
    read_json,
    read_jsonl,
    relative_to_run,
    run_lock,
    stable_paper_id,
    text,
    topic_tokens,
    utc_now,
)
from .providers import (
    PAPER_FIELDS,
    RECOMMENDATION_FIELDS,
    RELATION_FIELDS,
    S2_CITATIONS,
    S2_GET_PAPER,
    S2_RECOMMENDATIONS,
    S2_REFERENCES,
    S2_SEARCH,
)


def _authors(value: object) -> list[JsonObject]:
    result: list[JsonObject] = []
    seen: set[str] = set()
    for author in as_list(value):
        if is_mapping(author):
            source = as_mapping(author)
            name = normalize_space(source.get("name"))
            author_id = text(source.get("authorId") or source.get("author_id"))
        else:
            name = normalize_space(author)
            author_id = ""
        if not name:
            continue
        key = author_id.lower() or name.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append({"author_id": author_id or None, "name": name})
    return result


def _discovery(
    *,
    provider: str,
    kind: str,
    operation: str,
    query: str = "",
    purpose: str = "",
    rank: int | None = None,
    url: str = "",
    seed_id: str = "",
) -> JsonObject:
    return {
        "provider": provider,
        "kind": kind,
        "operation": operation,
        "query": query or None,
        "purpose": purpose or None,
        "rank": rank,
        "url": url or None,
        "seed_id": seed_id or None,
    }


def _pdf_candidates(
    *,
    open_access_pdf: object,
    external_ids: Mapping[str, object],
    discovery_url: str = "",
) -> list[JsonObject]:
    candidates: list[JsonObject] = []
    seen: set[str] = set()

    def add(url: object, source: str, priority: int) -> None:
        candidate_url = text(url)
        if (
            not candidate_url
            or not candidate_url.lower().startswith(("https://", "http://"))
            or candidate_url in seen
        ):
            return
        seen.add(candidate_url)
        candidates.append(
            {
                "url": candidate_url,
                "source": source,
                "priority": priority,
            }
        )

    if is_mapping(open_access_pdf):
        add(as_mapping(open_access_pdf).get("url"), "semantic_scholar_open_access", 1)
    elif isinstance(open_access_pdf, str):
        add(open_access_pdf, "semantic_scholar_open_access", 1)

    arxiv_id = text(external_ids.get("ArXiv"))
    if arxiv_id:
        normalized = normalize_arxiv(arxiv_id)
        add(f"https://arxiv.org/pdf/{normalized}.pdf", "arxiv_external_id", 2)
        add(
            f"https://export.arxiv.org/pdf/{normalized}.pdf",
            "arxiv_external_id_fallback",
            3,
        )

    if discovery_url.lower().split("?", 1)[0].endswith(".pdf"):
        add(discovery_url, "tavily_direct_pdf", 4)
    return candidates


def _normalise_s2_paper(raw: object, discovery: JsonObject) -> JsonObject:
    source = as_mapping(raw)
    external_ids = extract_external_ids(source.get("externalIds"))
    paper_id = text(source.get("paperId"))
    corpus_id = integer(source.get("corpusId"))
    if corpus_id is not None and "CorpusId" not in external_ids:
        external_ids["CorpusId"] = str(corpus_id)
    open_access_pdf = source.get("openAccessPdf")
    tldr = source.get("tldr")
    if is_mapping(tldr):
        tldr = as_mapping(tldr).get("text")
    paper: JsonObject = {
        "paper_id": paper_id or None,
        "title": normalize_space(source.get("title")),
        "authors": _authors(source.get("authors")),
        "year": integer(source.get("year")),
        "publication_date": text(source.get("publicationDate")) or None,
        "venue": normalize_space(source.get("venue")) or None,
        "abstract": normalize_space(source.get("abstract")) or None,
        "tldr": normalize_space(tldr) or None,
        "publication_types": [
            normalize_space(item)
            for item in as_list(source.get("publicationTypes"))
            if normalize_space(item)
        ],
        "fields_of_study": [
            normalize_space(item)
            for item in as_list(source.get("fieldsOfStudy"))
            if normalize_space(item)
        ],
        "citation_count": integer(source.get("citationCount")),
        "influential_citation_count": integer(
            source.get("influentialCitationCount")
        ),
        "reference_count": integer(source.get("referenceCount")),
        "external_ids": external_ids,
        "url": text(source.get("url")) or None,
        "open_access_pdf": open_access_pdf if is_mapping(open_access_pdf) else None,
        "pdf_candidates": _pdf_candidates(
            open_access_pdf=open_access_pdf,
            external_ids=external_ids,
        ),
        "discovery": [discovery],
        "seed_ids": [discovery["seed_id"]] if discovery.get("seed_id") else [],
        "is_seed": discovery.get("kind") == "tavily_seed"
        or (
            discovery.get("kind") == "semantic_search"
            and integer(discovery.get("rank")) is not None
            and (integer(discovery.get("rank")) or 0) <= 10
        ),
        "metadata_sources": ["semantic_scholar"],
    }
    paper["_identity_keys"] = identity_keys(paper)
    return paper


def _normalise_tavily_result(
    raw: object,
    discovery: JsonObject,
) -> JsonObject:
    source = as_mapping(raw)
    title_value = normalize_space(source.get("title"))
    url = text(source.get("url") or source.get("link"))
    content = normalize_space(source.get("content") or source.get("snippet"))
    external_ids = extract_ids_from_text(" ".join([url, title_value, content]))
    paper: JsonObject = {
        "paper_id": None,
        "title": title_value,
        "authors": [],
        "year": None,
        "publication_date": None,
        "venue": None,
        "abstract": None,
        "tavily_snippet": content or None,
        "tldr": None,
        "publication_types": [],
        "fields_of_study": [],
        "citation_count": None,
        "influential_citation_count": None,
        "reference_count": None,
        "external_ids": external_ids,
        "url": url or None,
        "open_access_pdf": None,
        "pdf_candidates": _pdf_candidates(
            open_access_pdf=None,
            external_ids=external_ids,
            discovery_url=url,
        ),
        "discovery": [discovery],
        "seed_ids": [],
        "is_seed": True,
        "metadata_sources": ["tavily"],
        "tavily_score": number(source.get("score")),
    }
    paper["_identity_keys"] = identity_keys(paper)
    return paper


def _merge_unique_objects(
    left: object,
    right: object,
    keys: tuple[str, ...],
) -> list[JsonObject]:
    result: list[JsonObject] = []
    seen: set[str] = set()
    for value in [*as_list(left), *as_list(right)]:
        item = as_mapping(value)
        if not item:
            continue
        signature = "\0".join(text(item.get(key)).lower() for key in keys)
        if not signature.strip("\0"):
            signature = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(item)
    return result


def _merge_paper(target: JsonObject, incoming: Mapping[str, object]) -> None:
    target["paper_id"] = target.get("paper_id") or incoming.get("paper_id")
    target["title"] = target.get("title") or incoming.get("title")
    target["year"] = target.get("year") or incoming.get("year")
    target["publication_date"] = target.get("publication_date") or incoming.get(
        "publication_date"
    )
    target["venue"] = target.get("venue") or incoming.get("venue")
    target["url"] = target.get("url") or incoming.get("url")
    target["tldr"] = target.get("tldr") or incoming.get("tldr")
    target["tavily_snippet"] = target.get("tavily_snippet") or incoming.get(
        "tavily_snippet"
    )
    target["open_access_pdf"] = target.get("open_access_pdf") or incoming.get(
        "open_access_pdf"
    )
    incoming_abstract = text(incoming.get("abstract"))
    if len(incoming_abstract) > len(text(target.get("abstract"))):
        target["abstract"] = incoming_abstract

    target["authors"] = _merge_unique_objects(
        target.get("authors"),
        incoming.get("authors"),
        ("author_id", "name"),
    )
    target["discovery"] = _merge_unique_objects(
        target.get("discovery"),
        incoming.get("discovery"),
        ("operation", "query", "rank", "url", "seed_id"),
    )
    target["pdf_candidates"] = _merge_unique_objects(
        target.get("pdf_candidates"),
        incoming.get("pdf_candidates"),
        ("url",),
    )

    for key in ("publication_types", "fields_of_study", "seed_ids", "metadata_sources"):
        values: list[object] = []
        seen: set[str] = set()
        for item in [*as_list(target.get(key)), *as_list(incoming.get(key))]:
            item_text = text(item)
            if not item_text or item_text.lower() in seen:
                continue
            seen.add(item_text.lower())
            values.append(item)
        target[key] = values

    external_ids = extract_external_ids(target.get("external_ids"))
    external_ids.update(extract_external_ids(incoming.get("external_ids")))
    target["external_ids"] = external_ids
    for key in (
        "citation_count",
        "influential_citation_count",
        "reference_count",
        "tavily_score",
    ):
        values = [
            value
            for value in (number(target.get(key)), number(incoming.get(key)))
            if value is not None
        ]
        if values:
            target[key] = max(values)
    target["is_seed"] = bool(target.get("is_seed") or incoming.get("is_seed"))
    keys = {
        text(item)
        for item in [
            *as_list(target.get("_identity_keys")),
            *as_list(incoming.get("_identity_keys")),
            *identity_keys(target),
        ]
        if text(item)
    }
    target["_identity_keys"] = sorted(keys)


def _deduplicate(candidates: list[JsonObject]) -> list[JsonObject]:
    groups: list[JsonObject] = []
    active: list[bool] = []
    owners: dict[str, int] = {}

    for candidate in candidates:
        keys = {
            text(key)
            for key in [
                *as_list(candidate.get("_identity_keys")),
                *identity_keys(candidate),
            ]
            if text(key)
        }
        matches = sorted({owners[key] for key in keys if key in owners and active[owners[key]]})
        if not matches:
            target_index = len(groups)
            groups.append(candidate)
            active.append(True)
        else:
            target_index = matches[0]
            _merge_paper(groups[target_index], candidate)
            for other_index in matches[1:]:
                _merge_paper(groups[target_index], groups[other_index])
                active[other_index] = False
                for key, owner in list(owners.items()):
                    if owner == other_index:
                        owners[key] = target_index
        for key in [
            *as_list(groups[target_index].get("_identity_keys")),
            *identity_keys(groups[target_index]),
        ]:
            key_text = text(key)
            if key_text:
                owners[key_text] = target_index

    return [paper for index, paper in enumerate(groups) if active[index]]


def _payloads(record: Mapping[str, object]) -> list[object]:
    payload = record.get("payload")
    return [payload] if payload is not None else []


def _has_named_list(value: object, names: set[str]) -> bool:
    mapping = as_mapping(value)
    if any(isinstance(mapping.get(name), list) for name in names):
        return True
    return any(
        _has_named_list(mapping.get(key), names)
        for key in ("result", "data", "payload", "response")
        if mapping.get(key) is not None
    )


def _content_texts(record: Mapping[str, object]) -> list[str]:
    result: list[str] = []
    for item in as_list(record.get("content")):
        value = as_mapping(item).get("text")
        if isinstance(value, str) and value.strip():
            result.append(value.strip())
    return result


def _label_value(source: str, labels: tuple[str, ...]) -> str:
    alternatives = "|".join(re.escape(label) for label in labels)
    match = re.search(
        rf"^[ \t]*(?:{alternatives})[ \t]*[:：][ \t]*(.*?)[ \t]*$",
        source,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def _label_integer(source: str, labels: tuple[str, ...]) -> int | None:
    value = _label_value(source, labels)
    match = re.search(r"-?\d[\d,]*", value)
    return integer(match.group(0).replace(",", "")) if match else None


def _text_authors(value: str) -> list[JsonObject]:
    return _authors(
        [
            author
            for author in re.split(r"\s*(?:,|，|;|；|\band\b)\s*", value)
            if author.strip()
        ]
    )


def _text_external_ids(source: str) -> JsonObject:
    result: JsonObject = {}
    for canonical, labels in (
        ("DOI", ("DOI",)),
        ("ArXiv", ("ArXiv", "arXiv")),
        ("PubMed", ("PMID", "PubMed")),
        ("CorpusId", ("Corpus ID", "CorpusId")),
    ):
        value = _label_value(source, labels)
        if value:
            result[canonical] = value
    return extract_external_ids(result)


def _text_paper(source: str, *, title: str = "") -> JsonObject:
    paper_title = title or _label_value(source, ("标题", "Title"))
    paper_id = _label_value(
        source,
        ("Semantic Scholar ID", "S2 ID", "ID"),
    )
    abstract = ""
    abstract_match = re.search(
        (
            r"^[ \t]*(?:摘要|Abstract)[ \t]*[:：][ \t]*\n?"
            r"(.*?)(?=^[ \t]*(?:Semantic Scholar ID|S2 ID|ID|DOI|"
            r"ArXiv|arXiv|PMID|PubMed|PDF|URL)[ \t]*[:：]|\Z)"
        ),
        source,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if abstract_match:
        abstract = normalize_space(abstract_match.group(1))
    pdf_url = _label_value(source, ("PDF", "Open Access PDF"))
    open_access_pdf = {"url": pdf_url} if pdf_url else None
    return {
        "paperId": paper_id or None,
        "title": normalize_space(paper_title),
        "authors": _text_authors(_label_value(source, ("作者", "Authors"))),
        "year": _label_integer(source, ("年份", "Year")),
        "venue": normalize_space(
            _label_value(source, ("发表于", "期刊/会议", "Venue"))
        )
        or None,
        "abstract": abstract or None,
        "citationCount": _label_integer(source, ("引用数", "引用", "Citations")),
        "referenceCount": _label_integer(
            source,
            ("参考文献数", "References"),
        ),
        "externalIds": _text_external_ids(source),
        "openAccessPdf": open_access_pdf,
    }


def _text_paper_list(source: str) -> list[JsonObject]:
    matches = list(
        re.finditer(
            r"^[ \t]*(\d+)\.[ \t]+(.+?)[ \t]*$",
            source,
            flags=re.MULTILINE,
        )
    )
    result: list[JsonObject] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        paper = _text_paper(
            source[match.start() : end],
            title=match.group(2),
        )
        if text(paper.get("paperId")) and text(paper.get("title")):
            result.append(paper)
    return result


def _query_purposes(plan: object) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in as_list(as_mapping(plan).get("queries")):
        item = as_mapping(entry)
        arguments = as_mapping(item.get("arguments"))
        query = normalize_space(arguments.get("query")).lower()
        if query:
            result[query] = text(item.get("purpose")) or "topic"
    return result


def _provider_signature(record: Mapping[str, object]) -> str:
    operation = text(record.get("operation"))
    arguments = as_mapping(record.get("input"))
    return _call_signature(operation, arguments)


def _latest_provider_records(records: list[object]) -> list[object]:
    latest: dict[str, object] = {}
    parse_errors: list[object] = []
    for raw_record in records:
        record = as_mapping(raw_record)
        if record.get("_parse_error"):
            parse_errors.append(raw_record)
            continue
        if text(record.get("operation")):
            latest[_provider_signature(record)] = raw_record
    return [*parse_errors, *latest.values()]


def _title_resolution_matches(expected: object, actual: object) -> bool:
    expected_title = normalize_title(expected)
    actual_title = normalize_title(actual)
    if not expected_title or not actual_title:
        return False
    if expected_title == actual_title:
        return True
    expected_tokens = set(expected_title.split())
    actual_tokens = set(actual_title.split())
    if len(expected_tokens) < 4 or len(actual_tokens) < 4:
        return False
    overlap = len(expected_tokens & actual_tokens)
    containment = overlap / min(len(expected_tokens), len(actual_tokens))
    sequence_ratio = SequenceMatcher(
        None,
        expected_title,
        actual_title,
    ).ratio()
    return sequence_ratio >= 0.88 or (overlap >= 4 and containment >= 0.85)


def _input_paper_id(record: Mapping[str, object]) -> str:
    input_value = as_mapping(record.get("input"))
    raw_id = text(input_value.get("paperId") or input_value.get("paper_id"))
    if not raw_id:
        return ""
    lowered = raw_id.lower()
    if lowered.startswith("doi:"):
        return f"doi:{normalize_doi(raw_id[4:])}"
    if lowered.startswith("arxiv:"):
        return f"arxiv:{normalize_arxiv(raw_id[6:]).lower()}"
    return f"s2:{raw_id.lower()}"


def _provider_records(
    records: list[object],
    query_purposes: Mapping[str, str],
) -> tuple[list[JsonObject], list[JsonObject], list[JsonObject]]:
    candidates: list[JsonObject] = []
    edges: list[JsonObject] = []
    failures: list[JsonObject] = []

    for raw_record in records:
        record = as_mapping(raw_record)
        if record.get("_parse_error"):
            failures.append(
                {
                    "stage": "provider_log",
                    "error": record.get("_parse_error"),
                    "raw": record.get("_raw"),
                }
            )
            continue
        operation = text(record.get("operation"))
        if not operation:
            continue
        if record.get("is_error") is True:
            failures.append(
                {
                    "stage": "provider_call",
                    "operation": operation,
                    "input": record.get("input"),
                    "error": text(record.get("error")) or "Provider API returned an error",
                }
            )
            continue
        input_value = as_mapping(record.get("input"))
        query = normalize_space(input_value.get("query"))
        purpose = query_purposes.get(query.lower(), "topic")
        payloads = _payloads(record)
        parsed = False

        if operation == "tavily.search":
            for payload in payloads:
                results = find_named_list(payload, {"results"})
                if not results:
                    parsed = parsed or _has_named_list(payload, {"results"})
                    continue
                parsed = True
                for rank, result in enumerate(results, 1):
                    result_mapping = as_mapping(result)
                    discovery = _discovery(
                        provider="tavily",
                        kind="tavily_seed",
                        operation=operation,
                        query=query,
                        purpose=purpose,
                        rank=rank,
                        url=text(result_mapping.get("url") or result_mapping.get("link")),
                    )
                    candidates.append(_normalise_tavily_result(result, discovery))

        elif operation == S2_SEARCH:
            expected_title = normalize_space(
                input_value.get("expectedTitle")
                or input_value.get("expected_title")
            )
            exact_resolution = (
                text(input_value.get("matchMode") or input_value.get("match_mode"))
                == "exact_title"
                or bool(expected_title)
                or (
                    (integer(input_value.get("limit")) or 100) <= 3
                    and query.lower() not in query_purposes
                )
            )
            if exact_resolution and not expected_title:
                expected_title = query
            matched_resolution = False
            saw_results = False
            for payload in payloads:
                results = find_named_list(payload, {"data", "papers", "results"})
                if not results:
                    parsed = parsed or _has_named_list(
                        payload,
                        {"data", "papers", "results"},
                    )
                    continue
                parsed = True
                saw_results = True
                for rank, result in enumerate(results, 1):
                    paper = find_paper_mapping(result)
                    if not paper:
                        continue
                    if exact_resolution and not _title_resolution_matches(
                        expected_title,
                        paper.get("title"),
                    ):
                        continue
                    matched_resolution = True
                    discovery = _discovery(
                        provider="semantic_scholar",
                        kind=(
                            "metadata_resolution"
                            if exact_resolution
                            else "semantic_search"
                        ),
                        operation=operation,
                        query=query,
                        purpose=(
                            "exact_title_resolution"
                            if exact_resolution
                            else purpose
                        ),
                        rank=rank,
                    )
                    candidates.append(_normalise_s2_paper(paper, discovery))
            if exact_resolution and saw_results and not matched_resolution:
                failures.append(
                    {
                        "stage": "title_resolution",
                        "operation": operation,
                        "input": input_value,
                        "error": (
                            "Semantic Scholar search returned no sufficiently "
                            "similar title."
                        ),
                    }
                )

        elif operation == S2_CITATIONS:
            seed_id = _input_paper_id(record)
            for payload in payloads:
                results = find_named_list(payload, {"data", "citations", "results"})
                if not results:
                    parsed = parsed or _has_named_list(
                        payload,
                        {"data", "citations", "results"},
                    )
                    continue
                parsed = True
                for rank, result in enumerate(results, 1):
                    item = as_mapping(result)
                    paper = find_paper_mapping(
                        item.get("citingPaper") or item.get("paper") or result
                    )
                    if not paper:
                        continue
                    discovery = _discovery(
                        provider="semantic_scholar",
                        kind="citation",
                        operation=operation,
                        rank=rank,
                        seed_id=seed_id,
                    )
                    normalized = _normalise_s2_paper(paper, discovery)
                    candidates.append(normalized)
                    related_id = stable_paper_id(normalized)
                    if related_id and seed_id:
                        edges.append(
                            {
                                "source_paper_id": related_id,
                                "target_paper_id": seed_id,
                                "relation": "cites",
                                "discovered_from": "semantic_scholar_citations",
                                "seed_id": seed_id,
                            }
                        )

        elif operation == S2_REFERENCES:
            seed_id = _input_paper_id(record)
            for payload in payloads:
                results = find_named_list(payload, {"data", "references", "results"})
                if not results:
                    parsed = parsed or _has_named_list(
                        payload,
                        {"data", "references", "results"},
                    )
                    continue
                parsed = True
                for rank, result in enumerate(results, 1):
                    item = as_mapping(result)
                    paper = find_paper_mapping(
                        item.get("citedPaper") or item.get("paper") or result
                    )
                    if not paper:
                        continue
                    discovery = _discovery(
                        provider="semantic_scholar",
                        kind="reference",
                        operation=operation,
                        rank=rank,
                        seed_id=seed_id,
                    )
                    normalized = _normalise_s2_paper(paper, discovery)
                    candidates.append(normalized)
                    related_id = stable_paper_id(normalized)
                    if related_id and seed_id:
                        edges.append(
                            {
                                "source_paper_id": seed_id,
                                "target_paper_id": related_id,
                                "relation": "cites",
                                "discovered_from": "semantic_scholar_references",
                                "seed_id": seed_id,
                            }
                        )

        elif operation == S2_RECOMMENDATIONS:
            positive_ids = [
                f"s2:{text(value).lower()}"
                for value in as_list(input_value.get("positivePaperIds"))
                if text(value)
            ]
            for payload in payloads:
                results = find_named_list(
                    payload,
                    {"recommendedPapers", "data", "papers", "results"},
                )
                if not results:
                    parsed = parsed or _has_named_list(
                        payload,
                        {"recommendedPapers", "data", "papers", "results"},
                    )
                    continue
                parsed = True
                for rank, result in enumerate(results, 1):
                    paper = find_paper_mapping(result)
                    if not paper:
                        continue
                    seed_id = positive_ids[0] if len(positive_ids) == 1 else ""
                    discovery = _discovery(
                        provider="semantic_scholar",
                        kind="recommendation",
                        operation=operation,
                        rank=rank,
                        seed_id=seed_id,
                    )
                    normalized = _normalise_s2_paper(paper, discovery)
                    candidates.append(normalized)
                    related_id = stable_paper_id(normalized)
                    for positive_id in positive_ids:
                        if related_id:
                            edges.append(
                                {
                                    "source_paper_id": positive_id,
                                    "target_paper_id": related_id,
                                    "relation": "recommended_with",
                                    "discovered_from": "semantic_scholar_recommendations",
                                    "seed_id": positive_id,
                                }
                            )

        elif operation == S2_GET_PAPER:
            for payload in payloads:
                paper = find_paper_mapping(payload)
                if not paper:
                    continue
                parsed = True
                discovery = _discovery(
                    provider="semantic_scholar",
                    kind="metadata_enrichment",
                    operation=operation,
                    seed_id=_input_paper_id(record),
                )
                candidates.append(_normalise_s2_paper(paper, discovery))

        if not parsed and operation.startswith("semantic_scholar."):
            if operation == S2_GET_PAPER:
                text_papers = [
                    paper
                    for body in _content_texts(record)
                    if text((paper := _text_paper(body)).get("paperId"))
                    and text(paper.get("title"))
                ]
            else:
                text_papers = [
                    paper
                    for body in _content_texts(record)
                    for paper in _text_paper_list(body)
                ]
            seed_id = _input_paper_id(record)
            positive_ids = [
                f"s2:{text(value).lower()}"
                for value in as_list(input_value.get("positivePaperIds"))
                if text(value)
            ]
            for rank, paper in enumerate(text_papers, 1):
                if operation == S2_CITATIONS:
                    kind = "citation"
                elif operation == S2_REFERENCES:
                    kind = "reference"
                elif operation == S2_RECOMMENDATIONS:
                    kind = "recommendation"
                elif operation == S2_GET_PAPER:
                    kind = "metadata_enrichment"
                else:
                    kind = "semantic_search"
                discovery_seed = (
                    positive_ids[0]
                    if kind == "recommendation" and len(positive_ids) == 1
                    else seed_id
                    if kind in {"citation", "reference", "metadata_enrichment"}
                    else ""
                )
                discovery = _discovery(
                    provider="semantic_scholar",
                    kind=kind,
                    operation=operation,
                    query=query if kind == "semantic_search" else "",
                    purpose=purpose,
                    rank=rank,
                    seed_id=discovery_seed,
                )
                normalized = _normalise_s2_paper(paper, discovery)
                candidates.append(normalized)
                related_id = stable_paper_id(normalized)
                if kind == "citation" and related_id and seed_id:
                    edges.append(
                        {
                            "source_paper_id": related_id,
                            "target_paper_id": seed_id,
                            "relation": "cites",
                            "discovered_from": "semantic_scholar_citations",
                            "seed_id": seed_id,
                        }
                    )
                elif kind == "reference" and related_id and seed_id:
                    edges.append(
                        {
                            "source_paper_id": seed_id,
                            "target_paper_id": related_id,
                            "relation": "cites",
                            "discovered_from": "semantic_scholar_references",
                            "seed_id": seed_id,
                        }
                    )
                elif kind == "recommendation" and related_id:
                    for positive_id in positive_ids:
                        edges.append(
                            {
                                "source_paper_id": positive_id,
                                "target_paper_id": related_id,
                                "relation": "recommended_with",
                                "discovered_from": "semantic_scholar_recommendations",
                                "seed_id": positive_id,
                            }
                        )
            parsed = bool(text_papers)

        if not parsed:
            failures.append(
                {
                    "stage": "provider_parse",
                    "operation": operation,
                    "input": input_value,
                    "error": "No structured paper records found in provider response",
                }
            )
    return candidates, edges, failures


def _metadata_status(paper: Mapping[str, object]) -> str:
    if not stable_paper_id(paper) or not normalize_space(paper.get("title")):
        return "invalid"
    has_authors = bool(as_list(paper.get("authors")))
    has_year = integer(paper.get("year")) is not None
    has_abstract = bool(normalize_space(paper.get("abstract")))
    if has_authors and has_year and has_abstract:
        return "complete"
    if has_authors or has_year:
        return "partial"
    return "minimal"


def _score_papers(
    papers: list[JsonObject],
    request: Mapping[str, object],
) -> None:
    tokens = topic_tokens(
        text(request.get("query")),
        as_list(request.get("facets")),
    )
    for paper in papers:
        title_tokens = topic_tokens(text(paper.get("title")))
        abstract_tokens = topic_tokens(text(paper.get("abstract")))
        if tokens:
            title_overlap = len(tokens & title_tokens) / len(tokens)
            abstract_overlap = len(tokens & abstract_tokens) / len(tokens)
        else:
            title_overlap = abstract_overlap = 0.0
        lexical_score = min(1.0, title_overlap * 0.7 + abstract_overlap * 0.3)

        direct_score = 0.0
        graph_seeds: set[str] = set()
        direct_search = False
        best_rank: int | None = None
        for raw_discovery in as_list(paper.get("discovery")):
            discovery = as_mapping(raw_discovery)
            rank = integer(discovery.get("rank"))
            kind = text(discovery.get("kind"))
            if kind == "semantic_search":
                direct_search = True
                if rank is not None:
                    best_rank = rank if best_rank is None else min(best_rank, rank)
                    direct_score = max(direct_score, 1.0 / (1.0 + rank / 10.0))
            elif kind == "tavily_seed":
                tavily_score = number(paper.get("tavily_score"))
                direct_score = max(direct_score, min(1.0, tavily_score or 0.5) * 0.6)
            seed_id = text(discovery.get("seed_id"))
            if seed_id:
                graph_seeds.add(seed_id)
        graph_score = min(1.0, len(graph_seeds) / 2.0)
        relevance_score = min(
            1.0,
            lexical_score * 0.55 + direct_score * 0.30 + graph_score * 0.15,
        )

        has_s2_id = bool(text(paper.get("paper_id")))
        if relevance_score >= 0.45 or (
            direct_search
            and best_rank is not None
            and best_rank <= 20
            and lexical_score >= 0.05
        ):
            tier = "core"
        elif (
            (relevance_score >= 0.22 and lexical_score >= 0.05)
            or len(graph_seeds) >= 2
            or (len(graph_seeds) >= 1 and lexical_score >= 0.05)
            or (
                bool(paper.get("is_seed"))
                and has_s2_id
                and lexical_score >= 0.05
            )
        ):
            tier = "related"
        else:
            tier = "boundary"

        paper["id"] = stable_paper_id(paper)
        paper["dedupe_key"] = next(
            (key for key in identity_keys(paper) if key.startswith(("s2:", "doi:", "arxiv:"))),
            f"title:{normalize_title(paper.get('title'))}",
        )
        paper["metadata_status"] = _metadata_status(paper)
        paper["relevance"] = {
            "tier": tier,
            "score": round(relevance_score, 6),
            "lexical_score": round(lexical_score, 6),
            "direct_score": round(direct_score, 6),
            "graph_support": len(graph_seeds),
        }


def _alias_map(papers: list[JsonObject]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for paper in papers:
        paper_id = text(paper.get("id"))
        if not paper_id:
            continue
        aliases[paper_id] = paper_id
        for key in [
            *as_list(paper.get("_identity_keys")),
            *identity_keys(paper),
        ]:
            key_text = text(key)
            if key_text:
                aliases[key_text] = paper_id
    return aliases


def _normalise_edges(
    raw_edges: list[JsonObject],
    aliases: Mapping[str, str],
    selected_ids: set[str],
) -> list[JsonObject]:
    edges: list[JsonObject] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_edge in raw_edges:
        source = aliases.get(text(raw_edge.get("source_paper_id")), "")
        target = aliases.get(text(raw_edge.get("target_paper_id")), "")
        relation = text(raw_edge.get("relation"))
        if (
            not source
            or not target
            or source == target
            or source not in selected_ids
            or target not in selected_ids
            or not relation
        ):
            continue
        key = (source, target, relation)
        if key in seen:
            continue
        seen.add(key)
        edge = dict(raw_edge)
        edge["source_paper_id"] = source
        edge["target_paper_id"] = target
        edge["schema_version"] = "xlab.paper_edge.v1"
        edges.append(edge)
    return edges


def _call_signature(operation: str, arguments: Mapping[str, object]) -> str:
    canonical = dict(arguments)
    if operation in {S2_CITATIONS, S2_REFERENCES, S2_RECOMMENDATIONS}:
        canonical.pop("fields", None)
    return f"{operation}\0{json.dumps(canonical, ensure_ascii=False, sort_keys=True)}"


def _lookup_argument(paper: Mapping[str, object]) -> str:
    paper_id = text(paper.get("paper_id"))
    if paper_id:
        return paper_id
    external_ids = extract_external_ids(paper.get("external_ids"))
    doi = text(external_ids.get("DOI"))
    arxiv = text(external_ids.get("ArXiv"))
    if doi:
        return f"DOI:{doi}"
    if arxiv:
        return f"ARXIV:{arxiv}"
    return ""


def _followups(
    *,
    papers: list[JsonObject],
    records: list[object],
    request: Mapping[str, object],
    stop: bool,
) -> list[JsonObject]:
    if stop:
        return []
    existing: set[str] = set()
    for raw_record in records:
        record = as_mapping(raw_record)
        operation = text(record.get("operation"))
        if not operation.startswith("semantic_scholar."):
            continue
        existing.add(_call_signature(operation, as_mapping(record.get("input"))))

    max_calls = integer(request.get("max_semantic_scholar_calls")) or 240
    remaining = max(0, max_calls - len(existing))
    if remaining == 0:
        return []

    actions: list[JsonObject] = []

    def add(operation: str, arguments: JsonObject, reason: str) -> None:
        nonlocal remaining
        if remaining <= 0:
            return
        signature = _call_signature(operation, arguments)
        if signature in existing:
            return
        existing.add(signature)
        remaining -= 1
        actions.append(
            {
                "provider": "semantic_scholar",
                "operation": operation,
                "arguments": arguments,
                "reason": reason,
            }
        )

    enrichment_limit = integer(request.get("max_enrichment_calls")) or 80
    enrichment_count = 0
    ranked = sorted(
        papers,
        key=lambda paper: (
            number(as_mapping(paper.get("relevance")).get("score")) or 0.0,
            integer(paper.get("citation_count")) or 0,
        ),
        reverse=True,
    )
    for paper in ranked:
        if enrichment_count >= enrichment_limit:
            break
        if text(paper.get("metadata_status")) == "complete":
            continue
        lookup = _lookup_argument(paper)
        if lookup:
            add(
                S2_GET_PAPER,
                {"paperId": lookup, "fields": PAPER_FIELDS},
                "Complete missing Semantic Scholar metadata.",
            )
        elif len(normalize_title(paper.get("title")).split()) >= 4:
            title = text(paper.get("title"))
            add(
                S2_SEARCH,
                {
                    "query": title,
                    "limit": 3,
                    "fields": PAPER_FIELDS,
                    "expectedTitle": title,
                    "matchMode": "exact_title",
                },
                "Resolve a web-search seed by exact paper title.",
            )
        enrichment_count += 1

    seed_limit = integer(request.get("seed_limit")) or 40
    seeds = [
        paper
        for paper in ranked
        if text(paper.get("paper_id"))
        and text(as_mapping(paper.get("relevance")).get("tier"))
        in {"core", "related"}
        and bool(paper.get("is_seed"))
    ]
    if len(seeds) < seed_limit:
        present = {text(paper.get("id")) for paper in seeds}
        seeds.extend(
            paper
            for paper in ranked
            if text(paper.get("paper_id"))
            and text(paper.get("id")) not in present
            and text(as_mapping(paper.get("relevance")).get("tier")) == "core"
        )
    seeds = seeds[:seed_limit]
    relation_limit = integer(request.get("relation_limit_per_seed")) or 200
    for seed in seeds:
        paper_id = text(seed.get("paper_id"))
        add(
            S2_REFERENCES,
            {
                "paperId": paper_id,
                "limit": relation_limit,
                "offset": 0,
                "fields": RELATION_FIELDS,
            },
            "Expand references from a relevant seed.",
        )
        add(
            S2_CITATIONS,
            {
                "paperId": paper_id,
                "limit": relation_limit,
                "offset": 0,
                "fields": RELATION_FIELDS,
            },
            "Expand citations from a relevant seed.",
        )

    recommendation_ids = [text(seed.get("paper_id")) for seed in seeds[:10]]
    if recommendation_ids:
        add(
            S2_RECOMMENDATIONS,
            {
                "positivePaperIds": recommendation_ids,
                "negativePaperIds": [],
                "limit": integer(request.get("recommendation_limit")) or 100,
                "fields": RECOMMENDATION_FIELDS,
            },
            "Find papers jointly related to the seed set.",
        )
    return actions


def _ingest_run_unlocked(run_dir: Path) -> JsonObject:
    paths = artifact_paths(run_dir)
    request = as_mapping(read_json(paths["request"], {}))
    plan = read_json(paths["query_plan"], {})
    records = _latest_provider_records(read_jsonl(paths["provider_results"]))
    candidates, raw_edges, failures = _provider_records(
        records,
        _query_purposes(plan),
    )
    papers = _deduplicate(candidates)
    _score_papers(papers, request)

    max_papers = integer(request.get("max_papers")) or 1500
    eligible = [
        paper
        for paper in papers
        if text(paper.get("metadata_status")) != "invalid"
        and text(as_mapping(paper.get("relevance")).get("tier"))
        in {"core", "related"}
    ]
    eligible.sort(
        key=lambda paper: (
            number(as_mapping(paper.get("relevance")).get("score")) or 0.0,
            text(paper.get("metadata_status")) == "complete",
            math.log1p(integer(paper.get("citation_count")) or 0),
        ),
        reverse=True,
    )
    selected = eligible[:max_papers]

    previous_collection = as_mapping(read_json(paths["collection"], {}))
    previous_by_id = {
        text(as_mapping(item).get("id")): as_mapping(item)
        for item in as_list(previous_collection.get("papers"))
        if text(as_mapping(item).get("id"))
    }
    for paper in selected:
        previous = previous_by_id.get(text(paper.get("id")))
        if previous and is_mapping(previous.get("download")):
            paper["download"] = previous.get("download")
        else:
            paper["download"] = {
                "status": "not_requested"
                if as_list(paper.get("pdf_candidates"))
                else "unavailable",
                "path": None,
                "url": None,
                "sha256": None,
                "bytes": None,
                "error": None,
            }

    aliases = _alias_map(papers)
    selected_ids = {text(paper.get("id")) for paper in selected}
    edges = _normalise_edges(raw_edges, aliases, selected_ids)
    connected_ids = {
        text(edge.get("source_paper_id"))
        for edge in edges
    } | {text(edge.get("target_paper_id")) for edge in edges}

    previous_state = as_mapping(read_json(paths["state"], {}))
    history = [
        as_mapping(item)
        for item in as_list(previous_state.get("history"))
        if is_mapping(item)
    ]
    previous_count = (
        integer(history[-1].get("selected_count")) if history else None
    )
    new_count = max(0, len(selected) - (previous_count or 0))
    novelty_ratio = (
        new_count / max(1, previous_count)
        if previous_count is not None
        else 1.0
    )
    history.append(
        {
            "round": len(history) + 1,
            "selected_count": len(selected),
            "candidate_count": len(papers),
            "new_count": new_count,
            "novelty_ratio": round(novelty_ratio, 6),
            "timestamp": utc_now(),
        }
    )
    target_papers = integer(request.get("target_papers")) or 500
    novelty_threshold = number(request.get("novelty_stop_ratio")) or 0.05
    low_novelty_rounds = 0
    for item in reversed(history):
        if (number(item.get("novelty_ratio")) or 0.0) < novelty_threshold:
            low_novelty_rounds += 1
        else:
            break
    expansion_operations = {
        S2_CITATIONS,
        S2_REFERENCES,
        S2_RECOMMENDATIONS,
    }
    failed_expansion_calls = [
        as_mapping(record)
        for record in records
        if as_mapping(record).get("is_error") is True
        and text(as_mapping(record).get("operation")) in expansion_operations
    ]
    semantic_call_count = len(
        {
            _provider_signature(as_mapping(record))
            for record in records
            if text(as_mapping(record).get("operation")).startswith(
                "semantic_scholar."
            )
        }
    )
    max_semantic_calls = (
        integer(request.get("max_semantic_scholar_calls")) or 240
    )
    has_graph_evidence = bool(edges)
    if len(selected) >= target_papers and has_graph_evidence:
        stop_reason = "target_reached"
    elif failed_expansion_calls and not has_graph_evidence:
        stop_reason = "provider_expansion_failed"
    elif (
        semantic_call_count >= max_semantic_calls
        and not has_graph_evidence
    ):
        stop_reason = "graph_expansion_exhausted"
    elif (
        semantic_call_count >= max_semantic_calls
        and len(selected) < target_papers
    ):
        stop_reason = "call_budget_exhausted"
    elif (
        low_novelty_rounds >= 2
        and bool(selected)
        and has_graph_evidence
    ):
        stop_reason = "marginal_yield_converged"
    elif len(papers) >= max_papers and has_graph_evidence:
        stop_reason = "candidate_limit_reached"
    else:
        stop_reason = ""

    followups = _followups(
        papers=papers,
        records=records,
        request=request,
        stop=bool(stop_reason),
    )
    if not followups and not stop_reason:
        stop_reason = (
            "no_new_followups"
            if has_graph_evidence
            else "no_graph_evidence"
        )

    tier_counts = {"core": 0, "related": 0, "boundary": 0}
    metadata_counts = {"complete": 0, "partial": 0, "minimal": 0, "invalid": 0}
    for paper in papers:
        tier = text(as_mapping(paper.get("relevance")).get("tier"))
        status = text(paper.get("metadata_status"))
        if tier in tier_counts:
            tier_counts[tier] += 1
        if status in metadata_counts:
            metadata_counts[status] += 1

    seed_ids = [
        text(paper.get("id"))
        for paper in selected
        if bool(paper.get("is_seed"))
    ]
    collection = {
        "schema_version": "xlab.paper_set.v3",
        "query": text(request.get("query")),
        "facets": as_list(request.get("facets")),
        "target_count": target_papers,
        "max_count": max_papers,
        "candidate_count": len(papers),
        "collected_count": len(selected),
        "seed_count": len(seed_ids),
        "edge_count": len(edges),
        "generated_at": utc_now(),
        "search": {
            "round": len(history),
            "stop_reason": stop_reason or None,
            "followup_count": len(followups),
            "novelty_ratio": round(novelty_ratio, 6),
        },
        "quality": {
            "tiers": tier_counts,
            "metadata": metadata_counts,
            "abstract_coverage": round(
                sum(bool(text(paper.get("abstract"))) for paper in selected)
                / max(1, len(selected)),
                6,
            ),
            "graph_connected_ratio": round(
                len(selected_ids & connected_ids) / max(1, len(selected)),
                6,
            ),
        },
        "files": {
            "papers": relative_to_run(paths["papers_jsonl"], run_dir),
            "edges": relative_to_run(paths["edges_jsonl"], run_dir),
            "seeds": relative_to_run(paths["seeds"], run_dir),
            "provider_results": relative_to_run(
                paths["provider_results"],
                run_dir,
            ),
            "followups": relative_to_run(paths["followups"], run_dir),
            "downloads": relative_to_run(paths["downloads"], run_dir),
            "failures": relative_to_run(paths["failures"], run_dir),
        },
        "papers": [
            {key: value for key, value in paper.items() if not key.startswith("_")}
            for paper in selected
        ],
    }
    state = {
        "schema_version": "xlab.paper_collect_state.v1",
        "history": history,
        "stop_reason": stop_reason or None,
        "updated_at": utc_now(),
    }
    atomic_write_jsonl(
        paths["candidates"],
        (
            {key: value for key, value in paper.items() if not key.startswith("_")}
            for paper in papers
        ),
    )
    atomic_write_jsonl(paths["papers_jsonl"], collection["papers"])
    atomic_write_jsonl(paths["edges_jsonl"], edges)
    atomic_write_json(
        paths["seeds"],
        {
            "schema_version": "xlab.paper_seeds.v1",
            "paper_ids": seed_ids,
        },
    )
    atomic_write_json(
        paths["followups"],
        {
            "schema_version": "xlab.paper_followups.v1",
            "stop_reason": stop_reason or None,
            "actions": followups,
        },
    )
    atomic_write_jsonl(paths["failures"], failures)
    atomic_write_json(paths["state"], state)
    atomic_write_json(paths["collection"], collection)
    return {
        "candidate_count": len(papers),
        "collected_count": len(selected),
        "seed_count": len(seed_ids),
        "edge_count": len(edges),
        "followup_count": len(followups),
        "stop_reason": stop_reason or None,
        "collection_path": str(paths["collection"]),
        "followups_path": str(paths["followups"]),
    }


def ingest_run(run_dir: Path) -> JsonObject:
    with run_lock(run_dir, "ingest"):
        return _ingest_run_unlocked(run_dir)
