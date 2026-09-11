"""Generate complete comparison tables in bounded, resumable batches."""
import hashlib
import json
from pathlib import Path

from ...common import atomic_write_json, read_json
from ..modules.pe import CLUSTER_TABLE_GENERATION
from .utils import extract_json


def generate_tables(analyzer, clusters):
    batch_size = 20
    directory = Path(analyzer.config.BasicInfo.base_dir) / "state" / "xcientist" / "table_batches"
    directory.mkdir(parents=True, exist_ok=True)
    groups = []
    results = {}
    for cluster in clusters:
        name = cluster["cluster_name"]
        papers = {paper["id"]: paper for paper in cluster.get("papers", [])}
        blocks = []
        for paper in papers.values():
            blocks.append({"id": paper["id"], "title": paper["title"], "keynote": analyzer.get_paper_keynote(paper["id"])})
        batches = [blocks[index:index + batch_size] for index in range(0, len(blocks), batch_size)]
        groups.append({"name": name, "summary": cluster.get("summary", ""), "batches": batches})
        results[name] = {"comparison_dimensions": [], "table_data": []}

    def task_for(group, index, dimensions):
        papers = group["batches"][index]
        prompt = CLUSTER_TABLE_GENERATION.format(
            cluster_name=group["name"], cluster_description=group["summary"],
            paper_content="\n".join(f"Paper ID: {p['id']}\nPaper Title: {p['title']}\nKeynote: {p['keynote']}\n" for p in papers),
        )
        prompt += f"\nReturn exactly {len(papers)} rows, one for every supplied paper ID, with no additional papers. Keep cells concise."
        if dimensions:
            prompt += "\nUse exactly these comparison_dimensions and column names: " + json.dumps(dimensions)
        signature = json.dumps({"version": 1, "prompt": prompt, "model": analyzer.config.APIInfo.llm_model_name,
                                "thinking": getattr(analyzer.config.APIInfo, "enable_thinking", None)}, sort_keys=True)
        return {"group": group["name"], "index": index, "papers": papers, "dimensions": dimensions,
                "prompt": prompt, "path": directory / (hashlib.sha256(signature.encode()).hexdigest() + ".json")}

    def run_tasks(tasks):
        completed = {}
        pending = []
        for task in tasks:
            if task["path"].exists():
                try:
                    completed[(task["group"], task["index"])] = validate_table(read_json(task["path"]), task)
                    continue
                except (ValueError, TypeError, KeyError):
                    pass
            pending.append(task)
        attempts = getattr(analyzer.config.ModuleInfo.WorkAnalyzer, "cluster_table_max_retry", 3)
        for attempt in range(attempts):
            if not pending:
                break
            responses = analyzer.chat_agent.batch_remote_chat(
                [task["prompt"] for task in pending],
                temperature=getattr(analyzer.config.ModuleInfo.WorkAnalyzer, "cluster_table_temperature", 0.3),
                desc=f"Comparison table batches {attempt + 1}/{attempts}",
            )
            failed = []
            for index, task in enumerate(pending):
                try:
                    response = responses[index] if index < len(responses) else None
                    table = validate_table(extract_json(response), task)
                    completed[(task["group"], task["index"])] = table
                    atomic_write_json(task["path"], table)
                except (ValueError, TypeError, KeyError) as error:
                    analyzer.logger.warning(f"Table batch {task['group']} / {task['index']} failed: {error}")
                    failed.append(task)
            pending = failed
        if pending:
            raise ValueError(f"{len(pending)} comparison table batches failed; completed batches are checkpointed")
        return completed

    first = run_tasks([task_for(group, 0, []) for group in groups if group["batches"]])
    remaining = []
    for group in groups:
        if not group["batches"]:
            continue
        results[group["name"]] = first[(group["name"], 0)]
        dimensions = results[group["name"]]["comparison_dimensions"]
        remaining.extend(task_for(group, index, dimensions) for index in range(1, len(group["batches"])))
    later = run_tasks(remaining)
    for group in groups:
        for index in range(1, len(group["batches"])):
            results[group["name"]]["table_data"].extend(later[(group["name"], index)]["table_data"])
    return results


def validate_table(table, task):
    if not isinstance(table, dict):
        raise ValueError("Comparison table must be an object")
    dimensions = table.get("comparison_dimensions")
    if not isinstance(dimensions, list) or not 3 <= len(dimensions) <= 5 or not all(isinstance(d, str) and d for d in dimensions):
        raise ValueError("Expected 3-5 named comparison dimensions")
    if task["dimensions"] and dimensions != task["dimensions"]:
        raise ValueError("Comparison dimensions changed between batches")
    rows = table.get("table_data")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Comparison rows must be objects")
    expected = {paper["id"]: paper["title"] for paper in task["papers"]}
    ids = [row.get("paper_id") for row in rows]
    if len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError("Comparison table has missing, duplicate, or unknown paper IDs")
    for row in rows:
        columns = row.get("columns")
        if not isinstance(columns, dict) or set(columns) != set(dimensions) or not all(isinstance(v, str) for v in columns.values()):
            raise ValueError("Comparison row has invalid columns")
        row["paper_title"] = expected[row["paper_id"]]
    return table
