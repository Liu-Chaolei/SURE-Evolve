"""Task-only evidence adapter. Never resolves a survey or literature resource."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .common import stable_signature
from .inputs import IdeaRequest
from .resources.ids import stable_evidence_id


@dataclass(frozen=True)
class TaskSource:
    signatures: dict[str, str]
    direct_parent_artifact_ids: tuple[str, ...]


class TaskContextRepository:
    def __init__(self, request: IdeaRequest):
        if request.survey_path is not None:
            raise ValueError("task_only must not receive a survey path")
        if not request.task_context.get("current_best") or not request.task_context.get("execution_contract"):
            raise ValueError("task_only requires the parent implementation and execution contract")
        self.topic = request.topic
        self.references: list[dict[str, Any]] = []
        self.evidence_items = []
        for key in ("task_card", "base_model_profile", "execution_contract", "current_best", "experiment_memory"):
            value = request.task_context.get(key)
            if not value:
                continue
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
            kind = "experiment_observation" if key == "experiment_memory" else "task_context"
            source_id = stable_signature({"key": key, "value": value})
            self.evidence_items.append({
                "evidence_id": stable_evidence_id(kind, text, paper_ids=(), source_id=source_id),
                "kind": kind, "text": text, "paper_ids": [], "source_id": source_id,
                "provenance": {"source": "sure_task", "title": key, "digest": source_id},
            })
        signature = stable_signature(self.evidence_items)
        self.source = TaskSource({"task_context": signature}, (signature,))

    def validation(self) -> dict[str, Any]:
        return {"passed": bool(self.evidence_items), "warnings": [], "blocking_errors": []}

    def source_context(self, request: IdeaRequest) -> dict[str, Any]:
        return {"topic": self.topic, "references": [], "selected_evidence": self.evidence_items,
                "source_signatures": self.source.signatures,
                "direct_parent_artifact_ids": list(self.source.direct_parent_artifact_ids),
                "research_policy": request.research_policy, "task_context": request.task_context}
