#!/usr/bin/env python3
"""Isolated SURE scoring process. No candidate or model-family code is imported."""

from __future__ import annotations
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from playground.sure_master.core.utils.metric import SureMetricRunner
from playground.sure_master.core.utils.task_cards import SureTaskCard


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    card = dict(request.pop("task_card"))
    card.pop("base_model", None)
    workspace, output, roles = (
        request.pop(k) for k in ("workspace", "output", "roles")
    )
    result = SureMetricRunner(**request).run(
        SureTaskCard(**card), workspace, output, roles
    )
    Path(sys.argv[2]).write_text(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
