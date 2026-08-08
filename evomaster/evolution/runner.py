"""CLI entry point for the optional EvoMaster evolution wrapper."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .manager import EvolutionManager, EvolutionRunConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvoMaster with a post-run evolution loop")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--disable-llm-analyzer", action="store_true")
    parser.add_argument("--images", nargs="+")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    manager = EvolutionManager(
        EvolutionRunConfig(
            agent_name=args.agent,
            config_path=Path(args.config),
            task_description=args.task,
            run_dir=Path(args.run_dir),
            project_root=project_root,
            iterations=args.iterations,
            images=args.images,
            use_llm_analyzer=not args.disable_llm_analyzer,
        )
    )
    return manager.run()


if __name__ == "__main__":
    sys.exit(main())
