# Repository Guidelines

## Project Structure & Module Organization

EvoMaster is a Python package centered on `evomaster/`, which contains the core agent framework: `agent/`, `core/`, `env/`, `evolution/`, `skills/`, `skills_ts/`, and `utils/`. Runnable examples and domain agents live in `playground/`, with matching YAML configurations in `configs/`. Documentation is in `docs/`, reusable external integrations are in `extensions/`, and images used by the README are in `assets/`. The main entry point is `run.py`.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install core runtime dependencies.
- `uv sync`: install from `pyproject.toml` when using `uv`.
- `python run.py --agent minimal --task "Your task"`: run the default minimal playground with `configs/minimal/config.yaml`.
- `python run.py --agent minimal --config configs/minimal/deepseek-v3.2-example.yaml --task "Your task"`: run with an explicit config.
- `pip install -r playground/<agent>/requirements.txt`: install optional dependencies for playgrounds such as `ml_master_2`, `asr_master`, or `minimal_kaggle`.

## Coding Style & Naming Conventions

Use Python 3.10+ and follow PEP 8 conventions: 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and descriptive config keys in YAML. Keep public APIs typed where practical, following existing dataclass and Pydantic patterns. Prompt templates use `.txt` files under each playground's `prompts/` directory. There is no repository-wide formatter configuration; if using `black`, keep formatting-only changes separate from behavioral edits.

## Testing Guidelines

The repository currently mixes `unittest` tests with playground-level smoke scripts. Run focused tests with commands such as `python -m unittest evomaster.agent.test_agent_context`. For playground changes, run the smallest relevant agent command with a safe local task and the matching config. Name new Python tests `test_*.py` or place test cases near the module when no dedicated test tree exists.

## Commit & Pull Request Guidelines

Recent history uses short imperative or descriptive subjects, for example `fix browsemaster mcp` and `feat: add run level self-evolution`. Keep commits scoped and mention the affected agent, skill, or config when useful. Pull requests should include the motivation, changed modules, commands run, and any required API keys or external services. For UI or documentation changes, include screenshots or rendered examples when applicable.

## Security & Configuration Tips

Do not commit real API keys, credentials, run artifacts, or local cache contents. Start from `.env.template` and localize secrets in `.env` or private YAML copies. Review config changes carefully because many playgrounds execute tools, shell commands, or remote model calls.
