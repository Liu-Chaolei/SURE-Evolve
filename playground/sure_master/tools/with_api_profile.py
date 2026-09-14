"""Launch controller (ZAI) or XLab (XI) with isolated, explicit API credentials."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

from dotenv import dotenv_values


def profile_environment(env_file: Path, role: str) -> dict[str, str]:
    if role not in {"controller", "xlab"}:
        raise ValueError("API role must be controller or xlab")
    prefix, model = ("ZAI", "glm-5.3-flash") if role == "controller" else ("XI", "gpt-6-astra")
    values = dotenv_values(env_file, interpolate=False)
    required = (prefix + "_API_KEY", prefix + "_BASE_URL")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError("Missing API configuration: " + ", ".join(missing))
    # A child never inherits the other provider's credentials or stale role models.
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ("ZAI_", "XI_", "OPENAI_", "LLM_", "XLAB_RESEARCH_IDEA_"))}
    env.update({key: str(values[key]) for key in required})
    base_url = env[required[1]].rstrip("/")
    if not urlsplit(base_url).path:
        base_url += "/v1"
    env[required[1]] = base_url
    env.update(OPENAI_API_KEY=env[required[0]], OPENAI_BASE_URL=env[required[1]],
               LLM_BASE_URL=env[required[1]], LLM_MODEL=model, SURE_AGENT_MODEL=model)
    if role == "xlab":
        env["XLAB_RESEARCH_IDEA_STREAM"] = "1"
        for phase in ("AGENT", "GENERATION", "EVALUATION", "FUSION"):
            env[f"XLAB_RESEARCH_IDEA_{phase}_MODEL"] = model
    return env


def apply_api_routing(config: dict, env_file: Path, python: str, wrapper: Path) -> dict:
    """Configure all SURE agents and both XLab commands without embedding keys."""
    config = deepcopy(config)
    config["api_profile"] = {"provider": "zai_controller_xi_xlab", "env_file": str(env_file.resolve())}
    old = next((v for v in config["llm"].values() if isinstance(v, dict)), {})
    config["llm"] = {"zai_flash": {**old, "provider": "openai", "model": "glm-5.3-flash",
                                  "api_key": "${ZAI_API_KEY}", "base_url": "${ZAI_BASE_URL}"},
                     "default": "zai_flash"}
    if "reseach" not in config["agents"]:
        config["agents"]["reseach"] = deepcopy(config["agents"]["draft"])
        for key in ("system_prompt_file", "user_prompt_file"):
            config["agents"]["reseach"][key] = config["agents"]["draft"][key].replace("draft_", "reseach_")
    for agent in config["agents"].values():
        agent["llm"] = "zai_flash"
    provider = config["xlab"]["idea_provider"]
    prefix = [python, str(wrapper.resolve()), "--env-file", str(env_file.resolve()), "--role", "xlab", "--"]
    for key in ("command", "preflight_command"):
        if provider.get(key):
            provider[key] = [*prefix, *provider[key]]
    provider["environment"] = {k: v for k, v in provider.get("environment", {}).items()
                               if not k.startswith(("ZAI_", "XI_", "OPENAI_", "LLM_", "XLAB_RESEARCH_IDEA_"))}
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--role", choices=("controller", "xlab"), required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    raise SystemExit(subprocess.run(command, env=profile_environment(args.env_file, args.role)).returncode)


if __name__ == "__main__":
    main()
