"""Run a command using an existing XLab API-key profile, without copying secrets."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def zai_environment(env_file: Path) -> dict[str, str]:
    """Load the explicitly selected ZAI profile, without inheriting old providers."""
    from dotenv import dotenv_values

    values = dotenv_values(env_file)
    missing = [key for key in ("ZAI_API_KEY", "ZAI_BASE_URL") if not values.get(key)]
    if missing:
        raise ValueError("Missing ZAI configuration: " + ", ".join(missing))
    env = dict(os.environ)
    env.update({key: str(values[key]) for key in ("ZAI_API_KEY", "ZAI_BASE_URL")})
    env.update(OPENAI_API_KEY=env["ZAI_API_KEY"], OPENAI_BASE_URL=env["ZAI_BASE_URL"],
               SURE_AGENT_MODEL="glm-5.3-flash", LLM_BASE_URL=env["ZAI_BASE_URL"],
               LLM_MODEL="glm-5.3-flash")
    for phase in ("AGENT", "GENERATION", "EVALUATION", "FUSION"):
        env[f"XLAB_RESEARCH_IDEA_{phase}_MODEL"] = (
            "glm-5.3" if phase in {"AGENT", "GENERATION"} else "glm-5.3-flash"
        )
    return env


def xlab_environment(agent_dir: Path) -> dict[str, str]:
    settings = json.loads((agent_dir / "settings.json").read_text())
    models = json.loads((agent_dir / "models.json").read_text())
    auth = json.loads((agent_dir / "auth.json").read_text())
    provider = settings.get("defaultProvider")
    profile = models.get("providers", {}).get(provider, {})
    credential = auth.get(provider, {})
    if credential.get("type") != "api_key" or not credential.get("key"):
        raise ValueError("The selected XLab profile needs an API-key credential for this Python bridge")
    if not profile.get("baseUrl") or not settings.get("defaultModel"):
        raise ValueError("The selected XLab profile needs a base URL and default model")
    env = dict(os.environ)
    if not env.get("OPENAI_API_KEY") and env.get("OPENAI_BASE_URL") and env["OPENAI_BASE_URL"].rstrip("/") != profile["baseUrl"].rstrip("/"):
        raise ValueError("Cannot use an XLab credential with a different OPENAI_BASE_URL")
    defaults = {"OPENAI_API_KEY": credential["key"], "OPENAI_BASE_URL": profile["baseUrl"],
                "SURE_AGENT_MODEL": settings["defaultModel"]}
    for key, value in defaults.items():
        if not env.get(key):
            env[key] = str(value)
    for phase in ("AGENT", "GENERATION", "EVALUATION", "FUSION"):
        env.setdefault(f"XLAB_RESEARCH_IDEA_{phase}_MODEL", settings["defaultModel"])
    if env.get("LLM_BASE_URL") and env["LLM_BASE_URL"].rstrip("/") != env["OPENAI_BASE_URL"].rstrip("/"):
        raise ValueError("XLab survey LLM_BASE_URL must match the selected credential's endpoint")
    env.setdefault("LLM_BASE_URL", env["OPENAI_BASE_URL"])
    env.setdefault("LLM_MODEL", env["SURE_AGENT_MODEL"])
    env.setdefault("XLAB_LITERATURE_SURVEY_USE_STREAM", "1")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    raise SystemExit(subprocess.run(command, env=xlab_environment(args.agent_dir)).returncode)


if __name__ == "__main__":
    main()
