"""Check the selected ZAI model endpoints before expensive evolution work."""

from __future__ import annotations

import json
from typing import Any


def check_zai_api(env: dict[str, str]) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(
        api_key=env["ZAI_API_KEY"],
        base_url=env["ZAI_BASE_URL"],
        timeout=90,
        max_retries=0,
    )
    results = []
    for model in ("glm-5.3", "glm-5.3-flash"):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": 'Return JSON {"ok":true} only.'}],
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
            if json.loads(response.choices[0].message.content).get("ok") is not True:
                raise ValueError("Model did not return the requested JSON")
            results.append({"model": model, "operation": "json", "status": "passed"})
        except Exception as exc:
            results.append(
                {
                    "model": model,
                    "operation": "json",
                    "status": "failed",
                    "http_status": getattr(exc, "status_code", None),
                    "error": str(exc).replace(env["ZAI_API_KEY"], "[REDACTED]")[:1200],
                }
            )
    if all(row["status"] == "passed" for row in results):
        try:
            chunks = client.chat.completions.create(
                model="glm-5.3-flash",
                messages=[{"role": "user", "content": "Call finish with ok=true."}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "description": "Finish this check",
                            "parameters": {
                                "type": "object",
                                "properties": {"ok": {"type": "boolean"}},
                                "required": ["ok"],
                            },
                        },
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "finish"}},
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=1024,
            )
            arguments = ""
            name = ""
            for chunk in chunks:
                if chunk.choices:
                    for call in chunk.choices[0].delta.tool_calls or []:
                        name += call.function.name or ""
                        arguments += call.function.arguments or ""
            if name != "finish" or json.loads(arguments).get("ok") is not True:
                raise ValueError("Flash streaming tool call was not valid")
            results.append(
                {
                    "model": "glm-5.3-flash",
                    "operation": "streaming_tools",
                    "status": "passed",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "model": "glm-5.3-flash",
                    "operation": "streaming_tools",
                    "status": "failed",
                    "error": str(exc).replace(env["ZAI_API_KEY"], "[REDACTED]")[:1200],
                }
            )
    return {
        "status": (
            "passed" if all(row["status"] == "passed" for row in results) else "blocked"
        ),
        "base_url": env["ZAI_BASE_URL"],
        "checks": results,
    }
