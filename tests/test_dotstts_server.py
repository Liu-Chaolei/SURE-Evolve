from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_dotstts_server_logs_full_traceback_on_tool_failure(monkeypatch, capsys) -> None:
    model_dir = Path("src/sure_eval/models/rednote-hilab__dots.tts-base").resolve()
    spec = importlib.util.spec_from_file_location("dotstts_server_test", model_dir / "server.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(model_dir))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    server = module.MCPServer()

    def fail_predict(_arguments):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(server.model, "predict", fail_predict)
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"arguments": {"text": "hello", "prompt_audio_path": "prompt.wav"}},
        }
    )

    stderr = capsys.readouterr().err
    assert response["error"]["message"] == "synthetic failure"
    assert "Traceback (most recent call last):" in stderr
    assert "RuntimeError: synthetic failure" in stderr
