from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "sure_eval"
    / "models"
    / "FunAudioLLM__Fun-CosyVoice3-0.5B-2512"
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_model_wrapper_writes_requested_tts_output_path(monkeypatch, tmp_path: Path) -> None:
    model_module = _load_module("cosyvoice3_model_under_test", MODEL_DIR / "model.py")

    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"prompt")
    output_path = tmp_path / "generated.wav"

    class FakeTensor:
        shape = (1, 4)

        def numel(self) -> int:
            return 4

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def dim(self) -> int:
            return 2

    saved: dict[str, object] = {}

    class FakeTorchaudio:
        @staticmethod
        def save(path: str, speech, sample_rate: int) -> None:
            saved["path"] = path
            saved["sample_rate"] = sample_rate
            Path(path).write_bytes(b"RIFF")

    monkeypatch.setitem(sys.modules, "torchaudio", FakeTorchaudio)

    wrapper = model_module.ModelWrapper()
    wrapper._model = type(
        "FakeCosyVoice",
        (),
        {
            "inference_zero_shot": lambda self, *args, **kwargs: iter(
                [{"tts_speech": FakeTensor(), "sample_rate": 24000}]
            )
        },
    )()
    wrapper.model_loaded = True

    result = wrapper.predict(
        {
            "text": "hello",
            "prompt_text": "speaker prompt",
            "prompt_audio_path": str(prompt_audio),
            "output_path": str(output_path),
        }
    )

    payload = result.to_dict()
    assert output_path.exists()
    assert saved == {"path": str(output_path), "sample_rate": 24000}
    assert payload["prediction"]["audio_path"] == str(output_path)
    assert payload["audio_path"] == str(output_path)
    assert payload["sample_rate"] == 24000


def test_model_wrapper_adds_cosyvoice3_endofprompt_for_dataset_prompt_text(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_module = _load_module("cosyvoice3_model_prompt_under_test", MODEL_DIR / "model.py")

    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"prompt")
    output_path = tmp_path / "generated.wav"
    calls: dict[str, object] = {}

    class FakeTensor:
        shape = (1, 4)

        def numel(self) -> int:
            return 4

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def dim(self) -> int:
            return 2

    class FakeTorchaudio:
        @staticmethod
        def save(path: str, speech, sample_rate: int) -> None:
            Path(path).write_bytes(b"RIFF")

    class FakeCosyVoice:
        sample_rate = 24000

        def inference_zero_shot(self, text, prompt_text, prompt_audio_path, stream=False):
            calls["text"] = text
            calls["prompt_text"] = prompt_text
            calls["prompt_audio_path"] = prompt_audio_path
            calls["stream"] = stream
            return iter([{"tts_speech": FakeTensor(), "sample_rate": 24000}])

    monkeypatch.setitem(sys.modules, "torchaudio", FakeTorchaudio)
    wrapper = model_module.ModelWrapper()
    wrapper._model = FakeCosyVoice()
    wrapper.model_loaded = True

    wrapper.predict(
        {
            "text": "target text",
            "prompt_text": "dataset reference text",
            "prompt_audio_path": str(prompt_audio),
            "output_path": str(output_path),
        }
    )

    assert calls["prompt_text"] == "You are a helpful assistant.<|endofprompt|>dataset reference text"


def test_server_returns_mcp_content_payload_with_audio_path(monkeypatch, tmp_path: Path) -> None:
    server_module = _load_module("cosyvoice3_server_under_test", MODEL_DIR / "server.py")

    output_path = tmp_path / "generated.wav"

    class FakeResult:
        def to_dict(self):
            return {
                "prediction": {"audio_path": str(output_path)},
                "audio_path": str(output_path),
                "sample_rate": 24000,
            }

    class FakeModel:
        def predict(self, arguments):
            Path(arguments["output_path"]).write_bytes(b"RIFF")
            return FakeResult()

    server = server_module.MCPServer()
    monkeypatch.setattr(server, "_load_model", lambda: FakeModel())

    response = server._handle_request(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "arguments": {
                    "text": "hello",
                    "prompt_text": "speaker prompt",
                    "prompt_audio_path": str(tmp_path / "prompt.wav"),
                    "output_path": str(output_path),
                }
            },
        }
    )

    assert response is not None
    assert response["id"] == 7
    content = response["result"]["content"]
    assert content and content[0]["type"] == "text"
    payload = json.loads(content[0]["text"])
    assert payload["prediction"]["audio_path"] == str(output_path)


def test_server_returns_top_level_error_and_logs_traceback(monkeypatch, capsys) -> None:
    server_module = _load_module("cosyvoice3_server_error_under_test", MODEL_DIR / "server.py")

    class FailingModel:
        def predict(self, arguments):
            raise RuntimeError("model exploded")

    server = server_module.MCPServer()
    monkeypatch.setattr(server, "_load_model", lambda: FailingModel())

    response = server._handle_request(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {"arguments": {"text": "hello"}},
        }
    )

    assert response is not None
    assert response["id"] == 8
    assert response["error"]["message"] == "model exploded"
    assert "Traceback (most recent call last)" in capsys.readouterr().err
