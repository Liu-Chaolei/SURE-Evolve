from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "sure_eval"
    / "models"
    / "openai__whisper-large-v3-turbo"
)


def load_server_module():
    sys.path.insert(0, str(MODEL_DIR))
    spec = importlib.util.spec_from_file_location("openai_whisper_mcp_server", MODEL_DIR / "server.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load Whisper server module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeModel:
    instances = []

    def __init__(self):
        self.loaded = False
        self.requests = []
        self.__class__.instances.append(self)

    def load(self):
        self.loaded = True

    def predict(self, request):
        self.requests.append(request)
        return {"text": "test transcript", "language": request.get("language")}

    def health(self):
        return {"loaded": self.loaded}


class WhisperMCPServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server_module = load_server_module()

    def setUp(self):
        FakeModel.instances.clear()
        self.server = self.server_module.MCPServer(model_factory=FakeModel)

    def test_initialize_and_notification(self):
        response = self.server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})

        self.assertEqual(response["id"], 1)
        self.assertEqual(response["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual(FakeModel.instances, [])
        self.assertIsNone(
            self.server.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"})
        )

    def test_lists_configured_transcription_tools(self):
        response = self.server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})

        names = {tool["name"] for tool in response["result"]["tools"]}
        self.assertEqual(names, {"asr_transcribe", "transcribe_audio", "healthcheck"})
        self.assertEqual(FakeModel.instances, [])

    def test_both_transcription_tools_dispatch_to_model(self):
        for tool_name in ("asr_transcribe", "transcribe_audio"):
            with self.subTest(tool_name=tool_name):
                server = self.server_module.MCPServer(model_factory=FakeModel)
                response = server.handle_request(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": {"audio_path": "/tmp/sample.wav", "language": "en"},
                        },
                    }
                )

                model = FakeModel.instances[-1]
                self.assertTrue(model.loaded)
                self.assertEqual(
                    model.requests,
                    [{"audio_path": "/tmp/sample.wav", "language": "en"}],
                )
                self.assertEqual(response["result"]["raw"]["text"], "test transcript")

    def test_healthcheck_does_not_load_model(self):
        response = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "healthcheck", "arguments": {}},
            }
        )

        self.assertEqual(response["result"]["raw"], {"loaded": False})
        self.assertEqual(FakeModel.instances, [])

    def test_unknown_tool_returns_json_rpc_error(self):
        response = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "missing", "arguments": {}},
            }
        )

        self.assertEqual(response["error"]["code"], -32601)


if __name__ == "__main__":
    unittest.main()
