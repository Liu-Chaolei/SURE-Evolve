"""Model routing and credential isolation between SURE and XLab."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from playground.sure_master.tools.with_api_profile import apply_api_routing, profile_environment


class ApiProfileTests(unittest.TestCase):
    def test_profiles_override_inherited_keys_and_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("ZAI_API_KEY=zai-secret\nZAI_BASE_URL=https://zai.example/v1\n"
                            "XI_API_KEY=xi-secret\nXI_BASE_URL=https://xi.example/v1\n")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "old", "XI_API_KEY": "wrong",
                                         "XLAB_RESEARCH_IDEA_AGENT_MODEL": "old"}):
                controller = profile_environment(path, "controller")
                xlab = profile_environment(path, "xlab")
            self.assertEqual(controller["OPENAI_API_KEY"], "zai-secret")
            self.assertEqual(controller["SURE_AGENT_MODEL"], "glm-5.3-flash")
            self.assertNotIn("XI_API_KEY", controller)
            self.assertNotIn("XLAB_RESEARCH_IDEA_AGENT_MODEL", controller)
            self.assertEqual(xlab["OPENAI_API_KEY"], "xi-secret")
            self.assertEqual(xlab["OPENAI_BASE_URL"], "https://xi.example/v1")
            self.assertNotIn("ZAI_API_KEY", xlab)
            for phase in ("AGENT", "GENERATION", "EVALUATION", "FUSION"):
                self.assertEqual(xlab[f"XLAB_RESEARCH_IDEA_{phase}_MODEL"], "gpt-6-astra")

    def test_missing_selected_key_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("ZAI_API_KEY=zai-secret\nZAI_BASE_URL=https://zai.example/v1\n")
            with patch.dict(os.environ, {"XI_API_KEY": "old", "OPENAI_API_KEY": "old"}):
                with self.assertRaisesRegex(ValueError, "XI_API_KEY"):
                    profile_environment(path, "xlab")

    def test_bare_provider_origin_uses_api_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("XI_API_KEY=xi-secret\nXI_BASE_URL=https://xi.example/\n")
            env = profile_environment(path, "xlab")
            self.assertEqual(env["OPENAI_BASE_URL"], "https://xi.example/v1")

    def test_configuration_keeps_secrets_out_and_routes_fallback_agents(self):
        agent = {"llm": "old", "system_prompt_file": "prompts/draft_system_prompt.txt",
                 "user_prompt_file": "prompts/draft_user_prompt.txt"}
        config = {"llm": {"old": {"model": "old", "api_key": "old", "max_tokens": 1024}},
                  "agents": {name: dict(agent) for name in
                             ("prefetch", "draft", "improve", "debug", "knowledge_promotion", "wisdom_promotion")},
                  "xlab": {"idea_provider": {"command": ["python", "client.py"],
                                             "preflight_command": ["python", "client.py", "--check-survey", "survey"],
                                             "environment": {"OPENAI_API_KEY": "stale-secret", "XLAB_SURE_RUN_ROOT": "runs"}}}}
        result = apply_api_routing(config, Path(".env"), "python", Path("wrapper.py"))
        self.assertTrue(all(a["llm"] == "zai_flash" for a in result["agents"].values()))
        self.assertIn("reseach", result["agents"])
        self.assertEqual(result["llm"]["zai_flash"]["api_key"], "${ZAI_API_KEY}")
        self.assertNotIn("stale-secret", json.dumps(result))
        provider = result["xlab"]["idea_provider"]
        self.assertEqual(provider["command"][-2:], ["python", "client.py"])
        self.assertIn("xlab", provider["command"])
        self.assertEqual(provider["environment"], {"XLAB_SURE_RUN_ROOT": "runs"})
