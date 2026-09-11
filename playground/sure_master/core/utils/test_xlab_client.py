from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from dataclasses import asdict
from pathlib import Path

from playground.sure_master.core.contracts import IdeaRequest, MetricSpec
from playground.sure_master.core.utils.fingerprints import digest
from playground.sure_master.core.xlab_client import XlabIdeaClient, XlabIdeaClientError


class XlabIdeaClientTests(unittest.TestCase):
    def _request(self) -> IdeaRequest:
        return IdeaRequest(
            request_id="request-1",
            sure_run_id="sure-1",
            task_id="task-1",
            task_description="test task",
            search_mode="ordinary",
            round_index=1,
            requested_idea_count=4,
            metric=MetricSpec(name="wer", direction="lower"),
            input_digest="sha256:request",
        )

    def _provider(self, root: Path) -> tuple[Path, Path]:
        counter = root / "counter.txt"
        script = root / "provider.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import json
                import pathlib
                import sys

                counter = pathlib.Path({str(counter)!r})
                count = int(counter.read_text() if counter.exists() else "0") + 1
                counter.write_text(str(count))
                envelope = json.loads(sys.stdin.readline())
                payload = envelope["payload"]
                ideas = []
                for index in range(1, 5):
                    artifact = {{"idea_id": f"idea-{{index}}", "index": index}}
                    ideas.append({{
                        "idea_id": f"idea-{{index}}",
                        "artifact_id": f"artifacts/idea-{{index}}.json",
                        "artifact_digest": "sha256:" + str(index),
                        "title": f"Idea {{index}}",
                        "candidate_type": "inference",
                        "hypothesis": f"Hypothesis {{index}}",
                        "mechanism": f"Mechanism {{index}}",
                        "axis": None,
                        "spec": {{"implementation_instructions": f"Implement {{index}}"}},
                    }})
                batch = {{
                    "schema_version": "sure.idea_batch.v1",
                    "status": "success",
                    "request_digest": payload["input_digest"],
                    "xlab_run_id": "xlab-1",
                    "workspace": "xlab-run",
                    "ideas": ideas,
                    "batch_digest": "sha256:batch",
                    "blockers": [],
                }}
                print(json.dumps({{
                    "protocol": "xlab.sure.jsonl.v1",
                    "operation": envelope["operation"],
                    "operation_id": envelope["operation_id"],
                    "status": "success",
                    "payload": batch,
                }}))
                """
            ),
            encoding="utf-8",
        )
        return script, counter

    def test_published_response_replays_without_provider_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, counter = self._provider(root)
            command = ["python3", str(script)]
            client = XlabIdeaClient(
                command,
                receipt_path="artifacts/receipts.json",
                workspace_root=root,
            )
            first = client.generate(self._request())
            second = client.generate(self._request())
            self.assertEqual(asdict(first), asdict(second))
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")

            restored = XlabIdeaClient(
                command,
                receipt_path="artifacts/receipts.json",
                workspace_root=root,
            )
            replayed = restored.generate(self._request())
            self.assertEqual(asdict(first), asdict(replayed))
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")

    def test_conflicting_request_and_incomplete_operation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, counter = self._provider(root)
            client = XlabIdeaClient(
                ["python3", str(script)],
                receipt_path="receipts.json",
                workspace_root=root,
            )
            client.generate(self._request())
            changed = IdeaRequest(**{**asdict(self._request()), "task_description": "changed"})
            with self.assertRaisesRegex(XlabIdeaClientError, "conflicts"):
                client.generate(changed)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")

            operation = client._operations["request-1"]
            operation.pop("response")
            operation.pop("response_digest")
            operation["status"] = "incomplete"
            client._flush_receipts()
            restored = XlabIdeaClient(
                ["python3", str(script)],
                receipt_path="receipts.json",
                workspace_root=root,
            )
            with self.assertRaisesRegex(XlabIdeaClientError, "reconcile"):
                restored.generate(self._request())
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")

    def test_receipt_tampering_and_path_escape_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, _ = self._provider(root)
            client = XlabIdeaClient(
                ["python3", str(script)],
                receipt_path="receipts.json",
                workspace_root=root,
            )
            client.generate(self._request())
            receipt = root / "receipts.json"
            document = json.loads(receipt.read_text(encoding="utf-8"))
            document["operations"]["request-1"]["response"]["payload"]["workspace"] = "tampered"
            receipt.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(XlabIdeaClientError, "digest mismatch"):
                XlabIdeaClient(
                    ["python3", str(script)],
                    receipt_path="receipts.json",
                    workspace_root=root,
                )
            with self.assertRaisesRegex(ValueError, "absolute paths"):
                XlabIdeaClient(
                    ["python3", str(script)],
                    receipt_path=root / "outside.json",
                    workspace_root=root,
                )
            with self.assertRaisesRegex(ValueError, "escapes"):
                XlabIdeaClient(
                    ["python3", str(script)],
                    receipt_path="../outside.json",
                    workspace_root=root,
                )

    def test_strict_single_response_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "provider.py"
            script.write_text(
                "import json,sys\ne=json.loads(sys.stdin.readline())\nprint('{}')\nprint('{}')\n",
                encoding="utf-8",
            )
            client = XlabIdeaClient(["python3", str(script)])
            with self.assertRaisesRegex(XlabIdeaClientError, "exactly one"):
                client.generate(self._request())


if __name__ == "__main__":
    unittest.main()
