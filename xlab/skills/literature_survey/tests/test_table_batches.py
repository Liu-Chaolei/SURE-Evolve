import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.xcientist.utils.cluster_tables import generate_tables, validate_table


class TableBatchesTest(unittest.TestCase):
    def test_complete_large_table_resumes_without_repeating_calls(self):
        dimensions = ["Method", "Data", "Evaluation"]
        batch_sizes = []

        def respond(prompts, **kwargs):
            responses = []
            for prompt in prompts:
                ids = re.findall(r"^Paper ID: (.+)$", prompt, re.MULTILINE)
                batch_sizes.append(len(ids))
                responses.append(json.dumps({"comparison_dimensions": dimensions, "table_data": [
                    {"paper_id": pid, "paper_title": "Model title", "columns": {d: "Evidence" for d in dimensions}} for pid in ids
                ]}))
            return responses

        with tempfile.TemporaryDirectory() as temporary:
            analyzer = SimpleNamespace(
                config=SimpleNamespace(BasicInfo=SimpleNamespace(base_dir=temporary), APIInfo=SimpleNamespace(llm_model_name="test"),
                                       ModuleInfo=SimpleNamespace(WorkAnalyzer=SimpleNamespace())),
                get_paper_keynote=Mock(return_value="Grounded note"),
                chat_agent=SimpleNamespace(batch_remote_chat=Mock(side_effect=respond)), logger=Mock(),
            )
            papers = [{"id": f"p{index}", "title": f"Title {index}"} for index in range(43)]
            clusters = [{"cluster_name": "Speech", "summary": "Speech methods", "papers": papers}]
            first = generate_tables(analyzer, clusters)
            self.assertEqual(batch_sizes, [20, 20, 3])
            self.assertEqual([row["paper_id"] for row in first["Speech"]["table_data"]], [p["id"] for p in papers])
            self.assertEqual(first["Speech"]["table_data"][0]["paper_title"], "Title 0")
            second = generate_tables(analyzer, clusters)
            self.assertEqual(first, second)
            self.assertEqual(analyzer.chat_agent.batch_remote_chat.call_count, 2)

    def test_missing_rows_cannot_pass_as_a_complete_table(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_table({"comparison_dimensions": ["A", "B", "C"], "table_data": []},
                           {"dimensions": [], "papers": [{"id": "p", "title": "Paper"}]})


if __name__ == "__main__":
    unittest.main()
