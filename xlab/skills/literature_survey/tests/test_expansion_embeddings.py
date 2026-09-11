import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.xcientist.modules.work_collector import WorkCollector


class ExpansionEmbeddingsTest(unittest.TestCase):
    def test_shared_candidates_encoded_once_with_seed_specific_selection(self):
        collector = WorkCollector.__new__(WorkCollector)
        collector.logger = Mock()
        collector.paper_graph_retriever = Mock()
        collector.paper_graph_retriever.search_by_node_id.side_effect = lambda node: [{"paper_title": node}]
        collector.get_paper_with_title_batch = Mock(return_value={
            name: {"paperId": name, "externalIds": {}} for name in ("candidate-a", "candidate-b")
        })
        collector.graph_paper_ids = set()
        collector.ignore_paper = set()
        collector.relatedness_cache = {}
        collector.advanced_filter_in_local_paper_graph_expansion = True
        options = SimpleNamespace(sentence_transformer_batch_size=64, related_work_top_k=1,
                                  related_work_threshold=0.5, RAG_source_use_embedding_filter=True,
                                  RAG_source_use_LLM_filter=True, relatedness_temperature=0.1,
                                  related_work_threshold_for_llm=4, RAG_source_downloadable_only=False)
        collector.config = SimpleNamespace(ModuleInfo=SimpleNamespace(WorkCollector=options))
        collector.data_manager = Mock()
        collector.data_manager.get_paper_title_abstract.side_effect = lambda paper: (paper, "Abstract")
        model = Mock()
        model.encode.side_effect = [torch.eye(2), torch.eye(2)]
        collector._get_embedding_model = Mock(return_value=model)
        collector.chat_agent = Mock()
        collector.chat_agent.batch_remote_chat.return_value = ['{"relevance_score":5}', '{"relevance_score":5}']
        result = collector.filter_papers_local_paper_graph(["candidate-a", "candidate-b"], ["seed-a", "seed-b"])
        self.assertEqual(set(result), {"candidate-a", "candidate-b"})
        self.assertEqual(model.encode.call_count, 2)
        self.assertEqual({(value["seed_paper_id"], value["related_paper_id"]) for value in collector.relatedness_cache.values()},
                         {("seed-a", "candidate-a"), ("seed-b", "candidate-b")})


if __name__ == "__main__":
    unittest.main()
