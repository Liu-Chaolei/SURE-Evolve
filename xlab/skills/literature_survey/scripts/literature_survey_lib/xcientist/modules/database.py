import os, torch
import json
import re
import unicodedata
import sqlite3
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from ..utils.rich_logger import get_logger
from ..utils.gpu_utils import load_sentence_transformer_auto

class Database:
    def __init__(self, config, work_collector):
        self.config = config
        self.logger = get_logger("Database")
        self.work_collector = work_collector
        self.valid_paper_ids = set()
        self.model, self.device = load_sentence_transformer_auto(
            config.ModuleInfo.WorkCollector.sentence_transformer_model,
            logger=self.logger,
        )
        self.use_title_in_draft = True
        # self.index_path = os.path.join(config.BasicInfo.cache_path, "paper_embedding_index.pt")
        # self.title_index_path = os.path.join(config.BasicInfo.cache_path, "paper_title_embedding_index.pt")
        self.emb_dict = None
        self.title_emb_dict = None
        self.reference_papers = dict(getattr(work_collector.data_manager, "xlab_paper_by_id", {}))
        self._title_index = {}
        self._identifier_index = {}
        self._suffix_index = {}
        self._resolution_cache = {}
        self.resolution_records = {}
        self.resolved_reference_papers = {}
        self.citation_aliases = {}
        alias_path = Path(config.BasicInfo.base_dir) / "state/xcientist/citation_aliases.json"
        if alias_path.exists():
            for entry in json.loads(alias_path.read_text())["aliases"]:
                paper = self.reference_papers.get(entry["paper_id"])
                if paper is None or title_key(paper["title"]) != title_key(entry["canonical_title"]):
                    raise ValueError("Citation alias does not match a paper in the input graph")
                self.citation_aliases[title_key(entry["alias"])] = entry["paper_id"]

    def build_with_graph(self):
        self.valid_paper_ids = set()
        paper_ids = list(self.work_collector.graph_paper_ids)
        self.logger.info(f"Building database with {len(paper_ids)} papers from the reference graph.")
        self.build(paper_ids)

    def _text_for(self, pid: str) -> str:
        if not self.work_collector.expand_in_local_paper_graph:
            node = self.work_collector.reference_graph.nodes.get(pid, {})
            title = node.get("title", "")
            abstract = node.get("abstract", "")
        else:
            abstract = ""
            title = ""
        if not abstract:
            try:
                title, abstract = self.work_collector.get_paper_title_abstract(pid)
            except Exception as e:
                self.logger.error(f"Error getting title and abstract for paper ID: {pid}: {e}")
                title, abstract = "", ""

        if not abstract or not title:
            self.logger.warning(f"No abstract found for paper ID: {pid}")
            return None, None, None
        return f"Title: {title}\nAbstract: {abstract}", title, abstract

    def build(self, paper_ids):
        # Reset state for a clean rebuild
        self.valid_paper_ids.clear()
        self._resolution_cache.clear()
        self.resolution_records.clear()
        self.resolved_reference_papers.clear()

        texts = []
        titles = []
        valid_paper_embed_ids = []
        for pid in paper_ids:
            text, title_only, abstract_only = self._text_for(pid)
            if text is not None:
                texts.append(text)
                self.valid_paper_ids.add(pid)
                valid_paper_embed_ids.append(pid)
                titles.append(title_only)
                self.reference_papers.setdefault(pid, {"id": pid, "title": title_only, "abstract": abstract_only})

        if not texts:
            self.logger.warning("No valid texts found for embedding. Database build aborted.")
            return
        self.logger.info(f"Encoding {len(valid_paper_embed_ids)} paper abstracts for database.")
        embs = self.model.encode(
            texts,
            convert_to_tensor=True,
            batch_size=self.config.ModuleInfo.WorkCollector.sentence_transformer_batch_size,
            show_progress_bar=True,
        )
        # Title-only embeddings stay aligned with the same ids
        title_embs = self.model.encode(
            titles,
            convert_to_tensor=True,
            batch_size=self.config.ModuleInfo.WorkCollector.sentence_transformer_batch_size,
            show_progress_bar=False,
        )

        self.emb_dict = {"ids": valid_paper_embed_ids, "embs": embs.cpu()}
        self.title_emb_dict = {"ids": valid_paper_embed_ids, "embs": title_embs.cpu()}
        self._title_index = {}
        self._identifier_index = {}
        self._suffix_index = {}
        for pid, paper in self.reference_papers.items():
            self._title_index.setdefault(title_key(paper["title"]), set()).add(pid)
            prefix, separator, suffix = paper["title"].partition(":")
            if separator and suffix.strip():
                self._suffix_index.setdefault(title_key(suffix), set()).add(pid)
            if separator and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.+\-]{2,39}", prefix.strip()):
                self._identifier_index.setdefault(title_key(prefix), set()).add(pid)
        retriever = getattr(self.config.ModuleInfo, "PaperGraphRetriever", None)
        graph_path = Path(str(retriever.db_path)) if retriever is not None else None
        if graph_path is not None and graph_path.is_file():
            with sqlite3.connect(f"file:{graph_path}?mode=ro", uri=True) as connection:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(nodes)")}
                if {"node_type", "paper_id", "full_name", "acronym"}.issubset(columns):
                    alias_column = "aliases" if "aliases" in columns else "NULL"
                    for pid, name, acronym, aliases in connection.execute(f"SELECT paper_id, full_name, acronym, {alias_column} FROM nodes WHERE node_type = 'Core'"):
                        if pid not in self.reference_papers:
                            continue
                        try:
                            names = json.loads(aliases) if aliases else []
                        except (ValueError, TypeError):
                            names = []
                        names = names if isinstance(names, list) else []
                        for value in [name, acronym, *names]:
                            if isinstance(value, str) and value.strip():
                                self._identifier_index.setdefault(title_key(value), set()).add(pid)

    def query(self, query_text: str, top_k: int = None):
        if self.emb_dict is None:
            raise RuntimeError("Database not built; call build/build_with_graph first.")
        if top_k is None:
            top_k = self.config.ModuleInfo.Database.default_top_k
        data = self.emb_dict
        embs = data["embs"].to(self.device)
        q = self.model.encode([query_text], convert_to_tensor=True)

        scores = util.pytorch_cos_sim(q, embs)[0]
        
        top_k = min(top_k, scores.shape[0])
        vals, idxs = torch.topk(scores, k=top_k)
        return [data["ids"][i] for i in idxs.cpu().tolist()], vals.cpu().tolist()

    def query_titles(self, query_text: str, top_k: int = None):
        """Query against the title-only embedding index."""
        if self.title_emb_dict is None:
            raise RuntimeError("Database not built; call build/build_with_graph first.")
        if top_k is None:
            top_k = self.config.ModuleInfo.Database.default_top_k
        data = self.title_emb_dict
        embs = data["embs"].to(self.device)
        q = self.model.encode([query_text], convert_to_tensor=True)

        scores = util.pytorch_cos_sim(q, embs)[0]

        top_k = min(top_k, scores.shape[0])
        vals, idxs = torch.topk(scores, k=top_k)
        return [data["ids"][i] for i in idxs.cpu().tolist()], vals.cpu().tolist()

    def query_and_text(self, query_text: str, top_k: int = None, include_paper_id: bool = False):
        if top_k is None:
            top_k = self.config.ModuleInfo.Database.default_top_k
        paper_ids, _ = self.query(query_text, top_k=top_k)
        # if self.config.BasicInfo.debug:
        #     self.logger.info(f"Database query for '{query_text}' returned paper IDs: {paper_ids}")
        texts = ""
        for pid in paper_ids:
            text, _, _ = self._text_for(pid)
            if not text:
                continue
            if self.use_title_in_draft and not include_paper_id:
                texts += f"{text}\n\n"
            else:
                texts += f"Paper_id: {pid}\n{text}\n\n"
        return texts

    def resolve_title_to_paper_id(self, title_text: str, min_title_similarity: float = 0.0, *, require_verified_identity: bool = True):
        """Given a title string, return (paper_id, matched_title, similarity) if confident; raise if likely hallucination."""
        if not isinstance(title_text, str) or not title_text.strip():
            raise ValueError("title_text must be a non-empty string")

        key = title_key(title_text)
        result = self._resolution_cache.get(key)
        if result is None:
            exact = self._title_index.get(key, set())
            identifier = title_key(title_text.partition(":")[0])
            identified = self._identifier_index.get(identifier, set())
            suffix_matches = self._suffix_index.get(key.lstrip(": "), set()) if key.startswith(":") else set()
            if key in self.citation_aliases:
                pid = self.citation_aliases[key]
                result = (pid, self.reference_papers[pid]["title"], 1.0, "verified_alias")
            elif len(exact) == 1:
                pid = next(iter(exact))
                result = (pid, self.reference_papers[pid]["title"], 1.0, "exact_title")
            elif len(suffix_matches) == 1:
                pid = next(iter(suffix_matches))
                result = (pid, self.reference_papers[pid]["title"], 1.0, "unique_title_suffix")
            elif len(identified) == 1:
                pid = next(iter(identified))
                result = (pid, self.reference_papers[pid]["title"], 1.0, "unique_identifier")
            elif len(exact) > 1 or len(identified) > 1 or len(suffix_matches) > 1:
                raise ValueError(f"Ambiguous citation identity: {title_text}")
            if result is not None:
                self._resolution_cache[key] = result
        if result is not None:
            pid, matched, similarity, method = result
            verified = method != "semantic" or not require_verified_identity
            self.resolution_records[key] = {"query": title_text, "paper_id": pid, "canonical_title": matched, "similarity": similarity, "method": method, "accepted": verified and similarity >= min_title_similarity}
            if not verified:
                raise ValueError(f"Semantic citation candidate requires identity verification: {title_text} -> {matched}")
            if similarity < min_title_similarity:
                raise ValueError(f"Title similarity {similarity:.3f} is below {min_title_similarity}")
            self.resolved_reference_papers[pid] = self.reference_papers[pid]
            self.valid_paper_ids.add(pid)
            return pid, matched, similarity

        # Find nearest paper using the title-only index to avoid abstract influence.
        paper_ids, sims = self.query_titles(title_text, top_k=1)
        if not paper_ids:
            raise ValueError("No papers available to match title")

        top_pid = paper_ids[0]
        top_sim = sims[0]

        try:
            matched_title, _ = self.work_collector.get_paper_title_abstract(top_pid)
            matched_title = matched_title or ""
        except Exception as e:
            raise ValueError(f"No title found for matched paper ID {top_pid}")

        # Recompute similarity between input title text and the retrieved title for a sanity check.
        q_vec = self.model.encode([title_text], convert_to_tensor=True, device=self.device)
        t_vec = self.model.encode([matched_title], convert_to_tensor=True, device=self.device)
        title_sim = util.cos_sim(q_vec, t_vec)[0][0].item()

        # if self.config.BasicInfo.debug:
        #     self.logger.info(
        #         f"Title similarity check: input '{title_text}' vs retrieved '{matched_title}' (ID {top_pid}) = {title_sim:.3f}"
        #     )


        self._resolution_cache[key] = (top_pid, matched_title, title_sim, "semantic")
        self.resolution_records[key] = {"query": title_text, "paper_id": top_pid, "canonical_title": matched_title, "similarity": title_sim, "method": "semantic", "accepted": not require_verified_identity and title_sim >= min_title_similarity}
        if require_verified_identity:
            raise ValueError(f"Semantic citation candidate requires identity verification: {title_text} -> {matched_title}")
        if title_sim < min_title_similarity:
            raise ValueError(
                f"Title mismatch: input '{title_text}' vs retrieved '{matched_title}' (ID {top_pid}) with similarity {title_sim:.3f} < {min_title_similarity}"
            )

        self.resolved_reference_papers[top_pid] = self.reference_papers[top_pid]
        return top_pid, matched_title, title_sim


def title_key(title):
    return " ".join(unicodedata.normalize("NFKC", title).replace("‐", "-").casefold().split())
