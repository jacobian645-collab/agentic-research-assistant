# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

from research_assistant.config import REPO_ROOT
from research_assistant.mcp_servers.retrieval_server import BM25, HashEmbedder, HybridIndex, tokenize

CORPUS = REPO_ROOT / "data" / "corpus"


def test_chunks_have_stable_ids():
    idx = HybridIndex.from_dir(CORPUS, None)
    ids = [c.doc_id for c in idx.chunks]
    assert len(ids) == len(set(ids)) and all(i.startswith("kb://") for i in ids)


def test_bm25_ranks_relevant_doc_first():
    idx = HybridIndex.from_dir(CORPUS, None)
    top = idx.search("agent runaway incident tool calls cost", top_k=3)
    assert top[0]["doc_id"].startswith("kb://incident-2026-03-agent-runaway.md")


def test_hybrid_rrf_returns_results():
    idx = HybridIndex.from_dir(CORPUS, HashEmbedder())
    res = idx.search("judge calibration kappa human labels", top_k=3)
    assert any("rag-evaluation-playbook" in r["doc_id"] for r in res)


def test_bm25_idf_prefers_rare_terms():
    bm = BM25([tokenize("common common rare"), tokenize("common common common")])
    s = bm.scores(tokenize("rare"))
    assert s[0] > s[1] == 0
