# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Retrieval (RAG) MCP server over an internal knowledge base.

Tools
-----
``search_documents(query, top_k)`` → ranked passages with stable ``kb://`` ids
``list_documents()``               → catalogue of indexed documents

Retrieval design (hybrid, the production default for enterprise RAG):
* Structure-aware chunking: split on markdown headings, then pack paragraphs into
  ~900-char windows with overlap so passages stay self-contained.
* Lexical: Okapi BM25 — strong on exact terms, ids, acronyms, product names.
* Semantic: dense embeddings (``RETRIEVAL_EMBEDDINGS=bedrock`` → Amazon Titan Text
  Embeddings v2; ``hash`` → deterministic hashing-trick vectors for offline tests).
* Fusion: Reciprocal Rank Fusion (k=60) — score-scale-free, no tuning needed.

In production swap the in-memory index for Amazon OpenSearch Serverless / Bedrock
Knowledge Bases / pgvector; the MCP tool contract stays identical, which is the point.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_assistant.config import get_settings
from research_assistant.mcp_servers._common import new_server, serve

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an and or of to in for on with is are was were be by as at from that this it its into".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Chunk:
    doc_id: str
    title: str
    section: str
    text: str
    tokens: list[str] = field(default_factory=list)


def chunk_document(path: Path, max_chars: int = 900, overlap: int = 150) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem
    sections = re.split(r"(?m)^(?=#{1,3}\s)", raw)
    chunks: list[Chunk] = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        heading = sec.splitlines()[0].lstrip("#").strip() if sec.startswith("#") else title
        buf = ""
        for para in re.split(r"\n\s*\n", sec):
            if len(buf) + len(para) > max_chars and buf:
                chunks.append(Chunk("", title, heading, buf.strip()))
                buf = buf[-overlap:] + "\n\n"
            buf += para + "\n\n"
        if buf.strip():
            chunks.append(Chunk("", title, heading, buf.strip()))
    for i, c in enumerate(chunks):
        c.doc_id = f"kb://{path.name}#{i}"
        c.tokens = tokenize(c.section + " " + c.text)
    return chunks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs, self.k1, self.b = docs, k1, b
        self.N = len(docs)
        self.avgdl = sum(map(len, docs)) / self.N if self.N else 0.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += (
                        self.idf[t]
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                    )
            out.append(s)
        return out


class HashEmbedder:
    """Deterministic feature-hashing embedder (unigrams + bigrams). Offline stand-in for a
    real embedding model so the hybrid path is exercised in CI."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            toks = tokenize(t)
            feats = toks + [a + "_" + b for a, b in itertools.pairwise(toks)]
            v = [0.0] * self.dim
            for f in feats:
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / n for x in v])
        return vecs


class BedrockEmbedder:
    def __init__(self, region: str, model_id: str = "amazon.titan-embed-text-v2:0") -> None:
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            body = json.dumps({"inputText": t[:8000], "dimensions": 512, "normalize": True})
            r = self.client.invoke_model(modelId=self.model_id, body=body)
            out.append(json.loads(r["body"].read())["embedding"])
        return out


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder: Any | None) -> None:
        self.chunks = chunks
        self.bm25 = BM25([c.tokens for c in chunks]) if chunks else None
        self.embedder = embedder
        self.vectors = (
            embedder.embed([c.section + "\n" + c.text for c in chunks]) if embedder and chunks else []
        )

    @classmethod
    def from_dir(cls, corpus_dir: Path, embedder: Any | None) -> HybridIndex:
        chunks: list[Chunk] = []
        for p in sorted(corpus_dir.glob("**/*")):
            if p.suffix.lower() in {".md", ".txt"}:
                chunks.extend(chunk_document(p))
        return cls(chunks, embedder)

    def search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[dict[str, Any]]:
        if not self.chunks or self.bm25 is None:
            return []
        lex = self.bm25.scores(tokenize(query))
        rankings = [sorted(range(len(lex)), key=lambda i: lex[i], reverse=True)]
        rankings[0] = [i for i in rankings[0] if lex[i] > 0]
        if self.vectors:
            qv = self.embedder.embed([query])[0]  # type: ignore[union-attr]
            sem = [_cos(qv, v) for v in self.vectors]
            rankings.append(
                [i for i in sorted(range(len(sem)), key=lambda i: sem[i], reverse=True) if sem[i] > 0]
            )
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking[: top_k * 4]):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        best = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "doc_id": self.chunks[i].doc_id,
                "title": self.chunks[i].title,
                "section": self.chunks[i].section,
                "score": round(s, 5),
                "text": self.chunks[i].text,
            }
            for i, s in best
        ]


def build_index() -> HybridIndex:
    s = get_settings()
    embedder: Any | None = None
    if s.retrieval_embeddings == "hash":
        embedder = HashEmbedder()
    elif s.retrieval_embeddings == "bedrock":
        embedder = BedrockEmbedder(s.aws_region)
    return HybridIndex.from_dir(Path(s.corpus_dir), embedder)


def build_server(host: str = "127.0.0.1", port: int = 8102):
    index = build_index()
    mcp = new_server(
        "knowledge-base",
        "Hybrid (BM25 + dense) search over the organisation's internal knowledge base.",
        host,
        port,
    )

    @mcp.tool()
    def search_documents(query: str, top_k: int = 5) -> dict[str, Any]:
        """Search internal documents. Returns passages with stable doc_id values (kb://file#chunk)
        that must be quoted exactly when cited."""
        k = max(1, min(int(top_k), 10))
        return {"query": query, "results": index.search(query, k)}

    @mcp.tool()
    def list_documents() -> dict[str, Any]:
        """List the documents available in the knowledge base."""
        titles: dict[str, str] = {}
        for c in index.chunks:
            titles.setdefault(c.doc_id.split("#")[0], c.title)
        return {"documents": [{"doc": d, "title": t} for d, t in titles.items()]}

    return mcp


def main() -> None:
    serve(build_server, default_port=8102)


if __name__ == "__main__":
    main()
