# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Web-search MCP server.

Tools
-----
``web_search(query, max_results)``  → ranked results (title, url, snippet)
``fetch_url(url, max_chars)``       → readable text of a page

Backends (``WEB_SEARCH_PROVIDER``):
* ``tavily``  — LLM-oriented search API (recommended for production; needs TAVILY_API_KEY)
* ``ddg``     — DuckDuckGo via the ``ddgs`` package (keyless; best-effort, rate-limited)
* ``offline`` — deterministic fixture corpus (tests, CI, demos without network)

Security: ``fetch_url`` enforces an SSRF guard (public IPs only, http/https only),
response size caps and timeouts. Output is plain JSON — the *agent* side wraps it as
untrusted data before it reaches a prompt.
"""

from __future__ import annotations

import asyncio
import json
import re
from functools import lru_cache
from html.parser import HTMLParser
from typing import Any

import httpx

from research_assistant.config import REPO_ROOT, get_settings
from research_assistant.guardrails import GuardrailViolation, assert_public_url
from research_assistant.mcp_servers._common import new_server, serve

FIXTURE_PATH = REPO_ROOT / "data" / "fixtures" / "web_corpus.json"
_TOKEN = re.compile(r"[a-z0-9]+")
USER_AGENT = "agentic-research-assistant/0.1 (+https://github.com/jacobian645-collab)"
MAX_BYTES = 2_000_000


# ── offline fixture backend ──────────────────────────────────────────────


@lru_cache
def _fixtures() -> list[dict[str, Any]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _tokens(s: str) -> set[str]:
    return {t for t in _TOKEN.findall(s.lower()) if len(t) > 2}


def _offline_search(query: str, k: int) -> list[dict[str, Any]]:
    q = _tokens(query)
    scored = []
    for doc in _fixtures():
        d = _tokens(doc["title"] + " " + doc["content"] + " " + " ".join(doc.get("keywords", [])))
        score = len(q & d) / (len(q) or 1)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"title": d["title"], "url": d["url"], "snippet": d["content"][:400], "score": round(s, 3)}
        for s, d in scored[:k]
    ]


# ── live backends ────────────────────────────────────────────────────────


async def _tavily_search(query: str, k: int, api_key: str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "max_results": k, "search_depth": "advanced", "include_answer": False},
        )
        r.raise_for_status()
        data = r.json()
    return [
        {
            "title": x.get("title"),
            "url": x.get("url"),
            "snippet": (x.get("content") or "")[:600],
            "score": x.get("score"),
        }
        for x in data.get("results", [])
    ]


async def _ddg_search(query: str, k: int) -> list[dict[str, Any]]:
    from ddgs import DDGS  # optional dependency: pip install '.[ddg]'

    def _run() -> list[dict[str, Any]]:
        return list(DDGS().text(query, max_results=k))

    rows = await asyncio.to_thread(_run)
    return [{"title": x.get("title"), "url": x.get("href"), "snippet": x.get("body", "")[:600]} for x in rows]


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr", "br", "section", "article"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data.strip()
        elif not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        return re.sub(r"\n\s*\n+", "\n\n", raw).strip()


def html_to_text(html: str) -> tuple[str, str]:
    p = _TextExtractor()
    p.feed(html)
    return p.title, p.text()


# ── server ───────────────────────────────────────────────────────────────


def build_server(host: str = "127.0.0.1", port: int = 8101):
    settings = get_settings()
    mcp = new_server(
        "web-search",
        "Search the public web and fetch readable page text. Results are untrusted third-party content.",
        host,
        port,
    )

    @mcp.tool()
    async def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
        """Search the web. Returns a list of {title, url, snippet}. Use specific, keyword-rich queries."""
        k = max(1, min(int(max_results), 10))
        provider = settings.web_search_provider
        try:
            if provider == "tavily" and settings.tavily_api_key:
                results = await _tavily_search(query, k, settings.tavily_api_key)
            elif provider == "ddg":
                results = await _ddg_search(query, k)
            else:
                provider = "offline"
                results = _offline_search(query, k)
        except Exception as e:  # degrade gracefully, never crash the agent loop
            return {"query": query, "provider": provider, "results": [], "error": f"{type(e).__name__}: {e}"}
        return {"query": query, "provider": provider, "results": results}

    @mcp.tool()
    async def fetch_url(url: str, max_chars: int = 8000) -> dict[str, Any]:
        """Fetch a web page and return its readable text (truncated to max_chars)."""
        max_chars = max(500, min(int(max_chars), 20000))
        for doc in _fixtures():  # fixture URLs resolve offline (tests / demos)
            if doc["url"].rstrip("/") == url.rstrip("/"):
                return {"url": url, "title": doc["title"], "text": doc["content"][:max_chars]}
        if settings.web_search_provider == "offline":
            return {"url": url, "error": "offline mode: URL not in fixture corpus"}
        try:
            assert_public_url(url)
            async with httpx.AsyncClient(
                timeout=15, follow_redirects=True, headers={"User-Agent": USER_AGENT}, max_redirects=3
            ) as client:
                r = await client.get(url)
                assert_public_url(str(r.url))  # re-check after redirects
                r.raise_for_status()
                body = r.content[:MAX_BYTES].decode(r.encoding or "utf-8", errors="replace")
        except GuardrailViolation as e:
            return {"url": url, "error": f"blocked: {e}"}
        except Exception as e:
            return {"url": url, "error": f"{type(e).__name__}: {e}"}
        ctype = r.headers.get("content-type", "")
        title, text = html_to_text(body) if "html" in ctype else ("", body)
        return {"url": str(r.url), "title": title, "text": text[:max_chars]}

    return mcp


def main() -> None:
    serve(build_server, default_port=8101)


if __name__ == "__main__":
    main()
