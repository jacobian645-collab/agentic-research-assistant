# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Deterministic guardrails — the controls that should NOT be delegated to an LLM.

Defence in depth for an agent that reads the open web and executes code:

1. Input validation        → ``validate_question``
2. Prompt-injection hygiene → ``wrap_untrusted`` + ``scan_for_injection`` on every tool result
3. SSRF protection          → ``assert_public_url`` (used by the web MCP server)
4. Grounding verification   → ``verify_evidence`` (claimed sources must appear in tool output)
5. Citation integrity       → ``check_citations`` (report cites only known sources)
6. Budgets                  → enforced in the graph (steps, tool calls, rounds, recursion_limit)
7. Sandboxed code execution → see mcp_servers/code_exec_server.py
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

from research_assistant.schemas import (
    CitationCheck,
    ReportDraft,
    Source,
    VerifiedEvidence,
    WorkerReport,
)

# ── 1. Input validation ──────────────────────────────────────────────────


class GuardrailViolation(ValueError):
    pass


def validate_question(question: str, max_chars: int) -> str:
    q = (question or "").strip()
    if len(q) < 5:
        raise GuardrailViolation("Question is too short.")
    if len(q) > max_chars:
        raise GuardrailViolation(f"Question exceeds {max_chars} characters.")
    return q


# ── 2. Prompt-injection hygiene ──────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"you are now (a|an|in) ",
    r"reveal (your|the) (system prompt|instructions|api key)",
    r"<\s*/?\s*system\s*>",
    r"BEGIN (NEW )?INSTRUCTIONS",
    r"(send|post|upload) (the |all )?(conversation|secrets?|credentials|api keys?) to",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def scan_for_injection(text: str) -> list[str]:
    """Cheap heuristic detector. Production: add a classifier (e.g. Bedrock Guardrails
    prompt-attack filter) — heuristics are a tripwire, not a guarantee."""
    return sorted({m.group(0).lower() for m in _INJECTION_RE.finditer(text or "")})


def wrap_untrusted(text: str, limit: int) -> str:
    text = text or ""
    if len(text) > limit:
        text = text[:limit] + f"\n…[truncated {len(text) - limit} chars]"
    text = text.replace("</untrusted_data>", "&lt;/untrusted_data&gt;")
    flags = scan_for_injection(text)
    warn = f' injection_suspected="{"; ".join(flags)}"' if flags else ""
    return f"<untrusted_data{warn}>\n{text}\n</untrusted_data>"


# ── 3. SSRF protection ───────────────────────────────────────────────────


def assert_public_url(url: str) -> None:
    """Reject non-HTTP(S) schemes and any host resolving to a private/loopback/link-local
    address (blocks cloud metadata endpoints such as 169.254.169.254)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise GuardrailViolation(f"Blocked URL scheme/host: {url!r}")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        raise GuardrailViolation(f"Cannot resolve host {parsed.hostname!r}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise GuardrailViolation(f"Blocked non-public address {ip} for {parsed.hostname!r}")


# ── 4. Grounding verification ────────────────────────────────────────────


def _norm(s: str) -> str:
    return s.strip().rstrip("/").lower()


def verify_evidence(report: WorkerReport, tool_outputs: list[str], used_code: bool) -> list[VerifiedEvidence]:
    """Mark each evidence item verified iff its source literally appears in a tool output.

    This catches the most damaging hallucination in research agents: fabricated URLs /
    document ids. It is cheap, deterministic and explainable.
    """
    corpus = _norm("\n".join(tool_outputs))
    out: list[VerifiedEvidence] = []
    for ev in report.evidence:
        src = _norm(ev.source)
        ok = (src == "code_execution" and used_code) or (len(src) > 3 and src in corpus)
        out.append(VerifiedEvidence(**ev.model_dump(), verified=ok))
    return out


def source_kind(locator: str) -> str:
    if locator.startswith("code_execution"):
        return "code"
    if locator.startswith(("http://", "https://")):
        return "web"
    return "document"


# ── 5. Citation integrity ────────────────────────────────────────────────


def check_citations(draft: ReportDraft, sources: list[Source], verified_ids: set[str]) -> CitationCheck:
    known = {s.id for s in sources}
    claims = [c for sec in draft.sections for c in sec.claims]
    cited = [c for c in claims if c.citations]
    unknown = sorted({cid for c in claims for cid in c.citations if cid not in known})
    unverified = sorted(
        {cid for c in claims for cid in c.citations if cid in known and cid not in verified_ids}
    )
    return CitationCheck(
        total_claims=len(claims),
        cited_claims=len(cited),
        unknown_citations=unknown,
        unverified_sources_cited=unverified,
    )
