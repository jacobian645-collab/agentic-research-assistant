# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Helpers shared by graph nodes: run context, event emission, source registry, formatting."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from typing import Any

from langgraph.runtime import Runtime

from research_assistant.guardrails import source_kind
from research_assistant.observability import UsageTracker
from research_assistant.schemas import Finding, ResearchReport, Source


@dataclass
class RunContext:
    """Per-run, non-persisted context (LangGraph ``context_schema``).

    Holds things that must not live in checkpointed state: live usage tracker, budget,
    caller identity. Passed via ``graph.ainvoke(..., context=RunContext(...))``.
    """

    tracker: UsageTracker | None = None
    max_cost_usd: float | None = None
    user_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def spent_usd(self) -> float:
        return float(self.tracker.summary()["est_cost_usd"]) if self.tracker else 0.0

    def over_budget(self) -> bool:
        return bool(self.max_cost_usd) and self.spent_usd() >= float(self.max_cost_usd or 0)


def ctx(runtime: Runtime[RunContext] | None) -> RunContext:
    c = getattr(runtime, "context", None)
    return c if isinstance(c, RunContext) else RunContext()


def emit(runtime: Runtime[RunContext] | None, event: str, **data: Any) -> None:
    """Push a custom progress event to ``stream_mode="custom"`` consumers (SSE clients)."""
    writer = getattr(runtime, "stream_writer", None)
    if writer is None:
        return
    with contextlib.suppress(Exception):  # streaming is best-effort; never break the run
        writer({"event": event, **data})


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(c.get("text") or (json.dumps(c.get("json")) if "json" in c else ""))
            else:
                parts.append(str(c))
        return "\n".join(p for p in parts if p)
    return str(content or "")


# ── Source registry ─────────────────────────────────────────────────────


def locator_for(finding: Finding, source: str) -> str:
    return f"code_execution:{finding.task_id}" if source.strip() == "code_execution" else source.strip()


@dataclass
class SourceRegistry:
    sources: list[Source]
    id_by_locator: dict[str, str]
    verified_ids: set[str]


def build_sources(findings: list[Finding]) -> SourceRegistry:
    """Deterministically number every distinct evidence source S1..Sn (stable across
    revisions, so the critic and the judge can check citations by id)."""
    sources: list[Source] = []
    id_by_loc: dict[str, str] = {}
    verified: set[str] = set()
    for f in findings:
        if f.status != "done":
            continue
        for ev in f.evidence:
            loc = locator_for(f, ev.source)
            if loc not in id_by_loc:
                sid = f"S{len(sources) + 1}"
                id_by_loc[loc] = sid
                kind = source_kind(loc)
                sources.append(Source(id=sid, locator=loc, kind=kind, title=f.source_titles.get(ev.source)))  # type: ignore[arg-type]
            if ev.verified:
                verified.add(id_by_loc[loc])
    for s in sources:
        s.verified = s.id in verified
    return SourceRegistry(sources, id_by_loc, verified)


def format_sources(reg: SourceRegistry) -> str:
    rows = []
    for s in reg.sources:
        flag = "verified" if s.id in reg.verified_ids else "UNVERIFIED"
        rows.append(f"[{s.id}] ({s.kind}, {flag}) {s.title + ' — ' if s.title else ''}{s.locator}")
    return "\n".join(rows) or "(no sources)"


def format_findings(
    findings: list[Finding], reg: SourceRegistry | None = None, max_chars: int = 12000
) -> str:
    reg = reg or build_sources(findings)
    blocks = []
    for f in findings:
        head = (
            f"### {f.task_id} ({f.agent}, round {f.round}, status={f.status}, "
            f"confidence={f.confidence:.2f}) — {f.objective}"
        )
        if f.status != "done":
            blocks.append(f"{head}\nFAILED: {f.error}")
            continue
        lines = [head, f"Summary: {f.summary}", "Evidence:"]
        for ev in f.evidence:
            sid = reg.id_by_locator.get(locator_for(f, ev.source), "?")
            claim = " ".join(ev.claim.split())
            lines.append(f"- {claim} [{sid}]{'' if ev.verified else ' (UNVERIFIED source)'}")
        if f.open_questions:
            lines.append("Open questions: " + "; ".join(f.open_questions))
        blocks.append("\n".join(lines))
    text = "\n\n".join(blocks) or "(no findings)"
    return text if len(text) <= max_chars else text[:max_chars] + "\n…[findings truncated]"


def report_to_text(report: ResearchReport) -> str:
    return report.to_markdown()
