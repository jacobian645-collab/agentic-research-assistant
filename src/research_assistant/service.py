# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Application service: owns MCP sessions + compiled graph; used by the CLI, API and evals."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Command

from research_assistant.agents.common import RunContext
from research_assistant.config import Settings
from research_assistant.graph import build_graph, initial_state, make_checkpointer, run_config
from research_assistant.guardrails import validate_question
from research_assistant.observability import UsageTracker, get_logger
from research_assistant.prompts import PROMPT_VERSION
from research_assistant.schemas import ResearchReport
from research_assistant.tools import MCPToolRegistry

log = get_logger("service")


@dataclass
class RunResult:
    thread_id: str
    status: str  # "completed" | "awaiting_approval"
    report: ResearchReport | None
    usage: dict[str, Any]
    trace: dict[str, Any] = field(default_factory=dict)
    interrupt: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "status": self.status,
            "report": self.report.model_dump() if self.report else None,
            "markdown": self.report.to_markdown() if self.report else None,
            "usage": self.usage,
            "trace": self.trace,
            "interrupt": self.interrupt,
        }


def _trace(state: dict[str, Any]) -> dict[str, Any]:
    """Compact, explainable run summary (what ran, what was verified, how it was judged)."""
    return {
        "prompt_version": PROMPT_VERSION,
        "plan": [t.model_dump() for t in state["plan"].subtasks] if state.get("plan") else [],
        "research_rounds": state.get("research_round", 0),
        "findings": [
            {
                "task_id": f.task_id,
                "agent": f.agent,
                "status": f.status,
                "round": f.round,
                "tool_calls": f.tool_calls,
                "evidence": len(f.evidence),
                "verified": sum(e.verified for e in f.evidence),
                "latency_s": f.latency_s,
            }
            for f in state.get("findings", [])
        ],
        "critiques": [
            {
                "revision": c.revision,
                "score": c.weighted_score,
                "verdict": c.verdict,
                "coverage": c.assessment.coverage,
                "groundedness": c.assessment.groundedness,
                "coherence": c.assessment.coherence,
                "citation_coverage": round(c.citation_check.citation_coverage, 2),
                "unknown_citations": c.citation_check.unknown_citations,
            }
            for c in state.get("critiques", [])
        ],
        "supervisor_notes": state.get("supervisor_notes", []),
        "errors": state.get("errors", []),
    }


class ResearchService:
    def __init__(self, settings: Settings, persistent_tools: bool = True) -> None:
        self.settings = settings
        self.registry = MCPToolRegistry(settings, persistent=persistent_tools)
        # Swap for langgraph-checkpoint-postgres / DynamoDB saver in production (durable HITL & replay).
        self.checkpointer = make_checkpointer()
        self.graph: Any = None

    async def __aenter__(self) -> ResearchService:
        await self.registry.__aenter__()
        self.graph = build_graph(self.settings, self.registry.toolsets(), checkpointer=self.checkpointer)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.registry.__aexit__(*exc)

    def _context(self, tracker: UsageTracker, user_id: str | None = None) -> RunContext:
        return RunContext(tracker=tracker, max_cost_usd=self.settings.max_cost_usd, user_id=user_id)

    async def _finish(self, thread_id: str, tracker: UsageTracker) -> RunResult:
        snap = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        state = snap.values
        if snap.next:  # paused at an interrupt
            intr = snap.tasks[0].interrupts[0].value if snap.tasks and snap.tasks[0].interrupts else None
            return RunResult(thread_id, "awaiting_approval", None, tracker.summary(), _trace(state), intr)
        return RunResult(thread_id, "completed", state.get("final_report"), tracker.summary(), _trace(state))

    async def run(self, question: str, thread_id: str | None = None, user_id: str | None = None) -> RunResult:
        q = validate_question(question, self.settings.max_question_chars)
        thread_id = thread_id or uuid.uuid4().hex
        tracker = UsageTracker()
        await self.graph.ainvoke(
            initial_state(q),
            config=run_config(self.settings, thread_id, [tracker]),
            context=self._context(tracker, user_id),
        )
        result = await self._finish(thread_id, tracker)
        log.info("run %s %s cost=$%s", thread_id, result.status, result.usage["est_cost_usd"])
        return result

    async def resume(self, thread_id: str, decision: dict[str, Any]) -> RunResult:
        tracker = UsageTracker()
        await self.graph.ainvoke(
            Command(resume=decision),
            config=run_config(self.settings, thread_id, [tracker]),
            context=self._context(tracker),
        )
        return await self._finish(thread_id, tracker)

    async def stream(self, question: str, thread_id: str | None = None) -> AsyncIterator[dict[str, Any]]:
        """Yield progress events (custom), node updates, then a final ``result`` event."""
        q = validate_question(question, self.settings.max_question_chars)
        thread_id = thread_id or uuid.uuid4().hex
        tracker = UsageTracker()
        yield {"event": "start", "thread_id": thread_id}
        async for ns, mode, chunk in self.graph.astream(
            initial_state(q),
            config=run_config(self.settings, thread_id, [tracker]),
            context=self._context(tracker),
            stream_mode=["custom", "updates"],
            subgraphs=True,
        ):
            if mode == "custom":
                yield chunk
            elif mode == "updates" and not ns:  # top-level node completions only
                for node in chunk:
                    yield {"event": "node", "node": node}
        result = await self._finish(thread_id, tracker)
        yield {"event": "result", **result.to_dict()}
