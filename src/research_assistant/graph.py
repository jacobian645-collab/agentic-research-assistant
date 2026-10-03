# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Graph assembly — the orchestration topology in one place.

```
START → planner → review_plan ─┬─(feedback)→ planner
                               └─→ supervisor ⇄ worker x N   (Send fan-out, parallel)
                                      │
                                      └─(no more work)→ synthesizer → critic ─┬─ accept → finalize → END
                                                              ▲               ├─ revise ─┘ (→ synthesizer)
                                                              │               └─ research_more → supervisor
```

Patterns combined: plan-and-execute, orchestrator-workers (supervisor + Send map-reduce),
evaluator-optimizer (critic loop), human-in-the-loop (interrupt), all bounded by explicit
budgets + LangGraph's ``recursion_limit``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from research_assistant.agents.common import RunContext
from research_assistant.agents.planner import make_plan_review, make_planner, route_after_review
from research_assistant.agents.supervisor import make_supervisor, route_from_supervisor
from research_assistant.agents.workers import make_worker_node
from research_assistant.agents.writer import make_critic, make_finalize, make_synthesizer, route_from_critic
from research_assistant.config import Settings
from research_assistant.llm import get_chat_model
from research_assistant.schemas import AgentName
from research_assistant.state import ResearchState

# Explicit allow-list of types that may be deserialised from checkpoints (supply-chain hardening:
# a tampered checkpoint store cannot instantiate arbitrary classes).
CHECKPOINT_TYPES = [
    ("research_assistant.schemas", name)
    for name in (
        "Finding",
        "VerifiedEvidence",
        "CritiqueRecord",
        "CritiqueAssessment",
        "CitationCheck",
        "ResearchPlan",
        "SubTask",
        "ResearchReport",
        "ReportSection",
        "Claim",
        "Source",
    )
]


def make_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))


LLM_RETRY = RetryPolicy(max_attempts=2, initial_interval=1.0, backoff_factor=2.0)


def build_graph(
    settings: Settings,
    toolsets: dict[AgentName, list[BaseTool]],
    *,
    checkpointer: Any | None = None,
    reasoning_llm: BaseChatModel | None = None,
    fast_llm: BaseChatModel | None = None,
):
    reasoning = reasoning_llm or get_chat_model("reasoning", settings)
    fast = fast_llm or get_chat_model("fast", settings)

    g = StateGraph(ResearchState, context_schema=RunContext)
    g.add_node("planner", make_planner(reasoning, settings), retry_policy=LLM_RETRY)
    g.add_node("review_plan", make_plan_review(settings))
    g.add_node("supervisor", make_supervisor(reasoning, settings), retry_policy=LLM_RETRY)
    g.add_node("worker", make_worker_node(fast, toolsets, settings))
    g.add_node("synthesizer", make_synthesizer(reasoning, settings), retry_policy=LLM_RETRY)
    g.add_node("critic", make_critic(reasoning, settings), retry_policy=LLM_RETRY)
    g.add_node("finalize", make_finalize(settings))

    g.add_edge(START, "planner")
    g.add_edge("planner", "review_plan")
    g.add_conditional_edges("review_plan", route_after_review, ["planner", "supervisor"])
    g.add_conditional_edges("supervisor", route_from_supervisor, ["worker", "synthesizer"])
    g.add_edge("worker", "supervisor")  # join: supervisor runs once after the whole wave
    g.add_edge("synthesizer", "critic")
    g.add_conditional_edges("critic", route_from_critic, ["finalize", "synthesizer", "supervisor"])
    g.add_edge("finalize", END)

    if checkpointer is None and settings.require_plan_approval:
        checkpointer = make_checkpointer()  # interrupts need a checkpointer
    return g.compile(checkpointer=checkpointer, name="research_assistant")


def initial_state(question: str) -> dict[str, Any]:
    return {"question": question, "findings": [], "critiques": [], "errors": [], "supervisor_notes": []}


def run_config(
    settings: Settings, thread_id: str | None = None, callbacks: list[Any] | None = None
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "recursion_limit": settings.recursion_limit,
        "max_concurrency": settings.max_parallel_workers + 2,
    }
    if thread_id:
        cfg["configurable"] = {"thread_id": thread_id}
    if callbacks:
        cfg["callbacks"] = callbacks
    return cfg
