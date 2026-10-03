# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Supervisor: dependency-aware dispatcher + LLM reflection.

Hybrid control flow — deterministic where possible, LLM where judgement is needed:

* **Deterministic**: which tasks are ready (dependencies satisfied), how many run in
  parallel, budget enforcement, cycle protection. No tokens spent, fully predictable.
* **LLM (reasoning tier)**: only once a wave is complete — "is this enough, or what
  specific follow-up research is needed?" (also consumes gaps raised by the critic).
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Send

from research_assistant.agents.common import RunContext, ctx, emit, format_findings
from research_assistant.config import Settings
from research_assistant.llm import structured
from research_assistant.prompts import SUPERVISOR_SYSTEM, SUPERVISOR_USER
from research_assistant.schemas import SubTask, SupervisorDecision
from research_assistant.state import ResearchState, WorkerInput

FINISHED = {"done", "failed", "skipped"}


def _ready(state: ResearchState, limit: int) -> tuple[list[str], dict[str, str]]:
    tasks, status = state.get("tasks", {}), dict(state.get("task_status", {}))
    finished = {tid for tid, s in status.items() if s in FINISHED}
    pending = [t for t in tasks.values() if status.get(t.id) == "pending"]
    ready = [t.id for t in pending if all(d in finished or d not in tasks for d in t.depends_on)]
    updates: dict[str, str] = {}
    if pending and not ready and not any(s == "running" for s in status.values()):
        # dependency cycle / unsatisfiable → skip rather than deadlock
        updates = {t.id: "skipped" for t in pending}
    return ready[:limit], updates


def make_supervisor(llm: BaseChatModel, settings: Settings):
    chain = structured(llm, SupervisorDecision, settings)

    async def supervisor(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
        ready, skipped = _ready(state, settings.max_parallel_workers)
        if ready:
            emit(runtime, "dispatch", tasks=ready)
            return {"dispatch": ready, "task_status": {**skipped, **dict.fromkeys(ready, "running")}}

        rnd = state.get("research_round", 0)
        gaps = state.get("critic_gaps") or []
        if rnd >= settings.max_research_rounds or ctx(runtime).over_budget():
            reason = "budget exhausted" if ctx(runtime).over_budget() else "max research rounds reached"
            return {"dispatch": [], "task_status": skipped, "critic_gaps": [], "supervisor_notes": [reason]}

        plan = state["plan"]
        prompt = SUPERVISOR_USER.format(
            question=state["question"],
            round=rnd,
            max_rounds=settings.max_research_rounds,
            task_ids=", ".join(state.get("tasks", {})),
            criteria="\n".join(f"- {c}" for c in plan.success_criteria),
            findings=format_findings(state.get("findings", [])),
        )
        if gaps:
            prompt += "\n\nREVIEWER-IDENTIFIED GAPS (from self-critique):\n" + "\n".join(
                f"- {g}" for g in gaps
            )
        decision: SupervisorDecision = await chain.ainvoke(
            [SystemMessage(SUPERVISOR_SYSTEM), HumanMessage(prompt)]
        )
        emit(runtime, "supervisor_decision", action=decision.action, gaps=decision.gaps)

        if decision.action == "research_more" and decision.new_subtasks:
            existing = state.get("tasks", {})
            new: dict[str, SubTask] = {}
            for t in decision.new_subtasks[:3]:
                if t.id in existing or t.id in new:
                    t.id = f"r{rnd + 1}_{len(new) + 1}"
                new[t.id] = t
            dispatch = [tid for tid, t in new.items() if not [d for d in t.depends_on if d in new]]
            dispatch = dispatch[: settings.max_parallel_workers]
            status = {tid: ("running" if tid in dispatch else "pending") for tid in new}
            return {
                "tasks": {**existing, **new},
                "task_status": {**skipped, **status},
                "dispatch": dispatch,
                "research_round": rnd + 1,
                "critic_gaps": [],
                "supervisor_notes": [decision.reasoning],
            }
        return {
            "dispatch": [],
            "task_status": skipped,
            "critic_gaps": [],
            "supervisor_notes": [decision.reasoning],
        }

    return supervisor


def route_from_supervisor(state: ResearchState) -> list[Send] | str:
    """Fan out ready tasks to parallel workers (map step) or move on to synthesis."""
    dispatch = state.get("dispatch") or []
    if not dispatch:
        return "synthesizer"
    findings = state.get("findings", [])
    sends = []
    for tid in dispatch:
        task = state["tasks"][tid]
        context = [f for f in findings if f.task_id in task.depends_on]
        payload: WorkerInput = {
            "task": task,
            "question": state["question"],
            "context": context,
            "round": state.get("research_round", 0),
        }
        sends.append(Send("worker", payload))
    return sends
