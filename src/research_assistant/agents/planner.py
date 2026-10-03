# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Planner + human-in-the-loop plan review."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from research_assistant.agents.common import RunContext, emit
from research_assistant.config import Settings
from research_assistant.llm import structured
from research_assistant.prompts import PLANNER_SYSTEM, PLANNER_USER
from research_assistant.schemas import ResearchPlan
from research_assistant.state import ResearchState


def make_planner(llm: BaseChatModel, settings: Settings):
    chain = structured(llm, ResearchPlan, settings)

    async def planner(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
        user = PLANNER_USER.format(question=state["question"])
        if state.get("plan_feedback"):
            prev = state["plan"].model_dump_json(indent=1) if state.get("plan") else ""
            user += (
                f"\n\nPREVIOUS PLAN:\n{prev}\n\n"
                f"HUMAN REVIEWER FEEDBACK (must be addressed):\n{state['plan_feedback']}"
            )
        plan: ResearchPlan = await chain.ainvoke([SystemMessage(PLANNER_SYSTEM), HumanMessage(user)])
        plan.subtasks = plan.subtasks[:6]  # hard cap on fan-out regardless of model output
        emit(
            runtime,
            "plan",
            subtasks=[t.model_dump() for t in plan.subtasks],
            question_type=plan.question_type,
        )
        # cancel tasks from a rejected earlier plan (merge reducer cannot delete keys)
        stale = {
            tid: "skipped" for tid in (state.get("tasks") or {}) if tid not in {t.id for t in plan.subtasks}
        }
        return {
            "plan": plan,
            "plan_feedback": None,
            "tasks": {t.id: t for t in plan.subtasks},
            "task_status": {**stale, **{t.id: "pending" for t in plan.subtasks}},
            "research_round": 0,
            "revision": 0,
        }

    return planner


def make_plan_review(settings: Settings):
    """Optional HITL checkpoint. ``interrupt`` persists state via the checkpointer and
    surfaces the plan to the caller; the run resumes with
    ``Command(resume={"approved": bool, "feedback": str | None, "plan": dict | None})``."""

    def review_plan(state: ResearchState) -> dict[str, Any]:
        if not settings.require_plan_approval:
            return {}
        decision = interrupt({"type": "plan_review", "plan": state["plan"].model_dump()})
        decision = decision if isinstance(decision, dict) else {"approved": bool(decision)}
        if decision.get("plan"):  # reviewer edited the plan directly
            plan = ResearchPlan.model_validate(decision["plan"])
            stale = {
                tid: "skipped" for tid in state.get("tasks", {}) if tid not in {t.id for t in plan.subtasks}
            }
            return {
                "plan": plan,
                "tasks": {t.id: t for t in plan.subtasks},
                "task_status": {**stale, **{t.id: "pending" for t in plan.subtasks}},
            }
        if not decision.get("approved", False):
            return {
                "plan_feedback": decision.get("feedback")
                or "Plan rejected; propose a different decomposition."
            }
        return {}

    return review_plan


def route_after_review(state: ResearchState) -> str:
    return "planner" if state.get("plan_feedback") else "supervisor"
