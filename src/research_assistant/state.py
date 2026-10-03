# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Graph state.

State design principles used here:

* **Reducers for anything written by parallel branches.** Workers fan out via the
  ``Send`` API and run in the same super-step; ``findings`` / ``task_status`` / usage
  use additive or merge reducers so concurrent writes never clobber each other.
* **Typed, minimal, serialisable.** Everything is checkpointable (Pydantic models or
  plain JSON types) so a run can be paused (HITL), resumed, replayed or inspected.
* **Raw tool output never enters the parent state.** Workers return condensed
  ``Finding`` records; the transcript stays inside the worker subgraph. This is the core
  context-engineering decision that keeps the synthesizer's prompt small and on-topic.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from research_assistant.schemas import CritiqueRecord, Finding, ResearchPlan, ResearchReport, SubTask

TaskStatus = Literal["pending", "running", "done", "failed", "skipped"]


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow-merge reducer (right wins). Safe for concurrent writes to different keys."""
    out = dict(left or {})
    out.update(right or {})
    return out


def sum_usage(left: dict[str, float] | None, right: dict[str, float] | None) -> dict[str, float]:
    out = dict(left or {})
    for k, v in (right or {}).items():
        out[k] = out.get(k, 0) + v
    return out


class ResearchState(TypedDict, total=False):
    # input
    question: str
    # planning
    plan: ResearchPlan
    plan_feedback: str | None  # human reviewer feedback → triggers re-planning (HITL)
    tasks: dict[str, SubTask]  # written only by planner / supervisor (single writer)
    task_status: Annotated[dict[str, TaskStatus], merge_dicts]  # written concurrently by workers
    dispatch: list[str]  # task ids the supervisor wants to run in the current wave
    research_round: int
    supervisor_notes: Annotated[list[str], operator.add]
    critic_gaps: list[str]  # missing information flagged by the critic → supervisor
    # research
    findings: Annotated[list[Finding], operator.add]
    # writing & critique
    draft: ResearchReport
    critiques: Annotated[list[CritiqueRecord], operator.add]
    revision: int
    final_report: ResearchReport
    # bookkeeping
    errors: Annotated[list[str], operator.add]
    usage: Annotated[dict[str, float], sum_usage]


class WorkerInput(TypedDict):
    """Payload sent to a worker via ``Send`` (isolated from the full parent state)."""

    task: SubTask
    question: str
    context: list[Finding]  # findings of tasks this one depends on
    round: int
