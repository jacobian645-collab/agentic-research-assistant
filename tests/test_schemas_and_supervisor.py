# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

import pytest
from langgraph.types import Send
from pydantic import ValidationError

from research_assistant.agents.supervisor import _ready, route_from_supervisor
from research_assistant.schemas import Claim, ReportSection, ResearchPlan, ResearchReport, Source, SubTask


def _plan(**kw):
    base = {"question_type": "analytical", "key_questions": ["q"], "success_criteria": ["c"]}
    return ResearchPlan(**base, **kw)


def test_plan_rejects_duplicate_ids_and_drops_unknown_deps():
    with pytest.raises(ValidationError):
        _plan(
            subtasks=[
                SubTask(id="t1", agent="web_researcher", objective="a"),
                SubTask(id="t1", agent="code_analyst", objective="b"),
            ]
        )
    p = _plan(subtasks=[SubTask(id="t1", agent="web_researcher", objective="a", depends_on=["t1", "zz"])])
    assert p.subtasks[0].depends_on == []


def test_ready_respects_dependencies_and_parallel_limit():
    tasks = {
        "t1": SubTask(id="t1", agent="web_researcher", objective="a"),
        "t2": SubTask(id="t2", agent="retrieval_researcher", objective="b"),
        "t3": SubTask(id="t3", agent="code_analyst", objective="c", depends_on=["t1", "t2"]),
    }
    state = {"tasks": tasks, "task_status": dict.fromkeys(tasks, "pending")}
    ready, _ = _ready(state, limit=4)
    assert ready == ["t1", "t2"]
    ready, _ = _ready(state, limit=1)
    assert ready == ["t1"]
    state["task_status"].update({"t1": "done", "t2": "failed"})  # failed deps don't block forever
    assert _ready(state, 4)[0] == ["t3"]


def test_dependency_cycle_is_skipped_not_deadlocked():
    tasks = {
        "a": SubTask(id="a", agent="web_researcher", objective="x", depends_on=["b"]),
        "b": SubTask(id="b", agent="web_researcher", objective="y", depends_on=["a"]),
    }
    ready, skipped = _ready({"tasks": tasks, "task_status": {"a": "pending", "b": "pending"}}, 4)
    assert ready == [] and skipped == {"a": "skipped", "b": "skipped"}


def test_route_fans_out_with_send():
    t = SubTask(id="t1", agent="web_researcher", objective="a")
    sends = route_from_supervisor({"dispatch": ["t1"], "tasks": {"t1": t}, "question": "q", "findings": []})
    assert isinstance(sends, list) and isinstance(sends[0], Send) and sends[0].node == "worker"
    assert route_from_supervisor({"dispatch": []}) == "synthesizer"


def test_report_markdown_lists_only_cited_references():
    r = ResearchReport(
        title="T",
        executive_summary="E",
        confidence="medium",
        question="Q",
        sections=[ReportSection(heading="H", claims=[Claim(text="c", citations=["S2"])])],
        references=[
            Source(id="S1", locator="https://a.org", kind="web"),
            Source(id="S2", locator="kb://x.md#0", kind="document", title="X"),
        ],
    )
    md = r.to_markdown()
    assert "[S2]" in md and "https://a.org" not in md
