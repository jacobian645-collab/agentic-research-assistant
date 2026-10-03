# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""End-to-end: real LangGraph graph + real MCP servers (stdio subprocesses) + fake LLM."""

from __future__ import annotations

import pytest

from research_assistant import observability
from research_assistant.service import ResearchService

Q = "What caused the March 2026 agent runaway incident and which controls prevent a repeat?"


async def test_full_run_completes_with_verified_citations(settings):
    async with ResearchService(settings) as svc:
        res = await svc.run(Q)
    assert res.status == "completed" and res.report is not None
    t = res.trace
    # plan → 2 parallel workers → dependent code task → supervisor-requested follow-up
    assert {f["task_id"] for f in t["findings"]} >= {"t1", "t2", "t3"}
    assert t["research_rounds"] == 1
    assert all(f["status"] == "done" for f in t["findings"])
    # self-critique loop: first draft revised, second accepted
    assert [c["verdict"] for c in t["critiques"]] == ["revise", "accept"]
    # every citation in the final report resolves to a verified source
    cited = res.report.cited_ids()
    valid = {r.id for r in res.report.references if r.verified}
    assert cited and cited <= valid
    assert res.usage["llm_calls"] > 0 and res.usage["tool_calls"]


async def test_least_privilege_tool_scoping(settings):
    async with ResearchService(settings) as svc:
        ts = svc.registry.toolsets()
    assert [t.name for t in ts["code_analyst"]] == ["run_python"]
    assert "run_python" not in {t.name for t in ts["web_researcher"]}
    assert "fetch_url" not in {t.name for t in ts["retrieval_researcher"]}


async def test_human_in_the_loop_plan_review(settings):
    settings.require_plan_approval = True
    async with ResearchService(settings) as svc:
        paused = await svc.run(Q)
        assert paused.status == "awaiting_approval" and paused.interrupt["type"] == "plan_review"
        again = await svc.resume(
            paused.thread_id, {"approved": False, "feedback": "add more internal sources"}
        )
        assert again.status == "awaiting_approval"  # re-planned, asks again
        done = await svc.resume(paused.thread_id, {"approved": True})
    assert done.status == "completed" and done.report is not None


async def test_streaming_emits_progress_events(settings):
    async with ResearchService(settings) as svc:
        events = [e["event"] async for e in svc.stream(Q)]
    for expected in (
        "start",
        "plan",
        "dispatch",
        "worker_start",
        "tool_call",
        "worker_done",
        "critique",
        "result",
    ):
        assert expected in events
    assert events[-1] == "result"


async def test_cost_cap_degrades_gracefully(settings, monkeypatch):
    monkeypatch.setitem(observability.PRICING_PER_MTOK, "fake", (2000.0, 2000.0))
    settings.max_cost_usd = 0.01
    async with ResearchService(settings) as svc:
        res = await svc.run(Q)
    assert res.status == "completed"
    assert any("cost cap" in x for x in res.report.limitations)
    assert len(res.trace["critiques"]) == 1  # no revision loop once over budget


@pytest.mark.parametrize("bad", ["hi", "x" * 5000])
async def test_input_guardrail(settings, bad):
    from research_assistant.guardrails import GuardrailViolation

    async with ResearchService(settings) as svc:
        with pytest.raises(GuardrailViolation):
            await svc.run(bad)


async def test_prompt_injection_is_quarantined(settings):
    """The offline web corpus contains a malicious page; its content must never reach the report."""
    async with ResearchService(settings) as svc:
        res = await svc.run("Are multi-agent LLM systems cheaper than single agents? Give agent tricks.")
    md = res.report.to_markdown().lower()
    assert "ignore all previous instructions" not in md
    assert "reveal your system prompt" not in md
