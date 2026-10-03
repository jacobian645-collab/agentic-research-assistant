# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Tool-calling specialist sub-agents.

Each specialist is a small LangGraph subgraph implementing a bounded ReAct loop:

    agent ──(tool_calls?)──▶ tools ──▶ agent ──▶ … ──▶ END
                └─(no tool calls / step budget hit)──▶ END

followed by a **structured extraction** step (``WorkerReport``) and **deterministic
grounding verification**. Only the condensed, verified ``Finding`` leaves the worker —
the raw transcript stays local (context isolation).

Why a hand-built loop instead of a prebuilt agent? We need hooks a prebuilt hides:
per-worker tool-call budgets, untrusted-output wrapping, injection scanning, structured
tool errors, tool-output capture for verification, and custom progress events.
"""

from __future__ import annotations

import asyncio
import json
import operator
import time
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from research_assistant.agents.common import RunContext, ctx, emit, message_text
from research_assistant.config import Settings
from research_assistant.guardrails import scan_for_injection, verify_evidence, wrap_untrusted
from research_assistant.llm import structured
from research_assistant.observability import get_logger
from research_assistant.prompts import INJECTION_NOTICE, WORKER_EXTRACT, WORKER_SYSTEM, WORKER_USER
from research_assistant.schemas import AgentName, Finding, WorkerReport
from research_assistant.state import WorkerInput

log = get_logger("worker")
TOOL_TIMEOUT_S = 60


class WorkerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    tool_calls: int
    tool_outputs: Annotated[list[str], operator.add]  # raw outputs, for verification
    used_code: bool
    injection_flags: Annotated[list[str], operator.add]


def build_worker_graph(agent: AgentName, llm: BaseChatModel, tools: list[BaseTool], settings: Settings):
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def call_model(state: WorkerState) -> dict[str, Any]:
        ai = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [ai], "steps": state["steps"] + 1}

    async def call_tools(state: WorkerState, runtime: Runtime[RunContext]) -> dict[str, Any]:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", []) or []
        budget = max(0, settings.max_tool_calls_per_worker - state["tool_calls"])
        if ctx(runtime).over_budget():
            budget = 0

        async def run_one(tc: dict[str, Any], allowed: bool) -> tuple[ToolMessage, str | None]:
            name = tc["name"]
            if not allowed:
                err: dict[str, Any] = {
                    "error": "tool_budget_exhausted",
                    "hint": "Answer now with the information you have.",
                }
                return ToolMessage(json.dumps(err), tool_call_id=tc["id"], name=name, status="error"), None
            tool = tools_by_name.get(name)
            if tool is None:  # model asked for a capability it was not granted
                err = {"error": "tool_not_permitted", "allowed": sorted(tools_by_name)}
                return ToolMessage(json.dumps(err), tool_call_id=tc["id"], name=name, status="error"), None
            emit(runtime, "tool_call", agent=agent, tool=name, args=tc.get("args", {}))
            try:
                result = await asyncio.wait_for(tool.ainvoke({**tc, "type": "tool_call"}), TOOL_TIMEOUT_S)
                raw = message_text(getattr(result, "content", result))
            except Exception as e:  # structured error → model can recover, loop stays bounded
                raw = json.dumps({"error": type(e).__name__, "detail": str(e)[:500]})
            return ToolMessage(
                wrap_untrusted(raw, settings.tool_output_char_limit), tool_call_id=tc["id"], name=name
            ), raw

        results = await asyncio.gather(*(run_one(tc, i < budget) for i, tc in enumerate(calls)))
        raws = [r for _, r in results if r]
        used_code = state.get("used_code", False) or any(
            '"exit_code": 0' in r for (m, r) in results if r and m.name == "run_python"
        )
        flags = [f for r in raws for f in scan_for_injection(r)]
        if flags:
            emit(runtime, "injection_suspected", agent=agent, patterns=flags)
            log.warning(
                "prompt-injection patterns in tool output", extra={"extra": {"agent": agent, "flags": flags}}
            )
        return {
            "messages": [m for m, _ in results],
            "tool_calls": state["tool_calls"] + min(len(calls), budget),
            "tool_outputs": raws,
            "used_code": used_code,
            "injection_flags": flags,
        }

    def route(state: WorkerState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and state["steps"] < settings.max_worker_steps:
            return "tools"
        return END

    g = StateGraph(WorkerState, context_schema=RunContext)
    g.add_node("agent", call_model)
    g.add_node("tools", call_tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(name=agent)


def _harvest_titles(raw_outputs: list[str]) -> dict[str, str]:
    titles: dict[str, str] = {}

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            loc = o.get("url") or o.get("doc_id")
            if isinstance(loc, str) and o.get("title"):
                titles.setdefault(loc, str(o["title"]))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for raw in raw_outputs:
        try:
            walk(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            continue
    return titles


def _transcript(messages: list[AnyMessage], tool_outputs: list[str], limit: int) -> str:
    """Compact, provider-neutral transcript for the extraction call (avoids replaying
    tool_use/tool_result blocks, which some providers reject without tool config)."""
    notes = [
        message_text(m.content) for m in messages if isinstance(m, AIMessage) and message_text(m.content)
    ]
    outs = [wrap_untrusted(o, limit) for o in tool_outputs]
    return "AGENT NOTES:\n" + "\n".join(notes) + "\n\nTOOL RESULTS:\n" + "\n\n".join(outs)


def make_worker_node(fast_llm: BaseChatModel, toolsets: dict[AgentName, list[BaseTool]], settings: Settings):
    graphs = {a: build_worker_graph(a, fast_llm, tools, settings) for a, tools in toolsets.items()}
    extract = structured(fast_llm, WorkerReport, settings)

    async def worker(inp: WorkerInput, runtime: Runtime[RunContext]) -> dict[str, Any]:
        task, t0 = inp["task"], time.perf_counter()
        emit(runtime, "worker_start", task_id=task.id, agent=task.agent, objective=task.objective)
        try:
            context = ""
            if inp["context"]:
                context = "CONTEXT FROM PRIOR TASKS:\n" + "\n".join(
                    f"- [{f.task_id}] {f.summary}" for f in inp["context"] if f.status == "done"
                )
            out = await graphs[task.agent].ainvoke(
                {
                    "messages": [
                        SystemMessage(WORKER_SYSTEM[task.agent] + "\n" + INJECTION_NOTICE),
                        HumanMessage(
                            WORKER_USER.format(
                                question=inp["question"],
                                task_id=task.id,
                                objective=task.objective,
                                context=context,
                            )
                        ),
                    ],
                    "steps": 0,
                    "tool_calls": 0,
                    "tool_outputs": [],
                    "used_code": False,
                    "injection_flags": [],
                },
                context=ctx(runtime),
                config={"recursion_limit": settings.max_worker_steps * 2 + 4},
            )
            report: WorkerReport = await extract.ainvoke(
                [
                    SystemMessage(
                        "You convert research transcripts into structured reports. " + INJECTION_NOTICE
                    ),
                    HumanMessage(
                        _transcript(out["messages"], out["tool_outputs"], settings.tool_output_char_limit)
                        + "\n\n"
                        + WORKER_EXTRACT.format(task_id=task.id)
                    ),
                ]
            )
            evidence = verify_evidence(report, out["tool_outputs"], out["used_code"])
            # quarantine: evidence carrying injection patterns never reaches the synthesizer
            clean = [e for e in evidence if not scan_for_injection(f"{e.claim} {e.quote or ''}")]
            if len(clean) < len(evidence):
                report.open_questions.append(
                    f"{len(evidence) - len(clean)} evidence item(s) quarantined: prompt-injection patterns."
                )
                emit(runtime, "evidence_quarantined", task_id=task.id, count=len(evidence) - len(clean))
            evidence = clean
            finding = Finding(
                task_id=task.id,
                agent=task.agent,
                objective=task.objective,
                status="done",
                summary=report.summary,
                evidence=evidence,
                confidence=report.confidence,
                open_questions=report.open_questions,
                tool_calls=out["tool_calls"],
                latency_s=round(time.perf_counter() - t0, 2),
                round=inp["round"],
                source_titles=_harvest_titles(out["tool_outputs"]),
            )
        except Exception as e:  # graceful degradation: one failed specialist ≠ failed run
            log.exception("worker %s failed", task.id)
            finding = Finding(
                task_id=task.id,
                agent=task.agent,
                objective=task.objective,
                status="failed",
                error=f"{type(e).__name__}: {e}"[:500],
                latency_s=round(time.perf_counter() - t0, 2),
                round=inp["round"],
            )
        emit(
            runtime,
            "worker_done",
            task_id=task.id,
            status=finding.status,
            evidence=len(finding.evidence),
            verified=sum(e.verified for e in finding.evidence),
            latency_s=finding.latency_s,
        )
        return {
            "findings": [finding],
            "task_status": {task.id: finding.status},
            "errors": [f"{task.id}: {finding.error}"] if finding.error else [],
        }

    return worker
