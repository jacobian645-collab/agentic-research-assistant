# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""FastAPI service.

Endpoints
---------
POST /v1/research                 → run to completion (or pause for plan approval)
POST /v1/research/stream          → Server-Sent Events: progress, tool calls, critique, result
POST /v1/research/{thread}/resume → human-in-the-loop decision on a paused plan
GET  /v1/research/{thread}        → current checkpointed state summary
GET  /healthz  /readyz            → liveness / readiness (MCP tools loaded)

Production notes: put behind an API gateway (authN/Z, WAF, rate limits); long research
runs belong on an async job queue (SQS + workers) with the SSE stream or webhooks for
progress — the synchronous endpoint is for short runs and demos.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from research_assistant.config import get_settings
from research_assistant.guardrails import GuardrailViolation
from research_assistant.observability import configure_logging, get_logger
from research_assistant.service import ResearchService, RunResult

log = get_logger("api")


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=4000)
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    approved: bool = True
    feedback: str | None = None
    plan: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    async with ResearchService(settings) as svc:
        app.state.service = svc
        app.state.background = set()
        log.info("service ready: provider=%s transport=%s", settings.llm_provider, settings.mcp_transport)
        yield


app = FastAPI(title="Agentic Research Assistant", version="0.1.0", lifespan=lifespan)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


def svc(request: Request) -> ResearchService:
    return request.app.state.service


async def _online_eval(service: ResearchService, question: str, result: RunResult) -> None:
    """Async LLM-as-judge on sampled production traffic (no reference answer → judge on
    groundedness/clarity/citations; completeness against the plan's success criteria)."""
    from research_assistant.eval.judge import LLMJudge
    from research_assistant.llm import get_chat_model

    try:
        judge = LLMJudge(get_chat_model("judge", service.settings), service.settings)
        v = await judge.grade(
            question,
            "(no reference — judge against the question only)",
            [],
            result.report.to_markdown() if result.report else "",
        )
        log.info(
            "online_eval",
            extra={
                "extra": {"thread_id": result.thread_id, "overall": v.overall, "groundedness": v.groundedness}
            },
        )
    except Exception:
        log.exception("online eval failed")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request) -> dict[str, Any]:
    s: ResearchService = request.app.state.service
    tools = {k: [t.name for t in v] for k, v in s.registry.tools_by_server.items()}
    if not all(tools.values()):
        raise HTTPException(503, detail={"tools": tools})
    return {"status": "ready", "tools": tools}


@app.post("/v1/research", dependencies=[Depends(require_key)])
async def research(req: ResearchRequest, service: ResearchService = Depends(svc)) -> dict[str, Any]:
    try:
        result = await service.run(req.question, req.thread_id)
    except GuardrailViolation as e:
        raise HTTPException(422, detail=str(e)) from e
    rate = service.settings.online_eval_sample_rate
    if result.report and rate > 0 and random.random() < rate:
        task = asyncio.create_task(_online_eval(service, req.question, result))
        app.state.background.add(task)
        task.add_done_callback(app.state.background.discard)
    return result.to_dict()


@app.post("/v1/research/stream", dependencies=[Depends(require_key)])
async def research_stream(req: ResearchRequest, service: ResearchService = Depends(svc)) -> StreamingResponse:
    async def gen() -> AsyncIterator[bytes]:
        try:
            async for ev in service.stream(req.question, req.thread_id):
                yield f"event: {ev.get('event', 'message')}\ndata: {json.dumps(ev, default=str)}\n\n".encode()
        except GuardrailViolation as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n".encode()
        except Exception as e:  # surface failures to the client instead of a dropped connection
            log.exception("stream failed")
            yield f"event: error\ndata: {json.dumps({'detail': type(e).__name__})}\n\n".encode()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/research/{thread_id}/resume", dependencies=[Depends(require_key)])
async def resume(
    thread_id: str, req: ResumeRequest, service: ResearchService = Depends(svc)
) -> dict[str, Any]:
    result = await service.resume(thread_id, req.model_dump())
    return result.to_dict()


@app.get("/v1/research/{thread_id}", dependencies=[Depends(require_key)])
async def get_state(thread_id: str, service: ResearchService = Depends(svc)) -> dict[str, Any]:
    snap = await service.graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snap.values:
        raise HTTPException(404, detail="unknown thread")
    v = snap.values
    return {
        "thread_id": thread_id,
        "next": list(snap.next),
        "task_status": v.get("task_status", {}),
        "revision": v.get("revision", 0),
        "final_report": v["final_report"].model_dump() if v.get("final_report") else None,
    }
