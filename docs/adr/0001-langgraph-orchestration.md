# ADR-0001 — LangGraph as the orchestration runtime

**Status:** Accepted · **Date:** 2026-10-03

## Context
We need multi-step, multi-agent control flow with parallel branches, loops (critique, re-research), human
approval pauses, durable state for resume/replay, streaming progress, and fine-grained hooks for budgets and
verification. Agent frameworks differ mainly in how much of this they hide.

## Decision
Use **LangGraph** (1.x, stable API) as a *low-level* runtime: explicit `StateGraph` with typed state and
reducers, `Send` for map-reduce fan-out, `interrupt`/`Command` for HITL, checkpointers for durability,
`context_schema` for per-run non-persisted context, `stream_mode=["custom","updates"]` for SSE. We build the
nodes ourselves instead of using prebuilt agents.

## Consequences
+ Control flow is code-reviewable and unit-testable; loops are explicitly bounded.
+ Durable execution and HITL come from the runtime, not bespoke plumbing.
+ Portable: runs in a plain Python process, LangGraph Platform/Studio (`langgraph.json`), or AgentCore Runtime.
− More code than a prebuilt agent; team must understand reducers and super-step semantics.
− Framework dependency — mitigated by keeping domain logic (guardrails, schemas, servers) framework-free.

## Alternatives
CrewAI/AutoGen (higher-level, less control over state & persistence); Bedrock Agents multi-agent collaboration
(managed, less loop control, AWS-only); hand-rolled asyncio orchestration (re-implements checkpointing, HITL,
streaming).
