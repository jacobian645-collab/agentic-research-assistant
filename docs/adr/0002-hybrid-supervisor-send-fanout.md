# ADR-0002 — Hybrid supervisor with `Send` fan-out

**Status:** Accepted · **Date:** 2026-10-03

## Context
Classic "LLM supervisor picks the next agent" routing costs a reasoning-tier call per hop, serialises work and
is non-deterministic where predictability matters most (budgets, ordering). Research sub-questions are mostly
independent and benefit from parallelism.

## Decision
1. A **planner** produces a typed `ResearchPlan` (tasks, agent per task, `depends_on`, success criteria).
2. The **supervisor dispatches deterministically**: every task whose dependencies are finished runs now, up to
   `MAX_PARALLEL_WORKERS`, via `Send("worker", payload)`; the `worker → supervisor` edge is the join.
3. The supervisor calls the LLM **only when a wave completes**, to decide whether gaps justify follow-up tasks
   (`SupervisorDecision`), bounded by `MAX_RESEARCH_ROUNDS`. Critic-identified gaps feed the same path.

## Consequences
+ Wall-clock ≈ number of *waves*, not number of tasks. Fewer expensive routing calls.
+ Dispatch logic is pure and unit-tested (cycle → `skipped`, failed deps don't block forever).
− Plan quality bounds research quality → mitigated by reflection rounds and optional human plan review.
− Workers can't talk to each other mid-flight (by design: context isolation); dependencies pass summaries only.
