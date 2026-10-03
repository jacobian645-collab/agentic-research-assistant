# Interview Guide — presenting this project as an AI Solution Architect

Use this to walk an interviewer (or an architecture review board) through the system. Numbers below come from
the repo's docs and cited public sources; re-check prices before quoting them.

---

## 1. The 60-second pitch

> "I built a reference architecture for a multi-agent research assistant on LangGraph. A planner decomposes the
> question into typed sub-tasks; a **hybrid supervisor** dispatches them in parallel — dependency logic in code,
> the LLM only decides whether gaps need more research. Each specialist — web, internal knowledge base, code —
> is a bounded tool-calling loop whose tools live behind **MCP servers**, so least privilege is structural:
> the web agent literally can't execute code. Every hop returns **validated structured output**, every citation
> is **verified in code** against raw tool output, and a **bounded self-critique** loop revises the draft.
> Quality is gated in CI by a **calibrated LLM-as-judge** with position-swap and deterministic metrics.
> It's Bedrock-first with model tiering — Sonnet for reasoning, Haiku for loops — which cuts cost about 39%,
> and it runs fully offline in CI with a deterministic fake model against the real MCP servers."

## 2. Whiteboard order (5 minutes)

1. **User → API (SSE) → graph.** Why streaming: 60-120 s runs.
2. **Planner → supervisor → N workers → supervisor** (draw the `Send` fan-out and the join).
3. **Workers → MCP servers** (three boxes, separate processes, separate networks; code server has none).
4. **Synthesizer → critic → (revise | research_more | accept)**; mark the bounds.
5. **Side panel:** budgets (steps, tools, rounds, revisions, $ cap), guardrails, checkpointer, judge loop.
6. **Deployment:** ECS/AgentCore, SQS for async, Aurora checkpointer, OpenSearch, Bedrock via PrivateLink.

## 3. Numbers worth remembering

| Fact | Value | Source |
|---|---|---|
| Multi-agent vs single-agent on Anthropic's internal research eval | +90.2% | Anthropic engineering blog, 2025 |
| Token usage: agents vs chat / multi-agent vs chat | ~4× / ~15× | same |
| Strong LLM judge agreement with humans | > 80% (≈ human-human) | Zheng et al., 2023 (MT-Bench) |
| RRF constant | k = 60 | Cormack et al., SIGIR 2009 |
| Bedrock regional vs global endpoint premium (Claude Sonnet 4.5+) | ~10% | Claude-on-Bedrock docs |
| This system's illustrative cost per run | ≈ $0.45 (≈ $0.73 single-tier) | architecture §11 |

## 4. Likely questions and strong answers

**Q: Why multi-agent at all? Isn't a single ReAct agent simpler?**
Yes — and for narrow tasks I'd use one. Multi-agent pays off when work is *breadth-first and parallelisable*
and exceeds one context window. It costs a lot more tokens, so the design contains that cost: cheap model in
loops, condensed findings, hard caps. Anthropic's own guidance is "start simple, add agency when it measurably
helps" — the eval harness is how I'd prove it helps.

**Q: Why not let the LLM supervisor route every step?**
Routing is mostly deterministic (dependencies, concurrency, budgets). Putting an LLM there adds latency, cost
and non-determinism with no quality gain. I reserve the LLM for the judgement call: "is anything material
missing?" That's ADR-0002.

**Q: Why MCP instead of plain function tools?**
Isolation and ownership. The code sandbox runs with no network and no credentials in its own container; the
web server has egress but no data access. Each server has its own IAM role and scales independently, and
other MCP clients can reuse them. And least privilege becomes structural — tools not bound to an agent can't be
called no matter what a prompt-injected page says.

**Q: How do you stop hallucinated citations?**
Three layers: the worker must copy a locator that appears *verbatim* in its tool output (checked in code); the
writer can cite only registry IDs (S1..Sn) built by code; the critic deterministically rejects unknown IDs and
low citation coverage. Citation validity is a release gate (≥ 95%).

**Q: How do you handle prompt injection from web pages?**
Defence in depth: delimit tool output as untrusted data with an explicit system rule; a pattern tripwire emits
events; flagged evidence is quarantined before synthesis; and least-privilege binding limits what a hijacked
agent could do anyway. In production add Bedrock Guardrails' prompt-attack filter. I'm explicit that pattern
matching is a tripwire, not a guarantee — capability restriction is the real control.

**Q: How do you know the LLM judge is right?**
I don't assume it. Separate stronger model, rubric with anchors, reasoning before scores, position swap for
pairwise, pinned prompt version, deterministic companion metrics, and calibration against ≥ 50 human labels
with κ ≥ 0.6 before it gates anything.

**Q: Self-critique uses the same model — isn't that circular?**
Partly, which is why it's bounded and paired with deterministic checks, and why release gating uses a separate
judge. Its main value is catching structural issues (unanswered sub-questions, missing citations) cheaply, and
routing *missing facts* back to research instead of letting the writer invent them.

**Q: What happens when Bedrock throttles under fan-out?**
Adaptive client retries, node-level retry policies, a `max_concurrency` cap, and a per-wave parallelism limit.
At scale: provisioned throughput or cross-region inference profiles for capacity, and SQS to smooth bursts.

**Q: How would you deploy this for 10,000 users?**
API on Fargate behind API Gateway + WAF + Cognito; accept-and-enqueue to SQS; autoscaled graph workers; MCP
servers as separate services (sandbox on AgentCore Code Interpreter or Firecracker); Aurora PostgreSQL
checkpointer; OpenSearch Serverless with document-level ACLs; Bedrock via PrivateLink with Guardrails;
OTel traces + per-run cost metrics; per-tenant rate limits and budgets. Or AgentCore Runtime + Gateway to
offload most of that.

**Q: Data residency for India / APAC customers?**
Configuration, not code: point model IDs at `apac.` inference profiles where available (~10% premium), keep
retrieval and checkpoint stores in-region, and record deviations when a model lacks an APAC profile.

**Q: What would you change for a regulated enterprise?**
Durable checkpointer for full audit replay, model-invocation logging with retention, PII redaction guardrails,
document-level authorization in retrieval, HITL approval on by default, signed tool-server images, and a model
risk record per prompt/model version linked to its eval report.

**Q: What are the weaknesses of your design?**
Plan quality bounds research quality; literal locator matching has false negatives; the fake LLM proves
plumbing, not quality; the bundled eval set is small; in-memory retrieval and checkpointer are demo-grade.
Each has a roadmap item in architecture §16.

## 5. Demo script (offline, 2 minutes)

```bash
make install
LLM_PROVIDER=fake ara run "What caused the March 2026 agent runaway incident and which controls prevent a repeat?"
# watch: plan → parallel workers → injection_suspected → supervisor follow-up → revise → accept
LLM_PROVIDER=fake ara graph            # Mermaid topology
LLM_PROVIDER=fake EVAL_MIN_OVERALL=2.5 ara eval
ara calibrate src/research_assistant/eval/datasets/calibration_sample.jsonl
```

Then switch `.env` to Bedrock and run the same question live.
