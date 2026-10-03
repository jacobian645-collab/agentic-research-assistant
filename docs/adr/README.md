# Architecture Decision Records

Format: [Michael Nygard ADR](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) —
Context → Decision → Consequences → Alternatives. ADRs are immutable once accepted; supersede, don't edit.

| # | Decision | Status |
|---|----------|--------|
| [0001](0001-langgraph-orchestration.md) | LangGraph as the orchestration runtime | Accepted |
| [0002](0002-hybrid-supervisor-send-fanout.md) | Hybrid supervisor: deterministic dispatch + LLM reflection, `Send` fan-out | Accepted |
| [0003](0003-mcp-tool-boundary.md) | MCP as the tool boundary; one server per capability; least-privilege binding | Accepted |
| [0004](0004-bedrock-first-model-tiering.md) | Bedrock-first, provider-agnostic, role-based model tiering | Accepted |
| [0005](0005-structured-output-and-grounding.md) | Structured output everywhere + deterministic grounding verification | Accepted |
| [0006](0006-bounded-self-critique-and-llm-judge.md) | Bounded self-critique at runtime; calibrated LLM-as-judge for release gating | Accepted |
| [0007](0007-sandboxed-code-execution.md) | Layered isolation for code execution | Accepted |
