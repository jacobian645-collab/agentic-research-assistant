# ADR-0003 — MCP as the tool boundary

**Status:** Accepted · **Date:** 2026-10-03

## Context
Tools carry the system's real risk: network egress, data access, code execution. In-process tool functions share
the orchestrator's credentials, dependencies, failure domain and release cadence.

## Decision
Expose every capability through a **Model Context Protocol server** — `web-search`, `knowledge-base`,
`code-sandbox` — one process/service per capability. The orchestrator connects with
`langchain-mcp-adapters` and binds **only the owning server's tools** to each specialist (`AGENT_SERVERS`).
Transports: `stdio` for dev/test (persistent sessions), **stateless Streamable HTTP** in containers.

## Consequences
+ Least privilege is structural, not prompt-based; injected content cannot reach unbound capabilities.
+ Per-server IAM role, network policy, resource limits, scaling and ownership.
+ Servers are reusable by any MCP client (IDEs, desktop assistants, other agents).
− Extra hop and process management; JSON-RPC overhead (negligible next to LLM latency).
− Tool outputs are third-party data → must be treated as untrusted (ADR-0005).
