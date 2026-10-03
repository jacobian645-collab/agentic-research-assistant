# ADR-0007 — Layered isolation for code execution

**Status:** Accepted · **Date:** 2026-10-03

## Context
The code analyst executes model-written Python. Model output must be treated as untrusted (OWASP LLM05), and
the input to the model may itself be attacker-controlled (indirect injection).

## Decision
Defence in depth inside the `code-sandbox` MCP server, each layer assuming the previous can fail:
1. AST policy: import allow-list, no dunder attribute access, no `exec/eval/open/__import__/getattr…`.
2. Fresh `python -I` process per call, empty temp CWD, scrubbed env (no credentials).
3. POSIX rlimits (CPU, address space, file size, open files) + wall-clock kill.
4. Container: internal-only network (no egress), read-only root FS, `cap_drop: ALL`, `no-new-privileges`,
   memory/CPU/PID limits.
5. Production multi-tenant: micro-VM isolation (Bedrock AgentCore Code Interpreter, Firecracker, gVisor) behind
   the *same* MCP tool contract.

## Consequences
+ A compromised snippet has no network, no secrets and bounded resources.
− AST deny-lists are bypassable in principle — which is why they are layer 1, not the only layer.
− Allowed libraries limit analysis (no plotting/file I/O) — acceptable for a research calculator.
