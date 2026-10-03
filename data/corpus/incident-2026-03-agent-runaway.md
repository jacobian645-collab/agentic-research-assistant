# Post-incident review: INC-2026-0312 research agent runaway

*Owner: SRE · Severity: SEV-2 · Classification: Internal · Fictional sample document for the demo knowledge base.*

## Summary

On 12 March 2026 a prototype research agent entered a loop in which it repeatedly called the web search tool with near-identical queries. Over six hours it made 412 tool calls and roughly 2,300 model calls, incurring USD 1,870 in model charges before an engineer stopped it manually.

## Root cause

The agent was a single ReAct loop with no maximum step count, no per-run cost budget and no loop detection. A fetch tool returned an error string that the model interpreted as an instruction to retry with a slightly different query, so the stop condition was never reached.

## Contributing factors

- No alert on per-run spend; the daily budget alert fired only the next morning.
- Tool errors were returned as free text rather than structured error objects.
- The agent ran with the same credentials as the developer, giving it broader access than needed.

## Remediation (completed)

1. Hard recursion / step limits on every agent graph, plus a per-worker tool-call cap.
2. Per-request cost cap of USD 0.50 with partial-answer fallback.
3. Structured tool errors and bounded retries with backoff.
4. Per-agent least-privilege credentials and per-tool IAM roles via MCP servers.
5. Real-time cost telemetry per run with alerting.
