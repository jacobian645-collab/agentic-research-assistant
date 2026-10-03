# LLM Cost Governance Policy (FY26)

*Owner: FinOps Guild · Classification: Internal · Fictional sample document for the demo knowledge base.*

## Targets per request

For research-assistant style agents the cost target is a p50 of USD 0.25 per request and a p95 of USD 0.45 per request. The hard cap is USD 0.50 per request, enforced in code; a run that would exceed the cap must stop and return a partial answer with a limitation note.

## Cost levers (in order of impact)

1. Model tiering: route tool-calling loops to the fast tier; reserve the reasoning tier for planning, synthesis and critique. In our benchmarks this cut cost per request by 58 percent with no measurable quality loss.
2. Context hygiene: sub-agents return condensed findings instead of raw tool transcripts; tool outputs are truncated to a fixed character budget.
3. Prompt caching for long, stable system prompts.
4. Bounded loops: caps on research rounds, revisions and tool calls per worker.

## Chargeback and monitoring

Token usage is tagged per application and per business unit through Bedrock application inference profiles and charged back monthly. Each application must emit tokens, calls and estimated cost per request to the central observability stack, and alert when the daily spend exceeds 150 percent of the trailing 7-day average.
