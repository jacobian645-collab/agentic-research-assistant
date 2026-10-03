# RAG and Agent Evaluation Playbook

*Owner: Applied AI Quality · Classification: Internal · Fictional sample document for the demo knowledge base.*

## What we measure

- Answer correctness against a reference answer and a list of required key points.
- Groundedness (faithfulness): every claim is supported by retrieved or cited evidence.
- Completeness: all sub-questions are addressed.
- Citation quality: citations exist, resolve to real sources and support the claim.
- Operational metrics: latency, tokens, cost per request, tool calls, failure rate.

## LLM-as-judge rules

1. Use a rubric with anchored 1-5 scales and ask the judge to reason before scoring.
2. Use a judge from a different model family, or at least a stronger model, than the system under test to limit self-preference bias.
3. For pairwise comparisons, run both orderings (A/B and B/A) and only count consistent verdicts; inconsistent verdicts are ties.
4. Calibrate before trusting: label at least 50 examples by hand and require Cohen's kappa of at least 0.6 (or Spearman correlation of at least 0.7) between judge and human scores.
5. Pin the judge model version and prompt version; re-calibrate when either changes.

## Release gating

Offline evaluation runs in CI on every pull request that touches prompts, models or graph logic. A drop of more than 0.2 in mean judge score versus the main branch blocks the merge. In production, sample 5 percent of traffic for asynchronous online judging and alert on a rolling-mean drop.
