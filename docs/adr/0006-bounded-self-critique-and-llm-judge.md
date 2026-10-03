# ADR-0006 — Bounded self-critique at runtime; calibrated LLM-as-judge for release gating

**Status:** Accepted · **Date:** 2026-10-03

## Context
We need (a) per-answer quality improvement and (b) system-level quality measurement that can block a bad
prompt/model change. These are different problems with different inputs and trust requirements.

## Decision
* **Runtime self-critique (evaluator-optimizer):** rubric (coverage, groundedness, coherence; reason before
  score) + deterministic citation check → `accept | revise | research_more`, bounded by `MAX_REVISIONS` and
  `MAX_RESEARCH_ROUNDS`. "Missing information" routes to new research, not to rewriting.
* **Offline LLM-as-judge:** reference-guided rubric (correctness, groundedness, completeness, citation quality,
  clarity) with a *separate* judge model; pairwise A/B mode with **position swap** (inconsistent → tie);
  deterministic companions (key-point recall, citation validity, cost, latency); CI gates incl. "no regression
  > 0.2 vs baseline".
* **Calibration before trust:** ≥ 50 human-labelled items, quadratic-weighted κ ≥ 0.6 (`ara calibrate`).
* **Online:** sample `ONLINE_EVAL_SAMPLE_RATE` of production traffic for asynchronous judging.

## Consequences
+ Quality regressions are caught before merge/deploy; judge bias is measured, not assumed away.
− Judge calls cost money → nightly Bedrock evals, fake-LLM plumbing checks on every PR.
− Self-critique by the same model family shares its blind spots → that is why the judge is separate.
