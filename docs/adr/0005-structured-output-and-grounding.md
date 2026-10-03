# ADR-0005 — Structured output everywhere + deterministic grounding verification

**Status:** Accepted · **Date:** 2026-10-03

## Context
Free-text hand-offs between agents are brittle (parsing) and unauditable. The most damaging research-agent
failure is a confident claim with a fabricated citation.

## Decision
* Every LLM hop that drives code returns a Pydantic model via tool-calling structured output, with validation
  retries. LLM-facing schemas are separate from system records enriched by code.
* **Grounding verification in code:** an evidence item is `verified` only if its source locator appears
  literally in that worker's raw tool output (or `code_execution` after a successful run).
* **Source registry:** code numbers sources `S1..Sn`; the writer may cite only those IDs; the critic checks
  unknown IDs and citation coverage deterministically.
* **Quarantine:** evidence whose text matches prompt-injection patterns is removed before synthesis.

## Consequences
+ Fabricated URLs/doc IDs are caught without an LLM; citation validity becomes a hard release metric.
+ Reports are machine-checkable and renderable to any format.
− Literal matching can reject legitimate paraphrased locators (false negatives) → acceptable: unverified
  evidence is flagged, not silently trusted.
