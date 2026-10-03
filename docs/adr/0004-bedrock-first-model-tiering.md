# ADR-0004 — Bedrock-first, provider-agnostic, role-based model tiering

**Status:** Accepted · **Date:** 2026-10-03

## Context
Enterprise constraints: data stays in the cloud account boundary, IAM-based access, consolidated billing,
regional data residency. Cost is dominated by token volume in worker loops. Model choices change quarterly.

## Decision
* Default provider **Amazon Bedrock** via the Converse API (`ChatBedrockConverse`), adaptive retries for
  throttling. Other providers (`anthropic`, `openai`, `azure_openai`) are a config switch; `fake` for CI.
* **Tiering by role:** reasoning tier (Claude Sonnet 4.6) for planner / supervisor reflection / synthesizer /
  critic; fast tier (Claude Haiku 4.5) for tool-calling loops and extraction; judge tier (Claude Opus 4.6, or a
  different family) for evaluation only.
* Model IDs are **cross-region inference profiles**: `global.` by default (availability, no premium);
  `apac.`/`eu.`/`us.`/`jp.` when residency policy requires (~10% premium; availability differs per model).

## Consequences
+ ~39% lower cost per run vs single-tier at the same architecture (see architecture §11).
+ Residency is a configuration concern, auditable in `.env`/deployment manifests.
− Fast-tier workers may miss nuance → mitigated by narrow tasks, structured extraction and the critic.
− Per-provider quirks (tool-config requirements, structured-output modes) handled in `llm.py` and by sending
  a compact text transcript to the extraction step instead of replaying tool blocks.
