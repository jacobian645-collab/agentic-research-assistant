# Evaluation Strategy

> "You can't improve what you don't measure — and you can't trust a measurement you haven't calibrated."

Agentic systems fail in ways unit tests cannot see: a plausible but wrong synthesis, a missing sub-question, a
citation that points nowhere, a 3× cost regression after a prompt tweak. This project treats evaluation as a
**release-gating subsystem**, with three layers:

| Layer | What it checks | When | Cost |
|---|---|---|---|
| **Deterministic tests** (`pytest`) | Graph mechanics, budgets, HITL, tool scoping, sandbox, SSRF, injection quarantine, API | Every commit (CI) | Free (fake LLM, real MCP servers) |
| **Offline LLM-as-judge** (`ara eval`) | Answer quality vs reference answers on a versioned dataset | Nightly + on demand (Bedrock); plumbing smoke on every PR | $ (judge + system calls) |
| **Online sampled judging** | Quality drift on real traffic (no reference) | Continuously, `ONLINE_EVAL_SAMPLE_RATE` | ¢ per sampled request |

---

## 1. Dataset

`src/research_assistant/eval/datasets/research_eval.jsonl` — one JSON object per line:

```json
{"id": "cost-calc",
 "question": "...",
 "reference_answer": "Expert-written answer with the key numbers...",
 "key_points": ["total about $0.495", "hard cap of USD 0.50", "..."],
 "tags": ["quantitative", "code", "kb", "web"]}
```

Design rules:

* **Coverage by capability:** every item is tagged with the specialists it should need (`web`, `kb`, `code`) so
  failures can be sliced by sub-agent.
* **Key points are atomic and checkable** — they feed both the judge (completeness) and a deterministic
  lexical recall metric (a sanity check on the judge).
* **Mix of difficulty:** single-source facts, cross-source synthesis (public + internal), numeric reasoning,
  and adversarial content (the offline corpus contains a prompt-injection page).
* **Version it** like code; never edit an item in place once it has baseline scores — add a new id.
* **Grow to ≥ 50 items** before using the judge as a hard release gate (the bundled 6 items are a seed set).

---

## 2. Judge design (`eval/judge.py`)

**Pointwise, reference-guided rubric** — five criteria scored 1-5 with explicit anchors:

| Criterion | Question the judge answers |
|---|---|
| correctness | Does it agree with the reference answer? Any contradictions? |
| groundedness | Are claims supported by the cited references (no fabrication)? |
| completeness | Are all required key points covered? |
| citation_quality | Are citations present, specific and relevant per claim? |
| clarity | Direct answer first, well structured, concise? |

The schema puts `reasoning` **before** the scores (chain-of-thought-then-score improves agreement with humans,
cf. G-Eval) and asks for `key_points_covered / key_points_missing` so failures are explainable.

**Pairwise mode** (`LLMJudge.compare`) for A/B testing two system versions: the comparison runs **twice with
positions swapped**; only a consistent verdict counts, otherwise it is a tie. `tests/test_eval.py` proves this
neutralises a judge that always prefers position A.

### Bias controls

| Known judge bias | Control in this repo |
|---|---|
| Position bias | Swap order in pairwise; inconsistent → tie |
| Verbosity bias | Rubric says "do not reward length"; clarity criterion rewards concision |
| Self-preference | Separate judge model (`JUDGE_MODEL`, optionally `JUDGE_PROVIDER` of another family) |
| Score drift | Pinned judge prompt version (`JUDGE_PROMPT_VERSION`) + model id stamped in every report |
| Leniency / miscalibration | Anchored scale + calibration against human labels (below) |
| Single-metric gaming | Deterministic companions: key-point recall, citation validity, cost, latency |

---

## 3. Calibration (`ara calibrate`)

Before a judge score may gate a release, measure agreement with humans on a labelled sample:

```bash
ara calibrate path/to/labels.jsonl     # rows: {"id", "human_score": 1-5, "judge_score": 1-5}
```

Reports quadratic-weighted **Cohen's κ**, **Spearman ρ**, exact and within-1 agreement, and mean bias
(judge − human). Policy used here: trusted only when **n ≥ 50 and κ ≥ 0.6**. Re-calibrate whenever the judge
model or judge prompt changes. (`calibration_sample.jsonl` is synthetic — it only demonstrates the format.)

---

## 4. Metrics and gates (`eval/run_eval.py`)

Per item: five judge scores + overall mean, key-point recall, **citation validity** (cited IDs that resolve to a
*verified* source), citation coverage, revisions, research rounds, cost, tokens, tool calls, latency.

Aggregate gates (CI exits non-zero if any fails):

| Gate | Default |
|---|---|
| Mean overall judge score | ≥ `EVAL_MIN_OVERALL` (3.5 default; 4.0 in the nightly Bedrock workflow) |
| Mean citation validity | ≥ 0.95 |
| Failure rate | ≤ 5% |
| p95 cost per request | ≤ `MAX_COST_USD` |
| No regression vs baseline (`--baseline`) | Δ overall ≥ −0.2 |

Outputs: `reports/latest.json` (machine-readable, includes judge reasoning per item) and `reports/latest.md`.

---

## 5. CI wiring

* `ci.yml` → `pytest` (fake LLM + real MCP servers) and an **eval plumbing smoke** with the fake LLM
  (`EVAL_MIN_OVERALL=2.5`): proves dataset → system → judge → report → gate works, at zero cost.
* `eval-bedrock.yml` → nightly / manual run on Bedrock via GitHub OIDC (no static keys), compares with the last
  accepted baseline, promotes the new baseline on `main`, uploads the report artifact.

## 6. Online evaluation

`POST /v1/research` samples `ONLINE_EVAL_SAMPLE_RATE` of successful responses and judges them asynchronously
(no reference answer → focus on groundedness, citations, clarity). Emit the scores as metrics and alert on a
rolling-mean drop; route low-scoring traces to a human review queue — they become new dataset items.

## 7. Reading a failing report

1. Low **correctness** with high **groundedness** → the evidence was wrong or incomplete → inspect the worker
   findings in the trace (wrong tool queries? retrieval miss?).
2. Low **groundedness** / citation validity → writer is going beyond findings → tighten synthesizer prompt or
   lower its temperature; check the critic threshold.
3. Low **completeness** with missing key points tagged `kb` → retrieval issue (chunking, query phrasing).
4. Cost p95 breach → look at `usage.by_model`: extra revisions? long tool outputs? reasoning tier used in loops?
