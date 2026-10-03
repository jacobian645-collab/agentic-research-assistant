# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Offline evaluation loop: dataset → system under test → LLM judge → metrics → gate.

Outputs ``reports/eval-<ts>.json`` + ``.md`` and returns ``passed`` for CI gating.
Combines judge scores with *deterministic* metrics (key-point recall, citation validity,
cost, latency) — never gate on an LLM judge alone.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_assistant.config import Settings
from research_assistant.eval.judge import CRITERIA, JUDGE_PROMPT_VERSION, LLMJudge
from research_assistant.llm import get_chat_model
from research_assistant.prompts import PROMPT_VERSION
from research_assistant.service import ResearchService


@dataclass
class EvalItem:
    id: str
    question: str
    reference_answer: str
    key_points: list[str]
    tags: list[str]


def load_dataset(path: Path, limit: int | None = None) -> list[EvalItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            items.append(
                EvalItem(d["id"], d["question"], d["reference_answer"], d["key_points"], d.get("tags", []))
            )
    return items[:limit] if limit else items


def _sig(s: str) -> set[str]:
    return {t for t in re.findall(r"\$?\d+(?:[.,]\d+)*|[a-z0-9]+", s.lower()) if len(t) > 2}


def key_point_recall(key_points: list[str], answer: str) -> float:
    """Deterministic lexical proxy for completeness (cheap sanity check on the judge)."""
    ans = _sig(answer)
    hits = sum(1 for k in key_points if len(_sig(k) & ans) >= 0.6 * max(1, len(_sig(k))))
    return hits / len(key_points) if key_points else 1.0


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p * (len(xs) - 1)))]


async def run_eval(
    settings: Settings,
    dataset: Path | None = None,
    limit: int | None = None,
    concurrency: int = 2,
    baseline: Path | None = None,
) -> dict[str, Any]:
    items = load_dataset(dataset or settings.eval_dataset, limit)
    judge = LLMJudge(get_chat_model("judge", settings), settings)
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict[str, Any]] = []

    async with ResearchService(settings) as svc:

        async def one(item: EvalItem) -> dict[str, Any]:
            async with sem:
                t0 = time.perf_counter()
                try:
                    res = await svc.run(item.question)
                    md = res.report.to_markdown() if res.report else ""
                    verdict = await judge.grade(item.question, item.reference_answer, item.key_points, md)
                    crit = res.trace.get("critiques", [])
                    last = crit[-1] if crit else {}
                    cited = res.report.cited_ids() if res.report else set()
                    # valid citation = refers to a known source whose locator appeared in a tool output
                    valid = {r.id for r in res.report.references if r.verified} if res.report else set()
                    verified = cited & valid
                    return {
                        "id": item.id,
                        "tags": item.tags,
                        "ok": True,
                        "scores": {c: getattr(verdict, c) for c in CRITERIA},
                        "overall": verdict.overall,
                        "key_point_recall": round(key_point_recall(item.key_points, md), 3),
                        "key_points_missing": verdict.key_points_missing,
                        "citation_validity": round(len(verified) / len(cited), 3) if cited else 0.0,
                        "citation_coverage": last.get("citation_coverage", 0.0),
                        "revisions": len(crit),
                        "research_rounds": res.trace.get("research_rounds", 0),
                        "cost_usd": res.usage["est_cost_usd"],
                        "tokens": res.usage["input_tokens"] + res.usage["output_tokens"],
                        "tool_calls": sum(res.usage["tool_calls"].values()),
                        "latency_s": round(time.perf_counter() - t0, 2),
                        "judge_reasoning": verdict.reasoning[:800],
                    }
                except Exception as e:
                    return {
                        "id": item.id,
                        "tags": item.tags,
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                        "latency_s": round(time.perf_counter() - t0, 2),
                    }

        rows = await asyncio.gather(*(one(i) for i in items))

    ok = [r for r in rows if r["ok"]]

    def mean(key: str, src: list[dict[str, Any]] | None = None) -> float:
        vals = [r[key] for r in (src or ok)]
        return round(statistics.fmean(vals), 3) if vals else 0.0

    summary: dict[str, Any] = {
        "n": len(rows),
        "succeeded": len(ok),
        "failure_rate": round(1 - len(ok) / len(rows), 3) if rows else 0,
        "overall_mean": mean("overall"),
        "criteria_mean": {
            c: round(statistics.fmean([r["scores"][c] for r in ok]), 3) if ok else 0 for c in CRITERIA
        },
        "pass_rate_overall_ge_4": round(sum(r["overall"] >= 4 for r in ok) / len(ok), 3) if ok else 0,
        "key_point_recall_mean": mean("key_point_recall"),
        "citation_validity_mean": mean("citation_validity"),
        "cost_usd_mean": mean("cost_usd"),
        "cost_usd_p95": _pct([r["cost_usd"] for r in ok], 0.95),
        "latency_s_p50": _pct([r["latency_s"] for r in ok], 0.5),
        "latency_s_p95": _pct([r["latency_s"] for r in ok], 0.95),
        "tool_calls_mean": mean("tool_calls"),
    }
    gates = {
        "overall_mean": summary["overall_mean"] >= settings.eval_min_overall,
        "citation_validity": summary["citation_validity_mean"] >= 0.95,
        "failure_rate": summary["failure_rate"] <= 0.05,
        "cost_cap": summary["cost_usd_p95"] <= settings.max_cost_usd,
    }
    regression = None
    base = await asyncio.to_thread(_load_baseline, baseline) if baseline else None
    if base:
        delta = round(summary["overall_mean"] - base["overall_mean"], 3)
        regression = {"baseline_overall": base["overall_mean"], "delta": delta}
        gates["no_regression"] = delta >= -0.2
    report = {
        "meta": {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "provider": settings.llm_provider,
            "model_reasoning": settings.model_reasoning,
            "model_fast": settings.model_fast,
            "judge_model": settings.judge_model,
            "prompt_version": PROMPT_VERSION,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "dataset": str(dataset or settings.eval_dataset),
        },
        "summary": summary,
        "gates": gates,
        "passed": all(gates.values()),
        "regression": regression,
        "rows": rows,
    }
    _write(report, settings.eval_output_dir)
    return report


def _load_baseline(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text())["summary"] if path.is_file() else None


def _write(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    (out_dir / f"eval-{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / "latest.json").write_text(json.dumps(report, indent=2, default=str))
    s = report["summary"]
    lines = [
        "# Evaluation report",
        "",
        f"- Provider: `{report['meta']['provider']}` · judge: `{report['meta']['judge_model']}`",
        f"- Prompt version: `{report['meta']['prompt_version']}` · "
        f"judge prompt: `{report['meta']['judge_prompt_version']}`",
        f"- **Result: {'PASS ✅' if report['passed'] else 'FAIL ❌'}**",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Items (ok/total) | {s['succeeded']}/{s['n']} |",
        f"| Overall judge score (1-5) | {s['overall_mean']} |",
        *[f"| · {k} | {v} |" for k, v in s["criteria_mean"].items()],
        f"| Key-point recall (lexical) | {s['key_point_recall_mean']} |",
        f"| Citation validity | {s['citation_validity_mean']} |",
        f"| Cost / request (mean · p95) | ${s['cost_usd_mean']} · ${s['cost_usd_p95']} |",
        f"| Latency p50 · p95 (s) | {s['latency_s_p50']} · {s['latency_s_p95']} |",
        "",
        "## Gates",
        "",
        *[f"- {'✅' if v else '❌'} {k}" for k, v in report["gates"].items()],
        "",
        "## Per item",
        "",
        "| id | overall | recall | citations | cost $ | latency s |",
        "|---|---|---|---|---|---|",
        *[
            f"| {r['id']} | {r.get('overall', 'ERR')} | {r.get('key_point_recall', '-')} | "
            f"{r.get('citation_validity', '-')} | {r.get('cost_usd', '-')} | {r['latency_s']} |"
            for r in report["rows"]
        ],
    ]
    (out_dir / f"eval-{stamp}.md").write_text("\n".join(lines) + "\n")
    (out_dir / "latest.md").write_text("\n".join(lines) + "\n")
