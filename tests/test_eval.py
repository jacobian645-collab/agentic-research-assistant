# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

from __future__ import annotations

import json

from research_assistant.eval.calibration import calibrate, quadratic_weighted_kappa, spearman
from research_assistant.eval.judge import LLMJudge
from research_assistant.eval.run_eval import key_point_recall, run_eval
from research_assistant.llm_fake import FakeResearchLLM


def test_kappa_and_spearman():
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) < 0
    assert round(spearman([1, 2, 3], [10, 20, 30]), 6) == 1.0


def test_calibration_file(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps({"human_score": s, "judge_score": s}) for s in [1, 2, 3, 4, 5] * 10))
    out = calibrate(p)
    assert out["trusted"] and out["quadratic_weighted_kappa"] == 1.0


def test_key_point_recall():
    assert key_point_recall(["position bias", "verbosity bias"], "It shows position bias.") == 0.5


class _AlwaysA(FakeResearchLLM):
    """A judge with pure position bias: always prefers whatever is shown first."""

    def _build_PairwiseVerdict(self, prompt):
        return {"reasoning": "first one looks better", "winner": "A"}


async def test_pairwise_position_swap_neutralises_position_bias(settings):
    judge = LLMJudge(_AlwaysA(role="judge"), settings)
    out = await judge.compare("q", "ref", "answer one", "answer two")
    assert out["consistent"] is False and out["winner"] == "tie"


async def test_pairwise_consistent_preference(settings):
    judge = LLMJudge(FakeResearchLLM(role="judge"), settings)
    ref = "position bias verbosity bias self-enhancement bias swap positions calibrate kappa"
    out = await judge.compare(
        "q", ref, "position bias verbosity bias self-enhancement swap positions kappa", "no idea"
    )
    assert out["consistent"] and out["winner"] == "answer_1"


async def test_offline_eval_pipeline_writes_reports(settings):
    settings.eval_min_overall = 2.0
    report = await run_eval(settings, limit=2, concurrency=2)
    assert report["summary"]["succeeded"] == 2
    assert report["gates"]["citation_validity"] and report["passed"]
    assert (settings.eval_output_dir / "latest.md").exists()
