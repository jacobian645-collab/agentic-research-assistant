# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""LLM-as-judge.

Two complementary modes:

* **Pointwise rubric grading** (``grade``) — absolute 1-5 scores on five criteria with
  anchored descriptions, reference answer + required key points supplied
  (reference-guided judging), reasoning produced *before* scores.
* **Pairwise comparison** (``compare``) — A vs B for regression testing two system
  versions; run in **both orders** and only count consistent verdicts (position-bias
  control, per Zheng et al. 2023).

Bias controls baked in: separate (stronger / different-family) judge model, temperature 0,
pinned prompt version, structured output, position swapping, and a calibration step
against human labels (see calibration.py) before scores are trusted for release gating.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_assistant.config import Settings
from research_assistant.llm import structured

JUDGE_PROMPT_VERSION = "judge-2026-10-03.1"

CRITERIA = ("correctness", "groundedness", "completeness", "citation_quality", "clarity")


class JudgeVerdict(BaseModel):
    reasoning: str = Field(description="Evaluate each criterion step by step BEFORE assigning scores.")
    correctness: int = Field(ge=1, le=5, description="Factual agreement with the reference answer.")
    groundedness: int = Field(
        ge=1, le=5, description="Claims are supported by the cited references; no fabrication."
    )
    completeness: int = Field(ge=1, le=5, description="Covers all required key points.")
    citation_quality: int = Field(
        ge=1, le=5, description="Citations present, specific and relevant to each claim."
    )
    clarity: int = Field(ge=1, le=5, description="Direct answer first, well structured, concise.")
    key_points_covered: list[str] = Field(default_factory=list)
    key_points_missing: list[str] = Field(default_factory=list)

    @property
    def overall(self) -> float:
        return round(sum(getattr(self, c) for c in CRITERIA) / len(CRITERIA), 3)


class PairwiseVerdict(BaseModel):
    reasoning: str = Field(description="Compare both answers against the reference before deciding.")
    winner: Literal["A", "B", "tie"]


JUDGE_SYSTEM = """You are an impartial expert evaluator of research reports.
Score the CANDIDATE ANSWER against the QUESTION, the REFERENCE ANSWER and the required KEY POINTS.
Scoring anchors for every criterion (integers only):
5 = fully meets the criterion, no issues | 4 = minor issues | 3 = partially meets, noticeable gaps |
2 = major problems | 1 = fails the criterion.
Rules: do not reward length; ignore style preferences unrelated to the criteria; penalise any claim that
contradicts the reference or cites a source that does not support it. Reason first, then score."""

JUDGE_USER = """QUESTION: {question}

REFERENCE ANSWER:
{reference}

KEY POINTS:
{key_points}

CANDIDATE ANSWER:
{answer}"""

PAIRWISE_SYSTEM = """You are an impartial expert evaluator. Decide which answer better addresses the QUESTION,
using the REFERENCE ANSWER as ground truth. Judge correctness and completeness first, then grounding and
clarity. Do not let answer order or length influence you. Answer 'tie' if they are equivalent."""

PAIRWISE_USER = """QUESTION: {question}

REFERENCE ANSWER:
{reference}

ANSWER A:
{a}

ANSWER B:
{b}"""


class LLMJudge:
    def __init__(self, llm: BaseChatModel, settings: Settings) -> None:
        self.grader = structured(llm, JudgeVerdict, settings)
        self.pairwise = structured(llm, PairwiseVerdict, settings)

    async def grade(self, question: str, reference: str, key_points: list[str], answer: str) -> JudgeVerdict:
        return await self.grader.ainvoke(
            [
                SystemMessage(JUDGE_SYSTEM),
                HumanMessage(
                    JUDGE_USER.format(
                        question=question,
                        reference=reference,
                        key_points="\n".join(f"- {k}" for k in key_points),
                        answer=answer,
                    )
                ),
            ]
        )

    async def compare(self, question: str, reference: str, answer_1: str, answer_2: str) -> dict[str, Any]:
        """Position-swapped pairwise comparison. Returns winner in terms of answer_1 / answer_2."""

        async def once(a: str, b: str) -> PairwiseVerdict:
            return await self.pairwise.ainvoke(
                [
                    SystemMessage(PAIRWISE_SYSTEM),
                    HumanMessage(PAIRWISE_USER.format(question=question, reference=reference, a=a, b=b)),
                ]
            )

        v1 = await once(answer_1, answer_2)  # answer_1 in position A
        v2 = await once(answer_2, answer_1)  # answer_1 in position B
        map1 = {"A": "answer_1", "B": "answer_2", "tie": "tie"}[v1.winner]
        map2 = {"A": "answer_2", "B": "answer_1", "tie": "tie"}[v2.winner]
        consistent = map1 == map2
        return {
            "winner": map1 if consistent else "tie",
            "consistent": consistent,
            "verdicts": [v1.model_dump(), v2.model_dump()],
        }
