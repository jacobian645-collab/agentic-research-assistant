# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Synthesizer, self-critique (evaluator-optimizer loop) and finalizer."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from research_assistant.agents.common import (
    RunContext,
    build_sources,
    ctx,
    emit,
    format_findings,
    format_sources,
)
from research_assistant.config import Settings
from research_assistant.guardrails import check_citations
from research_assistant.llm import structured
from research_assistant.prompts import (
    CRITIC_SYSTEM,
    CRITIC_USER,
    REVISION_BLOCK,
    SYNTH_SYSTEM,
    SYNTH_USER,
)
from research_assistant.schemas import CritiqueAssessment, CritiqueRecord, ReportDraft, ResearchReport
from research_assistant.state import ResearchState

RUBRIC_WEIGHTS = {"coverage": 0.4, "groundedness": 0.4, "coherence": 0.2}


def weighted_score(a: CritiqueAssessment) -> float:
    """Map 1-5 rubric scores to a 0-1 weighted score."""
    return round(sum(w * (getattr(a, k) - 1) / 4 for k, w in RUBRIC_WEIGHTS.items()), 3)


def make_synthesizer(llm: BaseChatModel, settings: Settings):
    chain = structured(llm, ReportDraft, settings)

    async def synthesizer(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
        findings = state.get("findings", [])
        reg = build_sources(findings)
        revision = state.get("revision", 0)
        revision_block = ""
        critiques = state.get("critiques", [])
        if critiques and state.get("draft") is not None:
            last = critiques[-1]
            cc = last.citation_check
            feedback = last.assessment.issues + last.assessment.revision_instructions
            if cc.unknown_citations:
                feedback.append(f"Remove or fix citations to unknown source ids: {cc.unknown_citations}")
            if cc.citation_coverage < 0.8:
                feedback.append("Add citations: at least 80% of claims must cite a source id.")
            revision_block = REVISION_BLOCK.format(
                revision=revision,
                previous=state["draft"].to_markdown(),
                feedback="\n".join(f"- {x}" for x in feedback),
            )
        prompt = SYNTH_USER.format(
            question=state["question"],
            criteria="\n".join(f"- {c}" for c in state["plan"].success_criteria),
            sources=format_sources(reg),
            findings=format_findings(findings, reg),
            revision_block=revision_block,
        )
        draft: ReportDraft = await chain.ainvoke([SystemMessage(SYNTH_SYSTEM), HumanMessage(prompt)])
        report = ResearchReport(
            **draft.model_dump(), question=state["question"], references=reg.sources, revision=revision
        )
        emit(runtime, "draft", revision=revision, sections=len(report.sections))
        return {"draft": report}

    return synthesizer


def make_critic(llm: BaseChatModel, settings: Settings):
    chain = structured(llm, CritiqueAssessment, settings)

    async def critic(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
        draft: ResearchReport = state["draft"]
        findings = state.get("findings", [])
        reg = build_sources(findings)
        revision = state.get("revision", 0)
        check = check_citations(draft, reg.sources, reg.verified_ids)  # deterministic, free
        assessment: CritiqueAssessment = await chain.ainvoke(
            [
                SystemMessage(CRITIC_SYSTEM),
                HumanMessage(
                    CRITIC_USER.format(
                        question=state["question"],
                        revision=revision,
                        criteria="\n".join(f"- {c}" for c in state["plan"].success_criteria),
                        findings=format_findings(findings, reg),
                        draft=draft.to_markdown(),
                        citation_check=check.model_dump_json(),
                    )
                ),
            ]
        )
        score = weighted_score(assessment)
        passed = score >= settings.critic_pass_threshold and check.passed
        can_research = state.get("research_round", 0) < settings.max_research_rounds and bool(
            assessment.missing_information
        )
        if passed or ctx(runtime).over_budget():
            verdict = "accept"
        elif can_research:
            verdict = "research_more"
        elif revision < settings.max_revisions:
            verdict = "revise"
        else:
            verdict = "accept"  # bounded loop: ship best effort, flagged in limitations
        record = CritiqueRecord(
            revision=revision,
            assessment=assessment,
            citation_check=check,
            weighted_score=score,
            verdict=verdict,
        )  # type: ignore[arg-type]
        emit(
            runtime,
            "critique",
            revision=revision,
            score=score,
            verdict=verdict,
            citation_coverage=round(check.citation_coverage, 2),
        )
        update: dict[str, Any] = {"critiques": [record]}
        if verdict in ("revise", "research_more"):
            update["revision"] = revision + 1
        if verdict == "research_more":
            update["critic_gaps"] = assessment.missing_information
        return update

    return critic


def route_from_critic(state: ResearchState) -> str:
    verdict = state["critiques"][-1].verdict
    return {"accept": "finalize", "revise": "synthesizer", "research_more": "supervisor"}[verdict]


def make_finalize(settings: Settings):
    def finalize(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
        return _finalize(state, runtime, settings.critic_pass_threshold)

    return finalize


def _finalize(state: ResearchState, runtime: Runtime[RunContext], threshold: float) -> dict[str, Any]:
    report: ResearchReport = state["draft"].model_copy(deep=True)
    last = state["critiques"][-1] if state.get("critiques") else None
    if last and (last.weighted_score < threshold or not last.citation_check.passed):
        report.limitations.append(
            f"Automated review did not reach the quality bar after {last.revision + 1} draft(s) "
            f"(score {last.weighted_score:.2f}); treat conclusions with caution."
        )
    if ctx(runtime).over_budget():
        report.limitations.append("Run stopped early at the per-request cost cap; answer may be partial.")
    failed = [f.task_id for f in state.get("findings", []) if f.status == "failed"]
    if failed:
        report.limitations.append(f"Some research tasks failed and were excluded: {', '.join(failed)}.")
    emit(runtime, "final", title=report.title)
    return {"final_report": report}
