# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Structured-output contracts.

Every LLM hop that feeds *code* (routing, fan-out, verification, rendering) returns a
validated Pydantic object, never free text. Two kinds of models live here:

* **LLM-facing schemas** (``ResearchPlan``, ``WorkerReport``, ``ReportDraft``,
  ``CritiqueAssessment`` …) — kept deliberately small and well-described, because the
  field descriptions *are* the prompt for structured output / tool calling.
* **System records** (``Finding``, ``Source``, ``ResearchReport``, ``CritiqueRecord``)
  — enriched by deterministic code with things the model must not be trusted to fill
  (verification flags, source IDs, timings, scores computed from rubric weights).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

AgentName = Literal["web_researcher", "retrieval_researcher", "code_analyst"]
AGENT_NAMES: tuple[AgentName, ...] = ("web_researcher", "retrieval_researcher", "code_analyst")

# ─────────────────────────────── Planning ────────────────────────────────


class SubTask(BaseModel):
    """A unit of work delegated to exactly one specialist sub-agent."""

    id: str = Field(description="Short unique id such as 't1', 't2'.")
    agent: AgentName = Field(
        description=(
            "Specialist to run this task. web_researcher = public/current web facts; "
            "retrieval_researcher = internal knowledge-base documents; "
            "code_analyst = calculations, data analysis, quantitative checks in Python."
        )
    )
    objective: str = Field(description="Precise, self-contained instruction for the specialist.")
    rationale: str = Field(default="", description="Why this task is needed to answer the question.")
    depends_on: list[str] = Field(
        default_factory=list, description="Ids of tasks whose findings this task needs as input."
    )


class ResearchPlan(BaseModel):
    """Decomposition of the user question into parallelisable sub-tasks."""

    question_type: Literal["factual", "comparative", "analytical", "quantitative", "exploratory"]
    key_questions: list[str] = Field(description="2-6 sub-questions that together answer the question.")
    subtasks: list[SubTask] = Field(min_length=1, max_length=8)
    success_criteria: list[str] = Field(
        description="Observable criteria a reviewer can use to judge whether the final answer is complete."
    )

    @field_validator("subtasks")
    @classmethod
    def _unique_ids_and_valid_deps(cls, v: list[SubTask]) -> list[SubTask]:
        ids = [t.id for t in v]
        if len(ids) != len(set(ids)):
            raise ValueError("subtask ids must be unique")
        known = set(ids)
        for t in v:
            t.depends_on = [d for d in t.depends_on if d in known and d != t.id]
        return v


class SupervisorDecision(BaseModel):
    """Supervisor reflection after a research wave completes."""

    reasoning: str = Field(description="Brief assessment of what is known and what is still missing.")
    action: Literal["synthesize", "research_more"]
    gaps: list[str] = Field(default_factory=list)
    new_subtasks: list[SubTask] = Field(
        default_factory=list, description="Follow-up tasks (only when action == research_more). Max 3."
    )


# ─────────────────────────────── Workers ─────────────────────────────────


class EvidenceItem(BaseModel):
    claim: str = Field(description="One atomic factual claim supported by the source.")
    source: str = Field(
        description=(
            "Exact URL or document id copied verbatim from a tool result, or the literal string "
            "'code_execution' for results computed with the Python tool."
        )
    )
    quote: str | None = Field(default=None, description="Short supporting excerpt from the tool output.")


class WorkerReport(BaseModel):
    """What a specialist returns after using its tools."""

    summary: str = Field(description="3-6 sentence answer to the task objective.")
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="Self-assessed confidence 0-1.")
    open_questions: list[str] = Field(default_factory=list)


class VerifiedEvidence(EvidenceItem):
    verified: bool = False  # set by code: source string actually appeared in a tool output


class Finding(BaseModel):
    """System record of one sub-agent run (LLM output + deterministic enrichment)."""

    task_id: str
    agent: AgentName
    objective: str
    status: Literal["done", "failed"] = "done"
    summary: str = ""
    evidence: list[VerifiedEvidence] = Field(default_factory=list)
    confidence: float = 0.0
    open_questions: list[str] = Field(default_factory=list)
    tool_calls: int = 0
    latency_s: float = 0.0
    error: str | None = None
    round: int = 0
    source_titles: dict[str, str] = Field(default_factory=dict)  # locator → title, harvested from tool output

    @property
    def verified_ratio(self) -> float:
        return sum(e.verified for e in self.evidence) / len(self.evidence) if self.evidence else 0.0


# ──────────────────────────────── Report ─────────────────────────────────


class Source(BaseModel):
    id: str  # "S1", "S2" … assigned deterministically
    locator: str  # URL, document id, or code_execution:<task>
    kind: Literal["web", "document", "code"]
    title: str | None = None
    verified: bool = False  # locator literally appeared in a tool output


class Claim(BaseModel):
    text: str = Field(description="One sentence. Must be supported by the cited sources.")
    citations: list[str] = Field(description="Source ids such as ['S1','S3']. Empty only for pure inference.")


class ReportSection(BaseModel):
    heading: str
    claims: list[Claim] = Field(min_length=1)


class ReportDraft(BaseModel):
    """Structured research report produced by the synthesizer."""

    title: str
    executive_summary: str = Field(
        description="4-6 sentence direct answer to the question, with key numbers."
    )
    sections: list[ReportSection] = Field(min_length=1, max_length=8)
    limitations: list[str] = Field(default_factory=list, description="Caveats, conflicting evidence, gaps.")
    confidence: Literal["low", "medium", "high"]


class ResearchReport(ReportDraft):
    question: str
    references: list[Source] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    revision: int = 0

    def cited_ids(self) -> set[str]:
        return {c for s in self.sections for cl in s.claims for c in cl.citations}

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", f"> **Question:** {self.question}", "", "## Executive summary", ""]
        lines += [self.executive_summary, ""]
        for sec in self.sections:
            lines += [f"## {sec.heading}", ""]
            for cl in sec.claims:
                cites = "".join(f"[{c}]" for c in cl.citations)
                lines.append(f"- {cl.text} {cites}".rstrip())
            lines.append("")
        if self.limitations:
            lines += ["## Limitations", ""] + [f"- {x}" for x in self.limitations] + [""]
        lines += [f"**Confidence:** {self.confidence}", ""]
        used = self.cited_ids()
        refs = [r for r in self.references if r.id in used]
        if refs:
            lines += ["## References", ""]
            lines += [f"- **[{r.id}]** {r.title + ' — ' if r.title else ''}{r.locator}" for r in refs]
        return "\n".join(lines).strip() + "\n"


# ─────────────────────────────── Critique ────────────────────────────────


class CritiqueAssessment(BaseModel):
    """Self-critique rubric. Reason first, then score (improves score calibration)."""

    reasoning: str = Field(description="Step-by-step review of the draft against the success criteria.")
    coverage: int = Field(ge=1, le=5, description="Does it fully answer every key question? 5 = complete.")
    groundedness: int = Field(
        ge=1, le=5, description="Are claims supported by the cited findings? 5 = no unsupported claims."
    )
    coherence: int = Field(
        ge=1, le=5, description="Logical structure, no contradictions, clear. 5 = excellent."
    )
    issues: list[str] = Field(default_factory=list, description="Concrete defects found.")
    revision_instructions: list[str] = Field(
        default_factory=list, description="Actionable edits for the writer if revision is needed."
    )
    missing_information: list[str] = Field(
        default_factory=list, description="Facts that require NEW research (not just rewriting)."
    )


class CitationCheck(BaseModel):
    """Deterministic checks that do not need an LLM."""

    total_claims: int
    cited_claims: int
    unknown_citations: list[str]
    unverified_sources_cited: list[str]

    @property
    def citation_coverage(self) -> float:
        return self.cited_claims / self.total_claims if self.total_claims else 0.0

    @property
    def passed(self) -> bool:
        return not self.unknown_citations and self.citation_coverage >= 0.8


class CritiqueRecord(BaseModel):
    revision: int
    assessment: CritiqueAssessment
    citation_check: CitationCheck
    weighted_score: float
    verdict: Literal["accept", "revise", "research_more"]
