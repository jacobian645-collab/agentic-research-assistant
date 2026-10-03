# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Deterministic offline chat model.

Lets the *entire* system — planner, parallel tool-calling workers talking to real MCP
servers, synthesizer, critic loop, judge — run in CI with zero API keys and zero cost.
It implements the same two capabilities the graph depends on (``bind_tools`` and
``with_structured_output`` via tool calling), so the code path exercised in tests is the
production code path; only the token generator is swapped.

It is intentionally simple and heuristic: it parses the labelled blocks our prompts
emit (``QUESTION:``, ``SOURCES:``, ``FINDINGS:`` …) and produces schema-valid objects.
"""

from __future__ import annotations

import contextlib
import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def _grab(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip() if m else default


def _sig_tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"\$?\d+(?:[.,]\d+)*|[a-z0-9]+", s.lower()) if len(t) > 2}


class FakeResearchLLM(BaseChatModel):
    role: str = "reasoning"

    @property
    def _llm_type(self) -> str:
        return "fake-research"

    def bind_tools(  # type: ignore[override]
        self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    # ── generation ─────────────────────────────────────────────────────
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools") or []
        names = [t["function"]["name"] for t in tools]
        prompt = "\n".join(_text(m.content) for m in messages)
        builder = getattr(self, f"_build_{names[0]}", None) if len(names) == 1 else None
        if builder is not None:
            args = builder(prompt)
            msg = AIMessage(
                content="",
                tool_calls=[{"name": names[0], "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}],
            )
        elif tools:
            msg = self._react_step(messages, tools)
        else:
            msg = AIMessage(content="OK")
        out_chars = len(json.dumps([tc["args"] for tc in msg.tool_calls])) + len(_text(msg.content))
        msg.usage_metadata = {
            "input_tokens": len(prompt) // 4,
            "output_tokens": out_chars // 4,
            "total_tokens": len(prompt) // 4 + out_chars // 4,
        }
        msg.response_metadata = {"model_name": f"fake-{self.role}"}
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # ── ReAct worker behaviour ─────────────────────────────────────────
    def _react_step(self, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
        prompt = "\n".join(_text(m.content) for m in messages if not isinstance(m, ToolMessage))
        objective = _grab(r"YOUR TASK \([^)]*\):\s*(.+)", prompt) or _grab(r"QUESTION.*?:\s*(.+)", prompt)
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        names = [t["function"]["name"] for t in tools]

        def call(name: str, args: dict[str, Any]) -> AIMessage:
            return AIMessage(
                content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}]
            )

        if not tool_msgs:
            if "web_search" in names:
                return call("web_search", {"query": objective, "max_results": 4})
            if "search_documents" in names:
                return call("search_documents", {"query": objective, "top_k": 4})
            if "run_python" in names:
                question = _grab(r"QUESTION \(overall\):\s*(.+)", prompt)
                nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", question)][:8]
                code = f"nums = {nums}\nprint('values:', nums)\nprint('sum:', round(sum(nums), 4))\n"
                return call("run_python", {"code": code})
        if len(tool_msgs) == 1 and "fetch_url" in names:
            urls = re.findall(r'"url":\s*"([^"]+)"', _text(tool_msgs[0].content))
            if urls:
                return call("fetch_url", {"url": urls[0], "max_chars": 3000})
        last = re.sub(r"\s+", " ", _text(tool_msgs[-1].content))[:300] if tool_msgs else "no results"
        return AIMessage(content=f"Based on the tool results: {last}")

    # ── structured-output builders (keyed by schema name) ──────────────
    def _build_ResearchPlan(self, prompt: str) -> dict[str, Any]:
        q = _grab(r"QUESTION:\s*(.+)", prompt, "the question")
        return {
            "question_type": "analytical",
            "key_questions": [
                f"What do authoritative public sources say about: {q}",
                f"What do internal standards say about: {q}",
            ],
            "subtasks": [
                {"id": "t1", "agent": "web_researcher", "objective": q, "rationale": "public evidence"},
                {"id": "t2", "agent": "retrieval_researcher", "objective": q, "rationale": "internal policy"},
                {
                    "id": "t3",
                    "agent": "code_analyst",
                    "objective": f"Quantitative check for: {q}",
                    "rationale": "verify numbers",
                    "depends_on": ["t1", "t2"],
                },
            ],
            "success_criteria": ["Answers the question directly", "Every claim cites a source"],
        }

    def _build_SupervisorDecision(self, prompt: str) -> dict[str, Any]:
        rnd = int(_grab(r"RESEARCH ROUND:\s*(\d+)", prompt, "0"))
        q = _grab(r"QUESTION:\s*(.+)", prompt, "the question")
        if rnd == 0:
            existing = _grab(r"EXISTING TASK IDS:\s*(.+)", prompt)
            new_id = next(f"f{i}" for i in range(1, 50) if f"f{i}" not in existing)
            return {
                "reasoning": "Internal guidance needs a second, differently-phrased lookup.",
                "action": "research_more",
                "gaps": ["internal standards detail"],
                "new_subtasks": [
                    {
                        "id": new_id,
                        "agent": "retrieval_researcher",
                        "objective": f"Policy, standards and controls related to: {q}",
                    }
                ],
            }
        return {"reasoning": "Findings cover the success criteria.", "action": "synthesize", "gaps": []}

    def _build_WorkerReport(self, prompt: str) -> dict[str, Any]:
        prompt = prompt.split("TOOL RESULTS:", 1)[-1]  # parse tool output only, not agent notes
        locs = re.findall(r'"(?:url|doc_id)":\s*"([^"]+)"', prompt)
        bodies = re.findall(r'"(?:snippet|text)":\s*"((?:[^"\\]|\\.)*)"', prompt)
        evidence = []
        seen: set[str] = set()
        for loc, body in zip(locs, bodies, strict=False):
            if loc in seen:
                continue
            seen.add(loc)
            with contextlib.suppress(json.JSONDecodeError):
                body = json.loads(f'"{body}"')
            clean = " ".join(re.sub(r"[#*`>]", "", body).split())
            sentence = " ".join(re.split(r"(?<=[.!?])\s", clean)[:2])[:400]
            evidence.append({"claim": sentence, "source": loc, "quote": body[:160]})
            if len(evidence) >= 3:
                break
        stdout = _grab(r'"stdout":\s*"([^"]*)"', prompt)
        if stdout:
            evidence.append(
                {
                    "claim": f"Computed result: {stdout.replace(chr(92) + 'n', ' ')}",
                    "source": "code_execution",
                }
            )
        summary = " ".join(e["claim"] for e in evidence)[:800] or "The tools returned no useful results."
        return {
            "summary": summary,
            "evidence": evidence,
            "confidence": 0.8 if evidence else 0.2,
            "open_questions": [],
        }

    def _build_ReportDraft(self, prompt: str) -> dict[str, Any]:
        q = _grab(r"QUESTION:\s*(.+)", prompt, "the question")
        findings_block = prompt.split("FINDINGS:", 1)[-1].split("REVISION:", 1)[0]
        sections: list[dict[str, Any]] = []
        for block in re.split(r"(?m)^### ", findings_block)[1:]:
            header = block.splitlines()[0]
            claims = [
                {"text": m.group(1).strip(), "citations": [m.group(2)]}
                for m in re.finditer(r"(?m)^- (.+?) \[(S\d+)\]", block)
            ][:4]
            if claims:
                sections.append(
                    {"heading": header.split("—")[-1].strip()[:80] or header[:80], "claims": claims}
                )
        if not sections:
            sections = [
                {
                    "heading": "Findings",
                    "claims": [{"text": "No supported findings were returned.", "citations": []}],
                }
            ]
        first = [c["text"] for s in sections for c in s["claims"]][:3]
        limitations = ["Generated by the offline fake model; content is extractive."]
        if "REVISION:" in prompt:
            limitations.append("Revised after self-critique.")
        return {
            "title": f"Research brief: {q[:80]}",
            "executive_summary": " ".join(first) or "Insufficient evidence.",
            "sections": sections[:8],
            "limitations": limitations,
            "confidence": "medium",
        }

    def _build_CritiqueAssessment(self, prompt: str) -> dict[str, Any]:
        first = _grab(r"REVISION:\s*(\d+)", prompt, "0") == "0"
        return {
            "reasoning": "Checked coverage of success criteria, support of each claim, and structure.",
            "coverage": 3 if first else 5,
            "groundedness": 4 if first else 5,
            "coherence": 4,
            "issues": ["Executive summary does not lead with a direct answer."] if first else [],
            "revision_instructions": ["Lead with the direct answer and key numbers."] if first else [],
            "missing_information": [],
        }

    def _build_JudgeVerdict(self, prompt: str) -> dict[str, Any]:
        kp_block = prompt.split("KEY POINTS:", 1)[-1].split("CANDIDATE ANSWER:", 1)[0]
        key_points = [k.strip("- ").strip() for k in kp_block.splitlines() if k.strip().startswith("-")]
        answer = _sig_tokens(prompt.split("CANDIDATE ANSWER:", 1)[-1])
        covered = [k for k in key_points if len(_sig_tokens(k) & answer) >= 0.5 * max(1, len(_sig_tokens(k)))]
        ratio = len(covered) / len(key_points) if key_points else 1.0
        correctness = 1 + round(4 * ratio)
        return {
            "reasoning": f"{len(covered)}/{len(key_points)} key points covered.",
            "correctness": correctness,
            "completeness": correctness,
            "groundedness": 4,
            "citation_quality": 4,
            "clarity": 4,
            "key_points_covered": covered,
            "key_points_missing": [k for k in key_points if k not in covered],
        }

    def _build_PairwiseVerdict(self, prompt: str) -> dict[str, Any]:
        a = prompt.split("ANSWER A:", 1)[-1].split("ANSWER B:", 1)[0]
        b = prompt.split("ANSWER B:", 1)[-1]
        ref = _sig_tokens(prompt.split("REFERENCE ANSWER:", 1)[-1].split("ANSWER A:", 1)[0])
        sa, sb = len(_sig_tokens(a) & ref), len(_sig_tokens(b) & ref)
        winner = "tie" if abs(sa - sb) <= 1 else ("A" if sa > sb else "B")
        return {"reasoning": f"Reference overlap A={sa}, B={sb}.", "winner": winner}
