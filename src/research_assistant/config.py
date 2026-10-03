# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Centralised, typed configuration (12-factor: everything overridable via env / .env).

Design notes
------------
* One ``Settings`` object is built at process start and passed explicitly
  (no hidden globals inside agents) so tests can inject variants cheaply.
* Model *tiering* is first-class: planning / critique / judging use a stronger
  "reasoning" model; tool-calling workers use a cheaper "fast" model. This is the
  single biggest cost lever in a multi-agent system.
* The judge is configured independently of the generator so it can be a different
  model (ideally a different family) to reduce self-preference bias.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["bedrock", "anthropic", "openai", "azure_openai", "fake"]

PACKAGE_DIR = Path(__file__).resolve().parent


def _repo_root() -> Path:
    """Source checkout → repo root; installed wheel / container → ARA_HOME or CWD."""
    if os.environ.get("ARA_HOME"):
        return Path(os.environ["ARA_HOME"])
    here = PACKAGE_DIR.parents[1]
    return here if (here / "data").is_dir() else Path.cwd()


REPO_ROOT = _repo_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LLM provider & model tiering ────────────────────────────────────────
    llm_provider: Provider = "bedrock"
    # Bedrock Converse model / inference-profile IDs. `global.` = global cross-region
    # inference (best availability, no premium). Use `us.` / `eu.` / `apac.` / `jp.`
    # for data-residency-pinned routing (≈10% premium on newer models).
    model_reasoning: str = "global.anthropic.claude-sonnet-4-6"
    model_fast: str = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    temperature: float = 0.0
    max_output_tokens: int = 4096
    aws_region: str = "us-west-2"
    llm_max_retries: int = 3
    llm_timeout_s: float = 120.0

    # Judge (offline + online evaluation). Different (stronger) model by default.
    judge_provider: Provider | None = None  # None → same as llm_provider
    judge_model: str = "global.anthropic.claude-opus-4-6-v1"

    # Azure OpenAI specifics (only used when provider == azure_openai)
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-10-21"

    # ── MCP tool servers ────────────────────────────────────────────────────
    mcp_transport: Literal["stdio", "streamable_http"] = "stdio"
    mcp_web_url: str = "http://localhost:8101/mcp"
    mcp_retrieval_url: str = "http://localhost:8102/mcp"
    mcp_code_url: str = "http://localhost:8103/mcp"
    python_executable: str = Field(default_factory=lambda: sys.executable)

    # Web search backend used by the web MCP server: tavily | ddg | offline
    web_search_provider: Literal["tavily", "ddg", "offline"] = "offline"
    tavily_api_key: str | None = None
    corpus_dir: Path = REPO_ROOT / "data" / "corpus"
    retrieval_embeddings: Literal["none", "hash", "bedrock"] = "hash"

    # ── Orchestration budgets / guardrails ─────────────────────────────────
    max_research_rounds: int = 2  # supervisor re-planning rounds after first wave
    max_revisions: int = 2  # critic → synthesizer revision loops
    max_parallel_workers: int = 4  # fan-out width per wave
    max_worker_steps: int = 6  # ReAct iterations per worker
    max_tool_calls_per_worker: int = 6
    tool_output_char_limit: int = 6000  # context-window hygiene
    recursion_limit: int = 60  # LangGraph super-step ceiling (hard stop)
    critic_pass_threshold: float = 0.75  # min weighted score to accept a draft
    max_cost_usd: float = 0.50  # per-request cost cap → graceful partial answer, never a runaway
    require_plan_approval: bool = False  # human-in-the-loop interrupt after planning
    max_question_chars: int = 2000

    # ── Evaluation ─────────────────────────────────────────────────────────
    eval_dataset: Path = (
        REPO_ROOT / "src" / "research_assistant" / "eval" / "datasets" / "research_eval.jsonl"
    )
    eval_output_dir: Path = REPO_ROOT / "reports"
    eval_min_overall: float = 3.5  # CI gate on mean judge score (1-5 scale)
    online_eval_sample_rate: float = 0.0  # fraction of API traffic judged asynchronously

    # ── Service / observability ────────────────────────────────────────────
    log_level: str = "INFO"
    log_json: bool = False
    api_key: str | None = None  # if set, API requires `x-api-key`

    @property
    def effective_judge_provider(self) -> Provider:
        return self.judge_provider or self.llm_provider


@lru_cache
def get_settings() -> Settings:
    return Settings()
