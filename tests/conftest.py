# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

from __future__ import annotations

import pytest

from research_assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Fully offline settings: fake LLM, fixture web search, stdio MCP servers."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider="fake",
        web_search_provider="offline",
        retrieval_embeddings="hash",
        mcp_transport="stdio",
        eval_output_dir=tmp_path / "reports",
    )
