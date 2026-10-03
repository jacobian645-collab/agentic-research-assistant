# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Graph factory for LangGraph Studio / LangGraph Platform (referenced from langgraph.json).

The platform supplies its own checkpointer and streaming, so we only build the graph.
MCP sessions are opened per tool call (``persistent=False``) because the platform owns
the process lifecycle.
"""

from __future__ import annotations

from research_assistant.config import get_settings
from research_assistant.graph import build_graph
from research_assistant.tools import MCPToolRegistry


async def make_graph():
    settings = get_settings()
    registry = await MCPToolRegistry(settings, persistent=False).__aenter__()
    return build_graph(settings, registry.toolsets())
