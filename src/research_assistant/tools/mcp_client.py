# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""MCP client side: connect to tool servers and hand each sub-agent ONLY its own tools.

Least privilege is enforced *structurally*: the retrieval agent never receives
``fetch_url``; the web agent never receives ``run_python``. A prompt-injected web page
therefore cannot make the web agent execute code — the capability simply isn't bound.
"""

from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from research_assistant.config import REPO_ROOT, Settings
from research_assistant.schemas import AgentName

# server → module (stdio)  |  agent → servers it may use
SERVER_MODULES = {
    "web": "research_assistant.mcp_servers.web_search_server",
    "retrieval": "research_assistant.mcp_servers.retrieval_server",
    "code": "research_assistant.mcp_servers.code_exec_server",
}
AGENT_SERVERS: dict[AgentName, tuple[str, ...]] = {
    "web_researcher": ("web",),
    "retrieval_researcher": ("retrieval",),
    "code_analyst": ("code",),
}


def _server_env(server: str, settings: Settings) -> dict[str, str]:
    """Explicit, minimal environment per server (stdio children do not inherit secrets
    by default — the code sandbox in particular gets no cloud credentials)."""
    base = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")]).strip(
            os.pathsep
        ),
        "HOME": os.environ.get("HOME", "/tmp"),
        "ARA_HOME": str(REPO_ROOT),
    }
    if server == "web":
        base["WEB_SEARCH_PROVIDER"] = settings.web_search_provider
        if settings.tavily_api_key:
            base["TAVILY_API_KEY"] = settings.tavily_api_key
    elif server == "retrieval":
        base.update(
            {
                "CORPUS_DIR": str(settings.corpus_dir),
                "RETRIEVAL_EMBEDDINGS": settings.retrieval_embeddings,
                "AWS_REGION": settings.aws_region,
            }
        )
        for k in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            if k in os.environ and settings.retrieval_embeddings == "bedrock":
                base[k] = os.environ[k]
    return base


def server_connections(settings: Settings) -> dict[str, dict[str, Any]]:
    if settings.mcp_transport == "streamable_http":
        urls = {
            "web": settings.mcp_web_url,
            "retrieval": settings.mcp_retrieval_url,
            "code": settings.mcp_code_url,
        }
        return {name: {"transport": "streamable_http", "url": url} for name, url in urls.items()}
    py = settings.python_executable or sys.executable
    return {
        name: {
            "transport": "stdio",
            "command": py,
            "args": ["-m", module],
            "env": _server_env(name, settings),
        }
        for name, module in SERVER_MODULES.items()
    }


class MCPToolRegistry:
    """Async context manager owning MCP sessions for the lifetime of the app.

    ``persistent=True`` keeps one session per server open (connection reuse; for stdio it
    avoids spawning a process per tool call). ``persistent=False`` opens a session per
    call — simpler lifecycle, used by the LangGraph Studio graph factory.
    """

    def __init__(self, settings: Settings, persistent: bool = True) -> None:
        self.settings = settings
        self.persistent = persistent
        self.client = MultiServerMCPClient(server_connections(settings))  # type: ignore[arg-type]
        self._stack: AsyncExitStack | None = None
        self.tools_by_server: dict[str, list[BaseTool]] = {}

    async def __aenter__(self) -> MCPToolRegistry:
        self._stack = AsyncExitStack()
        for server in SERVER_MODULES:
            if self.persistent:
                session = await self._stack.enter_async_context(self.client.session(server))
                tools = await load_mcp_tools(session)
            else:
                tools = await self.client.get_tools(server_name=server)
            self.tools_by_server[server] = tools
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._stack:
            await self._stack.aclose()

    def for_agent(self, agent: AgentName) -> list[BaseTool]:
        return [t for s in AGENT_SERVERS[agent] for t in self.tools_by_server.get(s, [])]

    def toolsets(self) -> dict[AgentName, list[BaseTool]]:
        return {a: self.for_agent(a) for a in AGENT_SERVERS}
