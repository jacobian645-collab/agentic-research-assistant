# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Shared bootstrap for the MCP tool servers.

Each server is an independent process exposing tools over the Model Context Protocol:

* ``stdio`` transport — spawned on demand by the agent runtime (local dev, tests).
* ``streamable-http`` transport — long-running service (docker-compose / Kubernetes),
  run **stateless** so replicas can sit behind a plain load balancer.

The process boundary is deliberate: tools get their own dependencies, IAM role,
network policy, resource limits and release cadence, and can be reused by any MCP
client (Claude Desktop, IDEs, other agents) — not only this graph.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from mcp.server.fastmcp import FastMCP


def serve(factory: Callable[[str, int], FastMCP], default_port: int) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    server = factory(args.host, args.port)
    server.run(transport=args.transport)


def new_server(name: str, instructions: str, host: str, port: int) -> FastMCP:
    return FastMCP(
        name,
        instructions=instructions,
        host=host,
        port=port,
        stateless_http=True,  # no sticky sessions → horizontal scaling
        json_response=True,
        log_level="WARNING",
    )
