# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Sandboxed Python-execution MCP server.

Tool
----
``run_python(code, timeout_s)`` → {stdout, stderr, exit_code, timed_out, duration_s}

Layered isolation (each layer assumes the previous one can fail):

1. **Static policy** — AST scan rejects non-allow-listed imports, dunder attribute access
   and dangerous builtins (``exec``, ``eval``, ``open``, ``__import__`` …).
2. **Process isolation** — fresh ``python -I`` interpreter per call, empty temp CWD,
   scrubbed environment (no credentials inherited).
3. **Resource limits** — CPU seconds, address space, file size, open files (POSIX rlimits)
   + wall-clock timeout with hard kill.
4. **Container / network** — in docker-compose this server runs with ``network_mode: none``,
   read-only root FS, dropped capabilities and memory/pids limits.

For multi-tenant production use a micro-VM sandbox (Firecracker / gVisor, e.g. Amazon
Bedrock AgentCore Code Interpreter, E2B, Modal). This server is the *interface*; the
isolation backend is swappable behind the same MCP tool contract.
"""

from __future__ import annotations

import ast
import asyncio
import os
import sys
import tempfile
import time
from typing import Any

from research_assistant.mcp_servers._common import new_server, serve

ALLOWED_IMPORTS = {
    "math",
    "statistics",
    "decimal",
    "fractions",
    "itertools",
    "functools",
    "collections",
    "datetime",
    "json",
    "re",
    "random",
    "string",
    "textwrap",
    "heapq",
    "bisect",
    "operator",
    "dataclasses",
    "typing",
    "numpy",
    "pandas",
    "csv",
    "io",
}
BLOCKED_CALLS = {
    "exec",
    "eval",
    "compile",
    "open",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "memoryview",
}
MAX_CODE_CHARS = 20_000
MAX_OUTPUT_CHARS = 10_000


class PolicyViolation(Exception):
    pass


def check_policy(code: str) -> None:
    if len(code) > MAX_CODE_CHARS:
        raise PolicyViolation(f"code exceeds {MAX_CODE_CHARS} chars")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise PolicyViolation(f"SyntaxError: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise PolicyViolation(f"import of '{a.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise PolicyViolation(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PolicyViolation(f"access to dunder attribute '{node.attr}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_CALLS:
            raise PolicyViolation(f"use of '{node.id}' is not allowed")


def _limits(cpu_s: int, mem_mb: int):  # pragma: no cover - runs in child process
    def apply() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            if sys.platform.startswith("linux"):
                resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
            os.setsid()
        except Exception:
            pass

    return apply


def _truncate(s: str) -> str:
    return (
        s
        if len(s) <= MAX_OUTPUT_CHARS
        else s[:MAX_OUTPUT_CHARS] + f"\n…[truncated {len(s) - MAX_OUTPUT_CHARS}]"
    )


async def execute(code: str, timeout_s: int = 10, mem_mb: int = 768) -> dict[str, Any]:
    timeout_s = max(1, min(int(timeout_s), 30))
    try:
        check_policy(code)
    except PolicyViolation as e:
        return {
            "stdout": "",
            "stderr": f"PolicyViolation: {e}",
            "exit_code": 2,
            "timed_out": False,
            "duration_s": 0.0,
        }
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ara-sbx-") as tmp:
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": tmp,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            code,
            cwd=tmp,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=_limits(timeout_s, mem_mb) if os.name == "posix" else None,
        )
        timed_out = False
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s + 1)
        except TimeoutError:
            timed_out = True
            proc.kill()
            out, err = await proc.communicate()
    rc = proc.returncode if not timed_out else -9
    stderr = err.decode(errors="replace")
    if rc in (-24, -9) and not stderr:  # SIGXCPU / SIGKILL from rlimits or wall-clock kill
        stderr = "Execution stopped: CPU/time limit exceeded."
        timed_out = True
    return {
        "stdout": _truncate(out.decode(errors="replace")),
        "stderr": _truncate(stderr),
        "exit_code": rc,
        "timed_out": timed_out,
        "duration_s": round(time.perf_counter() - t0, 3),
    }


def build_server(host: str = "127.0.0.1", port: int = 8103):
    mcp = new_server(
        "code-sandbox",
        "Execute short, self-contained Python snippets in an isolated sandbox (no network, no files).",
        host,
        port,
    )

    @mcp.tool()
    async def run_python(code: str, timeout_s: int = 10) -> dict[str, Any]:
        """Run a self-contained Python 3 script and return stdout/stderr. Always print() results.
        Allowed imports: math, statistics, decimal, fractions, itertools, collections, datetime, json, re,
        random, numpy, pandas (if installed). No network, filesystem or subprocess access."""
        return await execute(code, timeout_s)

    return mcp


def main() -> None:
    serve(build_server, default_port=8103)


if __name__ == "__main__":
    main()
