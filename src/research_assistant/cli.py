# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Command-line interface: ``ara run | eval | calibrate | serve | graph``."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer

from research_assistant.config import get_settings
from research_assistant.observability import configure_logging

app = typer.Typer(add_completion=False, help="Agentic Research Assistant (LangGraph multi-agent + MCP).")


@app.callback()
def _init() -> None:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)


@app.command()
def run(
    question: str = typer.Argument(..., help="Research question"),
    stream: bool = typer.Option(True, help="Print progress events while running"),
    out: Path | None = typer.Option(None, help="Write the markdown report to this file"),
    show_trace: bool = typer.Option(False, "--trace", help="Print the run trace JSON"),
) -> None:
    """Run one research question end-to-end."""
    from research_assistant.service import ResearchService

    async def _main() -> dict:
        async with ResearchService(get_settings()) as svc:
            if not stream:
                return (await svc.run(question)).to_dict()
            result: dict = {}
            async for ev in svc.stream(question):
                kind = ev.get("event")
                if kind == "result":
                    result = ev
                elif kind != "node":
                    detail = {k: v for k, v in ev.items() if k not in ("event", "args", "subtasks")}
                    typer.secho(
                        f"  • {kind:<20} {json.dumps(detail, default=str)[:140]}", fg="cyan", err=True
                    )
            return result

    result = asyncio.run(_main())
    md = result.get("markdown") or "(no report)"
    if out:
        out.write_text(md)
        typer.secho(f"Report written to {out}", fg="green", err=True)
    typer.echo(md)
    u = result.get("usage", {})
    typer.secho(
        f"\nllm_calls={u.get('llm_calls')} tokens={u.get('input_tokens', 0) + u.get('output_tokens', 0)} "
        f"tool_calls={sum((u.get('tool_calls') or {}).values())} est_cost=${u.get('est_cost_usd')} "
        f"time={u.get('wall_time_s')}s",
        fg="yellow",
        err=True,
    )
    if show_trace:
        typer.echo(json.dumps(result.get("trace"), indent=2, default=str))


@app.command("eval")
def eval_cmd(
    dataset: Path | None = typer.Option(None, help="JSONL dataset (defaults to bundled set)"),
    limit: int | None = typer.Option(None, help="Evaluate only the first N items"),
    concurrency: int = typer.Option(2, help="Parallel questions"),
    baseline: Path | None = typer.Option(None, help="Previous eval JSON for regression gating"),
) -> None:
    """Run the offline LLM-as-judge evaluation and exit non-zero if gates fail (CI)."""
    from research_assistant.eval.run_eval import run_eval

    report = asyncio.run(run_eval(get_settings(), dataset, limit, concurrency, baseline))
    typer.echo(
        json.dumps(
            {"summary": report["summary"], "gates": report["gates"], "passed": report["passed"]}, indent=2
        )
    )
    typer.secho(f"Full report: {get_settings().eval_output_dir / 'latest.md'}", fg="green", err=True)
    raise typer.Exit(code=0 if report["passed"] else 1)


@app.command()
def calibrate(path: Path = typer.Argument(..., help="JSONL with human_score and judge_score")) -> None:
    """Measure judge-vs-human agreement (weighted kappa, Spearman)."""
    from research_assistant.eval.calibration import calibrate as _cal

    typer.echo(json.dumps(_cal(path), indent=2))


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Start the FastAPI service."""
    import uvicorn

    uvicorn.run("research_assistant.api.app:app", host=host, port=port, reload=reload)


@app.command()
def graph() -> None:
    """Print the orchestration graph as Mermaid."""
    from research_assistant.config import Settings
    from research_assistant.graph import build_graph

    s = Settings(llm_provider="fake", _env_file=None)  # type: ignore[call-arg]
    g = build_graph(s, {"web_researcher": [], "retrieval_researcher": [], "code_analyst": []})
    sys.stdout.write(g.get_graph().draw_mermaid())


if __name__ == "__main__":
    app()
