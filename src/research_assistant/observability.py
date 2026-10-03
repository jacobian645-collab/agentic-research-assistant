# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Observability: structured logs + token/cost accounting per run.

* Tracing: LangChain/LangGraph emit OpenTelemetry-compatible run trees. Set
  ``LANGSMITH_TRACING=true`` + ``LANGSMITH_API_KEY`` (or point an OTel exporter at
  Langfuse / Arize / CloudWatch) — no code changes required.
* Metrics: ``UsageTracker`` is a callback handler attached per request. It aggregates
  tokens, LLM calls, tool calls and estimated USD cost by model, which the API returns
  and the eval harness reports (cost and latency are first-class quality metrics).
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# USD per 1M tokens (input, output). Indicative on-demand list prices — verify against
# https://aws.amazon.com/bedrock/pricing/ for your region / endpoint type before relying on them.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "fake": (0.0, 0.0),
}


def price_for(model: str) -> tuple[float, float]:
    m = model.lower()
    for key in sorted(PRICING_PER_MTOK, key=len, reverse=True):
        if key in m:
            return PRICING_PER_MTOK[key]
    return (0.0, 0.0)


class UsageTracker(BaseCallbackHandler):
    """Thread-safe aggregation of LLM + tool usage for a single graph run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.by_model: dict[str, dict[str, float]] = {}
        self.tool_calls: dict[str, int] = {}
        self.started = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                usage = getattr(msg, "usage_metadata", None) or {}
                meta = getattr(msg, "response_metadata", {}) or {}
                model = str(meta.get("model_id") or meta.get("model_name") or meta.get("model") or "unknown")
                with self._lock:
                    row = self.by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
                    row["calls"] += 1
                    row["input_tokens"] += usage.get("input_tokens", 0)
                    row["output_tokens"] += usage.get("output_tokens", 0)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        name = (serialized or {}).get("name", "tool")
        with self._lock:
            self.tool_calls[name] = self.tool_calls.get(name, 0) + 1

    def summary(self) -> dict[str, Any]:
        cost = 0.0
        tin = tout = calls = 0
        for model, row in self.by_model.items():
            pin, pout = price_for(model)
            cost += row["input_tokens"] / 1e6 * pin + row["output_tokens"] / 1e6 * pout
            tin += int(row["input_tokens"])
            tout += int(row["output_tokens"])
            calls += int(row["calls"])
        return {
            "llm_calls": calls,
            "input_tokens": tin,
            "output_tokens": tout,
            "est_cost_usd": round(cost, 5),
            "tool_calls": dict(self.tool_calls),
            "wall_time_s": round(time.perf_counter() - self.started, 2),
            "by_model": self.by_model,
        }


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        _JsonFormatter()
        if json_logs
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s │ %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "botocore", "urllib3", "mcp", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ara.{name}")
