# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from research_assistant import config


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "offline")
    monkeypatch.setenv("API_KEY", "secret")
    config.get_settings.cache_clear()
    from research_assistant.api.app import app

    with TestClient(app) as c:
        yield c
    config.get_settings.cache_clear()


H = {"x-api-key": "secret"}
Q = {"question": "Which MCP transports exist and why run tool servers as stateless HTTP services?"}


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz").json()
    assert r["status"] == "ready" and "run_python" in r["tools"]["code"]


def test_auth_required(client):
    assert client.post("/v1/research", json=Q).status_code == 401


def test_research_sync_and_state(client):
    r = client.post("/v1/research", json=Q, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed" and body["markdown"].startswith("# ")
    st = client.get(f"/v1/research/{body['thread_id']}", headers=H).json()
    assert st["final_report"]["title"]


def test_research_stream_sse(client):
    with client.stream("POST", "/v1/research/stream", json=Q, headers=H) as r:
        text = "".join(r.iter_text())
    assert "event: plan" in text and "event: tool_call" in text and "event: result" in text


def test_validation_error(client):
    assert client.post("/v1/research", json={"question": "hi"}, headers=H).status_code == 422
