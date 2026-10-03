.PHONY: install lint test eval-offline eval run serve up down graph clean

PY ?= python3

install:            ## editable install with dev tools
	$(PY) -m pip install -e ".[dev,ddg]"

lint:               ## ruff + mypy
	ruff check src tests && ruff format --check src tests && mypy src

test:               ## full offline test suite (fake LLM + real MCP servers)
	pytest

eval-offline:       ## evaluation pipeline smoke test, no API keys
	LLM_PROVIDER=fake EVAL_MIN_OVERALL=2.5 ara eval

eval:               ## real evaluation with configured provider (Bedrock by default)
	ara eval --baseline reports/baseline.json

run:                ## ask a question: make run Q="..."
	ara run "$(Q)"

serve:              ## API on :8000
	ara serve

up:                 ## API + MCP servers in containers
	docker compose up --build

down:
	docker compose down

graph:              ## print the LangGraph topology as Mermaid
	@LLM_PROVIDER=fake ara graph

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache reports
