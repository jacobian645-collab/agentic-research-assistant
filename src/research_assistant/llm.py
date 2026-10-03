# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Provider-agnostic chat-model factory (Bedrock first).

Roles map to model tiers:

======== ======================== =============================================
role     default (Bedrock)        used by
======== ======================== =============================================
reasoning Claude Sonnet 4.6        planner, supervisor reflection, synthesizer, critic
fast      Claude Haiku 4.5         tool-calling worker loops + evidence extraction
judge     Claude Opus 4.6          offline / online LLM-as-judge evaluation
======== ======================== =============================================

Everything downstream depends only on ``BaseChatModel`` (``bind_tools`` +
``with_structured_output``), so swapping provider is a config change, not a code change.
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError

from research_assistant.config import Provider, Settings

Role = Literal["reasoning", "fast", "judge"]
T = TypeVar("T", bound=BaseModel)


def _model_id(role: Role, settings: Settings) -> str:
    return {
        "reasoning": settings.model_reasoning,
        "fast": settings.model_fast,
        "judge": settings.judge_model,
    }[role]


def get_chat_model(role: Role, settings: Settings) -> BaseChatModel:
    provider: Provider = settings.effective_judge_provider if role == "judge" else settings.llm_provider
    model = _model_id(role, settings)
    common: dict[str, Any] = {"temperature": settings.temperature}

    if provider == "bedrock":
        from botocore.config import Config
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(  # type: ignore[call-arg]
            model=model,
            region_name=settings.aws_region,
            max_tokens=settings.max_output_tokens,
            # adaptive retry mode backs off on ThrottlingException — the most common
            # production failure on Bedrock under parallel fan-out.
            config=Config(
                retries={"max_attempts": settings.llm_max_retries + 2, "mode": "adaptive"},
                read_timeout=int(settings.llm_timeout_s),
                connect_timeout=10,
            ),
            **common,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(  # type: ignore[call-arg]
            model=model,
            max_tokens=settings.max_output_tokens,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout_s,
            **common,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout_s,
            **common,
        )
    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(  # type: ignore[call-arg]
            azure_deployment=model,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout_s,
            **common,
        )
    if provider == "fake":
        from research_assistant.llm_fake import FakeResearchLLM

        return FakeResearchLLM(role=role)
    raise ValueError(f"Unknown LLM provider: {provider}")


def structured(llm: BaseChatModel, schema: type[T], settings: Settings) -> Runnable[Any, T]:
    """Bind a Pydantic schema as the model's output contract, with validation retries.

    * Uses tool/function calling under the hood (portable across Bedrock Converse,
      Anthropic, OpenAI/Azure) rather than provider-specific JSON modes.
    * Retries on schema-validation failures — the LLM gets a fresh attempt instead of
      the graph crashing on a malformed object.
    """
    kwargs: dict[str, Any] = {}
    if type(llm).__name__ in {"ChatOpenAI", "AzureChatOpenAI"}:
        kwargs["method"] = "function_calling"  # strict json_schema rejects optional fields
    runnable = llm.with_structured_output(schema, **kwargs)
    from langchain_core.exceptions import OutputParserException

    return runnable.with_retry(  # type: ignore[return-value]
        retry_if_exception_type=(ValidationError, OutputParserException, ValueError),
        stop_after_attempt=settings.llm_max_retries,
    )
