"""LLM 调用封装。RAG 和 Agent 共用。

同步客户端；异步场景用 asyncio.to_thread 包一层，避免阻塞事件循环。

OpenAI-compatible relays differ in their SSE support. ``OPENAI_STREAM`` keeps
the established streaming default, while ``false`` uses a regular Chat
Completions response and preserves the same message contract for callers.
"""

import time
from typing import Any

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from app.config import settings
from app.services.provider_telemetry import (
    record_provider_attempt,
    record_provider_response,
)

_client = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    timeout=settings.openai_timeout_seconds,
    max_retries=settings.openai_max_retries,
)


def _build_message(
    content: str | None,
    raw_tool_calls: list[dict[str, str | None]],
) -> ChatCompletionMessage:
    """Create the common message type from streaming or JSON tool calls."""

    built_calls = [
        ChatCompletionMessageToolCall(
            id=slot["id"] or "",
            type="function",
            function=Function(
                name=slot["name"] or "",
                arguments=slot["arguments"] or "",
            ),
        )
        for slot in raw_tool_calls
    ]
    return ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=built_calls or None,
    )


def _aggregate_stream(stream) -> ChatCompletionMessage:
    """把流式分块聚合成一个完整的 message 对象。

    - content：逐块拼接
    - tool_calls：按 delta.tool_calls[i].index 分组累加（name/arguments 都是增量）
    """
    content_parts: list[str] = []
    # index -> {"id", "name", "arguments"}
    tool_calls: dict[int, dict] = {}
    usage: dict[str, Any] | None = None
    model: str | None = None

    for chunk in stream:
        raw_model = getattr(chunk, "model", None)
        if isinstance(raw_model, str) and raw_model.strip():
            model = raw_model.strip()
        raw_usage = getattr(chunk, "usage", None)
        if raw_usage is not None:
            if hasattr(raw_usage, "model_dump"):
                usage = raw_usage.model_dump(exclude_none=True)
            elif isinstance(raw_usage, dict):
                usage = raw_usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content_parts.append(delta.content)
        for tc in delta.tool_calls or []:
            slot = tool_calls.setdefault(
                tc.index, {"id": None, "name": "", "arguments": ""}
            )
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments

    telemetry_payload: dict[str, Any] = {}
    if usage is not None:
        telemetry_payload["usage"] = usage
    if model is not None:
        telemetry_payload["model"] = model
    record_provider_response(telemetry_payload)
    return _build_message(
        "".join(content_parts) or None,
        [
            {
                "id": slot["id"],
                "name": slot["name"],
                "arguments": slot["arguments"],
            }
            for _, slot in sorted(tool_calls.items())
        ],
    )


def _chat_completion_endpoint() -> str:
    """Accept either an OpenAI base URL or the full Chat Completions URL."""

    base_url = (
        settings.openai_base_url or "https://api.openai.com/v1"
    ).rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _should_retry_nonstream_error(error: Exception) -> bool:
    """Retry only transient failures that the OpenAI SDK would also retry."""

    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429 or error.response.status_code >= 500
    return isinstance(error, httpx.RequestError)


def _post_nonstream_with_retries(payload: dict[str, Any]) -> httpx.Response:
    """Preserve configured retry behavior when the SDK streaming path is off."""

    retry_count = max(0, int(settings.openai_max_retries))
    for attempt in range(retry_count + 1):
        record_provider_attempt(
            retry=attempt > 0,
            model=str(payload.get("model") or ""),
        )
        try:
            response = httpx.post(
                _chat_completion_endpoint(),
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json=payload,
                timeout=settings.openai_timeout_seconds,
            )
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.RequestError) as error:
            if attempt >= retry_count or not _should_retry_nonstream_error(error):
                raise
            # Keep retries bounded for interactive Agent requests while giving
            # relays a brief chance to recover from a transient 429/5xx.
            time.sleep(min(0.5 * (2**attempt), 4.0))

    raise RuntimeError("unreachable non-stream retry state")  # pragma: no cover


def _nonstream_chat_completion(
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float,
    tool_choice: str | dict,
) -> ChatCompletionMessage:
    """Call compatible non-SSE endpoints without OpenAI SDK transport headers."""

    payload: dict[str, Any] = {
        "model": settings.chat_model,
        "messages": messages,
    }
    if settings.openai_send_temperature:
        payload["temperature"] = temperature
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    response = _post_nonstream_with_retries(payload)
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("OpenAI-compatible response must be a JSON object")
    record_provider_response(body)
    try:
        message = body["choices"][0]["message"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            "OpenAI-compatible response is missing choices[0].message"
        ) from exc
    if not isinstance(message, dict):
        raise ValueError(
            "OpenAI-compatible response choices[0].message must be an object"
        )

    tool_calls: list[dict[str, str | None]] = []
    for raw_call in message.get("tool_calls") or []:
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict):
            continue
        tool_calls.append(
            {
                "id": str(raw_call.get("id") or ""),
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or ""),
            }
        )
    content = message.get("content")
    return _build_message(
        content if isinstance(content, str) else None,
        tool_calls,
    )


def chat_completion(
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.3,
    tool_choice: str | dict = "auto",
):
    """通用对话补全。

    messages: [{"role": "system"|"user"|"assistant"|"tool", "content": ...}, ...]
    tools:    function calling 的工具定义；传了 LLM 才可能返回 tool_calls
    返回原始的 message 对象（含 .content 和 .tool_calls）。
    """
    if not settings.openai_stream:
        return _nonstream_chat_completion(
            messages,
            tools,
            temperature,
            tool_choice,
        )

    kwargs = {
        "model": settings.chat_model,
        "messages": messages,
        "stream": True,
    }
    if settings.openai_send_temperature:
        kwargs["temperature"] = temperature
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    record_provider_attempt(model=settings.chat_model)
    response = _client.chat.completions.create(**kwargs)
    return _aggregate_stream(response)


def chat_completion_stream(messages: list[dict], temperature: float = 0.3):
    """流式对话补全：逐块 yield 文本增量，供 SSE 端点使用（不支持 tools）。"""
    if not settings.openai_stream:
        message = _nonstream_chat_completion(
            messages,
            None,
            temperature,
            "auto",
        )
        if message.content:
            yield message.content
        return

    kwargs = {
        "model": settings.chat_model,
        "messages": messages,
        "stream": True,
    }
    if settings.openai_send_temperature:
        kwargs["temperature"] = temperature
    record_provider_attempt(model=settings.chat_model)
    response = _client.chat.completions.create(
        **kwargs,
    )
    usage: dict[str, Any] | None = None
    model: str | None = None
    try:
        for chunk in response:
            raw_model = getattr(chunk, "model", None)
            if isinstance(raw_model, str) and raw_model.strip():
                model = raw_model.strip()
            raw_usage = getattr(chunk, "usage", None)
            if raw_usage is not None:
                if hasattr(raw_usage, "model_dump"):
                    usage = raw_usage.model_dump(exclude_none=True)
                elif isinstance(raw_usage, dict):
                    usage = raw_usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    finally:
        telemetry_payload: dict[str, Any] = {}
        if usage is not None:
            telemetry_payload["usage"] = usage
        if model is not None:
            telemetry_payload["model"] = model
        record_provider_response(telemetry_payload)
