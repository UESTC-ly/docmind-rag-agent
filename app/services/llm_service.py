"""LLM 调用封装。RAG 和 Agent 共用。

同步客户端；异步场景用 asyncio.to_thread 包一层，避免阻塞事件循环。

注意：当前中转端点（gzxsy.vip）无视 stream 参数、强制以 SSE 流式返回。
非流式请求收到 event-stream 时，openai SDK 不解析、直接把响应体当 str 返回，
导致 resp.choices 报错。故这里统一走 stream=True 消费，再把增量聚合回一个
标准 ChatCompletionMessage，对上层（RAG / Agent）保持返回契约不变。
"""

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from app.config import settings

_client = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    timeout=settings.openai_timeout_seconds,
    max_retries=settings.openai_max_retries,
)


def _aggregate_stream(stream) -> ChatCompletionMessage:
    """把流式分块聚合成一个完整的 message 对象。

    - content：逐块拼接
    - tool_calls：按 delta.tool_calls[i].index 分组累加（name/arguments 都是增量）
    """
    content_parts: list[str] = []
    # index -> {"id", "name", "arguments"}
    tool_calls: dict[int, dict] = {}

    for chunk in stream:
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

    built_calls = [
        ChatCompletionMessageToolCall(
            id=slot["id"] or "",
            type="function",
            function=Function(name=slot["name"], arguments=slot["arguments"]),
        )
        for _, slot in sorted(tool_calls.items())
    ]

    return ChatCompletionMessage(
        role="assistant",
        content="".join(content_parts) or None,
        tool_calls=built_calls or None,
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
    kwargs = {
        "model": settings.chat_model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,  # 中转强制流式，显式声明以正确解析
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    stream = _client.chat.completions.create(**kwargs)
    return _aggregate_stream(stream)


def chat_completion_stream(messages: list[dict], temperature: float = 0.3):
    """流式对话补全：逐块 yield 文本增量，供 SSE 端点使用（不支持 tools）。"""
    stream = _client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
