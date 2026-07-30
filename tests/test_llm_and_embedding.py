"""llm_service 协议适配 + embedding_service 分批 的测试。

用假的流式 chunk 验证 _aggregate_stream 的内容拼接与 tool_calls 按 index 合并；
用假 client 验证 SSE / 非 SSE Chat Completions 与 embedding 分批行为。
"""

import httpx
import pytest

from app.services import embedding_service, llm_service
from app.services.provider_telemetry import capture_provider_telemetry


# ── 假流式 chunk 构造 ────────────────────────────────────────
class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta):
        self.delta = delta


class _Chunk:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [_Choice(_Delta(content, tool_calls))]


class _TCDelta:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class _HttpResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class TestAggregateStream:
    def test_concatenates_content(self):
        stream = [_Chunk(content="你"), _Chunk(content="好"), _Chunk(content="！")]
        msg = llm_service._aggregate_stream(stream)
        assert msg.content == "你好！"
        assert msg.tool_calls is None

    def test_skips_chunks_without_choices(self):
        empty = type("E", (), {"choices": []})()
        stream = [empty, _Chunk(content="ok")]
        msg = llm_service._aggregate_stream(stream)
        assert msg.content == "ok"

    def test_merges_tool_call_deltas_by_index(self):
        # 工具名与参数分多个 chunk 增量到达，按 index 合并
        stream = [
            _Chunk(tool_calls=[_TCDelta(0, id="call_1", name="get_weather", arguments='{"ci')]),
            _Chunk(tool_calls=[_TCDelta(0, arguments='ty":"北京"}')]),
        ]
        msg = llm_service._aggregate_stream(stream)
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]
        assert tc.id == "call_1"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"city":"北京"}'

    def test_multiple_tool_calls_distinct_index(self):
        stream = [
            _Chunk(tool_calls=[_TCDelta(0, id="c0", name="skill_a", arguments="{}")]),
            _Chunk(tool_calls=[_TCDelta(1, id="c1", name="skill_b", arguments="{}")]),
        ]
        msg = llm_service._aggregate_stream(stream)
        assert [tc.function.name for tc in msg.tool_calls] == ["skill_a", "skill_b"]


class TestChatCompletionUsesStream:
    def test_passes_stream_true_and_aggregates(self, monkeypatch):
        captured = {}

        def _fake_create(**kwargs):
            captured.update(kwargs)
            return [_Chunk(content="hi")]

        monkeypatch.setattr(llm_service._client.chat.completions, "create", _fake_create)
        monkeypatch.setattr(llm_service.settings, "openai_stream", True)
        msg = llm_service.chat_completion([{"role": "user", "content": "x"}])
        assert msg.content == "hi"
        assert captured["stream"] is True

    def test_tools_trigger_tool_choice(self, monkeypatch):
        captured = {}

        def _fake_create(**kwargs):
            captured.update(kwargs)
            return [_Chunk(content="ok")]

        monkeypatch.setattr(llm_service._client.chat.completions, "create", _fake_create)
        monkeypatch.setattr(llm_service.settings, "openai_stream", True)
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        llm_service.chat_completion([{"role": "user", "content": "x"}], tools=tools)
        assert captured["tool_choice"] == "auto"

    def test_explicit_tool_choice_is_forwarded(self, monkeypatch):
        captured = {}

        def _fake_create(**kwargs):
            captured.update(kwargs)
            return [_Chunk(content="ok")]

        monkeypatch.setattr(llm_service._client.chat.completions, "create", _fake_create)
        monkeypatch.setattr(llm_service.settings, "openai_stream", True)
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        choice = {"type": "function", "function": {"name": "f"}}
        llm_service.chat_completion(
            [{"role": "user", "content": "x"}],
            tools=tools,
            tool_choice=choice,
        )
        assert captured["tool_choice"] == choice

    def test_non_stream_returns_provider_message(self, monkeypatch):
        captured = {}

        def _fake_post(url, *, headers, json, timeout):
            captured.update(
                {
                    "url": url,
                    "headers": headers,
                    "json": json,
                    "timeout": timeout,
                }
            )
            return _HttpResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "non-stream response",
                            }
                        }
                    ]
                }
            )

        monkeypatch.setattr(llm_service.httpx, "post", _fake_post)
        monkeypatch.setattr(llm_service.settings, "openai_stream", False)
        monkeypatch.setattr(
            llm_service.settings,
            "openai_base_url",
            "https://relay.test/v1",
        )

        message = llm_service.chat_completion(
            [{"role": "user", "content": "x"}]
        )
        assert message.content == "non-stream response"
        assert captured["url"] == "https://relay.test/v1/chat/completions"
        assert "stream" not in captured["json"]

    def test_non_stream_preserves_tools_and_parses_provider_tool_calls(
        self, monkeypatch
    ):
        captured = {}

        def _fake_post(_url, *, headers, json, timeout):
            captured.update({"headers": headers, "json": json, "timeout": timeout})
            return _HttpResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "tool result",
                                "tool_calls": [
                                    None,
                                    {"function": None},
                                    {
                                        "id": "call_1",
                                        "function": {
                                            "name": "search",
                                            "arguments": '{"q":"SciFact"}',
                                        },
                                    },
                                ],
                            }
                        }
                    ]
                }
            )

        tools = [
            {
                "type": "function",
                "function": {"name": "search", "parameters": {}},
            }
        ]
        choice = {"type": "function", "function": {"name": "search"}}
        monkeypatch.setattr(llm_service.httpx, "post", _fake_post)
        monkeypatch.setattr(llm_service.settings, "openai_stream", False)
        monkeypatch.setattr(
            llm_service.settings,
            "openai_send_temperature",
            True,
        )

        message = llm_service.chat_completion(
            [{"role": "user", "content": "x"}],
            tools=tools,
            temperature=0.7,
            tool_choice=choice,
        )

        assert captured["json"]["temperature"] == 0.7
        assert captured["json"]["tools"] == tools
        assert captured["json"]["tool_choice"] == choice
        assert message.content == "tool result"
        assert message.tool_calls[0].function.name == "search"

    @pytest.mark.parametrize(
        "payload",
        [
            {"choices": []},
            {"choices": [{"message": ["not", "an", "object"]}]},
        ],
    )
    def test_non_stream_rejects_invalid_provider_message(
        self, monkeypatch, payload
    ):
        monkeypatch.setattr(
            llm_service.httpx,
            "post",
            lambda *_args, **_kwargs: _HttpResponse(payload),
        )
        monkeypatch.setattr(llm_service.settings, "openai_stream", False)

        with pytest.raises(ValueError):
            llm_service.chat_completion([{"role": "user", "content": "x"}])

    def test_non_stream_accepts_a_full_chat_completions_url(self, monkeypatch):
        captured = {}

        def _fake_post(url, **_kwargs):
            captured["url"] = url
            return _HttpResponse(
                {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            )

        monkeypatch.setattr(llm_service.httpx, "post", _fake_post)
        monkeypatch.setattr(llm_service.settings, "openai_stream", False)
        monkeypatch.setattr(
            llm_service.settings,
            "openai_base_url",
            "https://relay.test/v1/chat/completions/",
        )

        assert llm_service.chat_completion(
            [{"role": "user", "content": "x"}]
        ).content == "ok"
        assert captured["url"] == "https://relay.test/v1/chat/completions"

    def test_non_stream_chat_sse_yields_one_content_chunk(self, monkeypatch):
        monkeypatch.setattr(
            llm_service.httpx,
            "post",
            lambda *_args, **_kwargs: _HttpResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "single response",
                            }
                        }
                    ]
                }
            ),
        )
        monkeypatch.setattr(llm_service.settings, "openai_stream", False)

        assert list(
            llm_service.chat_completion_stream(
                [{"role": "user", "content": "x"}]
            )
        ) == ["single response"]

    def test_omits_temperature_when_provider_requires_it(self, monkeypatch):
        captured = {}

        def _fake_post(_url, *, headers, json, timeout):
            captured.update(
                {
                    "headers": headers,
                    "json": json,
                    "timeout": timeout,
                }
            )
            return _HttpResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "configured",
                            }
                        }
                    ]
                }
            )

        monkeypatch.setattr(llm_service.httpx, "post", _fake_post)
        monkeypatch.setattr(llm_service.settings, "openai_stream", False)
        monkeypatch.setattr(
            llm_service.settings,
            "openai_send_temperature",
            False,
        )

        llm_service.chat_completion(
            [{"role": "user", "content": "x"}],
            temperature=0.7,
        )

        assert "temperature" not in captured["json"]

    def test_non_stream_retries_transient_provider_error(self, monkeypatch):
        calls = 0
        request = httpx.Request("POST", "https://relay.test/v1/chat/completions")

        def _fake_post(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                response = httpx.Response(503, request=request)
                raise httpx.HTTPStatusError(
                    "temporarily unavailable",
                    request=request,
                    response=response,
                )
            return _HttpResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "recovered",
                            }
                        }
                    ]
                }
            )

        monkeypatch.setattr(llm_service.httpx, "post", _fake_post)
        monkeypatch.setattr(llm_service.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(llm_service.settings, "openai_stream", False)
        monkeypatch.setattr(llm_service.settings, "openai_max_retries", 1)

        message = llm_service.chat_completion(
            [{"role": "user", "content": "retry"}]
        )

        assert message.content == "recovered"
        assert calls == 2

    def test_non_stream_retries_transport_error(self, monkeypatch):
        calls = 0
        request = httpx.Request("POST", "https://relay.test/v1/chat/completions")

        def _fake_post(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectError("offline", request=request)
            return _HttpResponse(
                {"choices": [{"message": {"role": "assistant", "content": "back"}}]}
            )

        monkeypatch.setattr(llm_service.httpx, "post", _fake_post)
        monkeypatch.setattr(llm_service.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(llm_service.settings, "openai_stream", False)
        monkeypatch.setattr(llm_service.settings, "openai_max_retries", 1)

        assert llm_service.chat_completion(
            [{"role": "user", "content": "retry"}]
        ).content == "back"
        assert calls == 2

    def test_non_stream_reports_only_explicit_provider_metering(self, monkeypatch):
        monkeypatch.setattr(
            llm_service.httpx,
            "post",
            lambda *_args, **_kwargs: _HttpResponse(
                {
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {
                        "prompt_tokens": 13,
                        "completion_tokens": 8,
                    },
                    "cost": {"amount": 0.004, "currency": "usd"},
                }
            ),
        )
        monkeypatch.setattr(llm_service.settings, "openai_stream", False)

        with capture_provider_telemetry() as telemetry:
            message = llm_service.chat_completion(
                [{"role": "user", "content": "metered"}]
            )

        assert message.content == "ok"
        assert telemetry.usage_public()["total_tokens"] == 21.0
        assert telemetry.usage_public()["request_count"] == 1
        assert telemetry.cost_public()["amount"] == 0.004

    def test_non_stream_records_retry_count_without_estimating_missing_usage(
        self, monkeypatch
    ):
        calls = 0
        request = httpx.Request("POST", "https://relay.test/v1/chat/completions")

        def _fake_post(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                response = httpx.Response(503, request=request)
                raise httpx.HTTPStatusError(
                    "temporarily unavailable",
                    request=request,
                    response=response,
                )
            return _HttpResponse(
                {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            )

        monkeypatch.setattr(llm_service.httpx, "post", _fake_post)
        monkeypatch.setattr(llm_service.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(llm_service.settings, "openai_stream", False)
        monkeypatch.setattr(llm_service.settings, "openai_max_retries", 1)

        with capture_provider_telemetry() as telemetry:
            llm_service.chat_completion([{"role": "user", "content": "retry"}])

        usage = telemetry.usage_public()
        assert usage["status"] == "unavailable"
        assert usage["request_count"] == 2
        assert usage["retry_count"] == 1

    def test_non_stream_does_not_retry_non_transient_http_error(
        self, monkeypatch
    ):
        calls = 0
        request = httpx.Request("POST", "https://relay.test/v1/chat/completions")

        def _fake_post(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            response = httpx.Response(401, request=request)
            raise httpx.HTTPStatusError(
                "unauthorized",
                request=request,
                response=response,
            )

        monkeypatch.setattr(llm_service.httpx, "post", _fake_post)
        monkeypatch.setattr(llm_service.settings, "openai_stream", False)
        monkeypatch.setattr(llm_service.settings, "openai_max_retries", 3)

        with pytest.raises(httpx.HTTPStatusError):
            llm_service.chat_completion([{"role": "user", "content": "x"}])

        assert calls == 1

    def test_stream_generator_yields_content_and_forwards_temperature(
        self, monkeypatch
    ):
        captured = {}

        def _fake_create(**kwargs):
            captured.update(kwargs)
            return [
                type("E", (), {"choices": []})(),
                _Chunk(content="stream"),
                _Chunk(content="ed"),
            ]

        monkeypatch.setattr(
            llm_service._client.chat.completions,
            "create",
            _fake_create,
        )
        monkeypatch.setattr(llm_service.settings, "openai_stream", True)
        monkeypatch.setattr(
            llm_service.settings,
            "openai_send_temperature",
            True,
        )

        assert list(
            llm_service.chat_completion_stream(
                [{"role": "user", "content": "x"}],
                temperature=0.4,
            )
        ) == ["stream", "ed"]
        assert captured["stream"] is True
        assert captured["temperature"] == 0.4


# ── embedding 分批 ──────────────────────────────────────────
class _EmbItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _EmbResp:
    def __init__(self, n):
        self.data = [_EmbItem([0.1, 0.2]) for _ in range(n)]


class TestEmbedTexts:
    def test_empty_returns_empty(self):
        assert embedding_service.embed_texts([]) == []

    def test_batches_by_batch_size(self, monkeypatch):
        calls = []

        def _fake_create(model, input):
            calls.append(len(input))
            return _EmbResp(len(input))

        monkeypatch.setattr(embedding_service._client.embeddings, "create", _fake_create)
        monkeypatch.setattr(embedding_service.settings, "embedding_batch_size", 2)
        # 5 条 → 分批 2/2/1
        vectors = embedding_service.embed_texts(["a", "b", "c", "d", "e"])
        assert len(vectors) == 5
        assert calls == [2, 2, 1]

    def test_embed_query_returns_single_vector(self, monkeypatch):
        monkeypatch.setattr(
            embedding_service._client.embeddings, "create",
            lambda model, input: _EmbResp(len(input)),
        )
        v = embedding_service.embed_query("问题")
        assert v == [0.1, 0.2]
