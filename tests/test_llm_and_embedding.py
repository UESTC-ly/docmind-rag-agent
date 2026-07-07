"""llm_service 流式聚合 + embedding_service 分批 的测试。

用假的流式 chunk 验证 _aggregate_stream 的内容拼接与 tool_calls 按 index 合并；
用假 client 验证 embed_texts 的分批与空输入。
"""

from app.services import embedding_service, llm_service


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
        msg = llm_service.chat_completion([{"role": "user", "content": "x"}])
        assert msg.content == "hi"
        assert captured["stream"] is True  # 中转强制流式，必须显式声明

    def test_tools_trigger_tool_choice(self, monkeypatch):
        captured = {}

        def _fake_create(**kwargs):
            captured.update(kwargs)
            return [_Chunk(content="ok")]

        monkeypatch.setattr(llm_service._client.chat.completions, "create", _fake_create)
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        llm_service.chat_completion([{"role": "user", "content": "x"}], tools=tools)
        assert captured["tool_choice"] == "auto"


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
