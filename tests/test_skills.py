"""5 个 Skills 的单元测试。

各技能的外部边界都 mock 在其模块命名空间里：
- graph / mindmap：fetch_document_text + chat_completion
- report：embed_query + search + chat_completion（多次）
- kb_search：embed_query + search
- web_search：ddgs.DDGS
- _helpers：SyncSessionLocal（用同步内存库）
"""

from app.skills import graph, kb_search, mindmap, report, web_search
from app.skills.base import SkillContext

CTX = SkillContext(user_id=1, document_id=10)


class _FakeMsg:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None


# ── graph ────────────────────────────────────────────────────
class TestGraphSkill:
    def test_extracts_nodes_edges_and_mermaid(self, monkeypatch):
        monkeypatch.setattr(graph, "fetch_document_text", lambda uid, did: "文档全文")
        monkeypatch.setattr(
            graph, "chat_completion",
            lambda msgs, temperature=0.2: _FakeMsg(
                '{"nodes":[{"id":"张三","type":"人物"}],'
                '"edges":[{"source":"张三","target":"公司A","relation":"任职"}]}'
            ),
        )
        result = graph.GraphSkill().run(CTX, document_id=10)
        assert result["type"] == "relation_graph"
        assert result["nodes"][0]["id"] == "张三"
        assert "张三" in result["mermaid"]
        assert "graph TD" in result["mermaid"]

    def test_strips_json_fence(self, monkeypatch):
        monkeypatch.setattr(graph, "fetch_document_text", lambda uid, did: "x")
        monkeypatch.setattr(
            graph, "chat_completion",
            lambda msgs, temperature=0.2: _FakeMsg('```json\n{"nodes":[],"edges":[]}\n```'),
        )
        result = graph.GraphSkill().run(CTX, document_id=10)
        assert result["type"] == "relation_graph"
        assert result["nodes"] == []

    def test_empty_document_returns_error(self, monkeypatch):
        monkeypatch.setattr(graph, "fetch_document_text", lambda uid, did: "")
        result = graph.GraphSkill().run(CTX, document_id=10)
        assert "error" in result

    def test_invalid_json_returns_error(self, monkeypatch):
        monkeypatch.setattr(graph, "fetch_document_text", lambda uid, did: "x")
        monkeypatch.setattr(
            graph, "chat_completion", lambda msgs, temperature=0.2: _FakeMsg("不是JSON")
        )
        result = graph.GraphSkill().run(CTX, document_id=10)
        assert "error" in result


# ── mindmap ──────────────────────────────────────────────────
class TestMindmapSkill:
    def test_returns_mermaid_content(self, monkeypatch):
        monkeypatch.setattr(mindmap, "fetch_document_text", lambda uid, did: "文档全文")
        monkeypatch.setattr(
            mindmap, "chat_completion",
            lambda msgs, temperature=0.2: _FakeMsg("mindmap\n  root((主题))"),
        )
        result = mindmap.MindmapSkill().run(CTX, document_id=10)
        assert result["type"] == "mindmap"
        assert result["format"] == "mermaid"
        assert "root((主题))" in result["content"]

    def test_strips_fence(self, monkeypatch):
        monkeypatch.setattr(mindmap, "fetch_document_text", lambda uid, did: "x")
        monkeypatch.setattr(
            mindmap, "chat_completion",
            lambda msgs, temperature=0.2: _FakeMsg("```\nmindmap\n  root((x))\n```"),
        )
        result = mindmap.MindmapSkill().run(CTX, document_id=10)
        assert not result["content"].startswith("```")

    def test_empty_document_returns_error(self, monkeypatch):
        monkeypatch.setattr(mindmap, "fetch_document_text", lambda uid, did: "")
        result = mindmap.MindmapSkill().run(CTX, document_id=10)
        assert "error" in result


# ── report（多步：检索→大纲→逐节）─────────────────────────────
class TestReportSkill:
    def test_generates_structured_report(self, monkeypatch):
        monkeypatch.setattr(report, "embed_query", lambda t: [0.1])
        monkeypatch.setattr(report, "search", lambda *a, **k: [
            {"content": "资料一", "document_id": 1, "chunk_index": 0, "score": 0.9},
        ])
        # 第一次调用返回大纲，后续返回各节正文
        calls = iter([
            _FakeMsg("小节一\n小节二"),   # outline
            _FakeMsg("第一节正文"),        # section 1
            _FakeMsg("第二节正文"),        # section 2
        ])
        monkeypatch.setattr(report, "chat_completion", lambda msgs, temperature=0.4: next(calls))
        result = report.ReportSkill().run(CTX, topic="AI 架构")
        assert result["type"] == "report"
        assert result["outline"] == ["小节一", "小节二"]
        assert "# AI 架构" in result["content"]
        assert "## 小节一" in result["content"]
        assert "第一节正文" in result["content"]

    def test_no_hits_returns_error(self, monkeypatch):
        monkeypatch.setattr(report, "embed_query", lambda t: [0.1])
        monkeypatch.setattr(report, "search", lambda *a, **k: [])
        result = report.ReportSkill().run(CTX, topic="不存在的主题")
        assert "error" in result

    def test_outline_capped_at_5_sections(self, monkeypatch):
        monkeypatch.setattr(report, "embed_query", lambda t: [0.1])
        monkeypatch.setattr(report, "search", lambda *a, **k: [
            {"content": "资料", "document_id": 1, "chunk_index": 0, "score": 0.9}
        ])
        # 大纲给 7 节，但应只取前 5
        outline = _FakeMsg("\n".join(f"节{i}" for i in range(7)))
        section = _FakeMsg("正文")
        seq = iter([outline] + [section] * 7)
        monkeypatch.setattr(report, "chat_completion", lambda msgs, temperature=0.4: next(seq))
        result = report.ReportSkill().run(CTX, topic="x")
        assert len(result["outline"]) == 5


# ── kb_search ────────────────────────────────────────────────
class TestKbSearchSkill:
    def test_returns_mapped_results(self, monkeypatch):
        monkeypatch.setattr(kb_search, "embed_query", lambda q: [0.1])
        monkeypatch.setattr(kb_search, "search", lambda *a, **k: [
            {"content": "命中片段", "document_id": 2, "chunk_index": 3, "score": 0.876},
        ])
        result = kb_search.KnowledgeBaseSearchSkill().run(CTX, query="问题")
        assert result["type"] == "kb_search"
        assert result["results"][0]["document_id"] == 2
        assert result["results"][0]["score"] == 0.876

    def test_uses_context_document_when_arg_absent(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(kb_search, "embed_query", lambda q: [0.1])

        def _fake_search(vec, user_id, top_k, document_id):
            captured["document_id"] = document_id
            return []

        monkeypatch.setattr(kb_search, "search", _fake_search)
        kb_search.KnowledgeBaseSearchSkill().run(CTX, query="q")
        # 未传 document_id 参数 → 回退到 context.document_id(=10)
        assert captured["document_id"] == 10


# ── web_search ───────────────────────────────────────────────
class TestWebSearchSkill:
    def test_maps_ddgs_results(self, monkeypatch):
        import ddgs

        class _FakeDDGS:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def text(self, query, max_results):
                return [{"title": "标题", "body": "摘要", "href": "http://x.com"}]

        monkeypatch.setattr(ddgs, "DDGS", _FakeDDGS)
        result = web_search.WebSearchSkill().run(CTX, query="最新新闻")
        assert result["type"] == "web_search"
        assert result["results"][0]["title"] == "标题"
        assert result["results"][0]["url"] == "http://x.com"

    def test_search_failure_returns_error(self, monkeypatch):
        import ddgs

        class _BoomDDGS:
            def __enter__(self): raise RuntimeError("网络不通")
            def __exit__(self, *a): return False

        monkeypatch.setattr(ddgs, "DDGS", _BoomDDGS)
        result = web_search.WebSearchSkill().run(CTX, query="q")
        assert "error" in result
        assert "网络不通" in result["error"]
