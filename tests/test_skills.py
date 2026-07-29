"""Skills 的单元测试。

各技能的外部边界都 mock 在其模块命名空间里：
- graph / mindmap：fetch_document_text + chat_completion
- report：embed_query + search + chat_completion（多次）
- weekly_report：fetch_material_text + chat_completion
- presentation：fetch_material_text + chat_completion + 标准库 PPTX 生成
- kb_search：embed_query + search
- web_search：ddgs.DDGS
- _helpers：SyncSessionLocal（用同步内存库）
"""

import base64
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

from app.skills import graph, kb_search, mindmap, presentation, report, weekly_report, web_search
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

    def test_mermaid_uses_safe_internal_node_ids(self):
        code = graph._to_mermaid(
            [{"id": '实体 A "核心"', "type": "概念"}],
            [
                {
                    "source": '实体 A "核心"',
                    "target": "实体/B",
                    "relation": "包含|依赖",
                }
            ],
        )
        assert 'n0["实体 A &quot;核心&quot;"]' in code
        assert 'n1["实体/B"]' in code
        assert 'n0 -->|"包含&#124;依赖"| n1' in code

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

    def test_uses_context_document_when_arg_absent(self, monkeypatch):
        captured = {}

        def _fake_fetch_document_text(uid, did):
            captured["document_id"] = did
            return "文档全文"

        monkeypatch.setattr(
            graph,
            "fetch_document_text",
            _fake_fetch_document_text,
        )
        monkeypatch.setattr(
            graph,
            "chat_completion",
            lambda msgs, temperature=0.2: _FakeMsg('{"nodes":[],"edges":[]}'),
        )
        result = graph.GraphSkill().run(CTX)
        assert result["type"] == "relation_graph"
        assert captured["document_id"] == 10

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
        assert result["artifact_kind"] == "file"
        assert result["download"]["filename"].endswith(".mmd")

    def test_strips_fence(self, monkeypatch):
        monkeypatch.setattr(mindmap, "fetch_document_text", lambda uid, did: "x")
        monkeypatch.setattr(
            mindmap, "chat_completion",
            lambda msgs, temperature=0.2: _FakeMsg("```\nmindmap\n  root((x))\n```"),
        )
        result = mindmap.MindmapSkill().run(CTX, document_id=10)
        assert not result["content"].startswith("```")

    def test_strips_mermaid_language_fence(self, monkeypatch):
        monkeypatch.setattr(mindmap, "fetch_document_text", lambda uid, did: "x")
        monkeypatch.setattr(
            mindmap,
            "chat_completion",
            lambda msgs, temperature=0.2: _FakeMsg(
                "```mermaid\nmindmap\n  root((x))\n```"
            ),
        )
        result = mindmap.MindmapSkill().run(CTX, document_id=10)
        assert result["content"].startswith("mindmap")
        assert not result["content"].startswith("mermaid\n")

    def test_empty_document_returns_error(self, monkeypatch):
        monkeypatch.setattr(mindmap, "fetch_document_text", lambda uid, did: "")
        result = mindmap.MindmapSkill().run(CTX, document_id=10)
        assert "error" in result

    def test_uses_context_document_when_arg_absent(self, monkeypatch):
        captured = {}

        def _fake_fetch_document_text(uid, did):
            captured["document_id"] = did
            return "文档全文"

        monkeypatch.setattr(
            mindmap,
            "fetch_document_text",
            _fake_fetch_document_text,
        )
        monkeypatch.setattr(
            mindmap,
            "chat_completion",
            lambda msgs, temperature=0.2: _FakeMsg("mindmap\n  root((主题))"),
        )
        result = mindmap.MindmapSkill().run(CTX)
        assert result["type"] == "mindmap"
        assert captured["document_id"] == 10


# ── report（多步：检索→大纲→逐节）─────────────────────────────
class TestReportSkill:
    def test_generates_structured_report(self, monkeypatch):
        monkeypatch.setattr(report, "retrieve_for_skill", lambda *a, **k: [
            {"content": "资料一", "document_id": 1, "chunk_index": 0, "score": 0.9},
        ])
        # 第一次调用返回大纲，后续返回各节正文
        calls = iter([
            _FakeMsg("小节一\n小节二"),   # outline
            _FakeMsg("第一节正文。[D1:C0]"),  # section 1
            _FakeMsg("第二节正文。[D1:C0]"),  # section 2
        ])
        monkeypatch.setattr(report, "chat_completion", lambda msgs, temperature=0.4: next(calls))
        result = report.ReportSkill().run(CTX, topic="AI 架构")
        assert result["type"] == "report"
        assert result["outline"] == ["小节一", "小节二"]
        assert "# AI 架构" in result["content"]
        assert "## 小节一" in result["content"]
        assert "第一节正文" in result["content"]
        assert result["artifact_kind"] == "file"
        assert result["download"]["filename"].endswith(".md")
        assert result["grounding"]["mode"] == "hybrid_rag"
        assert result["verification"]["passed"] is True

    def test_limits_search_to_context_document_when_available(self, monkeypatch):
        captured = {}
        def _fake_search(user_id, query, top_k, document_id=None):
            captured["document_id"] = document_id
            return [
                {
                    "content": "资料",
                    "document_id": document_id,
                    "chunk_index": 0,
                    "score": 0.9,
                }
            ]

        monkeypatch.setattr(report, "retrieve_for_skill", _fake_search)
        seq = iter([_FakeMsg("小节一"), _FakeMsg("正文")])
        monkeypatch.setattr(report, "chat_completion", lambda msgs, temperature=0.4: next(seq))

        result = report.ReportSkill().run(CTX, topic="当前文档报告")

        assert result["type"] == "report"
        assert captured["document_id"] == 10

    def test_no_hits_returns_error(self, monkeypatch):
        monkeypatch.setattr(report, "retrieve_for_skill", lambda *a, **k: [])
        result = report.ReportSkill().run(CTX, topic="不存在的主题")
        assert "error" in result

    def test_outline_capped_at_5_sections(self, monkeypatch):
        monkeypatch.setattr(report, "retrieve_for_skill", lambda *a, **k: [
            {"content": "资料", "document_id": 1, "chunk_index": 0, "score": 0.9}
        ])
        # 大纲给 7 节，但应只取前 5
        outline = _FakeMsg("\n".join(f"节{i}" for i in range(7)))
        section = _FakeMsg("正文")
        seq = iter([outline] + [section] * 7)
        monkeypatch.setattr(report, "chat_completion", lambda msgs, temperature=0.4: next(seq))
        result = report.ReportSkill().run(CTX, topic="x")
        assert len(result["outline"]) == 5


# ── weekly_report（Markdown 文件产出）──────────────────────────
class TestWeeklyReportSkill:
    def test_generates_downloadable_markdown(self, monkeypatch):
        monkeypatch.setattr(
            weekly_report,
            "fetch_retrieved_material",
            lambda user_id, query, document_id=None, max_chars=12000: (
                "材料内容",
                [10],
                [{"document_id": 10, "chunk_index": 1}],
            ),
        )
        monkeypatch.setattr(
            weekly_report,
            "chat_completion",
            lambda msgs, temperature=0.3: _FakeMsg(
                "# 周报\n\n## 本周完成\n完成 A。[D10:C1]"
            ),
        )

        result = weekly_report.WeeklyReportSkill().run(
            CTX, topic="DocMind", week="2026-W28"
        )

        assert result["type"] == "weekly_report"
        assert result["artifact_kind"] == "file"
        assert result["download"]["filename"].endswith(".md")
        assert result["download"]["encoding"] == "text"
        assert "完成 A" in result["download"]["content"]
        assert result["grounding"]["mode"] == "hybrid_rag"
        assert result["verification"]["passed"] is True

    def test_no_material_returns_error(self, monkeypatch):
        monkeypatch.setattr(
            weekly_report,
            "fetch_retrieved_material",
            lambda user_id, query, document_id=None, max_chars=12000: ("", [], []),
        )
        result = weekly_report.WeeklyReportSkill().run(CTX, topic="x")
        assert "error" in result


# ── presentation（PPTX 文件产出）──────────────────────────────
class TestPresentationSkill:
    def test_generates_downloadable_pptx(self, monkeypatch):
        monkeypatch.setattr(
            presentation,
            "fetch_retrieved_material",
            lambda user_id, query, document_id=None, max_chars=14000: (
                "材料内容",
                [10],
                [{"document_id": 10, "chunk_index": 1}],
            ),
        )
        monkeypatch.setattr(
            presentation,
            "chat_completion",
            lambda msgs, temperature=0.35: _FakeMsg(
                '{"slides":[{"title":"封面","bullets":["副标题"]},'
                '{"title":"进展","bullets":["完成 A。[D10:C1]","完成 B。[D10:C1]"]}]}'
            ),
        )

        result = presentation.PresentationSkill().run(
            CTX, topic="DocMind 汇报", slide_count=4
        )

        assert result["type"] == "presentation"
        assert result["artifact_kind"] == "file"
        assert result["download"]["filename"].endswith(".pptx")
        assert result["download"]["encoding"] == "base64"
        assert result["grounding"]["mode"] == "hybrid_rag"
        assert result["verification"]["passed"] is True

        pptx = base64.b64decode(result["download"]["content"])
        with ZipFile(BytesIO(pptx)) as zf:
            names = set(zf.namelist())
        assert "[Content_Types].xml" in names
        assert "ppt/presentation.xml" in names
        assert "ppt/slides/slide1.xml" in names

    def test_invalid_json_falls_back_to_single_slide(self, monkeypatch):
        monkeypatch.setattr(
            presentation,
            "fetch_retrieved_material",
            lambda user_id, query, document_id=None, max_chars=14000: (
                "材料内容",
                [10],
                [{"document_id": 10, "chunk_index": 1}],
            ),
        )
        monkeypatch.setattr(
            presentation,
            "chat_completion",
            lambda msgs, temperature=0.35: _FakeMsg("不是 JSON"),
        )
        result = presentation.PresentationSkill().run(CTX, topic="兜底")
        assert result["slides"][0]["title"] == "兜底"

    def test_no_material_returns_error(self, monkeypatch):
        monkeypatch.setattr(
            presentation,
            "fetch_retrieved_material",
            lambda user_id, query, document_id=None, max_chars=14000: ("", [], []),
        )
        result = presentation.PresentationSkill().run(CTX, topic="x")
        assert "error" in result


# ── kb_search ────────────────────────────────────────────────
class TestKbSearchSkill:
    def test_returns_mapped_results(self, monkeypatch):
        monkeypatch.setattr(
            kb_search,
            "run_pipeline_for_skill",
            lambda *a, **k: SimpleNamespace(
                fingerprint="test-fingerprint",
                hits=[
                    {
                        "content": "命中片段",
                        "document_id": 2,
                        "chunk_index": 3,
                        "citation_id": "D2:C3",
                        "score": 0.876,
                    }
                ],
            ),
        )
        result = kb_search.KnowledgeBaseSearchSkill().run(CTX, query="问题")
        assert result["type"] == "kb_search"
        assert result["results"][0]["document_id"] == 2
        assert result["results"][0]["score"] == 0.876
        assert result["grounding"]["mode"] == "quality_adaptive_rag"
        assert result["grounding"]["sources"][0]["chunk_index"] == 3
        assert result["workflow"]["status"] == "completed"

    def test_uses_context_document_when_arg_absent(self, monkeypatch):
        captured = {}
        def _fake_search(**kwargs):
            captured["document_id"] = kwargs["document_id"]
            return SimpleNamespace(fingerprint="test", hits=[])

        monkeypatch.setattr(kb_search, "run_pipeline_for_skill", _fake_search)
        monkeypatch.setattr(kb_search.settings, "quality_max_interventions", 0)
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
