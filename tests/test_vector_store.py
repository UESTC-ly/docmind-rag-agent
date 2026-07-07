"""Qdrant 封装测试。

打桩 _client，验证 point id 生成、upsert 的 payload 结构、search 的过滤条件
（user_id 必带、document_id 可选）与返回映射、delete 过滤。
"""

from app.services import vector_store


class _Hit:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class _QueryResult:
    def __init__(self, points):
        self.points = points


class TestPointId:
    def test_id_no_collision_across_docs(self):
        # 不同文档的同一 chunk_index 不能撞
        assert vector_store._point_id(1, 5) != vector_store._point_id(2, 5)

    def test_id_formula(self):
        assert vector_store._point_id(3, 7) == 3 * vector_store._ID_STRIDE + 7


class TestUpsert:
    def test_builds_points_with_payload(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(vector_store, "ensure_collection", lambda: None)
        monkeypatch.setattr(
            vector_store._client, "upsert",
            lambda collection_name, points: captured.update(points=points),
        )
        vector_store.upsert_chunks(
            document_id=1, user_id=9,
            chunks=["c0", "c1"], vectors=[[0.1], [0.2]],
        )
        pts = captured["points"]
        assert len(pts) == 2
        assert pts[0].payload == {
            "document_id": 1, "user_id": 9, "chunk_index": 0, "content": "c0",
        }
        assert pts[1].payload["chunk_index"] == 1


class TestSearch:
    def test_filters_by_user_only_when_no_document(self, monkeypatch):
        captured = {}

        def _fake_query(collection_name, query, query_filter, limit):
            captured["filter"] = query_filter
            return _QueryResult([
                _Hit(0.9, {"content": "x", "document_id": 1, "chunk_index": 0})
            ])

        monkeypatch.setattr(vector_store._client, "query_points", _fake_query)
        results = vector_store.search([0.1], user_id=9, top_k=5)
        # 只有一个 must 条件（user_id）
        assert len(captured["filter"].must) == 1
        assert results[0] == {
            "score": 0.9, "content": "x", "document_id": 1, "chunk_index": 0,
        }

    def test_adds_document_filter_when_given(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            vector_store._client, "query_points",
            lambda **kw: captured.update(f=kw["query_filter"]) or _QueryResult([]),
        )
        vector_store.search([0.1], user_id=9, top_k=5, document_id=3)
        # user_id + document_id 两个 must 条件
        assert len(captured["f"].must) == 2


class TestDelete:
    def test_delete_by_document_filter(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            vector_store._client, "delete",
            lambda collection_name, points_selector: captured.update(sel=points_selector),
        )
        vector_store.delete_document(42)
        assert captured["sel"].must[0].match.value == 42
