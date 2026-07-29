"""多路召回测试：RRF 融合（纯函数）+ 关键词检索（async DB）+ 融合合并。"""

import pytest

from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.user import User
from app.services.retrieval import (
    build_keyword_statement,
    fuse_dense_and_keyword,
    keyword_search,
    reciprocal_rank_fusion,
)
from app.services import rag_service, skill_retrieval


class TestRRF:
    def test_item_in_both_lists_ranks_higher(self):
        # A 在两个列表都靠前 → RRF 分最高
        fused = reciprocal_rank_fusion(
            [[("d", 1), ("d", 2), ("d", 3)], [("d", 1), ("d", 4)]], top_k=3
        )
        assert fused[0] == ("d", 1)

    def test_dedupes_across_lists(self):
        fused = reciprocal_rank_fusion([[("d", 1)], [("d", 1)]], top_k=5)
        assert fused == [("d", 1)]

    def test_respects_top_k(self):
        lists = [[("d", i) for i in range(10)]]
        assert len(reciprocal_rank_fusion(lists, top_k=3)) == 3

    def test_empty_input(self):
        assert reciprocal_rank_fusion([], top_k=5) == []

    def test_k_constant_affects_score_gap(self):
        # 较小 k 放大排名差异：rank1 vs rank2 差距更大
        one = reciprocal_rank_fusion([[("d", 1), ("d", 2)]], k=1, top_k=2)
        assert one == [("d", 1), ("d", 2)]


class TestFuse:
    def test_cross_document_same_index_not_merged(self):
        # 关键修复：doc1/chunk0 与 doc2/chunk0 是不同块，不能合并
        dense = [{"document_id": 1, "chunk_index": 0, "content": "A", "score": 0.9}]
        keyword = [
            {"document_id": 2, "chunk_index": 0, "content": "B", "keyword_score": 3}
        ]
        fused = fuse_dense_and_keyword(dense, keyword, top_k=5)
        assert len(fused) == 2
        keys = {(h["document_id"], h["chunk_index"]) for h in fused}
        assert keys == {(1, 0), (2, 0)}

    def test_dense_preserved_over_keyword(self):
        # 同一 key 同时出现，保留稠密（带 score）
        dense = [{"document_id": 1, "chunk_index": 0, "content": "A", "score": 0.9}]
        keyword = [
            {"document_id": 1, "chunk_index": 0, "content": "A", "keyword_score": 2}
        ]
        fused = fuse_dense_and_keyword(dense, keyword, top_k=5)
        assert len(fused) == 1
        assert "score" in fused[0]

    def test_keyword_only_hit_still_has_response_score(self):
        keyword = [
            {
                "document_id": 1,
                "chunk_index": 3,
                "content": "精确型号",
                "keyword_score": 2,
                "score": 0.0,
            }
        ]
        assert fuse_dense_and_keyword([], keyword, top_k=5)[0]["score"] == 0.0


class TestSkillRetrieval:
    def test_hybrid_mode_reuses_dense_keyword_and_rrf(self, monkeypatch):
        dense = [
            {"document_id": 1, "chunk_index": 0, "content": "语义", "score": 0.9}
        ]
        keyword = [
            {
                "document_id": 1,
                "chunk_index": 1,
                "content": "关键词",
                "keyword_score": 2,
                "score": 0.0,
            }
        ]
        captured = {}
        monkeypatch.setattr(skill_retrieval.settings, "retrieval_mode", "hybrid")
        monkeypatch.setattr(skill_retrieval, "embed_query", lambda query: [0.1])
        monkeypatch.setattr(
            skill_retrieval,
            "apply_document_policy_sync",
            lambda hits, **kwargs: hits,
        )
        monkeypatch.setattr(
            skill_retrieval,
            "search",
            lambda vector, user_id, top_k, document_id: dense,
        )

        def _keyword(query, user_id, document_id, limit, db=None):
            captured.update(
                query=query,
                user_id=user_id,
                document_id=document_id,
                limit=limit,
            )
            return keyword

        monkeypatch.setattr(skill_retrieval, "keyword_search_sync", _keyword)

        hits = skill_retrieval.retrieve_for_skill(
            user_id=7, query="型号", document_id=9, top_k=5
        )

        assert {(h["document_id"], h["chunk_index"]) for h in hits} == {
            (1, 0),
            (1, 1),
        }
        assert captured["document_id"] == 9

    def test_dense_mode_skips_keyword_path(self, monkeypatch):
        monkeypatch.setattr(skill_retrieval.settings, "retrieval_mode", "dense")
        monkeypatch.setattr(skill_retrieval, "embed_query", lambda query: [0.1])
        monkeypatch.setattr(
            skill_retrieval,
            "apply_document_policy_sync",
            lambda hits, **kwargs: hits,
        )
        monkeypatch.setattr(
            skill_retrieval,
            "search",
            lambda vector, user_id, top_k, document_id: [
                {
                    "document_id": 1,
                    "chunk_index": 0,
                    "content": "dense",
                    "score": 0.8,
                }
            ],
        )
        monkeypatch.setattr(
            skill_retrieval,
            "keyword_search_sync",
            lambda *args: (_ for _ in ()).throw(AssertionError("不应调用关键词检索")),
        )
        hits = skill_retrieval.retrieve_for_skill(1, "q")
        assert [hit["content"] for hit in hits] == ["dense"]
        assert hits[0]["retrieval_sources"] == ["dense"]
        assert hits[0]["reranker"] == "local"


async def _seed(db, contents):
    user = User(email="r@t.com", hashed_password="x")
    db.add(user)
    await db.flush()
    doc = Document(user_id=user.id, filename="d", file_path="p",
                   status=DocumentStatus.COMPLETED, chunk_count=len(contents))
    db.add(doc)
    await db.flush()
    for i, c in enumerate(contents):
        db.add(DocumentChunk(document_id=doc.id, chunk_index=i, content=c))
    await db.flush()
    return user.id, doc.id


@pytest.mark.asyncio
class TestKeywordSearch:
    async def test_ranks_by_hit_count(self, db_session):
        uid, did = await _seed(
            db_session,
            ["PostgreSQL 存储元数据", "Qdrant 存储向量", "无关内容"],
        )
        hits = await keyword_search(db_session, "存储 向量", uid, did, limit=10)
        # "Qdrant 存储向量" 命中"存储"和"向量"两词，排第一
        assert hits[0]["content"] == "Qdrant 存储向量"
        assert hits[0]["keyword_score"] >= hits[-1]["keyword_score"]

    async def test_no_match_returns_empty(self, db_session):
        uid, did = await _seed(db_session, ["苹果 香蕉"])
        assert await keyword_search(db_session, "xyz", uid, did, limit=10) == []

    async def test_empty_query_returns_empty(self, db_session):
        uid, did = await _seed(db_session, ["内容"])
        assert await keyword_search(db_session, "", uid, did, limit=10) == []

    async def test_respects_user_isolation(self, db_session):
        uid, did = await _seed(db_session, ["机密内容"])
        # 另一个用户搜不到
        hits = await keyword_search(db_session, "机密", uid + 999, None, limit=10)
        assert hits == []

    async def test_database_applies_candidate_limit(self, db_session):
        uid, did = await _seed(db_session, [f"共同词 内容{i}" for i in range(200)])
        hits = await keyword_search(db_session, "共同", uid, did, limit=7)
        assert len(hits) == 7


class TestKeywordSql:
    def test_postgresql_uses_fts_rank_and_limit(self):
        from sqlalchemy.dialects import postgresql

        stmt = build_keyword_statement("型号 ABC", 7, 9, 12, "postgresql")
        compiled = stmt.compile(dialect=postgresql.dialect())
        sql = str(compiled)
        assert "to_tsvector" in sql
        assert "'simple'::regconfig, document_chunks.content" in sql
        assert "coalesce" not in sql.lower()
        assert "websearch_to_tsquery" in sql
        assert "ts_rank_cd" in sql
        assert "documents.user_id =" in sql
        assert "document_chunks.document_id =" in sql
        assert "LIMIT" in sql
        assert 7 in compiled.params.values()
        assert 9 in compiled.params.values()
        assert 12 in compiled.params.values()

    def test_sqlite_uses_sql_score_not_python_scan(self):
        from sqlalchemy.dialects import sqlite

        stmt = build_keyword_statement("存储 向量", 3, None, 5, "sqlite")
        sql = str(
            stmt.compile(
                dialect=sqlite.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        )
        assert "CASE WHEN" in sql
        assert "ORDER BY keyword_score DESC" in sql
        assert "LIMIT 5" in sql


@pytest.mark.asyncio
async def test_online_and_sync_paths_produce_same_final_order(db_session, monkeypatch):
    dense = [
        {"document_id": 1, "chunk_index": 0, "content": "一般内容", "score": 0.9},
        {"document_id": 1, "chunk_index": 1, "content": "精确型号 X1", "score": 0.7},
    ]
    keyword = [
        {
            "document_id": 1,
            "chunk_index": 1,
            "content": "精确型号 X1",
            "keyword_score": 2,
            "score": 0.0,
        }
    ]
    monkeypatch.setattr(skill_retrieval.settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(skill_retrieval.settings, "reranker_mode", "local")
    monkeypatch.setattr(skill_retrieval, "embed_query", lambda query: [0.1])
    monkeypatch.setattr(
        skill_retrieval,
        "apply_document_policy_sync",
        lambda hits, **kwargs: hits,
    )
    monkeypatch.setattr(skill_retrieval, "search", lambda *args: dense)
    monkeypatch.setattr(
        skill_retrieval,
        "keyword_search_sync",
        lambda *args, **kwargs: keyword,
    )
    monkeypatch.setattr(rag_service, "embed_query", lambda query: [0.1])
    async def _policy(hits, **kwargs):
        return hits

    monkeypatch.setattr(rag_service, "apply_document_policy_async", _policy)
    monkeypatch.setattr(rag_service, "search", lambda *args: dense)

    async def _keyword(*args, **kwargs):
        return keyword

    monkeypatch.setattr(rag_service, "keyword_search", _keyword)

    sync_hits = skill_retrieval.retrieve_for_skill(7, "精确型号", top_k=2)
    async_hits = await rag_service.retrieve(db_session, 7, "精确型号", None)
    assert [hit["chunk_index"] for hit in async_hits] == [
        hit["chunk_index"] for hit in sync_hits
    ]
    assert [hit["rerank_score"] for hit in async_hits] == [
        hit["rerank_score"] for hit in sync_hits
    ]
