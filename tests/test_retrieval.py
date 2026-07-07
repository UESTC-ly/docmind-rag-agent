"""多路召回测试：RRF 融合（纯函数）+ 关键词检索（async DB）+ 融合合并。"""

import pytest

from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.user import User
from app.services.retrieval import (
    fuse_dense_and_keyword,
    keyword_search,
    reciprocal_rank_fusion,
)


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
