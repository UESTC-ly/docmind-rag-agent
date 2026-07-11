"""Skills 共用数据访问 helper 测试（_helpers）。

把 SyncSessionLocal 指向同步内存库并播种，验证：
拼全文（按 chunk_index 排序）、归属校验、缺失文档返回空、列用户文档。
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.user import User
from app.skills import _helpers


@pytest.fixture
def bind_db(monkeypatch):
    """内存库 + 把 _helpers.SyncSessionLocal 换成它。返回 sessionmaker 供播种。"""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(_helpers, "SyncSessionLocal", Session)
    return Session


def _seed(Session, user_id=1, chunks=("块0", "块1", "块2")):
    with Session() as s:
        user = User(id=user_id, email=f"u{user_id}@t.com", hashed_password="x")
        s.add(user)
        doc = Document(user_id=user_id, filename="d.txt", file_path="p",
                       status=DocumentStatus.COMPLETED, chunk_count=len(chunks))
        s.add(doc)
        s.flush()
        # 乱序插入，验证按 chunk_index 排序拼接
        for idx in reversed(range(len(chunks))):
            s.add(DocumentChunk(document_id=doc.id, chunk_index=idx, content=chunks[idx]))
        s.commit()
        return doc.id


class TestFetchDocumentText:
    def test_joins_chunks_in_order(self, bind_db):
        doc_id = _seed(bind_db, user_id=1, chunks=("第一块", "第二块", "第三块"))
        text = _helpers.fetch_document_text(1, doc_id)
        assert text == "第一块\n第二块\n第三块"

    def test_missing_document_returns_empty(self, bind_db):
        assert _helpers.fetch_document_text(1, 99999) == ""

    def test_other_users_document_returns_empty(self, bind_db):
        # 文档属于 user 1，user 2 无权访问
        doc_id = _seed(bind_db, user_id=1)
        assert _helpers.fetch_document_text(2, doc_id) == ""


class TestFetchUserDocuments:
    def test_lists_user_documents(self, bind_db):
        _seed(bind_db, user_id=1)
        docs = _helpers.fetch_user_documents(1)
        assert len(docs) == 1
        assert docs[0]["filename"] == "d.txt"
        assert "id" in docs[0]

    def test_empty_when_no_documents(self, bind_db):
        with bind_db() as s:
            s.add(User(id=5, email="empty@t.com", hashed_password="x"))
            s.commit()
        assert _helpers.fetch_user_documents(5) == []

    def test_excludes_documents_that_are_not_ready(self, bind_db):
        with bind_db() as s:
            user = User(id=6, email="pending@t.com", hashed_password="x")
            s.add(user)
            s.add(
                Document(
                    user_id=6,
                    filename="pending.txt",
                    file_path="p",
                    status=DocumentStatus.PENDING,
                    chunk_count=0,
                )
            )
            s.commit()
        assert _helpers.fetch_user_documents(6) == []
