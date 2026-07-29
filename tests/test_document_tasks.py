"""Celery 文档解析任务测试（process_document）。

用同步内存库播种文档，把 SyncSessionLocal 指向测试库，
mock 文本提取/向量化/Qdrant，验证：成功流转 COMPLETED、
空文本→FAILED、缺失文档、异常→FAILED 并记录原因。
"""

import pytest

from app.models.document import Document, DocumentStatus
from app.models.user import User
from app.tasks import document_tasks
from app.utils.file_parser import (
    LocatedChunk,
    ParsedDocument,
    ParsedParagraph,
)


@pytest.fixture
def bind_sync_db(sync_db, monkeypatch):
    """让任务内的 SyncSessionLocal() 返回同一个测试 session（不真正关闭）。"""
    class _Factory:
        def __call__(self):
            return sync_db

    monkeypatch.setattr(document_tasks, "SyncSessionLocal", _Factory())
    # 任务里会 db.close()，用 no-op 避免测试后续读不到
    monkeypatch.setattr(sync_db, "close", lambda: None)
    return sync_db


def _seed_doc(sync_db, path="f.txt"):
    user = User(email="t@t.com", hashed_password="x")
    sync_db.add(user)
    sync_db.flush()
    doc = Document(user_id=user.id, filename="f", file_path=path,
                   status=DocumentStatus.PENDING)
    sync_db.add(doc)
    sync_db.commit()
    return doc.id


def _parsed_document(text="一些文本内容"):
    return ParsedDocument(
        text=text,
        paragraphs=(
            ParsedParagraph(
                content=text,
                paragraph_index=1,
                char_start=0,
                char_end=len(text),
            ),
        ),
    )


def _located_chunk(content, *, page=2, paragraph=1, start=0):
    return LocatedChunk(
        content=content,
        page_start=page,
        page_end=page,
        paragraph_start=paragraph,
        paragraph_end=paragraph,
        char_start=start,
        char_end=start + len(content),
    )


class TestProcessDocument:
    def test_success_marks_completed(self, bind_sync_db, monkeypatch):
        doc_id = _seed_doc(bind_sync_db)
        monkeypatch.setattr(
            document_tasks,
            "extract_text_with_locations",
            lambda _path: _parsed_document("块1\n块2"),
        )
        monkeypatch.setattr(
            document_tasks,
            "split_text_with_locations",
            lambda _document, **_kwargs: [
                _located_chunk("块1", page=2, paragraph=3, start=0),
                _located_chunk("块2", page=3, paragraph=4, start=3),
            ],
        )
        monkeypatch.setattr(document_tasks, "embed_texts", lambda chunks: [[0.1], [0.2]])
        monkeypatch.setattr(document_tasks, "upsert_chunks", lambda **k: None)

        result = document_tasks.process_document(doc_id)
        assert result == {"status": "ok", "chunks": 2}
        doc = bind_sync_db.get(Document, doc_id)
        assert doc.status == DocumentStatus.COMPLETED
        assert doc.chunk_count == 2
        assert [(chunk.page_start, chunk.paragraph_start, chunk.char_end) for chunk in doc.chunks] == [
            (2, 3, 2),
            (3, 4, 5),
        ]

    def test_missing_document(self, bind_sync_db):
        result = document_tasks.process_document(99999)
        assert result["status"] == "error"
        assert "not found" in result["reason"]

    def test_empty_text_marks_failed(self, bind_sync_db, monkeypatch):
        doc_id = _seed_doc(bind_sync_db)
        monkeypatch.setattr(
            document_tasks,
            "extract_text_with_locations",
            lambda _path: ParsedDocument(text="   ", paragraphs=()),
        )
        result = document_tasks.process_document(doc_id)
        assert result["status"] == "error"
        doc = bind_sync_db.get(Document, doc_id)
        assert doc.status == DocumentStatus.FAILED
        assert doc.error_message

    def test_embedding_failure_marks_failed(self, bind_sync_db, monkeypatch):
        doc_id = _seed_doc(bind_sync_db)
        monkeypatch.setattr(
            document_tasks,
            "extract_text_with_locations",
            lambda _path: _parsed_document("文本"),
        )
        monkeypatch.setattr(
            document_tasks,
            "split_text_with_locations",
            lambda _document, **_kwargs: [_located_chunk("块")],
        )
        monkeypatch.setattr(
            document_tasks, "embed_texts",
            lambda chunks: (_ for _ in ()).throw(RuntimeError("embed 欠费")),
        )
        result = document_tasks.process_document(doc_id)
        assert result["status"] == "error"
        assert "embed 欠费" in result["reason"]
        doc = bind_sync_db.get(Document, doc_id)
        assert doc.status == DocumentStatus.FAILED
