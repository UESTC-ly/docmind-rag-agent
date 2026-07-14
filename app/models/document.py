import enum
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    literal_column,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class DocumentStatus(str, enum.Enum):
    """文档解析状态，对应 Celery 任务生命周期。"""

    PENDING = "pending"  # 已上传，等待解析
    PROCESSING = "processing"  # 解析中
    COMPLETED = "completed"  # 解析完成，已向量化
    FAILED = "failed"  # 解析失败


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(512))
    file_path: Mapped[str] = mapped_column(String(1024))
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus), default=DocumentStatus.PENDING, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """文档分块。文本存 PG，向量存 Qdrant，两边用 (document_id, chunk_index) 关联。"""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)  # 块在文档内的序号
    content: Mapped[str] = mapped_column(Text)  # 块的原始文本
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Mirror the PostgreSQL-only migration index in ORM metadata so
    # ``alembic check`` does not mistake the production index for drift.  The
    # DDL predicate keeps SQLite desktop/test schemas free of PG-only SQL.
    __table_args__ = (
        Index(
            "ix_document_chunks_content_fts",
            func.to_tsvector(
                literal_column("'simple'::regconfig"),
                content,
            ),
            postgresql_using="gin",
        ).ddl_if(dialect="postgresql"),
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")
