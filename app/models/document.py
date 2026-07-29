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
    # Optional provenance/version metadata. Unknown is allowed for ordinary
    # uploads, while known expired/superseded sources are excluded from RAG.
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    authority: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_fingerprint: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    effective_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    effective_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    source_status: Mapped[str] = mapped_column(
        String(32),
        default="unknown",
    )
    supersedes_document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """Document chunk plus stable extracted-text source coordinates.

    Coordinates are nullable because documents imported before the provenance
    migration keep their existing citation IDs and remain valid evidence.  New
    ingestion writes one-based page/paragraph ranges and zero-based,
    end-exclusive character spans in the normalized extracted text.
    """

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)  # 块在文档内的序号
    content: Mapped[str] = mapped_column(Text)  # 块的原始文本
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    paragraph_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    paragraph_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    locator_version: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )
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
