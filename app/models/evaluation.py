"""评估模块数据模型。

表结构：
  EvalDataset  ── 一个评估数据集（绑定某篇文档，含若干样本）
  EvalSample   ── 数据集里的一条样本：问题 + 标准答案 + 相关 chunk 索引
  EvalRun      ── 一次完整的评估运行，记录聚合分数
  EvalResult   ── 某次运行里单条样本的检索/生成明细分数
"""

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvalDataset(Base):
    __tablename__ = "eval_datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(Text)  # 数据集名称，如 "技术手册v1-eval"
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    samples: Mapped[list["EvalSample"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    runs: Mapped[list["EvalRun"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class EvalSample(Base):
    """一条评估样本（三元组）。

    relevant_chunk_ids 存 JSON 字符串，如 "[0, 3, 7]"，
    表示该问题的正确答案来自文档的第 0、3、7 块。
    """

    __tablename__ = "eval_samples"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("eval_datasets.id", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text)
    ground_truth_answer: Mapped[str] = mapped_column(Text)
    relevant_chunk_ids: Mapped[str] = mapped_column(Text)  # JSON: [int, ...]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    dataset: Mapped["EvalDataset"] = relationship(back_populates="samples")
    results: Mapped[list["EvalResult"]] = relationship(
        back_populates="sample", cascade="all, delete-orphan"
    )


class EvalRun(Base):
    """一次评估运行。聚合分数在全部样本跑完后写入。"""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("eval_datasets.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus), default=RunStatus.PENDING, index=True
    )
    # 聚合检索指标（完成后填入）
    hit_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    mrr: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 聚合生成指标
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    dataset: Mapped["EvalDataset"] = relationship(back_populates="runs")
    results: Mapped[list["EvalResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvalResult(Base):
    """单条样本在某次运行中的明细分数。"""

    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True
    )
    sample_id: Mapped[int] = mapped_column(
        ForeignKey("eval_samples.id", ondelete="CASCADE"), index=True
    )
    # 检索明细（该样本的）
    hit: Mapped[int] = mapped_column(Integer, default=0)    # 0 or 1
    reciprocal_rank: Mapped[float] = mapped_column(Float, default=0.0)
    recall_at_k: Mapped[float] = mapped_column(Float, default=0.0)
    precision_at_k: Mapped[float] = mapped_column(Float, default=0.0)
    # 生成明细
    faithfulness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 检索器实际返回的 chunk_index 列表（JSON），便于调试
    retrieved_chunk_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    # LLM 生成的答案（评估时实际跑一遍 RAG）
    generated_answer: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped["EvalRun"] = relationship(back_populates="results")
    sample: Mapped["EvalSample"] = relationship(back_populates="results")
