"""评估模块路由。

端点：
  POST /eval/datasets        — 从文档生成评估数据集（同步，LLM 出题）
  GET  /eval/datasets        — 列出当前用户的数据集
  POST /eval/runs            — 触发一次评估运行（同步执行，结果完成后返回）
  GET  /eval/runs/{run_id}   — 查看某次运行的聚合结果
  GET  /eval/runs/{run_id}/details — 查看逐条样本明细
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.evaluation import EvalDataset, EvalResult, EvalRun, RunStatus
from app.schemas.evaluation import (
    DatasetGenerateRequest,
    DatasetResponse,
    ResultDetailResponse,
    RunCreateRequest,
    RunResponse,
)
from app.utils.deps import get_current_user, get_db

router = APIRouter(prefix="/eval", tags=["evaluation"])


# ── 数据集端点 ───────────────────────────────────────────────────────────────

@router.post("/datasets", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def generate_dataset(
    req: DatasetGenerateRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """从文档自动生成评估数据集（LLM 反向出题）。"""
    from app.database import SyncSessionLocal
    from app.services.evaluation.dataset_gen import generate_dataset as gen

    # 确认文档属于当前用户
    result = await db.execute(
        select(Document).where(
            Document.id == req.document_id,
            Document.user_id == current_user.id,
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")

    # LLM 出题是同步阻塞调用，用 to_thread 避免阻塞事件循环
    def _sync():
        with SyncSessionLocal() as sync_db:
            dataset = gen(
                sync_db,
                user_id=current_user.id,
                document_id=req.document_id,
                dataset_name=req.name,
                sample_count=req.sample_count,
            )
            return {
                "id": dataset.id,
                "name": dataset.name,
                "document_id": dataset.document_id,
                "created_at": dataset.created_at,
                "sample_count": len(dataset.samples),
            }

    data = await asyncio.to_thread(_sync)
    return DatasetResponse(**data)


@router.get("/datasets", response_model=list[DatasetResponse])
async def list_datasets(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """列出当前用户的所有评估数据集。"""
    from sqlalchemy import func as sqlfunc
    from app.models.evaluation import EvalSample

    result = await db.execute(
        select(EvalDataset).where(EvalDataset.user_id == current_user.id)
    )
    datasets = result.scalars().all()

    out = []
    for ds in datasets:
        count_result = await db.execute(
            select(sqlfunc.count(EvalSample.id)).where(EvalSample.dataset_id == ds.id)
        )
        count = count_result.scalar_one()
        out.append(DatasetResponse(
            id=ds.id,
            name=ds.name,
            document_id=ds.document_id,
            created_at=ds.created_at,
            sample_count=count,
        ))
    return out


# ── 评估运行端点 ──────────────────────────────────────────────────────────────

@router.post("/runs", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
async def create_run(
    req: RunCreateRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """触发一次评估运行（同步执行，等待全部样本跑完后返回结果）。"""
    from app.database import SyncSessionLocal
    from app.services.evaluation.runner import run_evaluation

    # 确认数据集属于当前用户
    result = await db.execute(
        select(EvalDataset).where(
            EvalDataset.id == req.dataset_id,
            EvalDataset.user_id == current_user.id,
        )
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="数据集不存在")

    document_id = dataset.document_id

    # 先在异步层建 run 记录，拿到 run_id
    from app.models.evaluation import EvalRun
    run = EvalRun(dataset_id=req.dataset_id)
    db.add(run)
    await db.flush()
    await db.refresh(run)
    run_id = run.id
    await db.commit()

    # 在线程里执行同步评估（会阻塞，适合小数据集；大数据集改用 Celery task）
    def _sync():
        with SyncSessionLocal() as sync_db:
            run_evaluation(sync_db, run_id, current_user.id, document_id)

    await asyncio.to_thread(_sync)

    # 评估在独立的 sync session 里提交；本 async session 配了
    # expire_on_commit=False，身份映射里缓存的 run 仍是旧的 pending 快照。
    # 必须 expire 掉，重新 select 才会从库读到 completed 的最新状态。
    db.expire_all()
    result = await db.execute(select(EvalRun).where(EvalRun.id == run_id))
    updated_run = result.scalar_one()
    return updated_run


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: int,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """查看某次评估运行的聚合结果。"""
    from app.models.evaluation import EvalRun

    result = await db.execute(
        select(EvalRun)
        .join(EvalDataset, EvalRun.dataset_id == EvalDataset.id)
        .where(EvalRun.id == run_id, EvalDataset.user_id == current_user.id)
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="运行记录不存在")
    return run


@router.get("/runs/{run_id}/details", response_model=list[ResultDetailResponse])
async def get_run_details(
    run_id: int,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """查看某次运行的逐条样本明细分数。"""
    from app.models.evaluation import EvalRun

    # 先确认运行存在且属于当前用户
    result = await db.execute(
        select(EvalRun)
        .join(EvalDataset, EvalRun.dataset_id == EvalDataset.id)
        .where(EvalRun.id == run_id, EvalDataset.user_id == current_user.id)
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="运行记录不存在")

    # 查询所有明细结果
    results = await db.execute(
        select(EvalResult).where(EvalResult.run_id == run_id)
    )
    return results.scalars().all()
