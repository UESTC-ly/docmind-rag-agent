"""评估模块路由。

端点：
  POST /eval/datasets        — 从文档生成评估数据集（同步，LLM 出题）
  GET  /eval/datasets        — 列出当前用户的数据集
  POST /eval/runs            — 创建 pending run 并派发后台任务（202）
  GET  /eval/runs/{run_id}   — 查看某次运行的聚合结果
  GET  /eval/runs/{run_id}/details — 查看逐条样本明细
"""

import asyncio
import json
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.evaluation import (
    EvalDataset,
    EvalMetricResult,
    EvalRegressionGate,
    EvalRegressionResult,
    EvalResult,
    EvalRun,
    EvalSample,
    RunStatus,
)
from app.schemas.evaluation import (
    BadcaseDiffResponse,
    BadcaseRerunRequest,
    BadcaseResponse,
    DatasetGenerateRequest,
    DatasetResponse,
    ExperimentCreateRequest,
    MetricResultResponse,
    PipelineResponse,
    PipelineValidateRequest,
    PipelineValidationResponse,
    RegressionGateCreateRequest,
    RegressionGateResponse,
    RegressionResultResponse,
    ResultDetailResponse,
    RunCreateRequest,
    RunResponse,
)
from app.services.rag_pipeline import (
    PipelineSpec,
    ensure_index_compatible,
    pipeline_presets,
    resolve_pipeline_spec,
    validate_pipeline_components,
)
from app.services.evaluation.badcases import (
    classify_badcase,
    compare_badcase_sets,
    diagnose_badcases,
    source_links,
)
from app.utils.deps import get_current_user, get_db

router = APIRouter(prefix="/eval", tags=["evaluation"])


def _pipeline_values(pipeline_id: str) -> tuple[str, str, str]:
    try:
        spec = resolve_pipeline_spec(pipeline_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    payload = json.dumps(
        spec.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return spec.id, payload, spec.fingerprint


async def _owned_dataset(
    db: AsyncSession,
    dataset_id: int,
    user_id: int,
) -> EvalDataset:
    result = await db.execute(
        select(EvalDataset).where(
            EvalDataset.id == dataset_id,
            EvalDataset.user_id == user_id,
        )
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="数据集不存在",
        )
    return dataset


async def _owned_run(
    db: AsyncSession,
    run_id: int,
    user_id: int,
) -> EvalRun:
    result = await db.execute(
        select(EvalRun)
        .join(EvalDataset, EvalRun.dataset_id == EvalDataset.id)
        .where(EvalRun.id == run_id, EvalDataset.user_id == user_id)
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="运行记录不存在",
        )
    return run


def _new_pipeline_run(
    dataset_id: int,
    pipeline_id: str,
    *,
    experiment_key: str | None = None,
    run_label: str | None = None,
    comparison_role: str = "standalone",
    baseline_run_id: int | None = None,
) -> EvalRun:
    resolved_id, pipeline_spec, fingerprint = _pipeline_values(pipeline_id)
    return EvalRun(
        dataset_id=dataset_id,
        pipeline_id=resolved_id,
        pipeline_spec=pipeline_spec,
        pipeline_fingerprint=fingerprint,
        experiment_key=experiment_key,
        run_label=run_label,
        comparison_role=comparison_role,
        baseline_run_id=baseline_run_id,
        evaluation_scope="full",
    )


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
        out.append(
            DatasetResponse.model_validate(ds).model_copy(
                update={"sample_count": count}
            )
        )
    return out


# ── 评估运行端点 ──────────────────────────────────────────────────────────────


@router.get("/pipelines", response_model=list[PipelineResponse])
async def list_pipelines(current_user=Depends(get_current_user)):
    """List immutable RAG experiment presets and their exact fingerprints."""
    return [
        PipelineResponse(
            id=spec.id,
            label=spec.label,
            description=spec.description,
            retriever=spec.retriever,
            fusion=spec.fusion,
            reranker=spec.reranker,
            context_builder=spec.context_builder,
            top_k=spec.top_k,
            fingerprint=spec.fingerprint,
            spec=spec.to_dict(),
        )
        for spec in pipeline_presets().values()
    ]


@router.post(
    "/pipelines/validate",
    response_model=PipelineValidationResponse,
)
async def validate_pipeline(
    req: PipelineValidateRequest,
    current_user=Depends(get_current_user),
):
    """Validate schema, registered components, and optional runtime compatibility."""

    del current_user  # authentication is the authority boundary for this endpoint
    try:
        spec = PipelineSpec.from_dict(req.spec)
        validate_pipeline_components(spec)
        if req.check_runtime:
            ensure_index_compatible(spec)
    except (TypeError, ValueError) as exc:
        return PipelineValidationResponse(valid=False, errors=[str(exc)])
    return PipelineValidationResponse(
        valid=True,
        fingerprint=spec.fingerprint,
        spec=spec.to_dict(),
    )


@router.post(
    "/runs",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_run(
    req: RunCreateRequest,
    response: Response,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """创建评估运行并立即返回；实际执行由后台任务完成。"""
    from app.services.task_dispatcher import dispatch_evaluation

    dataset = await _owned_dataset(db, req.dataset_id, current_user.id)

    document_id = dataset.document_id
    if req.baseline_run_id is not None:
        baseline = await _owned_run(db, req.baseline_run_id, current_user.id)
        if baseline.dataset_id != dataset.id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="baseline_run_id 必须属于同一数据集",
            )
        if baseline.status != RunStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="基线运行尚未完成",
            )

    # 先在异步层建 run 记录，拿到 run_id
    run = _new_pipeline_run(
        req.dataset_id,
        req.pipeline_id,
        run_label=req.run_label,
        comparison_role=(
            "candidate" if req.baseline_run_id is not None else "standalone"
        ),
        baseline_run_id=req.baseline_run_id,
    )
    db.add(run)
    await db.flush()
    await db.refresh(run)
    run_id = run.id
    await db.commit()

    try:
        receipt = dispatch_evaluation(run_id, current_user.id, document_id)
    except Exception as exc:  # broker/local executor unavailable
        run.status = RunStatus.FAILED
        run.error_message = f"后台任务派发失败: {str(exc)[:1000]}"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="评估后台任务暂不可用",
        ) from exc

    response.headers["X-Task-ID"] = receipt.task_id
    response.headers["X-Task-Mode"] = receipt.mode
    return run


@router.post(
    "/experiments",
    response_model=list[RunResponse],
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_experiment(
    req: ExperimentCreateRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Dispatch paired runs over one dataset for direct pipeline comparison."""
    from app.services.task_dispatcher import dispatch_evaluation

    dataset = await _owned_dataset(db, req.dataset_id, current_user.id)
    pipeline_ids = list(dict.fromkeys(req.pipeline_ids))
    if len(pipeline_ids) != len(req.pipeline_ids):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="pipeline_ids 不能重复",
        )
    baseline_pipeline_id = req.baseline_pipeline_id or pipeline_ids[0]
    if baseline_pipeline_id not in pipeline_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="baseline_pipeline_id 必须包含在 pipeline_ids 中",
        )

    experiment_key = uuid4().hex
    runs = [
        _new_pipeline_run(
            req.dataset_id,
            pipeline_id,
            experiment_key=experiment_key,
            run_label=(
                f"{req.name}:{pipeline_id}" if req.name else pipeline_id
            ),
            comparison_role=(
                "baseline"
                if pipeline_id == baseline_pipeline_id
                else "candidate"
            ),
        )
        for pipeline_id in pipeline_ids
    ]
    db.add_all(runs)
    await db.flush()
    baseline_run = next(
        run for run in runs if run.pipeline_id == baseline_pipeline_id
    )
    for run in runs:
        if run is not baseline_run:
            run.baseline_run_id = baseline_run.id
    for run in runs:
        await db.refresh(run)
    await db.commit()

    dispatch_order = [baseline_run] + [
        run for run in runs if run is not baseline_run
    ]
    for run in dispatch_order:
        try:
            dispatch_evaluation(run.id, current_user.id, dataset.document_id)
        except Exception as exc:  # keep other paired runs observable
            run.status = RunStatus.FAILED
            run.error_message = f"后台任务派发失败: {str(exc)[:1000]}"
    await db.commit()
    return runs


@router.post(
    "/runs/{run_id}/rerun",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def rerun_badcases(
    run_id: int,
    req: BadcaseRerunRequest,
    response: Response,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rerun selected public cases without treating them as release evidence."""

    from app.services.task_dispatcher import dispatch_evaluation

    source = await _owned_run(db, run_id, current_user.id)
    if source.status != RunStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="只能从已完成运行中选择 Badcase 重跑",
        )
    sample_ids = list(dict.fromkeys(req.sample_ids))
    if len(sample_ids) != len(req.sample_ids):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="sample_ids 不能重复",
        )
    available = set(
        (
            await db.execute(
                select(EvalResult.sample_id).where(
                    EvalResult.run_id == source.id,
                    EvalResult.sample_id.in_(sample_ids),
                )
            )
        ).scalars()
    )
    if available != set(sample_ids):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="只能重跑该运行中已有的样本",
        )
    dataset = await _owned_dataset(db, source.dataset_id, current_user.id)

    if req.pipeline_id is not None:
        rerun = _new_pipeline_run(
            dataset.id,
            req.pipeline_id,
            run_label=req.run_label,
            comparison_role="diagnostic",
        )
    elif (
        source.pipeline_id
        and source.pipeline_spec
        and source.pipeline_fingerprint
    ):
        rerun = EvalRun(
            dataset_id=dataset.id,
            pipeline_id=source.pipeline_id,
            pipeline_spec=source.pipeline_spec,
            pipeline_fingerprint=source.pipeline_fingerprint,
            run_label=req.run_label,
            comparison_role="diagnostic",
        )
    else:
        rerun = _new_pipeline_run(
            dataset.id,
            "configured",
            run_label=req.run_label,
            comparison_role="diagnostic",
        )
    rerun.evaluation_scope = "subset"
    rerun.sample_filter = json.dumps(
        sample_ids,
        separators=(",", ":"),
    )
    rerun.source_run_id = source.id
    db.add(rerun)
    await db.flush()
    await db.refresh(rerun)
    await db.commit()

    try:
        receipt = dispatch_evaluation(
            rerun.id,
            current_user.id,
            dataset.document_id,
        )
    except Exception as exc:
        rerun.status = RunStatus.FAILED
        rerun.error_message = f"后台任务派发失败: {str(exc)[:1000]}"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="评估后台任务暂不可用",
        ) from exc
    response.headers["X-Task-ID"] = receipt.task_id
    response.headers["X-Task-Mode"] = receipt.mode
    return rerun


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: int,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """查看某次评估运行的聚合结果。"""
    return await _owned_run(db, run_id, current_user.id)


@router.get("/runs/{run_id}/details", response_model=list[ResultDetailResponse])
async def get_run_details(
    run_id: int,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """查看某次运行的逐条样本明细分数。"""
    # 先确认运行存在且属于当前用户
    await _owned_run(db, run_id, current_user.id)

    # 查询所有明细结果
    results = await db.execute(
        select(EvalResult).where(EvalResult.run_id == run_id)
    )
    return results.scalars().all()


@router.get(
    "/runs/{run_id}/metrics",
    response_model=list[MetricResultResponse],
)
async def get_run_metrics(
    run_id: int,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return versioned aggregate and sample metrics for audit/regression."""

    await _owned_run(db, run_id, current_user.id)
    results = await db.execute(
        select(EvalMetricResult)
        .where(EvalMetricResult.run_id == run_id)
        .order_by(
            EvalMetricResult.subject_type,
            EvalMetricResult.subject_id,
            EvalMetricResult.metric_name,
        )
    )
    return results.scalars().all()


@router.post(
    "/regression-gates",
    response_model=RegressionGateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_regression_gate(
    req: RegressionGateCreateRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a deterministic release gate for one public evaluation set."""

    await _owned_dataset(db, req.dataset_id, current_user.id)
    gate = EvalRegressionGate(
        dataset_id=req.dataset_id,
        name=req.name,
        metric_name=req.metric_name,
        metric_version=req.metric_version,
        comparison=req.comparison,
        threshold=req.threshold,
        severity=req.severity,
        slice_filter=(
            json.dumps(
                req.slice_filter,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if req.slice_filter
            else None
        ),
        enabled=req.enabled,
    )
    db.add(gate)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="同一数据集中的门禁名称不能重复",
        ) from exc
    await db.refresh(gate)
    await db.commit()
    return gate


@router.get(
    "/regression-gates",
    response_model=list[RegressionGateResponse],
)
async def list_regression_gates(
    dataset_id: int,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List the active regression contract for one owned dataset."""

    await _owned_dataset(db, dataset_id, current_user.id)
    results = await db.execute(
        select(EvalRegressionGate)
        .where(EvalRegressionGate.dataset_id == dataset_id)
        .order_by(EvalRegressionGate.id)
    )
    return results.scalars().all()


@router.get(
    "/runs/{run_id}/regression",
    response_model=list[RegressionResultResponse],
)
async def get_run_regression(
    run_id: int,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return persisted release-gate verdicts for a candidate run."""

    await _owned_run(db, run_id, current_user.id)
    results = await db.execute(
        select(EvalRegressionResult)
        .where(EvalRegressionResult.candidate_run_id == run_id)
        .order_by(EvalRegressionResult.gate_id)
    )
    return results.scalars().all()


def _json_value(raw: str | None, default):
    if not raw:
        return default
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return default
    return value


async def _collect_badcases(
    db: AsyncSession,
    run: EvalRun,
    dataset: EvalDataset,
    *,
    include_passed: bool,
    faithfulness_threshold: float,
    relevance_threshold: float,
    ranking_threshold: float,
) -> list[BadcaseResponse]:
    result_rows = (
        await db.execute(
            select(EvalResult, EvalSample)
            .join(EvalSample, EvalResult.sample_id == EvalSample.id)
            .where(EvalResult.run_id == run.id)
            .order_by(EvalResult.id)
        )
    ).all()
    metric_rows = (
        await db.execute(
            select(EvalMetricResult).where(
                EvalMetricResult.run_id == run.id,
                EvalMetricResult.subject_type == "sample",
            )
        )
    ).scalars()
    metrics_by_sample: dict[int, dict[str, float | None]] = {}
    for metric in metric_rows:
        metrics_by_sample.setdefault(metric.subject_id, {})[
            metric.metric_name
        ] = metric.score

    output: list[BadcaseResponse] = []
    for result, sample in result_rows:
        metrics = {
            "hit_at_k": float(result.hit),
            "reciprocal_rank": result.reciprocal_rank,
            "recall_at_k": result.recall_at_k,
            "precision_at_k": result.precision_at_k,
            "average_precision_at_k": result.average_precision_at_k,
            "ndcg_at_k": result.ndcg_at_k,
            "faithfulness": result.faithfulness_score,
            "answer_relevance": result.answer_relevancy_score,
            "citation_precision": result.citation_precision,
            "citation_recall": result.citation_recall,
            "unsupported_claim_rate": result.unsupported_claim_rate,
            **metrics_by_sample.get(sample.id, {}),
        }
        trace = _json_value(result.retrieval_trace, [])
        if not isinstance(trace, list):
            trace = []
        report = _json_value(result.citation_report, None)
        if not isinstance(report, dict):
            report = None
        relevant_chunks = _json_value(sample.relevant_chunk_ids, [])
        if not isinstance(relevant_chunks, list):
            relevant_chunks = []
        document_qrels = _json_value(sample.document_qrels, {})
        if isinstance(document_qrels, dict):
            relevant_documents = []
            for document_id, score in document_qrels.items():
                try:
                    if float(score) > 0:
                        relevant_documents.append(int(document_id))
                except (TypeError, ValueError):
                    continue
        else:
            relevant_documents = []
        categories = classify_badcase(
            metrics,
            task_type=dataset.task_type,
            faithfulness_threshold=faithfulness_threshold,
            relevance_threshold=relevance_threshold,
            ranking_threshold=ranking_threshold,
            answerable=sample.answerable,
            citation_report=report,
            retrieval_trace=trace,
            relevant_chunk_ids=relevant_chunks,
            relevant_document_ids=relevant_documents,
        )
        if not categories and not include_passed:
            continue
        output.append(
            BadcaseResponse(
                result_id=result.id,
                sample_id=sample.id,
                external_id=sample.external_id,
                question=sample.question,
                ground_truth_answer=sample.ground_truth_answer,
                answerable=sample.answerable,
                difficulty=sample.difficulty,
                slice_tags=sample.slice_tags,
                categories=categories,
                diagnosis=diagnose_badcases(categories),
                metrics=metrics,
                relevant_chunk_ids=relevant_chunks,
                retrieved_chunk_ids=_json_value(
                    result.retrieved_chunk_ids,
                    [],
                ),
                generated_answer=result.generated_answer,
                citation_report=report,
                retrieval_trace=trace,
                source_links=source_links(trace),
                rerun_request={
                    "method": "POST",
                    "path": f"/eval/runs/{run.id}/rerun",
                    "body": {"sample_ids": [sample.id]},
                },
            )
        )
    return output


@router.get(
    "/runs/{run_id}/badcases",
    response_model=list[BadcaseResponse],
)
async def get_run_badcases(
    run_id: int,
    include_passed: bool = False,
    faithfulness_threshold: float = Query(default=0.8, ge=0.0, le=1.0),
    relevance_threshold: float = Query(default=0.8, ge=0.0, le=1.0),
    ranking_threshold: float = Query(default=0.5, ge=0.0, le=1.0),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return reproducible per-sample failure categories and source jumps."""

    run = await _owned_run(db, run_id, current_user.id)
    dataset = await _owned_dataset(db, run.dataset_id, current_user.id)
    return await _collect_badcases(
        db,
        run,
        dataset,
        include_passed=include_passed,
        faithfulness_threshold=faithfulness_threshold,
        relevance_threshold=relevance_threshold,
        ranking_threshold=ranking_threshold,
    )


@router.get(
    "/runs/{run_id}/badcase-diff",
    response_model=BadcaseDiffResponse,
)
async def get_badcase_diff(
    run_id: int,
    faithfulness_threshold: float = Query(default=0.8, ge=0.0, le=1.0),
    relevance_threshold: float = Query(default=0.8, ge=0.0, le=1.0),
    ranking_threshold: float = Query(default=0.5, ge=0.0, le=1.0),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Compare candidate failures with its named baseline sample by sample."""

    candidate = await _owned_run(db, run_id, current_user.id)
    if candidate.baseline_run_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="该运行没有命名基线，无法计算 Badcase 差异",
        )
    baseline = await _owned_run(
        db,
        candidate.baseline_run_id,
        current_user.id,
    )
    if baseline.dataset_id != candidate.dataset_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="候选与基线必须属于同一数据集",
        )
    dataset = await _owned_dataset(
        db,
        candidate.dataset_id,
        current_user.id,
    )
    common = {
        "include_passed": True,
        "faithfulness_threshold": faithfulness_threshold,
        "relevance_threshold": relevance_threshold,
        "ranking_threshold": ranking_threshold,
    }
    baseline_rows = await _collect_badcases(db, baseline, dataset, **common)
    candidate_rows = await _collect_badcases(db, candidate, dataset, **common)
    baseline_map = {
        row.sample_id: row.categories for row in baseline_rows
    }
    candidate_map = {
        row.sample_id: row.categories for row in candidate_rows
    }
    compared = compare_badcase_sets(baseline_map, candidate_map)
    context = {
        row.sample_id: {
            "external_id": row.external_id,
            "question": row.question,
        }
        for row in [*baseline_rows, *candidate_rows]
    }
    for key in ("newly_introduced", "fixed", "persistent"):
        for row in compared[key]:
            row.update(context.get(row["sample_id"], {}))
    return BadcaseDiffResponse(
        baseline_run_id=baseline.id,
        candidate_run_id=candidate.id,
        **compared,
    )
