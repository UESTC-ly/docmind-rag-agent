"""Dispatch background work to Celery or the self-contained desktop executor."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import func, select, update

from app.config import settings
from app.utils.logging import logger


@dataclass(frozen=True)
class DispatchReceipt:
    task_id: str
    mode: str


_executor: ThreadPoolExecutor | None = None
_executor_lock = Lock()
_local_futures: dict[str, Future] = {}


def _local_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=max(1, settings.local_task_workers),
                thread_name_prefix="docmind-task",
            )
        return _executor


def _submit_local(
    function,
    *args,
    task_id: str | None = None,
    function_kwargs: dict[str, Any] | None = None,
    on_error: Callable[[BaseException], None] | None = None,
) -> DispatchReceipt:
    task_id = task_id or f"local-{uuid4()}"
    future = _local_executor().submit(function, *args, **(function_kwargs or {}))
    with _executor_lock:
        _local_futures[task_id] = future

    def _forget(_future: Future) -> None:
        with _executor_lock:
            _local_futures.pop(task_id, None)
        if _future.cancelled():
            return
        error = _future.exception()
        if error is not None:
            logger.error(
                "local background task failed",
                extra={"task_id": task_id, "error": str(error)},
            )
            if on_error is not None:
                on_error(error)

    future.add_done_callback(_forget)
    return DispatchReceipt(task_id=task_id, mode="local")


def shutdown_local_executor() -> None:
    """Drain desktop background work and release executor threads at shutdown."""
    global _executor
    with _executor_lock:
        executor = _executor
        _executor = None
    if executor is not None:
        executor.shutdown(wait=True)
    with _executor_lock:
        _local_futures.clear()


def dispatch_evaluation(run_id: int, user_id: int, document_id: int) -> DispatchReceipt:
    from app.tasks.evaluation_tasks import (
        execute_evaluation_task,
        run_evaluation_task,
    )

    if settings.task_execution_mode.lower() == "local":
        task_id = f"local-{uuid4()}"
        return _submit_local(
            execute_evaluation_task,
            run_id,
            user_id,
            document_id,
            task_id=task_id,
            function_kwargs={"task_id": task_id},
            on_error=lambda exc: _mark_local_evaluation_failed(run_id, exc),
        )
    result = run_evaluation_task.delay(run_id, user_id, document_id)
    return DispatchReceipt(task_id=str(result.id), mode="celery")


def dispatch_document(document_id: int) -> DispatchReceipt:
    from app.tasks.document_tasks import process_document

    if settings.task_execution_mode.lower() == "local":
        return _submit_local(process_document.run, document_id)
    result = process_document.delay(document_id)
    return DispatchReceipt(task_id=str(result.id), mode="celery")


def _mark_local_evaluation_failed(run_id: int, error: BaseException) -> None:
    """Persist failures that occur outside the runner's normal error boundary."""

    from app.database import SyncSessionLocal
    from app.models.evaluation import EvalRun, RunStatus

    try:
        with SyncSessionLocal() as db:
            db.execute(
                update(EvalRun)
                .where(
                    EvalRun.id == run_id,
                    EvalRun.status.in_((RunStatus.PENDING, RunStatus.RUNNING)),
                )
                .values(
                    status=RunStatus.FAILED,
                    error_message=f"本地后台任务失败: {str(error)[:1950]}",
                    completed_at=func.now(),
                    task_id=None,
                    lease_token=None,
                    heartbeat_at=None,
                )
            )
            db.commit()
    except Exception as persist_error:  # noqa: BLE001 - last-resort observability
        logger.error(
            "failed to persist local evaluation error",
            extra={"run_id": run_id, "error": str(persist_error)},
        )


def recover_local_evaluations() -> int:
    """Requeue durable evaluation rows after a desktop process restart.

    Local futures live in memory and cannot survive process exit.  At startup
    there are no valid old local workers, so every RUNNING row is safely reset
    before all PENDING rows are dispatched again through the lease/CAS runner.
    """

    if settings.task_execution_mode.lower() != "local":
        return 0

    from app.database import SyncSessionLocal
    from app.models.evaluation import EvalDataset, EvalRun, RunStatus

    with SyncSessionLocal() as db:
        db.execute(
            update(EvalRun)
            .where(EvalRun.status == RunStatus.RUNNING)
            .values(
                status=RunStatus.PENDING,
                error_message="桌面进程已重启，评估任务正在恢复。",
                completed_at=None,
                task_id=None,
                lease_token=None,
                heartbeat_at=None,
            )
        )
        rows = db.execute(
            select(
                EvalRun.id,
                EvalDataset.user_id,
                EvalDataset.document_id,
            )
            .join(EvalDataset, EvalRun.dataset_id == EvalDataset.id)
            .where(EvalRun.status == RunStatus.PENDING)
            .order_by(EvalRun.id)
        ).all()
        db.commit()

    for run_id, user_id, document_id in rows:
        dispatch_evaluation(run_id, user_id, document_id)
    return len(rows)
