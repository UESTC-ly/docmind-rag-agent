"""Celery entrypoint for asynchronous, idempotent evaluation runs."""

from app.celery_app import celery_app
from app.config import settings
from app.database import SyncSessionLocal
from app.services.evaluation.runner import run_evaluation


def execute_evaluation_task(
    run_id: int,
    user_id: int,
    document_id: int,
    task_id: str | None = None,
    retryable: bool = False,
    reclaim_running: bool = False,
) -> dict:
    """Execute one run with its own sync session (also reused by local mode)."""
    with SyncSessionLocal() as db:
        return run_evaluation(
            db,
            run_id,
            user_id,
            document_id,
            raise_on_error=True,
            retryable=retryable,
            task_id=task_id,
            reclaim_running=reclaim_running,
        )


@celery_app.task(
    bind=True,
    name="run_evaluation",
    max_retries=2,
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=settings.evaluation_task_soft_time_limit_seconds,
    time_limit=settings.evaluation_task_time_limit_seconds,
)
def run_evaluation_task(
    self,
    run_id: int,
    user_id: int,
    document_id: int,
) -> dict:
    """Run evaluation off-request; retry transient failures with bounded backoff."""
    retries = int(self.request.retries or 0)
    final_attempt = retries >= int(self.max_retries or 0)
    delivery_info = self.request.delivery_info or {}
    task_id = str(self.request.id or f"celery-run-{run_id}")
    try:
        return execute_evaluation_task(
            run_id,
            user_id,
            document_id,
            task_id=task_id,
            retryable=not final_attempt,
            reclaim_running=bool(delivery_info.get("redelivered")),
        )
    except Exception as exc:  # noqa: BLE001 - Celery task boundary
        if final_attempt:
            raise
        countdown = min(2 ** (self.request.retries + 1), 30)
        raise self.retry(exc=exc, countdown=countdown)
