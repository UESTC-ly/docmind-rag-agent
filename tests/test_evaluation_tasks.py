"""Background evaluation task dispatch, success, and retry tests."""

import threading

import pytest
from celery.exceptions import Retry

from app import database
from app.models.document import Document, DocumentStatus
from app.models.evaluation import EvalDataset, EvalRun, RunStatus
from app.models.user import User
from app.services import task_dispatcher
from app.tasks import evaluation_tasks


class _SessionContext:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, *_args):
        return False


def test_execute_task_opens_session_and_runs(monkeypatch):
    session = object()
    captured = {}
    monkeypatch.setattr(
        evaluation_tasks,
        "SyncSessionLocal",
        lambda: _SessionContext(session),
    )

    def _run(
        db,
        run_id,
        user_id,
        document_id,
        *,
        raise_on_error,
        retryable,
        task_id,
        reclaim_running,
    ):
        captured.update(
            db=db,
            run_id=run_id,
            user_id=user_id,
            document_id=document_id,
            raise_on_error=raise_on_error,
            retryable=retryable,
            task_id=task_id,
            reclaim_running=reclaim_running,
        )
        return {"status": "completed"}

    monkeypatch.setattr(evaluation_tasks, "run_evaluation", _run)
    assert evaluation_tasks.execute_evaluation_task(1, 2, 3) == {"status": "completed"}
    assert captured == {
        "db": session,
        "run_id": 1,
        "user_id": 2,
        "document_id": 3,
        "raise_on_error": True,
        "retryable": False,
        "task_id": None,
        "reclaim_running": False,
    }


def test_celery_task_retries_runner_failure(monkeypatch):
    captured_call = {}

    def _fail(*args, **kwargs):
        captured_call.update(args=args, kwargs=kwargs)
        raise RuntimeError("temporary database error")

    monkeypatch.setattr(
        evaluation_tasks,
        "execute_evaluation_task",
        _fail,
    )
    captured = {}

    def _retry(*, exc, countdown):
        captured.update(exc=exc, countdown=countdown)
        raise Retry(exc=exc)

    monkeypatch.setattr(evaluation_tasks.run_evaluation_task, "retry", _retry)
    with pytest.raises(Retry) as exc_info:
        evaluation_tasks.run_evaluation_task.run(1, 2, 3)
    assert "temporary database error" in str(exc_info.value)
    assert isinstance(captured["exc"], RuntimeError)
    assert captured["countdown"] == 2
    assert captured_call["kwargs"]["retryable"] is True


def test_celery_final_attempt_marks_terminal_without_another_retry(monkeypatch):
    monkeypatch.setattr(
        evaluation_tasks,
        "execute_evaluation_task",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("still broken")),
    )
    monkeypatch.setattr(
        evaluation_tasks.run_evaluation_task,
        "retry",
        lambda **kwargs: pytest.fail("final attempt must not schedule another retry"),
    )
    evaluation_tasks.run_evaluation_task.push_request(
        id="celery-final",
        retries=2,
        delivery_info={},
    )
    try:
        with pytest.raises(RuntimeError, match="still broken"):
            evaluation_tasks.run_evaluation_task.run(1, 2, 3)
    finally:
        evaluation_tasks.run_evaluation_task.pop_request()


def test_celery_dispatch_returns_receipt(monkeypatch):
    class _Result:
        id = "celery-123"

    monkeypatch.setattr(task_dispatcher.settings, "task_execution_mode", "celery")
    monkeypatch.setattr(
        evaluation_tasks.run_evaluation_task,
        "delay",
        lambda *args: _Result(),
    )
    receipt = task_dispatcher.dispatch_evaluation(1, 2, 3)
    assert receipt.task_id == "celery-123"
    assert receipt.mode == "celery"


def test_desktop_local_dispatch_runs_without_celery(monkeypatch):
    completed = threading.Event()
    captured = []

    def _execute(*args, **kwargs):
        captured.append((args, kwargs))
        completed.set()
        return {"status": "completed"}

    monkeypatch.setattr(task_dispatcher.settings, "task_execution_mode", "local")
    monkeypatch.setattr(evaluation_tasks, "execute_evaluation_task", _execute)
    try:
        receipt = task_dispatcher.dispatch_evaluation(4, 5, 6)
        assert receipt.mode == "local"
        assert receipt.task_id.startswith("local-")
        assert completed.wait(timeout=2)
        assert len(captured) == 1
        assert captured[0][0] == (4, 5, 6)
        assert captured[0][1]["task_id"] == receipt.task_id
    finally:
        task_dispatcher.shutdown_local_executor()
    assert task_dispatcher._executor is None
    assert task_dispatcher._local_futures == {}


def test_desktop_startup_requeues_pending_and_running_runs(sync_db, monkeypatch):
    user = User(email="recovery@example.com", hashed_password="x")
    sync_db.add(user)
    sync_db.flush()
    document = Document(
        user_id=user.id,
        filename="recovery.txt",
        file_path="recovery.txt",
        status=DocumentStatus.COMPLETED,
    )
    sync_db.add(document)
    sync_db.flush()
    dataset = EvalDataset(
        user_id=user.id,
        document_id=document.id,
        name="recovery",
    )
    sync_db.add(dataset)
    sync_db.flush()
    pending = EvalRun(dataset_id=dataset.id)
    sync_db.add(pending)
    sync_db.flush()
    running = EvalRun(
        dataset_id=dataset.id,
        status=RunStatus.RUNNING,
        task_id="old-local-task",
        lease_token="old-token",
    )
    sync_db.add(running)
    sync_db.commit()
    pending_id = pending.id
    dispatched = []

    monkeypatch.setattr(task_dispatcher.settings, "task_execution_mode", "local")
    monkeypatch.setattr(database, "SyncSessionLocal", lambda: _SessionContext(sync_db))
    monkeypatch.setattr(
        task_dispatcher,
        "dispatch_evaluation",
        lambda *args: dispatched.append(args),
    )

    assert task_dispatcher.recover_local_evaluations() == 2
    assert dispatched == [
        (pending_id, user.id, document.id),
        (running.id, user.id, document.id),
    ]
    sync_db.expire_all()
    recovered = sync_db.get(EvalRun, running.id)
    assert recovered.status == RunStatus.PENDING
    assert recovered.lease_token is None
