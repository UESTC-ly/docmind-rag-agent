"""Celery 实例。broker 和 result backend 都用 Redis。

启动 worker：celery -A app.celery_app worker --loglevel=info
"""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "docmind",
    broker=settings.redis_url,
    backend=settings.redis_url,
    # 显式导入任务模块。autodiscover 默认找 <包>.tasks，
    # 而我们的任务在 app.tasks.document_tasks，名字对不上会漏注册。
    include=["app.tasks.document_tasks", "app.tasks.evaluation_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,  # 记录 STARTED 状态，便于查进度
    # At-least-once delivery is paired with the EvalRun database lease/CAS.
    # A worker crash must put the message back on Redis instead of leaving a
    # permanent RUNNING row, and one worker should reserve only one long task.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_transport_options={"visibility_timeout": 3600},
    result_expires=3600,  # 结果保留 1 小时
)
