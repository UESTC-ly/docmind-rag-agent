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
    include=["app.tasks.document_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,  # 记录 STARTED 状态，便于查进度
    result_expires=3600,  # 结果保留 1 小时
)
