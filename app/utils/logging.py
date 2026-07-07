"""结构化日志配置（基于 loguru）。

- log_json=True：输出 JSON 行，便于 ELK/Loki 等采集与检索（生产）。
- log_json=False：彩色文本，便于本地开发肉眼看。

用法：
  from app.utils.logging import setup_logging, logger
  setup_logging()               # 应用启动时调一次
  logger.bind(request_id=rid).info("msg", extra_field=...)
"""

import sys

from loguru import logger

from app.config import settings


def setup_logging() -> None:
    """按配置初始化 loguru。移除默认 handler，装上我们要的格式。幂等。"""
    logger.remove()

    if settings.log_json:
        # serialize=True 让 loguru 输出 JSON（含 time/level/message/extra 等）
        logger.add(
            sys.stdout,
            level=settings.log_level,
            serialize=True,
            backtrace=False,
            diagnose=False,  # 生产关掉变量值回显，避免泄露敏感数据
        )
    else:
        logger.add(
            sys.stdout,
            level=settings.log_level,
            colorize=True,
            format=(
                "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
                "<cyan>{extra}</cyan> | {message}"
            ),
        )


__all__ = ["logger", "setup_logging"]
