"""请求日志中间件：为每个请求生成 request_id，记录方法/路径/状态/耗时。

request_id 也写回响应头 X-Request-ID，便于前后端串联排查。
"""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.utils.logging import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        # 绑定 request_id 到本请求的日志上下文
        req_logger = logger.bind(request_id=request_id)
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            req_logger.exception(
                "request failed",
                method=request.method,
                path=request.url.path,
                elapsed_ms=elapsed_ms,
            )
            raise

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        req_logger.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=elapsed_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response
