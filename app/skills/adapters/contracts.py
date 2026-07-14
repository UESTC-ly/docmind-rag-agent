"""Stable observation contract shared by external Skill adapters."""

from __future__ import annotations

import json
from typing import Any


def success(
    summary: str,
    *,
    data: dict[str, Any] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    next_actions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "status": "success",
        "summary": summary,
        "next_actions": next_actions or [],
        "artifacts": artifacts or [],
        **(data or {}),
    }


def failure(
    summary: str,
    *,
    root_cause: str,
    retry: str,
    stop_condition: str,
    disabled: bool = False,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "status": "error",
        "summary": summary,
        "error": root_cause,
        "root_cause": root_cause,
        "next_actions": [retry] if retry else [],
        "stop_condition": stop_condition,
        "artifacts": [],
        "disabled": disabled,
        **(data or {}),
    }


def _serialized(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError):
        return repr(value)


def _bounded_artifacts(value: Any) -> list[dict[str, Any]]:
    bounded: list[dict[str, Any]] = []
    for artifact in value if isinstance(value, list) else []:
        if not isinstance(artifact, dict):
            continue
        metadata: dict[str, Any] = {}
        for key in ("path", "mime_type", "name", "type", "url"):
            if key in artifact:
                metadata[key] = str(artifact[key])[:1000]
        if metadata:
            bounded.append(metadata)
        if len(bounded) >= 16:
            break
    return bounded


def bound_observation(result: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Bound untrusted tool output before it is appended to the LLM context."""
    limit = max(512, int(max_chars))
    raw = _serialized(result)
    if len(raw) <= limit:
        return result

    bounded: dict[str, Any] = {
        "ok": bool(result.get("ok")),
        "status": str(result.get("status") or "error")[:64],
        "summary": str(result.get("summary") or "Tool observation truncated")[:1000],
        "next_actions": [
            str(action)[:500]
            for action in (result.get("next_actions") or [])[:8]
        ],
        "artifacts": _bounded_artifacts(result.get("artifacts")),
        "truncated": True,
    }
    if not bounded["ok"]:
        error = str(result.get("root_cause") or result.get("error") or "")[:1000]
        bounded.update(
            error=error,
            root_cause=error,
            stop_condition=str(result.get("stop_condition") or "")[:500],
        )

    contract_keys = {
        "ok",
        "status",
        "summary",
        "next_actions",
        "artifacts",
        "error",
        "root_cause",
        "stop_condition",
    }
    data = {key: value for key, value in result.items() if key not in contract_keys}
    preview = _serialized(data)
    bounded["data_preview"] = preview

    encoded = _serialized(bounded)
    if len(encoded) > limit:
        excess = len(encoded) - limit
        bounded["data_preview"] = preview[: max(0, len(preview) - excess - 8)]
        encoded = _serialized(bounded)
    if len(encoded) > limit:
        # A hostile adapter may also inflate contract fields.  Fall back to a
        # small, still-valid observation rather than exceeding the hard cap.
        bounded = {
            "ok": bool(result.get("ok")),
            "status": "success" if result.get("ok") else "error",
            "summary": "Tool observation exceeded the configured size limit",
            "next_actions": [],
            "artifacts": [],
            "truncated": True,
        }
    return bounded


def normalize_observation(
    result: dict[str, Any],
    *,
    operation: str,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Keep legacy ``ok`` adapters compatible while enforcing rich observations."""

    normalized = dict(result)
    ok = bool(normalized.get("ok"))
    normalized.setdefault("status", "success" if ok else "error")
    normalized.setdefault(
        "summary",
        f"{operation} completed" if ok else f"{operation} failed",
    )
    normalized.setdefault("next_actions", [])
    normalized.setdefault("artifacts", [])
    if not ok:
        error = str(normalized.get("error") or normalized["summary"])
        normalized.setdefault("error", error)
        normalized.setdefault("root_cause", error)
        normalized.setdefault("stop_condition", "停止重复调用，直到配置或输入发生变化。")
    if max_chars is not None:
        return bound_observation(normalized, max_chars)
    return normalized


__all__ = ["bound_observation", "failure", "normalize_observation", "success"]
