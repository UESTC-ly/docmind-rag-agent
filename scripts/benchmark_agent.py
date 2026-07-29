#!/usr/bin/env python3
"""Run reproducible Agent acceptance scenarios through the public HTTP API."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import httpx

# Permit direct ``python scripts/benchmark_agent.py`` execution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.evaluation.agent_metrics import (
    aggregate_agent_metrics,
    score_agent_case,
)


_PUBLIC_PROVENANCE_FIELDS = (
    "source_name",
    "source_uri",
    "source_version",
    "license_name",
    "split",
    "corpus_fingerprint",
    "source_snapshot_fingerprint",
    "transform_spec",
)
DEFAULT_AGENT_REQUEST_TIMEOUT_SECONDS = 600.0


def _positive_timeout_seconds(value: str) -> float:
    """Parse a bounded benchmark request timeout from the CLI."""

    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "timeout_seconds must be a positive number"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "timeout_seconds must be greater than zero"
        )
    return parsed


class _ConcurrencyTracker:
    """Measure actual in-flight benchmark cases without inspecting providers."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = Lock()

    def start(self) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def finish(self) -> None:
        with self.lock:
            self.active = max(0, self.active - 1)


def _suite_fingerprint(
    provenance: dict[str, Any],
    cases: list[dict[str, Any]],
) -> str:
    payload = json.dumps(
        {"provenance": provenance, "cases": cases},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fingerprint_is_valid(value: Any) -> bool:
    fingerprint = str(value or "")
    return (
        len(fingerprint) == 64
        and all(
            character in "0123456789abcdefABCDEF"
            for character in fingerprint
        )
        and set(fingerprint) != {"0"}
    )


def _document_selector(
    payload: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    selector = payload.get("document_selector")
    if selector is None:
        selector = {
            key: provenance[key]
            for key in (
                "source_name",
                "split",
                "corpus_fingerprint",
                "source_snapshot_fingerprint",
            )
        }
    if not isinstance(selector, dict):
        raise ValueError("document_selector must be an object")
    required = (
        "source_name",
        "split",
        "corpus_fingerprint",
        "source_snapshot_fingerprint",
    )
    missing = [key for key in required if not selector.get(key)]
    if missing:
        raise ValueError(
            "document_selector missing: " + ", ".join(missing)
        )
    for field in ("corpus_fingerprint", "source_snapshot_fingerprint"):
        if selector[field] != provenance[field]:
            raise ValueError(
                f"document_selector {field} must match public provenance"
            )
    if selector["source_name"] != provenance["source_name"]:
        raise ValueError(
            "document_selector source_name must match public provenance"
        )
    if selector["split"] != provenance["split"]:
        raise ValueError("document_selector split must match public provenance")
    return dict(selector)


def load_suite(path: Path) -> dict[str, Any]:
    """Load a public, versioned Agent acceptance suite."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scenario suite must be a JSON object")
    if payload.get("contract") != "public_agent_scenarios_v2":
        raise ValueError(
            "scenario suite contract must be public_agent_scenarios_v2"
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("scenario suite requires public provenance")
    missing = [
        field for field in _PUBLIC_PROVENANCE_FIELDS
        if not provenance.get(field)
    ]
    if missing:
        raise ValueError(
            "scenario provenance missing: " + ", ".join(missing)
        )
    parsed_uri = urlparse(str(provenance["source_uri"]))
    if parsed_uri.scheme not in {"http", "https"} or not parsed_uri.netloc:
        raise ValueError("scenario source_uri must be a public HTTP(S) URL")
    for fingerprint_field in (
        "corpus_fingerprint",
        "source_snapshot_fingerprint",
    ):
        if not _fingerprint_is_valid(provenance[fingerprint_field]):
            raise ValueError(
                f"scenario {fingerprint_field} must be a non-placeholder "
                "SHA-256 hex value"
            )
    selector = _document_selector(payload, provenance)

    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("scenario file must contain a non-empty cases list")
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"case {index} must be an object")
        request = row.get("request")
        if not isinstance(request, dict) or not str(request.get("message") or "").strip():
            raise ValueError(f"case {index} requires request.message")
        if not str(row.get("source_sample_id") or "").strip():
            raise ValueError(f"case {index} requires source_sample_id")
        if "document_id" in request:
            raise ValueError(
                f"case {index} must not hardcode document_id; "
                "resolve it from document_selector at runtime"
            )
        cases.append(row)
    return {
        "contract": payload["contract"],
        "provenance": provenance,
        "document_selector": selector,
        "cases": cases,
        "suite_fingerprint": _suite_fingerprint(provenance, cases),
    }


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Compatibility helper returning only validated public cases."""

    return load_suite(path)["cases"]


def resolve_public_document_id(
    datasets: list[dict[str, Any]],
    selector: dict[str, Any],
) -> int:
    """Resolve one imported public corpus without embedding local database IDs."""

    matches = [
        row
        for row in datasets
        if isinstance(row, dict)
        and row.get("source_name") == selector["source_name"]
        and row.get("split") == selector["split"]
        and row.get("corpus_fingerprint")
        == selector["corpus_fingerprint"]
        and row.get("source_snapshot_fingerprint")
        == selector["source_snapshot_fingerprint"]
        and row.get("label_source") == "public_ground_truth"
        and row.get("release_eligible") is True
        and isinstance(row.get("document_id"), int)
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one imported public dataset matching "
            "document_selector; found "
            f"{len(matches)}"
        )
    return int(matches[0]["document_id"])


def bind_cases_to_document(
    cases: list[dict[str, Any]],
    document_id: int,
) -> list[dict[str, Any]]:
    """Bind a runtime-resolved document ID without mutating the frozen suite."""

    if document_id < 1:
        raise ValueError("resolved public document_id must be positive")
    bound: list[dict[str, Any]] = []
    for case in cases:
        row = dict(case)
        request = dict(row["request"])
        if "document_id" in request:
            raise ValueError("scenario request must not include document_id")
        request["document_id"] = document_id
        row["request"] = request
        bound.append(row)
    return bound


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(result)
    artifacts: list[dict[str, Any]] = []
    for raw in compact.get("artifacts") or []:
        artifact = dict(raw)
        if isinstance(artifact.get("download"), dict):
            download = dict(artifact["download"])
            content = download.pop("content", None)
            payload = str(content or "").encode("utf-8")
            if download.get("encoding") == "base64":
                try:
                    payload = base64.b64decode(str(content or ""), validate=True)
                except (binascii.Error, ValueError, TypeError):
                    # Preserve a deterministic audit hash even when a provider
                    # returned malformed base64; the artifact quality gate owns
                    # validity, while this harness only removes large payloads.
                    payload = str(content or "").encode("utf-8")
            download["content_bytes_omitted"] = len(payload)
            download["content_sha256"] = hashlib.sha256(payload).hexdigest()
            artifact["download"] = download
        artifacts.append(artifact)
    compact["artifacts"] = artifacts
    return compact


def _benchmark_error_result(error: Exception) -> dict[str, Any]:
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        kind = "http_status_error"
        message = f"Agent API returned HTTP {status_code}."
        retryable = status_code == 429 or status_code >= 500
    elif isinstance(error, httpx.TimeoutException):
        status_code = None
        kind = "timeout"
        message = "Agent API request timed out."
        retryable = True
    elif isinstance(error, httpx.RequestError):
        status_code = None
        kind = "transport_error"
        message = "Agent API request failed before a response was received."
        retryable = True
    else:
        status_code = None
        kind = "protocol_error"
        message = "Agent API returned an invalid response contract."
        retryable = False
    benchmark_error = {
        "contract": "agent_benchmark_error_v1",
        "kind": kind,
        "message": message,
        "retryable": retryable,
    }
    if status_code is not None:
        benchmark_error["http_status"] = status_code
    return {
        "status": "failed",
        "answer": "",
        "plan": {"status": "failed", "steps": []},
        "trace": [],
        "artifacts": [],
        "benchmark_error": benchmark_error,
    }


def _run_case(
    client: httpx.Client,
    case: dict[str, Any],
    index: int,
    tracker: _ConcurrencyTracker,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tracker.start()
    started = time.perf_counter()
    try:
        case_id = str(case.get("id") or f"case-{index}")
        request = dict(case["request"])
        request.setdefault(
            "run_id",
            f"agent-eval-{case_id}-{uuid.uuid4().hex[:10]}",
        )
        try:
            response = client.post("/agent/chat", json=request)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Agent API response must be an object")

            if result.get("status") == "waiting_approval" and case.get(
                "auto_approve"
            ):
                resumed = client.post(
                    "/agent/resume",
                    json={
                        "run_id": result["run_id"],
                        "approved": True,
                        "comment": (
                            "Agent benchmark scenario pre-authorized this action"
                        ),
                    },
                )
                resumed.raise_for_status()
                result = resumed.json()
                if not isinstance(result, dict):
                    raise ValueError(
                        "Agent resume response must be an object"
                    )
        except (httpx.HTTPError, ValueError, KeyError) as error:
            result = _benchmark_error_result(error)

        latency = time.perf_counter() - started
        expected = dict(case.get("expected") or {})
        expected["id"] = case_id
        score = score_agent_case(
            expected,
            result,
            latency_seconds=latency,
        )
        evidence = {
            "case_id": case_id,
            "source_sample_id": case.get("source_sample_id"),
            "request": {
                key: value
                for key, value in request.items()
                if key not in {"token", "authorization"}
            },
            "expected": expected,
            "score": score,
            "result": _compact_result(result),
        }
        return score, evidence
    finally:
        tracker.finish()


def run_cases(
    client: httpx.Client,
    cases: list[dict[str, Any]],
    *,
    provenance: dict[str, Any] | None = None,
    suite_fingerprint: str | None = None,
    concurrency: int = 1,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    tracker = _ConcurrencyTracker()
    started = time.perf_counter()
    indexed = list(enumerate(cases, start=1))
    rows: list[tuple[dict[str, Any], dict[str, Any]] | None] = [
        None
    ] * len(indexed)
    workers = min(concurrency, max(1, len(indexed)))
    if workers == 1:
        for position, (index, case) in enumerate(indexed):
            rows[position] = _run_case(client, case, index, tracker)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_case, client, case, index, tracker): position
                for position, (index, case) in enumerate(indexed)
            }
            for future, position in futures.items():
                rows[position] = future.result()

    completed = [row for row in rows if row is not None]
    scores = [row[0] for row in completed]
    evidence = [row[1] for row in completed]
    wall_time = max(0.0, time.perf_counter() - started)
    metrics = aggregate_agent_metrics(scores)
    release_evidence_eligible = provenance is not None and bool(
        suite_fingerprint
    )
    release_gate_passed = release_evidence_eligible and bool(scores) and all(
        bool(score.get("verified_task_completion")) for score in scores
    )

    return {
        "contract": "agent_benchmark_v3",
        # Retain the old field as the strict outcome, not merely a provenance
        # claim. The two explicit fields keep failed public runs auditable.
        "release_eligible": release_gate_passed,
        "release_evidence_eligible": release_evidence_eligible,
        "release_gate_passed": release_gate_passed,
        "provenance": provenance,
        "suite_fingerprint": suite_fingerprint,
        "metrics": metrics,
        "performance": {
            "contract": "agent_performance_v1",
            "configured_concurrency": workers,
            "max_observed_in_flight": tracker.max_active,
            "wall_time_seconds": round(wall_time, 6),
            "throughput_cases_per_second": round(
                len(scores) / wall_time, 6
            )
            if wall_time
            else 0.0,
            "provider_usage": metrics["provider_usage"],
            "provider_cost": metrics["provider_cost"],
        },
        "evidence": evidence,
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    performance = report.get("performance") or {}
    usage = performance.get("provider_usage") or {}
    cost = performance.get("provider_cost") or {}
    lines = [
        "# DocMind Agent 任务评测报告",
        "",
        "> 本报告评估 Agent 任务闭环，不把检索指标替代为 Agent 成功率。",
        "",
        "## 汇总",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 场景数 | {metrics['case_count']} |",
        f"| Verified Task Completion Rate | {metrics.get('verified_task_completion_rate', 0):.4f} |",
        f"| 任务成功率 | {metrics.get('task_success_rate', 0):.4f} |",
        f"| 计划完成率 | {metrics.get('mean_plan_completion_rate', 0):.4f} |",
        f"| 工具选择召回率 | {metrics.get('mean_tool_selection_recall', 0):.4f} |",
        f"| 可验证产物率 | {metrics.get('mean_verified_artifact_rate', 0):.4f} |",
        f"| 平均人工介入次数 | {metrics.get('mean_human_intervention_count', 0):.4f} |",
        f"| 平均延迟 | {metrics.get('mean_latency_seconds', 0):.3f}s |",
        f"| P95 延迟 | {metrics.get('p95_latency_seconds', 0):.3f}s |",
        f"| 并发配置 / 实测峰值 | {performance.get('configured_concurrency', 1)} / {performance.get('max_observed_in_flight', 1)} |",
        f"| 端到端吞吐 | {performance.get('throughput_cases_per_second', 0):.4f} cases/s |",
        f"| Provider Token Usage | {usage.get('status', 'unavailable')} |",
        f"| Provider Cost | {cost.get('status', 'unavailable')} |",
        f"| 公开证据来源合格 | {'是' if report.get('release_evidence_eligible') else '否'} |",
        f"| Agent 发布门禁 | {'通过' if report.get('release_gate_passed') else '未通过'} |",
        "",
        "说明：Token 用量和货币成本只统计 Agent API 明确返回的 provider telemetry；"
        "缺失时保持 unavailable，不从模型名、提示词或延迟推算。",
        "",
        "## 场景",
        "",
    ]
    for row in metrics.get("cases") or []:
        state = "通过" if row["task_success"] else "失败"
        lines.append(f"- **{row['case_id']}**：{state}")
        for reason in row.get("failure_reasons") or []:
            lines.append(f"  - {reason}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--token",
        default=os.getenv("DOCMIND_BENCHMARK_TOKEN"),
        help="Bearer token; defaults to DOCMIND_BENCHMARK_TOKEN",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发执行的公开 Agent 场景数（默认 1）",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_timeout_seconds,
        default=DEFAULT_AGENT_REQUEST_TIMEOUT_SECONDS,
        help=(
            "单个 Agent HTTP 请求的端到端等待上限，"
            f"默认 {int(DEFAULT_AGENT_REQUEST_TIMEOUT_SECONDS)} 秒"
        ),
    )
    args = parser.parse_args()
    if not args.token:
        parser.error("--token or DOCMIND_BENCHMARK_TOKEN is required")
    if args.concurrency < 1:
        parser.error("--concurrency 必须至少为 1")

    suite = load_suite(args.scenarios)
    with httpx.Client(
        base_url=args.base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {args.token}"},
        timeout=args.timeout_seconds,
    ) as client:
        datasets_response = client.get("/eval/datasets")
        datasets_response.raise_for_status()
        datasets = datasets_response.json()
        if not isinstance(datasets, list):
            raise ValueError("/eval/datasets must return a list")
        document_id = resolve_public_document_id(
            datasets,
            suite["document_selector"],
        )
        report = run_cases(
            client,
            bind_cases_to_document(suite["cases"], document_id),
            provenance=suite["provenance"],
            suite_fingerprint=suite["suite_fingerprint"],
            concurrency=args.concurrency,
        )
    report["public_demo"] = {
        "document_selector": suite["document_selector"],
        "resolved_document_id": document_id,
        "source_sample_ids": [
            row.get("source_sample_id") for row in suite["cases"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
