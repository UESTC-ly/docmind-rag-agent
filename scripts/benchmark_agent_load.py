#!/usr/bin/env python3
"""Run repeatable public Agent load stages through the HTTP contract.

This is intentionally a black-box harness: it measures response latency,
in-flight concurrency, outcome gates, and provider telemetry returned by
``/agent/chat``. It never estimates tokens, currency cost, or provider-side
request concurrency.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.evaluation.agent_metrics import aggregate_agent_metrics


def _benchmark_module():
    path = Path(__file__).with_name("benchmark_agent.py")
    spec = importlib.util.spec_from_file_location("benchmark_agent", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _unit_interval(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be between 0 and 1") from exc
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be positive") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _concurrency_levels(value: str) -> list[int]:
    try:
        levels = [_positive_int(item.strip()) for item in value.split(",")]
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError(
            "concurrency levels must be comma-separated positive integers"
        ) from exc
    if not levels:
        raise argparse.ArgumentTypeError("at least one concurrency level is required")
    return levels


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return round(ordered[index], 6)


def _run_stage(
    benchmark,
    client: httpx.Client,
    cases: list[dict[str, Any]],
    *,
    concurrency: int,
    repetitions: int,
    phase: str,
) -> dict[str, Any]:
    tracker = benchmark._ConcurrencyTracker()
    scheduled = [
        (repetition, index, case)
        for repetition in range(1, repetitions + 1)
        for index, case in enumerate(cases, start=1)
    ]
    rows: list[tuple[dict[str, Any], dict[str, Any]] | None] = [None] * len(
        scheduled
    )
    started = time.perf_counter()
    workers = min(concurrency, max(1, len(scheduled)))
    if workers == 1:
        for position, (_, index, case) in enumerate(scheduled):
            rows[position] = benchmark._run_case(client, case, index, tracker)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(benchmark._run_case, client, case, index, tracker): position
                for position, (_, index, case) in enumerate(scheduled)
            }
            for future, position in futures.items():
                rows[position] = future.result()
    wall_time = max(0.0, time.perf_counter() - started)
    completed = [row for row in rows if row is not None]
    scores = [row[0] for row in completed]
    evidence: list[dict[str, Any]] = []
    for (repetition, _, _), row in zip(scheduled, completed, strict=True):
        score, receipt = row
        evidence.append(
            {
                "phase": phase,
                "repetition": repetition,
                "case_id": score.get("case_id"),
                "score": score,
                "receipt": receipt,
            }
        )
    metrics = aggregate_agent_metrics(scores)
    latencies = [float(score.get("latency_seconds") or 0.0) for score in scores]
    failures = [
        item
        for item in evidence
        if not bool(item["score"].get("verified_task_completion"))
    ]
    return {
        "phase": phase,
        "configured_concurrency": concurrency,
        "actual_max_in_flight": tracker.max_active,
        "repetitions": repetitions,
        "case_count": len(scores),
        "wall_time_seconds": round(wall_time, 6),
        "throughput_cases_per_second": (
            round(len(scores) / wall_time, 6) if wall_time else 0.0
        ),
        "latency_seconds": {
            "mean": metrics.get("mean_latency_seconds", 0.0),
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
        },
        "verified_task_completion_rate": metrics.get(
            "verified_task_completion_rate",
            0.0,
        ),
        "task_success_rate": metrics.get("task_success_rate", 0.0),
        "failure_count": len(failures),
        "failure_class_counts": metrics.get("failure_class_counts", {}),
        "provider_usage": metrics["provider_usage"],
        "provider_cost": metrics["provider_cost"],
        "provider_model": metrics["provider_model"],
        "evidence": evidence,
    }


def _gate_stage(
    stage: dict[str, Any],
    *,
    minimum_vtcr: float,
    maximum_p95_seconds: float | None,
    require_token_usage: bool,
    require_provider_cost: bool,
) -> dict[str, Any]:
    """Apply explicit, reproducible load gates without inventing telemetry."""

    target_in_flight = min(
        int(stage["configured_concurrency"]),
        int(stage["case_count"]),
    )
    checks = [
        {
            "name": "verified_task_completion_rate",
            "passed": float(stage["verified_task_completion_rate"])
            >= minimum_vtcr,
            "actual": stage["verified_task_completion_rate"],
            "threshold": f">={minimum_vtcr}",
        },
        {
            "name": "configured_concurrency_reached",
            "passed": int(stage["actual_max_in_flight"]) >= target_in_flight,
            "actual": stage["actual_max_in_flight"],
            "threshold": f">={target_in_flight}",
        },
    ]
    if maximum_p95_seconds is not None:
        checks.append(
            {
                "name": "p95_latency_seconds",
                "passed": float(stage["latency_seconds"]["p95"])
                <= maximum_p95_seconds,
                "actual": stage["latency_seconds"]["p95"],
                "threshold": f"<={maximum_p95_seconds}",
            }
        )
    if require_token_usage:
        checks.append(
            {
                "name": "provider_token_usage",
                "passed": stage["provider_usage"].get("status") == "observed",
                "actual": stage["provider_usage"].get(
                    "status",
                    "unavailable",
                ),
                "threshold": "observed",
            }
        )
    if require_provider_cost:
        checks.append(
            {
                "name": "provider_currency_cost",
                "passed": stage["provider_cost"].get("status") == "observed",
                "actual": stage["provider_cost"].get(
                    "status",
                    "unavailable",
                ),
                "threshold": "observed",
            }
        )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def run_load(
    client: httpx.Client,
    cases: list[dict[str, Any]],
    *,
    concurrency_levels: list[int],
    repetitions: int,
    warmup_repetitions: int = 0,
    minimum_vtcr: float = 1.0,
    maximum_p95_seconds: float | None = None,
    require_token_usage: bool = False,
    require_provider_cost: bool = False,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("load benchmark requires at least one case")
    if not concurrency_levels:
        raise ValueError("load benchmark requires at least one concurrency level")
    if repetitions < 1 or warmup_repetitions < 0:
        raise ValueError("invalid repetition count")
    if not 0 <= minimum_vtcr <= 1:
        raise ValueError("minimum_vtcr must be between 0 and 1")
    if maximum_p95_seconds is not None and maximum_p95_seconds <= 0:
        raise ValueError("maximum_p95_seconds must be positive")

    benchmark = _benchmark_module()
    warmup = (
        _run_stage(
            benchmark,
            client,
            cases,
            concurrency=1,
            repetitions=warmup_repetitions,
            phase="warmup",
        )
        if warmup_repetitions
        else None
    )
    stages = [
        _run_stage(
            benchmark,
            client,
            cases,
            concurrency=concurrency,
            repetitions=repetitions,
            phase="measurement",
        )
        for concurrency in concurrency_levels
    ]
    for stage in stages:
        stage["release_gate"] = _gate_stage(
            stage,
            minimum_vtcr=minimum_vtcr,
            maximum_p95_seconds=maximum_p95_seconds,
            require_token_usage=require_token_usage,
            require_provider_cost=require_provider_cost,
        )
    return {
        "contract": "agent_load_benchmark_v1",
        "policy": {
            "minimum_verified_task_completion_rate": minimum_vtcr,
            "maximum_p95_seconds": maximum_p95_seconds,
            "require_configured_concurrency_reached": True,
            "require_provider_token_usage": require_token_usage,
            "require_provider_currency_cost": require_provider_cost,
        },
        "warmup": warmup,
        "stages": stages,
        "release_gate_passed": all(
            stage["release_gate"]["passed"] for stage in stages
        ),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DocMind Agent 并发与压力评测报告",
        "",
        "> 仅统计 Agent API 明确返回的 token/cost telemetry；未返回时保持 unavailable。",
        "",
        (
            "| 并发配置 / 实测峰值 | 场景执行数 | VTCR | P95 延迟 | 吞吐 | "
            "请求 / 重试 | Provider Token | Provider Cost | Provider Model | "
            "失败分类 | Gate |"
        ),
        "|---|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for stage in report["stages"]:
        usage = stage["provider_usage"]
        cost = stage["provider_cost"]
        model = stage["provider_model"]
        usage_value = (
            f"{usage.get('total_tokens')} ({usage.get('status')})"
            if isinstance(usage.get("total_tokens"), (int, float))
            else usage.get("status", "unavailable")
        )
        cost_value = (
            f"{cost.get('currency')} {cost.get('amount')}"
            if isinstance(cost.get("amount"), (int, float))
            and cost.get("currency")
            else cost.get("status", "unavailable")
        )
        failure_classes = ", ".join(
            f"{name}={count}"
            for name, count in stage["failure_class_counts"].items()
            if name != "passed"
        ) or "无"
        lines.append(
            "| "
            f"{stage['configured_concurrency']} / {stage['actual_max_in_flight']} | "
            f"{stage['case_count']} | "
            f"{stage['verified_task_completion_rate']:.4f} | "
            f"{stage['latency_seconds']['p95']:.3f}s | "
            f"{stage['throughput_cases_per_second']:.4f} cases/s | "
            f"{usage.get('request_count', 'unavailable')} / "
            f"{usage.get('retry_count', 'unavailable')} | "
            f"{usage_value} | "
            f"{cost_value} | "
            + (
                ", ".join(model.get("provider_reported_models") or [])
                or "provider-unavailable"
            )
            + f" | {failure_classes} | "
            + f"{'通过' if stage['release_gate']['passed'] else '未通过'} |"
        )
    lines.extend(
        [
            "",
            f"**压力发布门禁：{'通过' if report['release_gate_passed'] else '未通过'}**",
        ]
    )
    lines.extend(["", "## 失败凭据", ""])
    failures = [
        item
        for stage in report["stages"]
        for item in stage["evidence"]
        if not item["score"].get("verified_task_completion")
    ]
    if not failures:
        lines.append("- 无。")
    else:
        for item in failures:
            reasons = item["score"].get("failure_reasons") or ["未分类失败"]
            lines.append(
                f"- `{item['case_id']}` (第 {item['repetition']} 次)："
                f"`{item['score'].get('failure_class', 'unknown')}`；"
                + "；".join(reasons)
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run staged public Agent load benchmarks."
    )
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=os.getenv("DOCMIND_BENCHMARK_TOKEN"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--concurrency", type=_concurrency_levels, default=[1, 2, 4])
    parser.add_argument("--repetitions", type=_positive_int, default=1)
    parser.add_argument("--warmup-repetitions", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=600.0)
    parser.add_argument("--min-vtcr", type=_unit_interval, default=1.0)
    parser.add_argument("--max-p95-seconds", type=_positive_float)
    parser.add_argument("--require-token-usage", action="store_true")
    parser.add_argument("--require-provider-cost", action="store_true")
    args = parser.parse_args()
    if not args.token:
        parser.error("--token or DOCMIND_BENCHMARK_TOKEN is required")
    if args.warmup_repetitions < 0:
        parser.error("--warmup-repetitions must not be negative")

    benchmark = _benchmark_module()
    suite = benchmark.load_suite(args.scenarios)
    with httpx.Client(
        base_url=args.base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {args.token}"},
        timeout=args.timeout_seconds,
    ) as client:
        datasets = client.get("/eval/datasets")
        datasets.raise_for_status()
        document_id = benchmark.resolve_public_document_id(
            datasets.json(),
            suite["document_selector"],
        )
        report = run_load(
            client,
            benchmark.bind_cases_to_document(suite["cases"], document_id),
            concurrency_levels=args.concurrency,
            repetitions=args.repetitions,
            warmup_repetitions=args.warmup_repetitions,
            minimum_vtcr=args.min_vtcr,
            maximum_p95_seconds=args.max_p95_seconds,
            require_token_usage=args.require_token_usage,
            require_provider_cost=args.require_provider_cost,
        )
    report["suite"] = {
        "provenance": suite["provenance"],
        "suite_fingerprint": suite["suite_fingerprint"],
        "document_selector": suite["document_selector"],
        "resolved_document_id": document_id,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            [
                {
                    "concurrency": stage["configured_concurrency"],
                    "verified_task_completion_rate": stage[
                        "verified_task_completion_rate"
                    ],
                    "p95_latency_seconds": stage["latency_seconds"]["p95"],
                    "throughput_cases_per_second": stage[
                        "throughput_cases_per_second"
                    ],
                }
                for stage in report["stages"]
            ]
            + [{"release_gate_passed": report["release_gate_passed"]}],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
