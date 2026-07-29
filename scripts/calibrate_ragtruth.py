#!/usr/bin/env python3
"""Calibrate the configured faithfulness judge on official RAGTruth JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.evaluation.dataset_import import source_snapshot_fingerprint
from app.services.evaluation.judge_calibration import (
    RAGTRUTH_TRANSFORM_CONTRACT,
    calibrate_faithfulness_judge,
    load_ragtruth_cases,
)


def _report_fingerprint(report: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in report.items()
        if key not in {"generated_at", "report_fingerprint"}
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    state = "可作为发布门禁证据" if report["release_gate_eligible"] else "不可作为发布门禁证据"
    lines = [
        "# DocMind RAGTruth Judge 校准报告",
        "",
        f"- 结论：**{state}**",
        f"- Judge：`{report['judge']['model']}` / `{report['judge']['rubric_version']}`",
        f"- Split：`{report['provenance']['split']}`",
        f"- 原始快照：`{report['provenance']['source_snapshot_fingerprint']}`",
        f"- 报告指纹：`{report['report_fingerprint']}`",
        "",
        "## 汇总",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 样本数 | {metrics['case_count']} |",
        f"| 可用 Judge 结果 | {metrics['available_count']} |",
        f"| Judge 不可用 | {metrics['unavailable_count']} |",
        f"| Coverage | {metrics['coverage'] if metrics['coverage'] is not None else 'N/A'} |",
        f"| Accuracy | {metrics['accuracy'] if metrics['accuracy'] is not None else 'N/A'} |",
        f"| Precision | {metrics['precision'] if metrics['precision'] is not None else 'N/A'} |",
        f"| Recall | {metrics['recall'] if metrics['recall'] is not None else 'N/A'} |",
        f"| Specificity | {metrics['specificity'] if metrics['specificity'] is not None else 'N/A'} |",
        f"| Balanced Accuracy | {metrics['balanced_accuracy'] if metrics['balanced_accuracy'] is not None else 'N/A'} |",
        "",
        "## 说明",
        "",
        "- 本报告校准 generation-level faithfulness judge，不代表 claim citation judge 已校准。",
        "- `implicit_true` 表示可能真实但未出现在上下文；DocMind 严格依据性契约仍判为 unsupported。",
        "- 缺失或异常的 Judge 返回保持 unavailable，不会折算为 0 分或通过。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate DocMind's faithfulness judge on RAGTruth."
    )
    parser.add_argument("--source-info", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--minimum-cases", type=int, default=100)
    parser.add_argument(
        "--minimum-balanced-accuracy",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Bounded concurrent judge calls (default: 4).",
    )
    parser.add_argument(
        "--task-types",
        help="Optional comma-separated RAGTruth task types.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    if args.max_cases <= 0:
        parser.error("--max-cases must be positive")

    task_types = (
        {
            value.strip()
            for value in args.task_types.split(",")
            if value.strip()
        }
        if args.task_types
        else None
    )
    cases = load_ragtruth_cases(
        args.source_info,
        args.responses,
        split=args.split,
        max_cases=args.max_cases,
        task_types=task_types,
    )
    report = calibrate_faithfulness_judge(
        cases,
        score_threshold=args.score_threshold,
        minimum_cases=args.minimum_cases,
        minimum_balanced_accuracy=args.minimum_balanced_accuracy,
        max_workers=args.workers,
    )
    report.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "provenance": {
                "source_name": "RAGTruth",
                "source_uri": "https://github.com/ParticleMedia/RAGTruth",
                "source_version": "official repository JSONL snapshot",
                "license_name": "MIT",
                "split": args.split,
                "source_snapshot_fingerprint": source_snapshot_fingerprint(
                    [args.source_info, args.responses]
                ),
                "transform_spec": {
                    "contract": RAGTRUTH_TRANSFORM_CONTRACT,
                    "max_cases": args.max_cases,
                    "task_types": sorted(task_types or []),
                    "quality_filter": ["good"],
                },
            },
            "judge": {
                "model": settings.chat_model,
                "rubric_version": "faithfulness_v2",
            },
        }
    )
    report["report_fingerprint"] = _report_fingerprint(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "release_gate_eligible": report["release_gate_eligible"],
                "metrics": report["metrics"],
                "report_fingerprint": report["report_fingerprint"],
                "evidence": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
