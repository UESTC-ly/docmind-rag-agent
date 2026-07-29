#!/usr/bin/env python3
"""Calibrate citation and answer-relevance judges on public human labels."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.evaluation.dataset_import import source_snapshot_fingerprint
from app.services.evaluation.public_judge_calibration import (
    calibrate_public_binary_judge,
    evaluate_alce_citation,
    evaluate_ares_relevance,
    load_alce_citation_cases,
    load_ares_relevance_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alce", type=Path, required=True)
    parser.add_argument("--ares", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=100)
    parser.add_argument("--minimum-cases", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    citation = calibrate_public_binary_judge(
        load_alce_citation_cases(args.alce, max_cases=args.max_cases),
        target="citation_correctness_claim_evidence",
        evaluator=evaluate_alce_citation,
        minimum_cases=args.minimum_cases,
        max_workers=args.workers,
    )
    relevance = calibrate_public_binary_judge(
        load_ares_relevance_cases(args.ares, max_cases=args.max_cases),
        target="answer_relevance_query_answer",
        evaluator=evaluate_ares_relevance,
        minimum_cases=args.minimum_cases,
        max_workers=args.workers,
    )
    report = {
        "contract": "docmind_public_judge_suite_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "alce": {
                "source_uri": "https://github.com/princeton-nlp/ALCE",
                "license_name": "MIT",
                "source_snapshot_fingerprint": source_snapshot_fingerprint(
                    [args.alce]
                ),
            },
            "ares": {
                "source_uri": "https://github.com/stanford-futuredata/ARES",
                "license_name": "Apache-2.0",
                "source_snapshot_fingerprint": source_snapshot_fingerprint(
                    [args.ares]
                ),
            },
        },
        "citation": citation,
        "answer_relevance": relevance,
        "all_release_gate_eligible": (
            citation["release_gate_eligible"]
            and relevance["release_gate_eligible"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "citation": citation["metrics"],
                "citation_release_gate_eligible": citation[
                    "release_gate_eligible"
                ],
                "answer_relevance": relevance["metrics"],
                "answer_relevance_release_gate_eligible": relevance[
                    "release_gate_eligible"
                ],
                "evidence": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
