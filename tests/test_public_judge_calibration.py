import csv
import json

from app.services.evaluation.public_judge_calibration import (
    calibrate_public_binary_judge,
    load_alce_citation_cases,
    load_ares_relevance_cases,
)


def test_public_judge_loaders_preserve_human_labels(tmp_path):
    alce = tmp_path / "alce.json"
    alce.write_text(
        json.dumps(
            {
                "asqa": {
                    "model": {
                        "q1": {
                            "sentences": [
                                {
                                    "text": "Claim [1]",
                                    "citations": [
                                        {
                                            "text": "Evidence",
                                            "citation_precision_score": 2,
                                        },
                                        {
                                            "text": "Noise",
                                            "citation_precision_score": 0,
                                        },
                                    ],
                                }
                            ]
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    ares = tmp_path / "ares.tsv"
    with ares.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "id",
                "Query",
                "Answer",
                "Answer_Relevance_Label",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "1",
                "Query": "Q1",
                "Answer": "A1",
                "Answer_Relevance_Label": "1.0",
            }
        )
        writer.writerow(
            {
                "id": "2",
                "Query": "Q2",
                "Answer": "A2",
                "Answer_Relevance_Label": "0.0",
            }
        )

    citation_cases = load_alce_citation_cases(alce, max_cases=2)
    relevance_cases = load_ares_relevance_cases(ares, max_cases=2)

    assert [row["gold_positive"] for row in citation_cases] == [True, False]
    assert citation_cases[0]["claim"] == "Claim"
    assert [row["gold_positive"] for row in relevance_cases] == [True, False]


def test_public_binary_calibration_fails_closed_on_unavailable_result():
    cases = [
        {
            "case_id": "positive",
            "slice": "public",
            "gold_positive": True,
            "gold_label": 1,
            "input_fingerprint": "a" * 64,
        },
        {
            "case_id": "negative",
            "slice": "public",
            "gold_positive": False,
            "gold_label": 0,
            "input_fingerprint": "b" * 64,
        },
    ]
    values = {"positive": 1.0, "negative": None}
    report = calibrate_public_binary_judge(
        cases,
        target="fixture",
        evaluator=lambda case: values[case["case_id"]],
        minimum_cases=2,
        max_workers=2,
    )

    assert report["metrics"]["coverage"] == 0.5
    assert report["release_gate_eligible"] is False
    assert "claim" not in report["evidence"][0]
