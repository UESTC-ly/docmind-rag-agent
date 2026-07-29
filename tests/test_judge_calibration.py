import json

import pytest

from app.services.evaluation.generation_judge import JudgeResult
from app.services.evaluation.judge_calibration import (
    calibrate_faithfulness_judge,
    load_ragtruth_cases,
)


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_ragtruth_cases_preserves_strict_grounding_labels(tmp_path):
    sources = tmp_path / "source_info.jsonl"
    responses = tmp_path / "response.jsonl"
    _write_jsonl(
        sources,
        [
            {
                "source_id": "s1",
                "task_type": "QA",
                "source": "MARCO",
                "source_info": {
                    "question": "Q",
                    "passages": "public passage",
                },
                "prompt": "answer from the passage",
            },
            {
                "source_id": "s2",
                "task_type": "Summary",
                "source": "CNN/DM",
                "source_info": "public article",
                "prompt": "summarize",
            },
        ],
    )
    _write_jsonl(
        responses,
        [
            {
                "id": "r1",
                "source_id": "s1",
                "model": "public-generator",
                "labels": [],
                "split": "test",
                "quality": "good",
                "response": "supported",
            },
            {
                "id": "r2",
                "source_id": "s2",
                "model": "public-generator",
                "labels": [{"text": "outside", "implicit_true": True}],
                "split": "test",
                "quality": "good",
                "response": "true elsewhere but unsupported here",
            },
            {
                "id": "r3",
                "source_id": "s2",
                "model": "public-generator",
                "labels": [{"text": "fabricated"}],
                "split": "train",
                "quality": "good",
                "response": "wrong split",
            },
            {
                "id": "r4",
                "source_id": "s2",
                "model": "public-generator",
                "labels": [],
                "split": "test",
                "quality": "truncated",
                "response": "wrong quality",
            },
        ],
    )

    cases = load_ragtruth_cases(sources, responses, split="test")

    assert [row["response_id"] for row in cases] == ["r1", "r2"]
    assert cases[0]["context"] == "public passage"
    assert cases[0]["gold_strict_unsupported"] is False
    assert cases[1]["gold_strict_unsupported"] is True
    assert cases[1]["gold_hallucinated"] is False
    assert cases[1]["implicit_true_count"] == 1
    assert len(cases[1]["input_fingerprint"]) == 64


def test_load_ragtruth_cases_validates_selection(tmp_path):
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="required"):
        load_ragtruth_cases(missing, missing)

    sources = tmp_path / "source_info.jsonl"
    responses = tmp_path / "response.jsonl"
    _write_jsonl(sources, [])
    _write_jsonl(responses, [])
    with pytest.raises(ValueError, match="no calibration"):
        load_ragtruth_cases(sources, responses)
    with pytest.raises(ValueError, match="positive"):
        load_ragtruth_cases(sources, responses, max_cases=0)


def test_load_ragtruth_cases_rejects_malformed_jsonl(tmp_path):
    sources = tmp_path / "source_info.jsonl"
    responses = tmp_path / "response.jsonl"
    sources.write_text("{not-json}\n", encoding="utf-8")
    _write_jsonl(responses, [])

    with pytest.raises(ValueError, match="not valid JSON"):
        load_ragtruth_cases(sources, responses)

    _write_jsonl(sources, [["not", "an", "object"]])
    with pytest.raises(ValueError, match="JSON object"):
        load_ragtruth_cases(sources, responses)


def test_load_ragtruth_cases_filters_unusable_sources_and_tasks(tmp_path):
    sources = tmp_path / "source_info.jsonl"
    responses = tmp_path / "response.jsonl"
    _write_jsonl(
        sources,
        [
            {
                "source_id": "",
                "task_type": "QA",
                "source_info": "missing id",
            },
            {
                "source_id": "empty",
                "task_type": "QA",
                "source_info": None,
            },
            {
                "source_id": "dict-context",
                "task_type": "QA",
                "source_info": {"question": "Q", "evidence": ["A"]},
            },
            {
                "source_id": "summary",
                "task_type": "Summary",
                "source_info": "public article",
            },
        ],
    )
    _write_jsonl(
        responses,
        [
            {
                "id": "unknown-source",
                "source_id": "missing",
                "split": "test",
                "quality": "good",
                "response": "ignored",
            },
            {
                "id": "empty-response",
                "source_id": "dict-context",
                "split": "test",
                "quality": "good",
                "response": "",
            },
            {
                "id": "filtered-task",
                "source_id": "summary",
                "split": "test",
                "quality": "good",
                "response": "ignored",
            },
            {
                "id": "kept",
                "source_id": "dict-context",
                "split": "test",
                "quality": "good",
                "labels": "invalid-label-container",
                "response": "supported",
            },
        ],
    )

    cases = load_ragtruth_cases(
        sources,
        responses,
        split="test",
        task_types={"qa"},
    )

    assert [row["response_id"] for row in cases] == ["kept"]
    assert cases[0]["context"] == '{"evidence":["A"],"question":"Q"}'
    assert cases[0]["label_count"] == 0


def _case(case_id, unsupported, task_type="QA"):
    return {
        "response_id": case_id,
        "source_id": f"s-{case_id}",
        "task_type": task_type,
        "source": "public",
        "generator_model": "generator",
        "context": f"context {case_id}",
        "response": f"response {case_id}",
        "gold_strict_unsupported": unsupported,
        "gold_hallucinated": unsupported,
        "label_count": int(unsupported),
        "implicit_true_count": 0,
        "input_fingerprint": case_id.rjust(64, "0"),
    }


def test_calibration_reports_confusion_matrix_and_release_decision():
    cases = [
        _case("1", False),
        _case("2", True),
        _case("3", False, "Summary"),
        _case("4", True, "Summary"),
    ]
    scores = iter([0.95, 0.2, 0.9, 0.1])

    report = calibrate_faithfulness_judge(
        cases,
        evaluator=lambda contexts, answer: next(scores),
        minimum_cases=4,
        minimum_balanced_accuracy=1.0,
    )

    assert report["release_gate_eligible"] is True
    assert report["metrics"] == {
        "case_count": 4,
        "available_count": 4,
        "unavailable_count": 0,
        "positive_count": 2,
        "negative_count": 2,
        "true_positive": 2,
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
        "coverage": 1.0,
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "specificity": 1.0,
        "f1": 1.0,
        "balanced_accuracy": 1.0,
    }
    assert report["slices"]["QA"]["case_count"] == 2
    assert "context" not in report["evidence"][0]
    assert "response" not in report["evidence"][0]


def test_calibration_keeps_unavailable_judge_result_unknown():
    cases = [_case("1", False), _case("2", True)]
    results = iter(
        [
            JudgeResult(
                score=None,
                reason="provider unavailable",
                status="unavailable",
                model="judge",
                rubric_version="faithfulness_v2",
                input_fingerprint="a" * 64,
            ),
            0.1,
        ]
    )

    report = calibrate_faithfulness_judge(
        cases,
        evaluator=lambda contexts, answer: next(results),
        minimum_cases=2,
        minimum_balanced_accuracy=0,
    )

    assert report["release_gate_eligible"] is False
    assert report["metrics"]["unavailable_count"] == 1
    assert report["metrics"]["coverage"] == 0.5
    assert report["evidence"][0]["predicted_unsupported"] is None
    assert report["evidence"][0]["correct"] is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"score_threshold": 1.1}, "score_threshold"),
        ({"minimum_cases": 0}, "minimum_cases"),
        ({"minimum_balanced_accuracy": -0.1}, "minimum_balanced_accuracy"),
        ({"max_workers": 0}, "max_workers"),
    ],
)
def test_calibration_rejects_invalid_thresholds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        calibrate_faithfulness_judge([_case("1", False)], **kwargs)


def test_calibration_parallel_results_keep_case_order():
    cases = [_case(str(index), index % 2 == 0) for index in range(1, 9)]

    report = calibrate_faithfulness_judge(
        cases,
        evaluator=lambda contexts, answer: (
            0.1 if int(answer.rsplit(" ", 1)[-1]) % 2 == 0 else 0.9
        ),
        minimum_cases=8,
        minimum_balanced_accuracy=1.0,
        max_workers=4,
    )

    assert [row["response_id"] for row in report["evidence"]] == [
        str(index) for index in range(1, 9)
    ]
    assert report["release_gate_eligible"] is True
