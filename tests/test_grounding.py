"""Claim-level semantic grounding and fail-closed delivery tests."""

from app.services.evaluation.grounding import (
    CANONICAL_REFUSAL,
    enforce_grounded_answer,
    evaluate_grounding,
    prepare_grounded_delivery,
)


HITS = [
    {
        "document_id": 1,
        "chunk_index": 0,
        "content": "DocMind 使用混合检索。",
        "score": 0.9,
    },
    {
        "document_id": 2,
        "chunk_index": 3,
        "content": "旧版说明 DocMind 只使用向量检索。",
        "score": 0.8,
    },
]


def _judge(
    *,
    verdict="entailed",
    claim_type="fact",
    citation_verdict="supports",
    conflicts=None,
):
    def run(payload):
        return {
            "claims": [
                {
                    "claim_index": row["claim_index"],
                    "claim_type": claim_type,
                    "verdict": verdict,
                    "citation_verdicts": [
                        {
                            "citation_id": evidence["citation_id"],
                            "verdict": citation_verdict,
                        }
                        for evidence in row["cited_evidence"]
                    ],
                    "reason": "fixture verdict",
                }
                for row in payload["claims"]
            ],
            "conflicts": conflicts or [],
        }

    return run


def test_entailing_citation_produces_semantic_grounding_metrics():
    report = evaluate_grounding(
        "DocMind 使用混合检索。[D1:C0]",
        HITS,
        judge=_judge(),
    )

    assert report["contract"] == "claim_grounding_v1"
    assert report["semantic_entailment_checked"] is True
    assert report["groundedness"] == 1.0
    assert report["faithfulness"] == 1.0
    assert report["citation_correctness"] == 1.0
    assert report["passed"] is True
    assert report["claims"][0]["semantically_supported"] is True
    assert report["claims"][0]["evidence_state"] == "supported"
    assert report["claim_state_counts"]["supported"] == 1
    assert len(report["judge_input_fingerprint"]) == 64


def test_citation_presence_does_not_pass_when_evidence_is_irrelevant():
    report = evaluate_grounding(
        "DocMind 已经实现量子计算。[D1:C0]",
        HITS,
        judge=_judge(
            verdict="unsupported",
            citation_verdict="irrelevant",
        ),
    )

    assert report["structural_passed"] is True
    assert report["groundedness"] == 0.0
    assert report["citation_correctness"] == 0.0
    assert report["passed"] is False


def test_reasonable_inference_must_be_explicitly_labeled():
    hidden = evaluate_grounding(
        "DocMind 的检索效果更好。[D1:C0]",
        HITS,
        judge=_judge(
            verdict="reasonable_inference",
            claim_type="inference",
        ),
    )
    explicit = evaluate_grounding(
        "推测：DocMind 的检索效果可能更好。[D1:C0]",
        HITS,
        judge=_judge(
            verdict="reasonable_inference",
            claim_type="inference",
        ),
    )

    assert hidden["claims"][0]["explicit_inference"] is False
    assert hidden["claims"][0]["evidence_state"] == "inference"
    assert hidden["passed"] is False
    assert explicit["passed"] is True


def test_evidence_conflict_blocks_delivery_even_when_claim_is_supported():
    report = evaluate_grounding(
        "DocMind 使用混合检索。[D1:C0][D2:C3]",
        HITS,
        judge=_judge(
            conflicts=[
                {
                    "citation_ids": ["D1:C0", "D2:C3"],
                    "reason": "版本说明冲突",
                }
            ]
        ),
    )

    assert report["conflict_count"] == 1
    assert report["conflicts_disclosed"] is False
    assert report["passed"] is False


def test_disclosed_conflict_can_pass_without_silently_choosing_a_side():
    report = evaluate_grounding(
        "不同版本文档的说法存在冲突：新版称使用混合检索，旧版称只使用向量检索。"
        "[D1:C0][D2:C3]",
        HITS,
        judge=_judge(
            conflicts=[
                {
                    "citation_ids": ["D1:C0", "D2:C3"],
                    "reason": "版本说明冲突",
                }
            ]
        ),
    )

    assert report["conflict_count"] == 1
    assert report["conflicts_disclosed"] is True
    assert report["passed"] is True


def test_unavailable_judge_is_not_converted_to_zero_score():
    def broken(_payload):
        raise TimeoutError("judge timed out")

    report = evaluate_grounding(
        "DocMind 使用混合检索。[D1:C0]",
        HITS,
        judge=broken,
    )

    assert report["judge_status"] == "unavailable"
    assert report["semantic_entailment_checked"] is False
    assert report["groundedness"] is None
    assert report["faithfulness"] is None
    assert report["citation_correctness"] is None
    assert report["passed"] is False
    assert enforce_grounded_answer("原答案", report) == CANONICAL_REFUSAL


def test_fail_closed_removes_only_unsupported_claims():
    answers = "有依据。[D1:C0]\n没有依据。[D1:C0]"

    def mixed(payload):
        return {
            "claims": [
                {
                    "claim_index": 0,
                    "claim_type": "fact",
                    "verdict": "entailed",
                    "citation_verdicts": [
                        {"citation_id": "D1:C0", "verdict": "supports"}
                    ],
                    "reason": "ok",
                },
                {
                    "claim_index": 1,
                    "claim_type": "fact",
                    "verdict": "unsupported",
                    "citation_verdicts": [
                        {"citation_id": "D1:C0", "verdict": "irrelevant"}
                    ],
                    "reason": "not in source",
                },
            ],
            "conflicts": [],
        }

    report = evaluate_grounding(answers, HITS, judge=mixed)
    sanitized = enforce_grounded_answer(answers, report)
    assert "有依据" in sanitized
    assert "没有依据" not in sanitized


def test_refusal_has_no_fake_semantic_score():
    report = evaluate_grounding(CANONICAL_REFUSAL, HITS, judge=_judge())
    assert report["refused"] is True
    assert report["judge_status"] == "not_applicable"
    assert report["groundedness"] is None
    assert report["passed"] is True


def test_delivery_repairs_once_then_rechecks():
    calls = []

    def evaluator(answer, _hits):
        calls.append(answer)
        if "没有依据" in answer:
            return {
                "passed": False,
                "semantic_entailment_checked": True,
                "conflict_count": 0,
                "groundedness": 0.5,
                "citation_correctness": 0.5,
                "unsupported_claim_count": 1,
                "claims": [
                    {
                        "text": "有依据。[D1:C0]",
                        "semantically_supported": True,
                    },
                    {
                        "text": "没有依据。[D1:C0]",
                        "semantically_supported": False,
                    },
                ],
            }
        return {
            "passed": True,
            "semantic_entailment_checked": True,
            "conflict_count": 0,
            "claims": [],
        }

    answer, report = prepare_grounded_delivery(
        "有依据。[D1:C0]\n没有依据。[D1:C0]",
        HITS,
        evaluator=evaluator,
    )
    assert answer == "有依据。[D1:C0]"
    assert len(calls) == 2
    assert report["delivery_action"] == "removed_unsupported_claims"


def test_delivery_refuses_on_conflict_without_repairing():
    def evaluator(answer, _hits):
        if answer == CANONICAL_REFUSAL:
            return {
                "passed": True,
                "semantic_entailment_checked": False,
                "claims": [],
            }
        return {
            "passed": False,
            "semantic_entailment_checked": True,
            "conflict_count": 1,
            "claims": [],
        }

    answer, report = prepare_grounded_delivery(
        "冲突结论。[D1:C0][D2:C3]",
        HITS,
        evaluator=evaluator,
    )
    assert answer == CANONICAL_REFUSAL
    assert report["delivery_action"] == "refused_due_to_conflicting_evidence"
    assert report["conflict_count"] == 1
    assert report["initial_verification"]["conflict_count"] == 1
