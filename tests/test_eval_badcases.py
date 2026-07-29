"""Badcase taxonomy stays deterministic and task-aware."""

from app.services.evaluation.badcases import (
    classify_badcase,
    compare_badcase_sets,
    diagnose_badcases,
    source_links,
)


def test_retrieval_and_grounding_failures_are_separated():
    categories = classify_badcase(
        {
            "hit_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "faithfulness": 0.5,
            "answer_relevance": 0.9,
            "groundedness": 0.0,
            "citation_correctness": 0.5,
            "refusal_correctness": 1.0,
        },
        task_type="rag_qa",
    )
    assert categories == [
        "retrieval_miss",
        "incomplete_recall",
        "relevant_passage_ranked_too_low",
        "faithfulness_failure",
        "unsupported_generation",
        "citation_does_not_entail_claim",
    ]


def test_retrieval_only_task_does_not_report_missing_generation_judge():
    categories = classify_badcase(
        {
            "hit_at_k": 1.0,
            "recall_at_k": 1.0,
            "ndcg_at_k": 1.0,
            "groundedness": None,
        },
        task_type="retrieval",
    )
    assert categories == []


def test_source_links_dedupe_and_point_to_exact_chunk():
    links = source_links(
        [
            {"document_id": 4, "chunk_index": 2, "source_version": "v2"},
            {"document_id": 4, "chunk_index": 2},
        ]
    )
    assert links == [
        {
            "citation_id": "D4:C2",
            "document_id": 4,
            "chunk_index": 2,
            "document_name": None,
            "source_version": "v2",
            "source_status": "unknown",
            "jump_url": "/documents/4/chunks/2",
        }
    ]


def test_refusal_freshness_inference_and_judge_disagreement_are_explicit():
    categories = classify_badcase(
        {
            "hit_at_k": 1.0,
            "recall_at_k": 1.0,
            "precision_at_k": 1.0,
            "ndcg_at_k": 1.0,
            "faithfulness": 0.95,
            "answer_relevance": 1.0,
            "groundedness": 0.0,
            "citation_correctness": 0.0,
        },
        task_type="rag_qa",
        answerable=False,
        citation_report={
            "refused": False,
            "claims": [
                {
                    "claim_type": "inference",
                    "explicit_inference": False,
                }
            ],
        },
        retrieval_trace=[
            {
                "document_id": 1,
                "chunk_index": 0,
                "source_status": "superseded",
            }
        ],
        relevant_chunk_ids=[0],
    )

    assert "stale_source_used" in categories
    assert "should_refuse_but_answered" in categories
    assert "inference_not_labeled" in categories
    assert "evaluator_disagreement" in categories
    assert diagnose_badcases(["stale_source_used"])[0]["layer"] == "freshness"


def test_badcase_set_comparison_separates_introduced_fixed_and_persistent():
    diff = compare_badcase_sets(
        {
            1: [],
            2: ["retrieval_miss"],
            3: ["faithfulness_failure"],
            4: [],
        },
        {
            1: ["unsupported_generation"],
            2: [],
            3: ["faithfulness_failure", "answer_irrelevant"],
            4: [],
        },
    )

    assert [row["sample_id"] for row in diff["newly_introduced"]] == [1]
    assert [row["sample_id"] for row in diff["fixed"]] == [2]
    assert [row["sample_id"] for row in diff["persistent"]] == [3]
    assert diff["unchanged_passed_count"] == 1
