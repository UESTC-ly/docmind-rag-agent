"""检索指标纯函数单元测试。

覆盖 hit@k / MRR / recall@k / precision@k / AP@k / nDCG@k 的正常路径与边界：
命中、未命中、空 relevant、空 retrieved、多命中排序。
"""

import pytest

from app.services.evaluation.retrieval_metrics import (
    average_precision_at_k,
    compute_retrieval_metrics,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)


class TestHitAtK:
    def test_returns_1_when_relevant_in_retrieved(self):
        # Arrange
        relevant = {3, 8}
        retrieved = [1, 3, 5]
        # Act / Assert
        assert hit_at_k(relevant, retrieved) == 1

    def test_returns_0_when_no_overlap(self):
        assert hit_at_k({3, 8}, [1, 2, 5]) == 0

    def test_returns_0_when_retrieved_empty(self):
        assert hit_at_k({3}, []) == 0

    def test_returns_0_when_relevant_empty(self):
        assert hit_at_k(set(), [1, 2, 3]) == 0


class TestReciprocalRank:
    def test_first_position_hit_gives_1(self):
        assert reciprocal_rank({5}, [5, 1, 2]) == 1.0

    def test_second_position_hit_gives_half(self):
        # 第一个命中在位置 2（1-indexed）→ 1/2
        assert reciprocal_rank({5, 8}, [3, 8, 1, 5]) == 0.5

    def test_no_hit_gives_0(self):
        assert reciprocal_rank({9}, [1, 2, 3]) == 0.0

    def test_empty_retrieved_gives_0(self):
        assert reciprocal_rank({9}, []) == 0.0


class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        # 2 个相关块全部命中 → 1.0
        assert recall_at_k({1, 2}, [1, 2, 5]) == 1.0

    def test_partial_recall(self):
        # 命中 1/2 → 0.5
        assert recall_at_k({1, 9}, [1, 5, 7]) == 0.5

    def test_empty_relevant_returns_0(self):
        assert recall_at_k(set(), [1, 2]) == 0.0


class TestPrecisionAtK:
    def test_all_retrieved_relevant(self):
        assert precision_at_k({1, 2}, [1, 2]) == 1.0

    def test_half_retrieved_relevant(self):
        # 检索 4 个，命中 2 个 → 0.5
        assert precision_at_k({1, 2}, [1, 2, 8, 9]) == 0.5

    def test_empty_retrieved_returns_0(self):
        assert precision_at_k({1}, []) == 0.0


class TestAveragePrecisionAtK:
    def test_perfect_ranking_gives_one(self):
        assert average_precision_at_k({1, 2}, [1, 2, 9]) == 1.0

    def test_late_hits_are_discounted_and_missing_hits_count_zero(self):
        # Relevant 1 is found at rank 2; relevant 2 is not retrieved.
        assert average_precision_at_k({1, 2}, [9, 1, 8]) == pytest.approx(0.25)

    def test_empty_relevance_returns_zero(self):
        assert average_precision_at_k(set(), [1, 2]) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking_gives_one(self):
        assert ndcg_at_k({1, 2}, [1, 2, 9]) == 1.0

    def test_relevant_item_ranked_later_reduces_score(self):
        high = ndcg_at_k({1}, [1, 9, 8])
        low = ndcg_at_k({1}, [9, 8, 1])
        assert high == 1.0
        assert 0.0 < low < high

    def test_empty_input_returns_zero(self):
        assert ndcg_at_k({1}, []) == 0.0
        assert ndcg_at_k(set(), [1]) == 0.0

    def test_graded_qrels_reward_putting_highest_grade_first(self):
        qrels = {1: 2, 2: 1}
        assert ndcg_at_k(qrels, [1, 2]) == 1.0
        assert ndcg_at_k(qrels, [2, 1]) < 1.0


def test_duplicate_retrieval_does_not_inflate_metrics():
    metrics = compute_retrieval_metrics({1, 2}, [1, 1, 9])
    assert metrics["recall_at_k"] == 0.5
    assert metrics["precision_at_k"] == 0.5
    assert metrics["average_precision_at_k"] == 0.5


class TestComputeRetrievalMetrics:
    def test_returns_all_six_metrics(self):
        # Arrange
        relevant = {1, 2}
        retrieved = [1, 5, 2, 9]
        # Act
        m = compute_retrieval_metrics(relevant, retrieved)
        # Assert
        assert m["hit"] == 1.0
        assert m["reciprocal_rank"] == 1.0  # 位置 1 命中
        assert m["recall_at_k"] == 1.0      # 2/2 命中
        assert m["precision_at_k"] == 0.5   # 2/4 命中
        assert m["average_precision_at_k"] == pytest.approx(5 / 6)
        assert m["ndcg_at_k"] == pytest.approx(0.9197207891)

    def test_complete_miss_all_zero(self):
        m = compute_retrieval_metrics({100}, [1, 2, 3])
        assert m == {
            "hit": 0.0,
            "reciprocal_rank": 0.0,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "average_precision_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }
