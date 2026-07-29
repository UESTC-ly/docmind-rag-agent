"""检索质量指标——纯函数，无 I/O 依赖，全部可单元测试。

术语约定：
  relevant_ids  : set[int] — 该问题对应的正确 chunk_index 集合（ground truth）
  retrieved_ids : list[int] — 检索器按排名顺序返回的 chunk_index 列表（最多 top_k 个）

指标说明：
  hit_rate       — top-k 内是否命中至少一个相关块（0 / 1）
  reciprocal_rank — 第一个命中位置的倒数（未命中为 0）；多条样本取均值 = MRR
  recall_at_k    — 命中相关块数 / 总相关块数
  precision_at_k — 命中相关块数 / k
  average_precision_at_k — 每次相关命中位置的 precision 均值（未召回项记 0）
  ndcg_at_k      — 按排名折损的归一化累计增益（二值 qrels）
"""

import math
from collections.abc import Mapping


Relevance = set[int] | Mapping[int, int | float]


def _relevance_scores(relevance: Relevance) -> dict[int, float]:
    if isinstance(relevance, Mapping):
        return {
            int(document_id): float(score)
            for document_id, score in relevance.items()
            if float(score) > 0
        }
    return {document_id: 1.0 for document_id in relevance}


def _unique_ranked(retrieved_ids: list[int]) -> list[int]:
    """Drop duplicate IDs without changing their first-seen rank.

    A retriever must not gain recall, precision, AP, or nDCG by returning the
    same relevant chunk more than once.
    """

    return list(dict.fromkeys(retrieved_ids))


def hit_at_k(relevant_ids: set[int], retrieved_ids: list[int]) -> int:
    """top-k 内是否命中至少一个相关块。返回 0 或 1。"""
    for rid in _unique_ranked(retrieved_ids):
        if rid in relevant_ids:
            return 1
    return 0


def reciprocal_rank(relevant_ids: set[int], retrieved_ids: list[int]) -> float:
    """第一个命中位置的倒数。未命中返回 0.0。

    例：retrieved = [3, 8, 1, 5]，relevant = {5, 8}
    → 第一个命中是位置 2（1-indexed）的 8，倒数 = 0.5
    """
    for rank, rid in enumerate(_unique_ranked(retrieved_ids), start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(relevant_ids: set[int], retrieved_ids: list[int]) -> float:
    """命中相关块数 / 全部相关块数。relevant_ids 为空时返回 0.0。"""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in _unique_ranked(retrieved_ids) if rid in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(relevant_ids: set[int], retrieved_ids: list[int]) -> float:
    """命中相关块数 / 实际返回块数（k）。retrieved 为空时返回 0.0。"""
    unique = _unique_ranked(retrieved_ids)
    if not unique:
        return 0.0
    hits = sum(1 for rid in unique if rid in relevant_ids)
    return hits / len(unique)


def average_precision_at_k(
    relevant_ids: set[int],
    retrieved_ids: list[int],
) -> float:
    """Return binary average precision over the supplied ranked cutoff.

    The denominator is the total number of known relevant items, so relevant
    items not retrieved within the cutoff contribute zero. Duplicate retrieved
    IDs are ignored after their first occurrence.
    """

    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, rid in enumerate(_unique_ranked(retrieved_ids), start=1):
        if rid not in relevant_ids:
            continue
        hits += 1
        precision_sum += hits / rank
    return precision_sum / len(relevant_ids)


def ndcg_at_k(relevant_ids: Relevance, retrieved_ids: list[int]) -> float:
    """Return standard nDCG with binary or graded relevance labels."""

    unique = _unique_ranked(retrieved_ids)
    qrels = _relevance_scores(relevant_ids)
    if not qrels or not unique:
        return 0.0
    dcg = sum(
        (2.0 ** qrels.get(rid, 0.0) - 1.0) / math.log2(rank + 1)
        for rank, rid in enumerate(unique, start=1)
        if rid in qrels
    )
    ideal_scores = sorted(qrels.values(), reverse=True)[: len(unique)]
    ideal_dcg = sum(
        (2.0 ** score - 1.0) / math.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def compute_retrieval_metrics(
    relevant_ids: Relevance,
    retrieved_ids: list[int],
) -> dict[str, float]:
    """Compute deterministic binary metrics plus graded nDCG."""

    qrels = _relevance_scores(relevant_ids)
    binary_relevant = set(qrels)
    return {
        "hit": float(hit_at_k(binary_relevant, retrieved_ids)),
        "reciprocal_rank": reciprocal_rank(binary_relevant, retrieved_ids),
        "recall_at_k": recall_at_k(binary_relevant, retrieved_ids),
        "precision_at_k": precision_at_k(binary_relevant, retrieved_ids),
        "average_precision_at_k": average_precision_at_k(
            binary_relevant,
            retrieved_ids,
        ),
        "ndcg_at_k": ndcg_at_k(qrels, retrieved_ids),
    }
