"""检索质量指标——纯函数，无 I/O 依赖，全部可单元测试。

术语约定：
  relevant_ids  : set[int] — 该问题对应的正确 chunk_index 集合（ground truth）
  retrieved_ids : list[int] — 检索器按排名顺序返回的 chunk_index 列表（最多 top_k 个）

指标说明：
  hit_rate       — top-k 内是否命中至少一个相关块（0 / 1）
  reciprocal_rank — 第一个命中位置的倒数（未命中为 0）；多条样本取均值 = MRR
  recall_at_k    — 命中相关块数 / 总相关块数
  precision_at_k — 命中相关块数 / k
"""


def hit_at_k(relevant_ids: set[int], retrieved_ids: list[int]) -> int:
    """top-k 内是否命中至少一个相关块。返回 0 或 1。"""
    for rid in retrieved_ids:
        if rid in relevant_ids:
            return 1
    return 0


def reciprocal_rank(relevant_ids: set[int], retrieved_ids: list[int]) -> float:
    """第一个命中位置的倒数。未命中返回 0.0。

    例：retrieved = [3, 8, 1, 5]，relevant = {5, 8}
    → 第一个命中是位置 2（1-indexed）的 8，倒数 = 0.5
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(relevant_ids: set[int], retrieved_ids: list[int]) -> float:
    """命中相关块数 / 全部相关块数。relevant_ids 为空时返回 0.0。"""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(relevant_ids: set[int], retrieved_ids: list[int]) -> float:
    """命中相关块数 / 实际返回块数（k）。retrieved 为空时返回 0.0。"""
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return hits / len(retrieved_ids)


def compute_retrieval_metrics(
    relevant_ids: set[int],
    retrieved_ids: list[int],
) -> dict[str, float]:
    """一次性计算四项检索指标，返回字典，方便存库。"""
    return {
        "hit": float(hit_at_k(relevant_ids, retrieved_ids)),
        "reciprocal_rank": reciprocal_rank(relevant_ids, retrieved_ids),
        "recall_at_k": recall_at_k(relevant_ids, retrieved_ids),
        "precision_at_k": precision_at_k(relevant_ids, retrieved_ids),
    }
