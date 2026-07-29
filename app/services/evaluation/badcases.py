"""Deterministic badcase taxonomy and baseline/candidate diagnosis."""

from __future__ import annotations

from typing import Any

_DIAGNOSIS = {
    "document_miss": (
        "retrieval",
        "相关文档没有进入返回结果。",
        ["检查查询改写", "扩大文档召回", "检查文档过滤与权限策略"],
    ),
    "retrieval_miss": (
        "retrieval",
        "Top-K 中没有任何已标注相关片段。",
        ["检查 embedding 与分块", "尝试 hybrid 召回", "分析查询词与语料词汇差异"],
    ),
    "incomplete_recall": (
        "retrieval",
        "只召回了部分相关片段。",
        ["增大候选集", "检查分块边界", "补充 query expansion"],
    ),
    "relevant_passage_ranked_too_low": (
        "ranking",
        "相关片段存在但排序质量低。",
        ["检查融合权重", "比较 reranker", "查看逐阶段 ranking trace"],
    ),
    "chunk_boundary_failure": (
        "chunking",
        "命中了相关文档但没有命中标注支持片段，可能是分块边界问题。",
        ["调整 chunk size/overlap", "保留标题与段落边界", "增加邻块扩展"],
    ),
    "reranking_failure": (
        "ranking",
        "相关候选在融合阶段进入 Top-K，但被 reranker 降到 Top-K 之外。",
        ["回放 reranker 输入", "检查模型/权重", "增加 reranker 回归门禁"],
    ),
    "irrelevant_noisy_context": (
        "context",
        "返回上下文中无关片段比例过高。",
        ["收紧候选集", "增加 context relevance 过滤", "检查 query expansion"],
    ),
    "stale_source_used": (
        "freshness",
        "结果使用了过期、被替代或尚未生效的来源。",
        ["修复文档版本元数据", "阻断过期来源", "重建受 freshness policy 约束的索引"],
    ),
    "faithfulness_failure": (
        "generation",
        "答案歪曲、扩大或超出检索证据。",
        ["缩小结论范围", "重新生成并逐 claim 校验", "无充分证据时拒答"],
    ),
    "answer_irrelevant": (
        "generation",
        "答案没有直接覆盖用户问题。",
        ["检查任务理解", "增加问题要点覆盖检查", "重新规划回答结构"],
    ),
    "grounding_judge_unavailable": (
        "evaluator",
        "语义依据评估器不可用，不能把缺失值当作通过或 0 分。",
        ["恢复 judge 服务", "保留本次输入指纹后重跑", "阻断语义发布门"],
    ),
    "unsupported_generation": (
        "grounding",
        "至少一个事实性结论没有被证据支持。",
        ["补充检索", "删除或收窄结论", "明确标记推测或拒答"],
    ),
    "citation_does_not_entail_claim": (
        "citation",
        "引用存在，但引用内容并不支持所附结论。",
        ["替换引用", "拆分复合结论", "重新执行 claim-evidence 对齐"],
    ),
    "should_refuse_but_answered": (
        "refusal",
        "公开标注为不可回答，但系统仍给出了事实性答案。",
        ["提高拒答阈值", "检查空证据处理", "增加不可回答回归样本"],
    ),
    "could_answer_but_refused": (
        "refusal",
        "公开标注为可回答，但系统错误拒答。",
        ["检查召回缺失", "降低不必要拒答", "分析 grounding judge 误判"],
    ),
    "conflict_not_detected": (
        "conflict",
        "样本要求识别来源冲突，但系统未正确披露。",
        ["扩大冲突证据检索", "检查版本/权威策略", "并列呈现冲突双方"],
    ),
    "inference_not_labeled": (
        "grounding",
        "模型给出了推测性结论，但没有显式标记为推测。",
        ["强制推测前缀", "将事实与推测分栏", "不满足时移除结论"],
    ),
    "evaluator_disagreement": (
        "evaluator",
        "独立生成评估与 claim-level grounding 给出相反结论。",
        ["进入人工校准集", "比较 rubric 与输入", "不得直接作为发布通过证据"],
    ),
}


def classify_badcase(
    metrics: dict[str, float | None],
    *,
    task_type: str | None,
    faithfulness_threshold: float = 0.8,
    relevance_threshold: float = 0.8,
    ranking_threshold: float = 0.5,
    answerable: bool | None = None,
    citation_report: dict[str, Any] | None = None,
    retrieval_trace: list[dict[str, Any]] | None = None,
    relevant_chunk_ids: list[int] | None = None,
    relevant_document_ids: list[int] | None = None,
) -> list[str]:
    """Return stable machine-readable root-cause categories."""

    categories: list[str] = []
    trace = retrieval_trace or []
    retrieved_document_ids = {
        int(hit["document_id"])
        for hit in trace
        if isinstance(hit, dict)
        and isinstance(hit.get("document_id"), int)
    }
    if relevant_document_ids and not (
        set(relevant_document_ids) & retrieved_document_ids
    ):
        categories.append("document_miss")
    if metrics.get("hit_at_k") == 0:
        categories.append("retrieval_miss")
    recall = metrics.get("recall_at_k")
    if recall is not None and recall < 1.0:
        categories.append("incomplete_recall")
    ndcg = metrics.get("ndcg_at_k")
    if ndcg is not None and ndcg < ranking_threshold:
        categories.append("relevant_passage_ranked_too_low")
    precision = metrics.get("precision_at_k")
    if precision is not None and precision < 0.2:
        categories.append("irrelevant_noisy_context")
    if (
        metrics.get("hit_at_k") == 0
        and relevant_chunk_ids
        and retrieved_document_ids
        and not relevant_document_ids
    ):
        categories.append("chunk_boundary_failure")
    relevant_chunks = set(relevant_chunk_ids or [])
    if any(
        int(hit.get("chunk_index", -1)) in relevant_chunks
        and int(hit.get("fused_rank", 10**9))
        <= max(len(trace), 1)
        and int(hit.get("final_rank", 10**9)) > max(len(trace), 1)
        for hit in trace
        if isinstance(hit, dict)
    ):
        categories.append("reranking_failure")
    if any(
        str(hit.get("source_status") or "unknown")
        in {"expired", "superseded", "not_yet_effective"}
        for hit in trace
        if isinstance(hit, dict)
    ):
        categories.append("stale_source_used")

    retrieval_only = task_type in {"retrieval", "document_retrieval"}
    if not retrieval_only:
        faithfulness = metrics.get("faithfulness")
        if faithfulness is not None and faithfulness < faithfulness_threshold:
            categories.append("faithfulness_failure")
        relevance = metrics.get("answer_relevance")
        if relevance is not None and relevance < relevance_threshold:
            categories.append("answer_irrelevant")
        groundedness = metrics.get("groundedness")
        if groundedness is None:
            categories.append("grounding_judge_unavailable")
        elif groundedness < 1.0:
            categories.append("unsupported_generation")
        citation_correctness = metrics.get("citation_correctness")
        if (
            citation_correctness is not None
            and citation_correctness < 1.0
        ):
            categories.append("citation_does_not_entail_claim")
        refused = bool((citation_report or {}).get("refused"))
        if answerable is False and not refused:
            categories.append("should_refuse_but_answered")
        elif answerable is True and refused:
            categories.append("could_answer_but_refused")
        conflict = metrics.get("conflict_detection_accuracy")
        if conflict is not None and conflict < 1.0:
            categories.append("conflict_not_detected")
        if any(
            claim.get("claim_type") == "inference"
            and not claim.get("explicit_inference")
            for claim in (citation_report or {}).get("claims") or []
            if isinstance(claim, dict)
        ):
            categories.append("inference_not_labeled")
        if (
            faithfulness is not None
            and groundedness is not None
            and (
                (faithfulness >= faithfulness_threshold and groundedness < 1.0)
                or (
                    faithfulness < faithfulness_threshold
                    and groundedness == 1.0
                )
            )
        ):
            categories.append("evaluator_disagreement")
    return list(dict.fromkeys(categories))


def diagnose_badcases(categories: list[str]) -> list[dict[str, Any]]:
    """Attach stable explanations and next actions to machine categories."""

    return [
        {
            "category": category,
            "layer": _DIAGNOSIS[category][0],
            "explanation": _DIAGNOSIS[category][1],
            "suggested_actions": list(_DIAGNOSIS[category][2]),
        }
        for category in categories
        if category in _DIAGNOSIS
    ]


def compare_badcase_sets(
    baseline: dict[int, list[str]],
    candidate: dict[int, list[str]],
) -> dict[str, list[dict[str, Any]] | int]:
    """Classify newly introduced, fixed, and persistent sample failures."""

    introduced: list[dict[str, Any]] = []
    fixed: list[dict[str, Any]] = []
    persistent: list[dict[str, Any]] = []
    unchanged_passed = 0
    for sample_id in sorted(set(baseline) | set(candidate)):
        before = set(baseline.get(sample_id) or [])
        after = set(candidate.get(sample_id) or [])
        row = {
            "sample_id": sample_id,
            "baseline_categories": sorted(before),
            "candidate_categories": sorted(after),
            "introduced_categories": sorted(after - before),
            "fixed_categories": sorted(before - after),
        }
        if not before and after:
            introduced.append(row)
        elif before and not after:
            fixed.append(row)
        elif before and after:
            persistent.append(row)
        else:
            unchanged_passed += 1
    return {
        "newly_introduced": introduced,
        "fixed": fixed,
        "persistent": persistent,
        "unchanged_passed_count": unchanged_passed,
    }


def source_links(retrieval_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build exact document/chunk jump targets without exposing source text."""

    links: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for hit in retrieval_trace:
        try:
            key = (int(hit["document_id"]), int(hit["chunk_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        link = {
            "citation_id": hit.get("citation_id") or f"D{key[0]}:C{key[1]}",
            "document_id": key[0],
            "chunk_index": key[1],
            "document_name": hit.get("document_name"),
            "source_version": hit.get("source_version"),
            "source_status": hit.get("source_status", "unknown"),
            "jump_url": hit.get("jump_url")
            or f"/documents/{key[0]}/chunks/{key[1]}",
        }
        link.update(
            {
                field: hit[field]
                for field in (
                    "page_start",
                    "page_end",
                    "paragraph_start",
                    "paragraph_end",
                    "char_start",
                    "char_end",
                    "locator_version",
                )
                if hit.get(field) is not None
            }
        )
        links.append(link)
    return links


__all__ = [
    "classify_badcase",
    "compare_badcase_sets",
    "diagnose_badcases",
    "source_links",
]
