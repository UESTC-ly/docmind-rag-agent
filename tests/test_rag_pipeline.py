"""Versioned RAG pipeline registry, execution, and reproducibility tests."""

import pytest

from app.services import rag_pipeline
from app.services.rag_pipeline import (
    AsyncRetrievalRuntime,
    CandidateSet,
    PipelineSpec,
    RetrieverPlugin,
    SyncRetrievalRuntime,
    create_component_registry,
    execute_async_pipeline,
    execute_sync_pipeline,
    pipeline_presets,
    resolve_pipeline_spec,
)


def _sync_runtime(calls: list[str]) -> SyncRetrievalRuntime:
    def embed(query):
        calls.append(f"embed:{query}")
        return [0.1]

    def dense(vector, user_id, limit, document_id):
        calls.append(f"dense:{limit}")
        return [
            {
                "document_id": 1,
                "chunk_index": 0,
                "content": "dense result",
                "score": 0.9,
            }
        ]

    def keyword(query, user_id, document_id, limit):
        calls.append(f"keyword:{limit}")
        return [
            {
                "document_id": 1,
                "chunk_index": 1,
                "content": "keyword result",
                "score": 0.0,
                "keyword_score": 2.0,
            }
        ]

    return SyncRetrievalRuntime(embed, dense, keyword)


def test_presets_cover_dense_hybrid_and_reranked_variants():
    presets = pipeline_presets()
    assert presets["dense"].retriever == "dense"
    assert presets["dense"].reranker == "off"
    assert presets["hybrid"].fusion == "rrf"
    assert presets["hybrid-rerank"].reranker == "local"


def test_pipeline_fingerprint_changes_with_output_affecting_config():
    original = pipeline_presets()["hybrid"]
    modified = PipelineSpec.from_dict(
        {**original.to_dict(), "rrf_k": original.rrf_k + 1}
    )
    renamed = PipelineSpec.from_dict(
        {**original.to_dict(), "label": "仅修改展示名称"}
    )
    assert original.fingerprint != modified.fingerprint
    assert original.fingerprint == renamed.fingerprint


def test_sync_pipeline_executes_selected_components_and_returns_trace():
    calls: list[str] = []
    execution = execute_sync_pipeline(
        spec="hybrid-rerank",
        user_id=7,
        query="DocMind",
        document_id=3,
        top_k=2,
        runtime=_sync_runtime(calls),
    )
    assert any(call.startswith("keyword:") for call in calls)
    assert len(execution.hits) == 2
    assert execution.hits[0]["citation_id"].startswith("D1:C")
    assert execution.hits[0]["pipeline_id"] == "hybrid-rerank"
    assert execution.trace[0]["component"] == "hybrid"
    assert execution.trace[-1]["component"] == "evidence"
    assert "[D1:C" in execution.context()


def test_dense_pipeline_skips_keyword_component():
    calls: list[str] = []
    execution = execute_sync_pipeline(
        spec="dense",
        user_id=7,
        query="DocMind",
        document_id=None,
        top_k=1,
        runtime=_sync_runtime(calls),
    )
    assert not any(call.startswith("keyword:") for call in calls)
    assert execution.hits[0]["retrieval_sources"] == ["dense"]
    assert execution.hits[0]["reranker"] == "off"


@pytest.mark.asyncio
async def test_async_pipeline_uses_same_contract():
    async def embed(query):
        return [0.1]

    async def dense(vector, user_id, limit, document_id):
        return [
            {
                "document_id": 2,
                "chunk_index": 3,
                "content": "async",
                "score": 0.7,
            }
        ]

    async def keyword(query, user_id, document_id, limit):
        return []

    execution = await execute_async_pipeline(
        spec="dense",
        user_id=1,
        query="q",
        document_id=None,
        top_k=1,
        runtime=AsyncRetrievalRuntime(embed, dense, keyword),
    )
    assert execution.hits[0]["citation_id"] == "D2:C3"
    assert execution.spec.fingerprint == execution.fingerprint


def test_custom_retriever_can_be_registered_without_editing_executor():
    registry = create_component_registry()

    def sync_custom(request, spec, runtime):
        return CandidateSet(
            dense=[
                {
                    "document_id": 9,
                    "chunk_index": 8,
                    "content": "plugin",
                    "score": 1.0,
                }
            ],
            keyword=[],
        )

    async def async_custom(request, spec, runtime):
        return sync_custom(request, spec, None)

    registry.register_retriever(
        "custom",
        RetrieverPlugin(sync=sync_custom, async_=async_custom),
    )
    spec = PipelineSpec.from_dict(
        {
            **pipeline_presets()["dense"].to_dict(),
            "id": "custom-test",
            "retriever": "custom",
        }
    )
    execution = execute_sync_pipeline(
        spec=spec,
        user_id=1,
        query="q",
        document_id=None,
        top_k=1,
        runtime=_sync_runtime([]),
        registry=registry,
    )
    assert execution.hits[0]["content"] == "plugin"


def test_index_manifest_mismatch_fails_before_retrieval(monkeypatch):
    spec = pipeline_presets()["dense"]
    monkeypatch.setattr(rag_pipeline.settings, "embedding_dim", spec.embedding_dim + 1)
    with pytest.raises(ValueError, match="index manifest"):
        execute_sync_pipeline(
            spec=spec,
            user_id=1,
            query="q",
            document_id=None,
            top_k=1,
            runtime=_sync_runtime([]),
        )


def test_unknown_pipeline_and_fields_are_rejected():
    with pytest.raises(ValueError, match="unknown RAG pipeline"):
        resolve_pipeline_spec("missing")
    with pytest.raises(ValueError, match="unknown pipeline fields"):
        PipelineSpec.from_dict(
            {**pipeline_presets()["dense"].to_dict(), "surprise": True}
        )
