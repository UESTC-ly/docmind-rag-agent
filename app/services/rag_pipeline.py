"""Versioned, pluggable RAG pipeline execution.

The pipeline keeps retrieval I/O behind injected runtimes so browser chat,
sync Agent Skills, and offline evaluation execute the same component contract
without introducing a third retrieval implementation.  Exact component and
index settings are serializable and fingerprinted for reproducible runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from app.config import settings
from app.services.document_policy import DOCUMENT_POLICY_VERSION
from app.services.evidence import attach_citation_ids, build_evidence_context
from app.services.reranker import rerank
from app.services.retrieval import fuse_dense_and_keyword

Hit = dict[str, Any]
Embedding = list[float]


@dataclass(frozen=True)
class PipelineSpec:
    """Immutable retrieval/generation-context configuration snapshot."""

    id: str
    label: str
    description: str
    retriever: str
    fusion: str
    reranker: str
    context_builder: str
    top_k: int
    dense_candidates: int
    keyword_candidates: int
    rrf_k: int
    reranker_candidate_limit: int
    reranker_rrf_weight: float
    reranker_dense_weight: float
    reranker_keyword_weight: float
    reranker_lexical_weight: float
    reranker_provider_fingerprint: str | None
    reranker_model: str | None
    embedding_model: str
    embedding_dim: int
    vector_collection: str
    generation_model: str
    document_policy_version: str = DOCUMENT_POLICY_VERSION
    generation_prompt_version: str = "citation-v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported pipeline schema_version")
        for name, value in (
            ("id", self.id),
            ("retriever", self.retriever),
            ("fusion", self.fusion),
            ("reranker", self.reranker),
            ("context_builder", self.context_builder),
            ("embedding_model", self.embedding_model),
            ("vector_collection", self.vector_collection),
            ("generation_model", self.generation_model),
            ("document_policy_version", self.document_policy_version),
        ):
            if not value or len(value) > 128:
                raise ValueError(f"invalid pipeline {name}")
        if not 1 <= self.top_k <= 100:
            raise ValueError("pipeline top_k must be between 1 and 100")
        if not 1 <= self.dense_candidates <= 1000:
            raise ValueError("dense_candidates must be between 1 and 1000")
        if not 0 <= self.keyword_candidates <= 1000:
            raise ValueError("keyword_candidates must be between 0 and 1000")
        if not 1 <= self.rrf_k <= 10000:
            raise ValueError("rrf_k must be between 1 and 10000")
        if not 1 <= self.reranker_candidate_limit <= 1000:
            raise ValueError(
                "reranker_candidate_limit must be between 1 and 1000"
            )
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        for optional_name, optional_value in (
            (
                "reranker_provider_fingerprint",
                self.reranker_provider_fingerprint,
            ),
            ("reranker_model", self.reranker_model),
        ):
            if optional_value is not None and len(optional_value) > 500:
                raise ValueError(f"invalid pipeline {optional_name}")
        weights = self.reranker_weights
        if any(value < 0 for value in weights.values()):
            raise ValueError("reranker weights cannot be negative")
        if sum(weights.values()) <= 0:
            raise ValueError("at least one reranker weight must be positive")

    @property
    def reranker_weights(self) -> dict[str, float]:
        return {
            "rrf": self.reranker_rrf_weight,
            "dense": self.reranker_dense_weight,
            "keyword": self.reranker_keyword_weight,
            "lexical": self.reranker_lexical_weight,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def execution_dict(self) -> dict[str, Any]:
        """Return fields that affect output, excluding display metadata."""
        data = self.to_dict()
        data.pop("id", None)
        data.pop("label", None)
        data.pop("description", None)
        return data

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.execution_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PipelineSpec":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                "unknown pipeline fields: " + ", ".join(sorted(unknown))
            )
        try:
            return cls(**dict(raw))
        except TypeError as exc:
            raise ValueError(f"invalid pipeline spec: {exc}") from exc


@dataclass(frozen=True)
class RetrievalRequest:
    user_id: int
    query: str
    document_id: int | None
    top_k: int


@dataclass(frozen=True)
class CandidateSet:
    dense: list[Hit]
    keyword: list[Hit]


@dataclass(frozen=True)
class SyncRetrievalRuntime:
    embed_query: Callable[[str], Embedding]
    dense_search: Callable[[Embedding, int, int, int | None], list[Hit]]
    keyword_search: Callable[[str, int, int | None, int], list[Hit]]


@dataclass(frozen=True)
class AsyncRetrievalRuntime:
    embed_query: Callable[[str], Awaitable[Embedding]]
    dense_search: Callable[
        [Embedding, int, int, int | None],
        Awaitable[list[Hit]],
    ]
    keyword_search: Callable[
        [str, int, int | None, int],
        Awaitable[list[Hit]],
    ]


SyncRetriever = Callable[
    [RetrievalRequest, PipelineSpec, SyncRetrievalRuntime],
    CandidateSet,
]
AsyncRetriever = Callable[
    [RetrievalRequest, PipelineSpec, AsyncRetrievalRuntime],
    Awaitable[CandidateSet],
]
Fusion = Callable[[CandidateSet, PipelineSpec, int], list[Hit]]
Reranker = Callable[[str, list[Hit], PipelineSpec, int], list[Hit]]
ContextBuilder = Callable[[list[Hit]], str]


@dataclass(frozen=True)
class RetrieverPlugin:
    sync: SyncRetriever
    async_: AsyncRetriever


class RagComponentRegistry:
    """Registry used by built-ins and external retrieval method plugins."""

    def __init__(self) -> None:
        self.retrievers: dict[str, RetrieverPlugin] = {}
        self.fusions: dict[str, Fusion] = {}
        self.rerankers: dict[str, Reranker] = {}
        self.context_builders: dict[str, ContextBuilder] = {}
        self.pipeline_factories: dict[str, Callable[[], PipelineSpec]] = {}

    @staticmethod
    def _register(target: dict, name: str, component: Any) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("component name cannot be empty")
        if normalized in target:
            raise ValueError(f"duplicate RAG component: {normalized}")
        target[normalized] = component

    def register_retriever(self, name: str, plugin: RetrieverPlugin) -> None:
        self._register(self.retrievers, name, plugin)

    def register_fusion(self, name: str, component: Fusion) -> None:
        self._register(self.fusions, name, component)

    def register_reranker(self, name: str, component: Reranker) -> None:
        self._register(self.rerankers, name, component)

    def register_context_builder(
        self,
        name: str,
        component: ContextBuilder,
    ) -> None:
        self._register(self.context_builders, name, component)

    def register_pipeline(
        self,
        name: str,
        factory: Callable[[], PipelineSpec],
    ) -> None:
        if not callable(factory):
            raise TypeError("pipeline factory must be callable")
        self._register(self.pipeline_factories, name, factory)

    @staticmethod
    def _require(target: dict, kind: str, name: str):
        try:
            return target[name]
        except KeyError as exc:
            raise ValueError(f"unknown RAG {kind} component: {name}") from exc

    def retriever(self, name: str) -> RetrieverPlugin:
        return self._require(self.retrievers, "retriever", name)

    def fusion(self, name: str) -> Fusion:
        return self._require(self.fusions, "fusion", name)

    def reranker(self, name: str) -> Reranker:
        return self._require(self.rerankers, "reranker", name)

    def context_builder(self, name: str) -> ContextBuilder:
        return self._require(self.context_builders, "context builder", name)

    def plugin_pipelines(self) -> dict[str, PipelineSpec]:
        specs: dict[str, PipelineSpec] = {}
        for name, factory in self.pipeline_factories.items():
            spec = factory()
            if not isinstance(spec, PipelineSpec):
                raise TypeError(f"RAG pipeline factory {name} did not return PipelineSpec")
            if spec.id != name:
                raise ValueError(
                    f"RAG pipeline factory {name} returned mismatched id {spec.id}"
                )
            validate_pipeline_components(spec, self)
            specs[name] = spec
        return specs


@dataclass(frozen=True)
class PipelineExecution:
    spec: PipelineSpec
    hits: list[Hit]
    trace: list[dict[str, Any]]

    @property
    def fingerprint(self) -> str:
        return self.spec.fingerprint

    def context(self, registry: RagComponentRegistry | None = None) -> str:
        active = registry or component_registry
        return active.context_builder(self.spec.context_builder)(self.hits)


def _dense_limit(request: RetrievalRequest, spec: PipelineSpec) -> int:
    return max(
        request.top_k,
        min(spec.dense_candidates, spec.reranker_candidate_limit),
    )


def _keyword_limit(spec: PipelineSpec) -> int:
    return min(spec.keyword_candidates, spec.reranker_candidate_limit)


def _dense_sync(
    request: RetrievalRequest,
    spec: PipelineSpec,
    runtime: SyncRetrievalRuntime,
) -> CandidateSet:
    vector = runtime.embed_query(request.query)
    dense = runtime.dense_search(
        vector,
        request.user_id,
        _dense_limit(request, spec),
        request.document_id,
    )
    return CandidateSet(dense=dense, keyword=[])


async def _dense_async(
    request: RetrievalRequest,
    spec: PipelineSpec,
    runtime: AsyncRetrievalRuntime,
) -> CandidateSet:
    vector = await runtime.embed_query(request.query)
    dense = await runtime.dense_search(
        vector,
        request.user_id,
        _dense_limit(request, spec),
        request.document_id,
    )
    return CandidateSet(dense=dense, keyword=[])


def _hybrid_sync(
    request: RetrievalRequest,
    spec: PipelineSpec,
    runtime: SyncRetrievalRuntime,
) -> CandidateSet:
    dense = _dense_sync(request, spec, runtime).dense
    keyword = runtime.keyword_search(
        request.query,
        request.user_id,
        request.document_id,
        _keyword_limit(spec),
    )
    return CandidateSet(dense=dense, keyword=keyword)


async def _hybrid_async(
    request: RetrievalRequest,
    spec: PipelineSpec,
    runtime: AsyncRetrievalRuntime,
) -> CandidateSet:
    dense = (await _dense_async(request, spec, runtime)).dense
    keyword = await runtime.keyword_search(
        request.query,
        request.user_id,
        request.document_id,
        _keyword_limit(spec),
    )
    return CandidateSet(dense=dense, keyword=keyword)


def _dense_fusion(
    candidates: CandidateSet,
    spec: PipelineSpec,
    limit: int,
) -> list[Hit]:
    output: list[Hit] = []
    for rank, raw in enumerate(candidates.dense[:limit], start=1):
        hit = dict(raw)
        dense_score = float(hit.get("score", 0.0) or 0.0)
        hit.update(
            dense_rank=rank,
            dense_score=dense_score,
            retrieval_sources=["dense"],
            rrf_score=1.0 / (spec.rrf_k + rank),
            fused_rank=rank,
        )
        output.append(hit)
    return output


def _rrf_fusion(
    candidates: CandidateSet,
    spec: PipelineSpec,
    limit: int,
) -> list[Hit]:
    return fuse_dense_and_keyword(
        candidates.dense,
        candidates.keyword,
        limit,
        rrf_k=spec.rrf_k,
    )


def _rerank_with_mode(mode: str) -> Reranker:
    def _component(
        query: str,
        candidates: list[Hit],
        spec: PipelineSpec,
        top_k: int,
    ) -> list[Hit]:
        return rerank(
            query,
            candidates,
            top_k,
            mode=mode,
            candidate_limit=spec.reranker_candidate_limit,
            weights=spec.reranker_weights,
        )

    return _component


def create_component_registry() -> RagComponentRegistry:
    registry = RagComponentRegistry()
    registry.register_retriever(
        "dense",
        RetrieverPlugin(sync=_dense_sync, async_=_dense_async),
    )
    registry.register_retriever(
        "hybrid",
        RetrieverPlugin(sync=_hybrid_sync, async_=_hybrid_async),
    )
    registry.register_fusion("dense", _dense_fusion)
    registry.register_fusion("rrf", _rrf_fusion)
    registry.register_reranker("off", _rerank_with_mode("off"))
    registry.register_reranker("local", _rerank_with_mode("local"))
    registry.register_reranker("http", _rerank_with_mode("http"))
    registry.register_context_builder("evidence", build_evidence_context)
    return registry


component_registry = create_component_registry()


def _provider_fingerprint(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _base_spec(**overrides: Any) -> PipelineSpec:
    values: dict[str, Any] = {
        "id": "configured",
        "label": "当前配置",
        "description": "使用部署环境当前配置的检索管线。",
        "retriever": settings.retrieval_mode,
        "fusion": "rrf" if settings.retrieval_mode == "hybrid" else "dense",
        "reranker": settings.reranker_mode,
        "context_builder": "evidence",
        "top_k": settings.retrieval_top_k,
        "dense_candidates": settings.dense_candidates,
        "keyword_candidates": settings.keyword_candidates,
        "rrf_k": settings.rrf_k,
        "reranker_candidate_limit": settings.reranker_candidate_limit,
        "reranker_rrf_weight": settings.reranker_rrf_weight,
        "reranker_dense_weight": settings.reranker_dense_weight,
        "reranker_keyword_weight": settings.reranker_keyword_weight,
        "reranker_lexical_weight": settings.reranker_lexical_weight,
        "reranker_provider_fingerprint": _provider_fingerprint(
            settings.reranker_http_url
        ),
        "reranker_model": settings.reranker_http_model,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "vector_collection": settings.qdrant_collection,
        "generation_model": settings.chat_model,
    }
    values.update(overrides)
    return PipelineSpec(**values)


_plugin_load_lock = threading.RLock()
_loaded_plugin_modules: set[str] = set()
_loading_plugin_modules: set[str] = set()


def _ensure_configured_plugins_loaded() -> None:
    with _plugin_load_lock:
        pending = [
            name
            for name in settings.rag_plugin_module_list
            if name not in _loaded_plugin_modules
            and name not in _loading_plugin_modules
        ]
        if not pending:
            return

        # Registration is a process-start operation, but multiple request
        # threads can reach the pipeline catalog simultaneously.  Snapshot all
        # mutable registry maps so a failed module cannot leave components
        # behind and then fail with "duplicate component" on the next retry.
        snapshots = {
            "retrievers": dict(component_registry.retrievers),
            "fusions": dict(component_registry.fusions),
            "rerankers": dict(component_registry.rerankers),
            "context_builders": dict(component_registry.context_builders),
            "pipeline_factories": dict(component_registry.pipeline_factories),
        }
        _loading_plugin_modules.update(pending)
        try:
            from app.services.rag_plugins import load_rag_plugin_modules

            load_rag_plugin_modules(component_registry, pending)
        except Exception:
            for attribute, snapshot in snapshots.items():
                target = getattr(component_registry, attribute)
                target.clear()
                target.update(snapshot)
            raise
        else:
            _loaded_plugin_modules.update(pending)
        finally:
            _loading_plugin_modules.difference_update(pending)


def validate_pipeline_components(
    spec: PipelineSpec,
    registry: RagComponentRegistry | None = None,
) -> None:
    active = registry or component_registry
    active.retriever(spec.retriever)
    active.fusion(spec.fusion)
    active.reranker(spec.reranker)
    active.context_builder(spec.context_builder)


def pipeline_presets(
    registry: RagComponentRegistry | None = None,
) -> dict[str, PipelineSpec]:
    """Return fresh snapshots so settings changes never mutate old runs."""
    active = registry or component_registry
    if registry is None:
        _ensure_configured_plugins_loaded()
    presets = {
        "configured": _base_spec(),
        "dense": _base_spec(
            id="dense",
            label="Dense 基线",
            description="仅向量召回，不使用关键词融合和二阶段排序。",
            retriever="dense",
            fusion="dense",
            reranker="off",
        ),
        "hybrid": _base_spec(
            id="hybrid",
            label="Hybrid 基线",
            description="向量与关键词召回，经 RRF 融合，不做二阶段排序。",
            retriever="hybrid",
            fusion="rrf",
            reranker="off",
        ),
        "hybrid-rerank": _base_spec(
            id="hybrid-rerank",
            label="Hybrid + Rerank",
            description="向量与关键词 RRF 融合，再使用本地确定性 reranker。",
            retriever="hybrid",
            fusion="rrf",
            reranker="local",
        ),
    }
    plugin_specs = active.plugin_pipelines()
    conflicts = sorted(set(presets) & set(plugin_specs))
    if conflicts:
        raise ValueError(
            "RAG plugin pipeline IDs conflict with built-ins: "
            + ", ".join(conflicts)
        )
    presets.update(plugin_specs)
    return presets


def resolve_pipeline_spec(
    value: PipelineSpec | Mapping[str, Any] | str | None,
) -> PipelineSpec:
    if value is None or value == "default":
        return pipeline_presets()["configured"]
    if isinstance(value, PipelineSpec):
        return value
    if isinstance(value, str):
        try:
            return pipeline_presets()[value]
        except KeyError as exc:
            raise ValueError(f"unknown RAG pipeline: {value}") from exc
    return PipelineSpec.from_dict(value)


def ensure_index_compatible(spec: PipelineSpec) -> None:
    """Fail queued reproducible runs if their runtime contract became stale."""
    current = {
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "vector_collection": settings.qdrant_collection,
    }
    expected = {
        "embedding_model": spec.embedding_model,
        "embedding_dim": spec.embedding_dim,
        "vector_collection": spec.vector_collection,
    }
    if current != expected:
        raise ValueError(
            "pipeline index manifest does not match the active embedding/index "
            "configuration; rebuild or select a compatible index"
        )
    if spec.generation_model != settings.chat_model:
        raise ValueError(
            "pipeline generation model does not match the active chat model"
        )
    if spec.reranker == "http" and (
        spec.reranker_provider_fingerprint
        != _provider_fingerprint(settings.reranker_http_url)
        or spec.reranker_model != settings.reranker_http_model
    ):
        raise ValueError(
            "pipeline HTTP reranker provider/model does not match the active "
            "runtime configuration"
        )


def _request(
    spec: PipelineSpec,
    user_id: int,
    query: str,
    document_id: int | None,
    top_k: int | None,
) -> RetrievalRequest:
    limit = top_k or spec.top_k
    if not 1 <= limit <= 100:
        raise ValueError("top_k must be between 1 and 100")
    if not query.strip():
        raise ValueError("retrieval query cannot be empty")
    return RetrievalRequest(
        user_id=user_id,
        query=query,
        document_id=document_id,
        top_k=limit,
    )


def _finish_execution(
    request: RetrievalRequest,
    spec: PipelineSpec,
    candidates: CandidateSet,
    registry: RagComponentRegistry,
) -> PipelineExecution:
    fusion_limit = max(
        request.top_k,
        min(
            spec.reranker_candidate_limit,
            max(len(candidates.dense) + len(candidates.keyword), request.top_k),
        ),
    )
    fused = registry.fusion(spec.fusion)(candidates, spec, fusion_limit)
    ranked = registry.reranker(spec.reranker)(
        request.query,
        fused,
        spec,
        request.top_k,
    )
    fingerprint = spec.fingerprint
    hits: list[Hit] = []
    for raw in attach_citation_ids(ranked):
        hit = dict(raw)
        hit["pipeline_id"] = spec.id
        hit["pipeline_fingerprint"] = fingerprint
        hits.append(hit)
    trace: list[dict[str, Any]] = [
        {
            "stage": "retrieve",
            "component": spec.retriever,
            "dense_candidates": len(candidates.dense),
            "keyword_candidates": len(candidates.keyword),
        },
        {
            "stage": "fuse",
            "component": spec.fusion,
            "output_candidates": len(fused),
        },
        {
            "stage": "rerank",
            "component": spec.reranker,
            "output_candidates": len(hits),
        },
        {
            "stage": "context",
            "component": spec.context_builder,
            "generation_prompt_version": spec.generation_prompt_version,
        },
    ]
    return PipelineExecution(spec=spec, hits=hits, trace=trace)


def execute_sync_pipeline(
    *,
    spec: PipelineSpec | Mapping[str, Any] | str | None,
    user_id: int,
    query: str,
    document_id: int | None,
    top_k: int | None,
    runtime: SyncRetrievalRuntime,
    registry: RagComponentRegistry | None = None,
) -> PipelineExecution:
    resolved = resolve_pipeline_spec(spec)
    ensure_index_compatible(resolved)
    request = _request(resolved, user_id, query, document_id, top_k)
    active = registry or component_registry
    validate_pipeline_components(resolved, active)
    candidates = active.retriever(resolved.retriever).sync(
        request,
        resolved,
        runtime,
    )
    return _finish_execution(request, resolved, candidates, active)


async def execute_async_pipeline(
    *,
    spec: PipelineSpec | Mapping[str, Any] | str | None,
    user_id: int,
    query: str,
    document_id: int | None,
    top_k: int | None,
    runtime: AsyncRetrievalRuntime,
    registry: RagComponentRegistry | None = None,
) -> PipelineExecution:
    resolved = resolve_pipeline_spec(spec)
    ensure_index_compatible(resolved)
    request = _request(resolved, user_id, query, document_id, top_k)
    active = registry or component_registry
    validate_pipeline_components(resolved, active)
    candidates = await active.retriever(resolved.retriever).async_(
        request,
        resolved,
        runtime,
    )
    return await asyncio.to_thread(
        _finish_execution,
        request,
        resolved,
        candidates,
        active,
    )
