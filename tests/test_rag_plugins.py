"""Trusted RAG plugin loading and pipeline catalog contracts."""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

import pytest

from app.services import rag_pipeline
from app.services.rag_pipeline import (
    PipelineSpec,
    create_component_registry,
    pipeline_presets,
)
from app.services.rag_plugins import load_rag_plugin_modules


def test_explicit_plugin_module_registers_component_and_pipeline(monkeypatch):
    module_name = "tests.fake_docmind_rag_plugin"
    module = ModuleType(module_name)

    def register(registry):
        registry.register_fusion(
            "plugin_passthrough",
            lambda candidates, spec, limit: list(candidates.dense[:limit]),
        )
        base = pipeline_presets(registry=registry)["dense"].to_dict()
        base.update(
            id="plugin-dense",
            label="Plugin Dense",
            description="Plugin-provided pipeline",
            fusion="plugin_passthrough",
        )
        registry.register_pipeline(
            "plugin-dense",
            lambda: PipelineSpec.from_dict(base),
        )

    module.register = register
    monkeypatch.setitem(sys.modules, module_name, module)
    registry = create_component_registry()

    loaded = load_rag_plugin_modules(registry, [module_name])
    presets = pipeline_presets(registry=registry)

    assert loaded == [
        {
            "module": module_name,
            "status": "loaded",
            "pipelines": ["plugin-dense"],
        }
    ]
    assert registry.fusion("plugin_passthrough")
    assert presets["plugin-dense"].fusion == "plugin_passthrough"
    assert len(presets["plugin-dense"].fingerprint) == 64


def test_plugin_without_register_contract_is_rejected(monkeypatch):
    module_name = "tests.invalid_docmind_rag_plugin"
    monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))

    with pytest.raises(ValueError, match="register"):
        load_rag_plugin_modules(create_component_registry(), [module_name])


def _configure_global_plugin_test(monkeypatch, module_name):
    registry = create_component_registry()
    monkeypatch.setattr(rag_pipeline, "component_registry", registry)
    monkeypatch.setattr(rag_pipeline, "_loaded_plugin_modules", set())
    monkeypatch.setattr(rag_pipeline, "_loading_plugin_modules", set())
    monkeypatch.setattr(
        rag_pipeline.settings,
        "rag_plugin_modules",
        module_name,
    )
    return registry


def test_configured_plugin_loading_is_concurrent_and_idempotent(monkeypatch):
    module_name = "tests.concurrent_docmind_rag_plugin"
    module = ModuleType(module_name)
    calls = 0
    base = pipeline_presets(registry=create_component_registry())["dense"].to_dict()

    def register(registry):
        nonlocal calls
        calls += 1
        time.sleep(0.02)
        registry.register_fusion(
            "concurrent_passthrough",
            lambda candidates, spec, limit: list(candidates.dense[:limit]),
        )
        registry.register_pipeline(
            "concurrent-dense",
            lambda: PipelineSpec.from_dict(
                {
                    **base,
                    "id": "concurrent-dense",
                    "label": "Concurrent Dense",
                    "description": "Concurrency-safe plugin",
                    "fusion": "concurrent_passthrough",
                }
            ),
        )

    module.register = register
    monkeypatch.setitem(sys.modules, module_name, module)
    _configure_global_plugin_test(monkeypatch, module_name)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: pipeline_presets(), range(16)))

    assert calls == 1
    assert all("concurrent-dense" in presets for presets in results)
    assert rag_pipeline._loaded_plugin_modules == {module_name}
    assert not rag_pipeline._loading_plugin_modules


def test_failed_configured_plugin_rolls_back_and_can_retry(monkeypatch):
    module_name = "tests.retryable_docmind_rag_plugin"
    module = ModuleType(module_name)
    attempts = 0
    base = pipeline_presets(registry=create_component_registry())["dense"].to_dict()

    def register(registry):
        nonlocal attempts
        attempts += 1
        registry.register_fusion(
            "retry_passthrough",
            lambda candidates, spec, limit: list(candidates.dense[:limit]),
        )
        if attempts == 1:
            raise RuntimeError("temporary registration failure")
        registry.register_pipeline(
            "retry-dense",
            lambda: PipelineSpec.from_dict(
                {
                    **base,
                    "id": "retry-dense",
                    "label": "Retry Dense",
                    "description": "Retry-safe plugin",
                    "fusion": "retry_passthrough",
                }
            ),
        )

    module.register = register
    monkeypatch.setitem(sys.modules, module_name, module)
    registry = _configure_global_plugin_test(monkeypatch, module_name)

    with pytest.raises(RuntimeError, match="temporary registration failure"):
        pipeline_presets()

    assert "retry_passthrough" not in registry.fusions
    assert not rag_pipeline._loaded_plugin_modules
    assert not rag_pipeline._loading_plugin_modules

    presets = pipeline_presets()

    assert attempts == 2
    assert presets["retry-dense"].fusion == "retry_passthrough"
    assert rag_pipeline._loaded_plugin_modules == {module_name}
