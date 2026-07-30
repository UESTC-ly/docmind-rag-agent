"""HTTP Agent benchmark harness contracts."""

import importlib.util
import hashlib
import json
from pathlib import Path
from threading import Barrier

import argparse
import httpx
import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_agent.py"
    spec = importlib.util.spec_from_file_location("benchmark_agent", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_harness_executes_case_and_omits_download_payload():
    benchmark = _module()

    def handler(request: httpx.Request):
        assert request.url.path == "/agent/chat"
        return httpx.Response(
            200,
            json={
                "run_id": "run-1",
                "thread_id": "run-1",
                "conversation_id": 1,
                "status": "completed",
                "answer": "报告已生成",
                "plan": {
                    "status": "completed",
                    "steps": [{"id": "step-1", "status": "completed"}],
                },
                "trace": [{"skill": "generate_report", "ok": True}],
                "artifacts": [
                    {
                        "type": "report",
                        "verification": {"passed": True},
                        "download": {"content": "large-payload", "filename": "x.md"},
                    }
                ],
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
    )
    report = benchmark.run_cases(
        client,
        [
            {
                "id": "report",
                "request": {"message": "生成报告"},
                "expected": {
                    "required_skills": ["generate_report"],
                    "required_artifact_types": ["report"],
                    "require_verified_artifact": True,
                },
            }
        ],
    )
    client.close()

    assert report["contract"] == "agent_benchmark_v3"
    assert report["release_eligible"] is False
    assert report["metrics"]["task_success_rate"] == 1.0
    assert report["metrics"]["verified_task_completion_rate"] == 1.0
    assert report["performance"]["configured_concurrency"] == 1
    assert report["performance"]["max_observed_in_flight"] == 1
    assert report["performance"]["provider_usage"]["status"] == "unavailable"
    assert report["performance"]["provider_cost"]["status"] == "unavailable"
    download = report["evidence"][0]["result"]["artifacts"][0]["download"]
    assert "content" not in download
    assert download["content_bytes_omitted"] == len("large-payload")
    assert download["content_sha256"] == hashlib.sha256(
        b"large-payload"
    ).hexdigest()


def test_agent_request_timeout_is_positive_and_long_task_safe():
    benchmark = _module()

    assert benchmark.DEFAULT_AGENT_REQUEST_TIMEOUT_SECONDS == 600.0
    assert benchmark._positive_timeout_seconds("720") == 720.0
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._positive_timeout_seconds("0")
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._positive_timeout_seconds("not-a-number")


def test_public_scenario_suite_requires_auditable_provenance(tmp_path):
    benchmark = _module()
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "contract": "public_agent_scenarios_v2",
                "provenance": {
                    "source_name": "CMRC 2018",
                    "source_uri": "https://ymcui.com/cmrc2018/",
                    "source_version": "2018",
                    "license_name": "CC BY-SA 4.0",
                    "split": "dev",
                    "corpus_fingerprint": "a" * 64,
                    "source_snapshot_fingerprint": "b" * 64,
                    "transform_spec": {"contract": "fixture_v1"},
                },
                "cases": [
                    {
                        "id": "public-1",
                        "source_sample_id": "cmrc-1",
                        "request": {"message": "生成证据报告"},
                        "expected": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite = benchmark.load_suite(suite_path)

    assert suite["provenance"]["source_name"] == "CMRC 2018"
    assert suite["cases"][0]["source_sample_id"] == "cmrc-1"
    assert len(suite["suite_fingerprint"]) == 64


@pytest.mark.parametrize(
    ("filename", "source_name", "case_count"),
    [
        ("scenarios.ms_marco_v21.json", "MS MARCO", 2),
        ("scenarios.cmrc2018.json", "CMRC 2018", 2),
    ],
)
def test_checked_in_public_agent_suites_are_fully_auditable(
    filename,
    source_name,
    case_count,
):
    benchmark = _module()
    suite = benchmark.load_suite(
        Path(__file__).parents[1] / "benchmarks" / "agent" / filename
    )

    assert suite["provenance"]["source_name"] == source_name
    assert len(suite["cases"]) == case_count
    assert len({item["id"] for item in suite["cases"]}) == case_count
    assert len(
        {item["source_sample_id"] for item in suite["cases"]}
    ) == case_count
    assert all(
        item["expected"]["answer_contains_any"] for item in suite["cases"]
    )
    assert all(
        item["expected"]["required_skill_order"]
        == [
            "select_evaluated_rag_pipeline",
            "generate_verified_research_report",
        ]
        for item in suite["cases"]
    )
    assert all(
        item["expected"]["require_evidence_gate"] is True
        and item["expected"]["require_verified_artifact"] is True
        and item["expected"]["max_interventions"] == 0
        for item in suite["cases"]
    )


def test_public_scenario_suite_rejects_duplicate_case_or_sample_ids(tmp_path):
    benchmark = _module()
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "contract": "public_agent_scenarios_v2",
                "provenance": {
                    "source_name": "public",
                    "source_uri": "https://example.test/public",
                    "source_version": "v1",
                    "license_name": "CC0",
                    "split": "test",
                    "corpus_fingerprint": "a" * 64,
                    "source_snapshot_fingerprint": "b" * 64,
                    "transform_spec": {"contract": "fixture_v1"},
                },
                "cases": [
                    {
                        "id": "same",
                        "source_sample_id": "sample-1",
                        "request": {"message": "task"},
                    },
                    {
                        "id": "same",
                        "source_sample_id": "sample-2",
                        "request": {"message": "task"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicates id"):
        benchmark.load_suite(suite_path)


def test_public_scenario_suite_rejects_placeholder_fingerprint(tmp_path):
    benchmark = _module()
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "contract": "public_agent_scenarios_v2",
                "provenance": {
                    "source_name": "public",
                    "source_uri": "https://example.test/public",
                    "source_version": "v1",
                    "license_name": "CC0",
                    "split": "test",
                    "corpus_fingerprint": "0" * 64,
                    "source_snapshot_fingerprint": "b" * 64,
                    "transform_spec": {"contract": "fixture_v1"},
                },
                "cases": [
                    {
                        "source_sample_id": "1",
                        "request": {"message": "task"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        benchmark.load_suite(suite_path)
    except ValueError as exc:
        assert "non-placeholder" in str(exc)
    else:
        raise AssertionError("placeholder fingerprint must be rejected")


def test_public_suite_rejects_hardcoded_document_id(tmp_path):
    benchmark = _module()
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "contract": "public_agent_scenarios_v2",
                "provenance": {
                    "source_name": "CMRC 2018",
                    "source_uri": "https://ymcui.com/cmrc2018/",
                    "source_version": "2018",
                    "license_name": "CC BY-SA 4.0",
                    "split": "dev",
                    "corpus_fingerprint": "a" * 64,
                    "source_snapshot_fingerprint": "b" * 64,
                    "transform_spec": {"contract": "fixture_v1"},
                },
                "cases": [
                    {
                        "id": "public-1",
                        "source_sample_id": "cmrc-1",
                        "request": {
                            "message": "生成证据报告",
                            "document_id": 42,
                        },
                        "expected": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not hardcode document_id"):
        benchmark.load_suite(suite_path)


def test_runtime_document_resolution_is_exact_and_binds_without_mutation():
    benchmark = _module()
    selector = {
        "source_name": "SciFact",
        "split": "test",
        "corpus_fingerprint": "a" * 64,
        "source_snapshot_fingerprint": "b" * 64,
    }
    dataset = {
        **selector,
        "document_id": 17,
        "label_source": "public_ground_truth",
        "release_eligible": True,
    }
    cases = [{"request": {"message": "生成公开研究报告"}}]

    assert benchmark.resolve_public_document_id([dataset], selector) == 17
    bound = benchmark.bind_cases_to_document(cases, 17)

    assert cases == [{"request": {"message": "生成公开研究报告"}}]
    assert bound[0]["request"]["document_id"] == 17
    with pytest.raises(ValueError, match="expected exactly one"):
        benchmark.resolve_public_document_id([], selector)
    with pytest.raises(ValueError, match="expected exactly one"):
        benchmark.resolve_public_document_id([dataset, dataset], selector)


def test_parallel_cases_record_actual_in_flight_peak():
    benchmark = _module()
    barrier = Barrier(2, timeout=1)

    def handler(request: httpx.Request):
        assert request.url.path == "/agent/chat"
        barrier.wait()
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "answer": "已完成",
                "plan": {"steps": []},
                "trace": [],
                "artifacts": [],
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
    )
    report = benchmark.run_cases(
        client,
        [
            {"id": "first", "request": {"message": "任务一"}},
            {"id": "second", "request": {"message": "任务二"}},
        ],
        concurrency=2,
    )
    client.close()

    assert report["performance"]["configured_concurrency"] == 2
    assert report["performance"]["max_observed_in_flight"] == 2
    assert report["performance"]["throughput_cases_per_second"] > 0


def test_http_failure_is_retained_as_a_badcase_instead_of_aborting():
    benchmark = _module()

    def handler(request: httpx.Request):
        assert request.url.path == "/agent/chat"
        return httpx.Response(503, json={"detail": "provider unavailable"})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
    )
    report = benchmark.run_cases(
        client,
        [{"id": "provider-down", "request": {"message": "生成报告"}}],
    )
    client.close()

    assert report["metrics"]["task_success_rate"] == 0.0
    assert report["metrics"]["failed_case_ids"] == ["provider-down"]
    evidence = report["evidence"][0]
    assert evidence["result"]["status"] == "failed"
    assert evidence["result"]["benchmark_error"] == {
        "contract": "agent_benchmark_error_v1",
        "kind": "http_status_error",
        "message": "Agent API returned HTTP 503.",
        "retryable": True,
        "http_status": 503,
    }
    assert any(
        reason.startswith("Agent API 请求失败")
        for reason in evidence["score"]["failure_reasons"]
    )


def test_release_gate_requires_public_provenance_and_verified_completion():
    benchmark = _module()

    def handler(request: httpx.Request):
        assert request.url.path == "/agent/chat"
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "answer": "已完成",
                "plan": {"status": "completed", "steps": []},
                "trace": [],
                "artifacts": [],
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
    )
    report = benchmark.run_cases(
        client,
        [{"id": "verified", "request": {"message": "完成任务"}}],
        provenance={"source_name": "public-fixture"},
        suite_fingerprint="a" * 64,
    )
    client.close()

    assert report["release_evidence_eligible"] is True
    assert report["release_gate_passed"] is True
    assert report["release_eligible"] is True
