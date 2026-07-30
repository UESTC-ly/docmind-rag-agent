"""Staged Agent load harness contracts."""

import importlib.util
import time
from pathlib import Path
from threading import Lock


def _module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_agent_load.py"
    spec = importlib.util.spec_from_file_location("benchmark_agent_load", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Tracker:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = Lock()

    def start(self):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def finish(self):
        with self.lock:
            self.active -= 1


class _Benchmark:
    _ConcurrencyTracker = _Tracker

    @staticmethod
    def _run_case(_client, case, _index, tracker):
        tracker.start()
        try:
            time.sleep(0.01)
            score = {
                "case_id": case["id"],
                "task_success": True,
                "verified_task_completion": True,
                "latency_seconds": 0.01,
                "provider_usage": {
                    "status": "observed",
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                    "request_count": 1,
                },
                "provider_cost": {
                    "status": "unavailable",
                    "request_count": 1,
                },
                "provider_model": {
                    "status": "observed",
                    "configured_request_models": ["gpt-5.6-terra"],
                    "provider_reported_models": ["gpt-5.6-terra"],
                    "request_count": 1,
                    "response_count": 1,
                    "observed_response_count": 1,
                },
            }
            return score, {"case_id": case["id"], "result": {}}
        finally:
            tracker.finish()


def test_staged_load_reports_actual_in_flight_and_explicit_telemetry(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "_benchmark_module", lambda: _Benchmark)

    report = module.run_load(
        object(),
        [{"id": "a"}, {"id": "b"}],
        concurrency_levels=[1, 2],
        repetitions=2,
        warmup_repetitions=1,
    )

    assert report["contract"] == "agent_load_benchmark_v1"
    assert report["warmup"]["case_count"] == 2
    assert [stage["case_count"] for stage in report["stages"]] == [4, 4]
    assert report["stages"][0]["actual_max_in_flight"] == 1
    assert report["stages"][1]["actual_max_in_flight"] == 2
    assert report["stages"][1]["provider_usage"]["total_tokens"] == 20.0
    assert report["stages"][1]["provider_cost"]["status"] == "unavailable"
    assert report["stages"][1]["provider_model"]["provider_reported_models"] == [
        "gpt-5.6-terra"
    ]
    assert report["stages"][1]["failure_class_counts"] == {"passed": 4}
    assert report["release_gate_passed"] is True
    assert report["stages"][1]["release_gate"]["passed"] is True
    assert "Provider Token" in module._render_markdown(report)
    assert "压力发布门禁：通过" in module._render_markdown(report)


def test_load_argument_parsers_reject_invalid_values():
    module = _module()

    assert module._concurrency_levels("1,2,4") == [1, 2, 4]
    try:
        module._concurrency_levels("0,2")
    except Exception as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero concurrency must be rejected")


def test_load_gate_can_require_real_token_and_currency_telemetry(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "_benchmark_module", lambda: _Benchmark)

    report = module.run_load(
        object(),
        [{"id": "a"}],
        concurrency_levels=[1],
        repetitions=1,
        require_token_usage=True,
        require_provider_cost=True,
    )

    assert report["release_gate_passed"] is False
    checks = {
        item["name"]: item
        for item in report["stages"][0]["release_gate"]["checks"]
    }
    assert checks["provider_token_usage"]["passed"] is True
    assert checks["provider_currency_cost"] == {
        "name": "provider_currency_cost",
        "passed": False,
        "actual": "unavailable",
        "threshold": "observed",
    }


def test_load_report_separates_provider_failures_from_agent_outcomes(monkeypatch):
    module = _module()

    class _FailureBenchmark(_Benchmark):
        @staticmethod
        def _run_case(_client, case, _index, tracker):
            tracker.start()
            try:
                failure_class = case["failure_class"]
                score = {
                    "case_id": case["id"],
                    "task_success": False,
                    "verified_task_completion": False,
                    "failure_class": failure_class,
                    "failure_reasons": [failure_class],
                    "latency_seconds": 0.01,
                    "provider_usage": {"status": "unavailable"},
                    "provider_cost": {"status": "unavailable"},
                    "provider_model": {"status": "unavailable"},
                }
                return score, {"case_id": case["id"], "result": {}}
            finally:
                tracker.finish()

    monkeypatch.setattr(module, "_benchmark_module", lambda: _FailureBenchmark)
    report = module.run_load(
        object(),
        [
            {"id": "provider", "failure_class": "provider_failure"},
            {"id": "quality", "failure_class": "agent_outcome_failure"},
        ],
        concurrency_levels=[2],
        repetitions=1,
    )

    stage = report["stages"][0]
    assert stage["failure_class_counts"] == {
        "agent_outcome_failure": 1,
        "provider_failure": 1,
    }
    rendered = module._render_markdown(report)
    assert "provider_failure" in rendered
    assert "agent_outcome_failure" in rendered
