"""Provider telemetry must stay explicit, scoped, and non-estimating."""

from app.services.provider_telemetry import (
    ProviderTelemetry,
    capture_provider_telemetry,
    public_telemetry,
    record_provider_attempt,
    record_provider_response,
)


def test_collects_explicit_usage_cost_and_retries_only():
    with capture_provider_telemetry() as telemetry:
        record_provider_attempt()
        record_provider_response(
            {
                "model": "gpt-5.6-terra",
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                },
                "cost": {"amount": 0.0025, "currency": "usd"},
            }
        )
        record_provider_attempt(retry=True, model="gpt-5.6-terra")
        record_provider_response(
            {"model": "gpt-5.6-terra", "usage": {"total_tokens": 5}}
        )

    usage = telemetry.usage_public()
    cost = telemetry.cost_public()
    model = telemetry.model_public()

    assert usage == {
        "status": "observed",
        "request_count": 2,
        "retry_count": 1,
        "response_count": 2,
        "observed_response_count": 2,
        "input_tokens": 11.0,
        "output_tokens": 7.0,
        "total_tokens": 23.0,
    }
    assert cost["status"] == "partial"
    assert cost["amount"] == 0.0025
    assert cost["currency"] == "USD"
    assert cost["request_count"] == 2
    assert cost["retry_count"] == 1
    assert model == {
        "status": "observed",
        "request_count": 2,
        "response_count": 2,
        "configured_request_models": ["gpt-5.6-terra"],
        "provider_reported_models": ["gpt-5.6-terra"],
        "observed_response_count": 2,
    }


def test_missing_provider_metering_stays_unavailable_and_retains_attempts():
    with capture_provider_telemetry() as telemetry:
        record_provider_attempt()
        record_provider_response({"choices": [{"message": {"content": "ok"}}]})

    usage = telemetry.usage_public()
    cost = telemetry.cost_public()

    assert usage["status"] == "unavailable"
    assert usage["request_count"] == 1
    assert "token usage" in usage["reason"]
    assert cost["status"] == "unavailable"
    assert cost["request_count"] == 1
    assert "currency cost" in cost["reason"]


def test_resume_state_accumulates_without_estimating_multi_currency_costs():
    first = ProviderTelemetry()
    first.record_attempt()
    first.record_response(
        {
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "cost": {"amount": 1.0, "currency": "USD"},
        }
    )
    resumed = ProviderTelemetry.from_state(first.to_state())
    resumed.record_attempt()
    resumed.record_response(
        {
            "usage": {"total_tokens": 9},
            "cost": {"amount": 2.0, "currency": "CNY"},
        }
    )

    usage, cost, model = public_telemetry(
        {"provider_telemetry": resumed.to_state()}
    )

    assert usage["total_tokens"] == 14.0
    assert usage["request_count"] == 2
    assert cost["status"] == "partial"
    assert cost["currencies"] == ["CNY", "USD"]
    assert "exchange rate" in cost["reason"]
    assert model["status"] == "unavailable"
