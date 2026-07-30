"""Request-scoped, provider-reported LLM telemetry.

The application must never estimate token usage or currency cost from a model
name, prompt length, or latency. This module only aggregates fields explicitly
returned by an upstream OpenAI-compatible provider.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping


_ACTIVE_TELEMETRY: ContextVar["ProviderTelemetry | None"] = ContextVar(
    "docmind_provider_telemetry",
    default=None,
)
_USAGE_INPUT_KEYS = ("input_tokens", "prompt_tokens")
_USAGE_OUTPUT_KEYS = ("output_tokens", "completion_tokens")
_USAGE_TOTAL_KEYS = ("total_tokens",)


def _non_negative_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return value if value >= 0 else None


def _first_number(payload: Mapping[str, Any], keys: tuple[str, ...]) -> float | int | None:
    for key in keys:
        value = _non_negative_number(payload.get(key))
        if value is not None:
            return value
    return None


def _explicit_cost(payload: Mapping[str, Any]) -> tuple[float | int, str] | None:
    candidates = [payload.get("provider_cost"), payload.get("cost")]
    usage = payload.get("usage")
    if isinstance(usage, Mapping):
        candidates.extend([usage.get("provider_cost"), usage.get("cost")])
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        amount = _non_negative_number(candidate.get("amount"))
        currency = str(candidate.get("currency") or "").strip().upper()
        if amount is not None and currency:
            return amount, currency
    return None


def _model_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized[:200] if normalized else None


def _model_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    counts: dict[str, int] = {}
    for raw_model, raw_count in value.items():
        model = _model_name(raw_model)
        count = _non_negative_number(raw_count)
        if model is not None and count is not None:
            counts[model] = int(count)
    return counts


@dataclass
class ProviderTelemetry:
    """Mutable aggregate carried by one outer Agent run."""

    request_count: int = 0
    retry_count: int = 0
    response_count: int = 0
    usage_response_count: int = 0
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    total_tokens: float = 0.0
    input_tokens_seen: bool = False
    output_tokens_seen: bool = False
    total_tokens_seen: bool = False
    costs: dict[str, float] = field(default_factory=dict)
    cost_response_count: int = 0
    requested_models: dict[str, int] = field(default_factory=dict)
    response_models: dict[str, int] = field(default_factory=dict)
    model_response_count: int = 0

    @classmethod
    def from_state(cls, value: Any) -> "ProviderTelemetry":
        if not isinstance(value, Mapping):
            return cls()
        costs_raw = value.get("costs")
        costs: dict[str, float] = {}
        if isinstance(costs_raw, Mapping):
            for currency, amount in costs_raw.items():
                valid_amount = _non_negative_number(amount)
                normalized_currency = str(currency or "").strip().upper()
                if valid_amount is not None and normalized_currency:
                    costs[normalized_currency] = float(valid_amount)
        telemetry = cls(
            request_count=int(_non_negative_number(value.get("request_count")) or 0),
            retry_count=int(_non_negative_number(value.get("retry_count")) or 0),
            response_count=int(_non_negative_number(value.get("response_count")) or 0),
            usage_response_count=int(
                _non_negative_number(value.get("usage_response_count")) or 0
            ),
            input_tokens=float(_non_negative_number(value.get("input_tokens")) or 0),
            output_tokens=float(
                _non_negative_number(value.get("output_tokens")) or 0
            ),
            total_tokens=float(_non_negative_number(value.get("total_tokens")) or 0),
            input_tokens_seen=bool(value.get("input_tokens_seen")),
            output_tokens_seen=bool(value.get("output_tokens_seen")),
            total_tokens_seen=bool(value.get("total_tokens_seen")),
            costs=costs,
            cost_response_count=int(
                _non_negative_number(value.get("cost_response_count")) or 0
            ),
            requested_models=_model_counts(value.get("requested_models")),
            response_models=_model_counts(value.get("response_models")),
            model_response_count=int(
                _non_negative_number(value.get("model_response_count")) or 0
            ),
        )
        return telemetry

    def record_attempt(
        self,
        *,
        retry: bool = False,
        model: str | None = None,
    ) -> None:
        self.request_count += 1
        if retry:
            self.retry_count += 1
        normalized_model = _model_name(model)
        if normalized_model is not None:
            self.requested_models[normalized_model] = (
                self.requested_models.get(normalized_model, 0) + 1
            )

    def record_response(self, payload: Mapping[str, Any] | None) -> None:
        self.response_count += 1
        raw = payload if isinstance(payload, Mapping) else {}
        usage = raw.get("usage")
        usage_data = usage if isinstance(usage, Mapping) else raw
        response_model = _model_name(raw.get("model"))
        if response_model is not None:
            self.response_models[response_model] = (
                self.response_models.get(response_model, 0) + 1
            )
            self.model_response_count += 1

        input_tokens = _first_number(usage_data, _USAGE_INPUT_KEYS)
        output_tokens = _first_number(usage_data, _USAGE_OUTPUT_KEYS)
        total_tokens = _first_number(usage_data, _USAGE_TOTAL_KEYS)
        if input_tokens is not None:
            self.input_tokens += float(input_tokens)
            self.input_tokens_seen = True
        if output_tokens is not None:
            self.output_tokens += float(output_tokens)
            self.output_tokens_seen = True
        if total_tokens is not None:
            self.total_tokens += float(total_tokens)
            self.total_tokens_seen = True
        elif input_tokens is not None and output_tokens is not None:
            self.total_tokens += float(input_tokens) + float(output_tokens)
            self.total_tokens_seen = True
        if (
            input_tokens is not None
            or output_tokens is not None
            or total_tokens is not None
        ):
            self.usage_response_count += 1

        cost = _explicit_cost(raw)
        if cost is not None:
            amount, currency = cost
            self.costs[currency] = self.costs.get(currency, 0.0) + float(amount)
            self.cost_response_count += 1

    def to_state(self) -> dict[str, Any]:
        return {
            "contract": "provider_telemetry_v1",
            "request_count": self.request_count,
            "retry_count": self.retry_count,
            "response_count": self.response_count,
            "usage_response_count": self.usage_response_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_tokens_seen": self.input_tokens_seen,
            "output_tokens_seen": self.output_tokens_seen,
            "total_tokens_seen": self.total_tokens_seen,
            "costs": dict(sorted(self.costs.items())),
            "cost_response_count": self.cost_response_count,
            "requested_models": dict(sorted(self.requested_models.items())),
            "response_models": dict(sorted(self.response_models.items())),
            "model_response_count": self.model_response_count,
        }

    def usage_public(self) -> dict[str, Any]:
        base = {
            "request_count": self.request_count,
            "retry_count": self.retry_count,
            "response_count": self.response_count,
            "observed_response_count": self.usage_response_count,
        }
        if not self.usage_response_count:
            return {
                "status": "unavailable",
                **base,
                "reason": (
                    "Provider responses did not include explicit token usage."
                    if self.response_count
                    else "No provider response was received for this Agent run."
                ),
            }
        payload: dict[str, Any] = {
            "status": (
                "observed"
                if self.usage_response_count == self.response_count
                else "partial"
            ),
            **base,
        }
        if self.input_tokens_seen:
            payload["input_tokens"] = round(self.input_tokens, 6)
        if self.output_tokens_seen:
            payload["output_tokens"] = round(self.output_tokens, 6)
        if self.total_tokens_seen:
            payload["total_tokens"] = round(self.total_tokens, 6)
        if payload["status"] == "partial":
            payload["reason"] = (
                "Some provider responses did not include explicit token usage."
            )
        return payload

    def cost_public(self) -> dict[str, Any]:
        base = {
            "request_count": self.request_count,
            "retry_count": self.retry_count,
            "response_count": self.response_count,
            "observed_response_count": self.cost_response_count,
        }
        if not self.cost_response_count:
            return {
                "status": "unavailable",
                **base,
                "reason": (
                    "Provider responses did not include explicit currency cost."
                    if self.response_count
                    else "No provider response was received for this Agent run."
                ),
            }
        if len(self.costs) != 1:
            return {
                "status": "partial",
                **base,
                "currencies": sorted(self.costs),
                "reason": (
                    "Provider reported multiple currencies; DocMind does not "
                    "apply an inferred exchange rate."
                ),
            }
        currency, amount = next(iter(self.costs.items()))
        payload: dict[str, Any] = {
            "status": (
                "observed"
                if self.cost_response_count == self.response_count
                else "partial"
            ),
            **base,
            "currency": currency,
            "amount": round(amount, 6),
        }
        if payload["status"] == "partial":
            payload["reason"] = (
                "Some provider responses did not include explicit currency cost."
            )
        return payload

    def model_public(self) -> dict[str, Any]:
        base = {
            "request_count": self.request_count,
            "response_count": self.response_count,
            "configured_request_models": sorted(self.requested_models),
            "provider_reported_models": sorted(self.response_models),
            "observed_response_count": self.model_response_count,
        }
        if not self.model_response_count:
            return {
                "status": "unavailable",
                **base,
                "reason": (
                    "Provider responses did not include an explicit model name."
                    if self.response_count
                    else "No provider response was received for this Agent run."
                ),
            }
        payload: dict[str, Any] = {
            "status": (
                "observed"
                if self.model_response_count == self.response_count
                else "partial"
            ),
            **base,
        }
        if payload["status"] == "partial":
            payload["reason"] = (
                "Some provider responses did not include an explicit model name."
            )
        return payload


@contextmanager
def capture_provider_telemetry(
    initial_state: Mapping[str, Any] | None = None,
) -> Iterator[ProviderTelemetry]:
    """Install a run-local aggregate for all nested LLM calls."""

    telemetry = ProviderTelemetry.from_state(initial_state)
    token = _ACTIVE_TELEMETRY.set(telemetry)
    try:
        yield telemetry
    finally:
        _ACTIVE_TELEMETRY.reset(token)


def record_provider_attempt(
    *,
    retry: bool = False,
    model: str | None = None,
) -> None:
    telemetry = _ACTIVE_TELEMETRY.get()
    if telemetry is not None:
        telemetry.record_attempt(retry=retry, model=model)


def record_provider_response(payload: Mapping[str, Any] | None) -> None:
    telemetry = _ACTIVE_TELEMETRY.get()
    if telemetry is not None:
        telemetry.record_response(payload)


def current_provider_telemetry_state() -> dict[str, Any] | None:
    """Return the active aggregate for a graph node state update.

    LangGraph interrupts are checkpoint tasks. Persisting telemetry with a
    separate ``update_state`` call after ``invoke`` would create a newer
    checkpoint without that pending task. Nodes therefore include this
    snapshot in their own atomic state update.
    """

    telemetry = _ACTIVE_TELEMETRY.get()
    return telemetry.to_state() if telemetry is not None else None


def public_telemetry(
    state: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    telemetry = ProviderTelemetry.from_state(
        (state or {}).get("provider_telemetry") if state else None
    )
    return (
        telemetry.usage_public(),
        telemetry.cost_public(),
        telemetry.model_public(),
    )


__all__ = [
    "ProviderTelemetry",
    "capture_provider_telemetry",
    "current_provider_telemetry_state",
    "public_telemetry",
    "record_provider_attempt",
    "record_provider_response",
]
