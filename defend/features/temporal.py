"""Entity-relative temporal features."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _bounded_fraction(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and 0 <= result <= 1 else None


def temporal_features(
    event: dict[str, Any],
    prior_times: list[datetime],
    typical_hour: float,
    typical_amount: float,
) -> dict[str, float]:
    timestamp = parse_time(str(event["ts"]))
    chain = event.get("mandate_chain") or {}
    intent = chain.get("intent") or {}
    signed_at = intent.get("signed_at")
    ttl = max(1.0, float(intent.get("ttl_seconds") or 1))
    age = 0.0 if not signed_at else max(0.0, (timestamp - parse_time(str(signed_at))).total_seconds())
    hour = float((event.get("txn") or {}).get("hour_local", timestamp.hour))
    hour_distance = abs(hour - typical_hour)
    hour_distance = min(hour_distance, 24 - hour_distance) / 12
    amount = float((event.get("txn") or {}).get("amount_minor") or 0)
    amount_z = abs(amount - typical_amount) / max(1.0, typical_amount * 0.5)
    velocity_1h = sum(0 <= (timestamp - value).total_seconds() <= 3_600 for value in prior_times)
    velocity_24h = sum(0 <= (timestamp - value).total_seconds() <= 86_400 for value in prior_times)
    txn = event.get("txn") or {}
    velocity_1h = max(
        velocity_1h,
        int(txn.get("authorisation_attempts_10m") or 0),
        int(txn.get("cashout_velocity_1h") or 0),
    )
    velocity_24h = max(velocity_24h, int(txn.get("authorisation_attempts_24h") or velocity_1h))
    prior_utilisation = _bounded_fraction(txn.get("prior_utilisation_pct"))
    current_utilisation = _bounded_fraction(txn.get("current_utilisation_pct"))
    utilisation_regime_change = float(
        prior_utilisation is not None
        and current_utilisation is not None
        and current_utilisation >= 0.90
        and current_utilisation - prior_utilisation >= 0.70
        and velocity_1h >= 5
    )
    return {
        "ttl_utilisation_pct": min(age / ttl, 10.0),
        "hour_deviation": float(hour_distance),
        "amount_zscore_entity": min(float(amount_z), 20.0),
        "velocity_1h": float(velocity_1h),
        "velocity_24h": float(velocity_24h),
        "credit_utilisation_regime_change": utilisation_regime_change,
    }
