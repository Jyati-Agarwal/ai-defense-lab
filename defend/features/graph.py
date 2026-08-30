"""Small graph statistics already observable at authorization time."""

from __future__ import annotations

from typing import Any


def graph_features(event: dict[str, Any]) -> dict[str, float]:
    txn = event.get("txn") or {}
    return {
        "beneficiary_in_degree": float(txn.get("beneficiary_in_degree") or 0),
        "shared_device_cluster_size": float(txn.get("shared_device_cluster_size") or 1),
        "connected_component_size": float(txn.get("connected_component_size") or 1),
    }
