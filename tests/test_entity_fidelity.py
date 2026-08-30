import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.download_retail_reference import DATASET_ID, EXPECTED_ROWS, SOURCE_URL
from sim.entity_fidelity import (
    FEATURES,
    _ids_hash,
    _empirical_bayes_log_scales,
    _empirical_bayes_short_probabilities,
    _nearest_correlation_psd,
    _orientation_free_bootstrap,
    _protocol_hash,
    aggregate_retail_rows,
    customer_features,
    customer_partitions,
    customer_split,
    fit_behavior_profile,
    fit_calibrations,
    load_retail_partition,
    run_entity_fidelity_v3_final,
    verify_v3_artifacts,
    verify_reference_provenance,
)
from sim.population import COPULA_PRIMITIVES


def test_retail_lines_are_filtered_then_aggregated_to_invoices() -> None:
    timestamp = pd.Timestamp("2011-01-02 12:30:00")
    rows = [
        {"InvoiceNo": "100", "Quantity": 2, "UnitPrice": 10, "CustomerID": 7, "InvoiceDate": timestamp},
        {"InvoiceNo": "100", "Quantity": 1, "UnitPrice": 5, "CustomerID": 7, "InvoiceDate": timestamp},
        {"InvoiceNo": "C101", "Quantity": 1, "UnitPrice": 1, "CustomerID": 7, "InvoiceDate": timestamp},
        {"InvoiceNo": "102", "Quantity": 1, "UnitPrice": 1, "CustomerID": None, "InvoiceDate": timestamp},
        {"InvoiceNo": "103", "Quantity": 0, "UnitPrice": 1, "CustomerID": 7, "InvoiceDate": timestamp},
        {"InvoiceNo": "104", "Quantity": 1, "UnitPrice": 0, "CustomerID": 7, "InvoiceDate": timestamp},
    ]
    invoices, counts = aggregate_retail_rows(rows)
    assert invoices.to_dict("records") == [
        {"customer_id": "7", "ts": timestamp, "amount": 25.0}
    ]
    assert counts == {
        "input_line_items": 6,
        "excluded_cancellation_line_items": 1,
        "excluded_missing_customer_line_items": 1,
        "excluded_nonpositive_quantity_line_items": 1,
        "excluded_nonpositive_price_line_items": 1,
        "usable_line_items": 2,
        "usable_invoices": 1,
        "usable_customers": 1,
    }


def test_customer_splits_are_deterministic_disjoint_and_namespaced() -> None:
    ids = [f"customer-{number}" for number in range(500)]
    first = customer_split(ids, seed=19, namespace="real")
    second = customer_split(reversed(ids), seed=19, namespace="real")
    synthetic = customer_split(ids, seed=19, namespace="synthetic")
    assert first == second
    assert first[0].isdisjoint(first[1])
    assert first[0] | first[1] == set(ids)
    assert first != synthetic

    partitions = customer_partitions(ids, seed=19, namespace="real-v2")
    assert partitions == customer_partitions(reversed(ids), seed=19, namespace="real-v2")
    assert set(partitions) == {"calibration", "diagnostic", "sealed_final"}
    assert set.union(*partitions.values()) == set(ids)
    assert all(
        partitions[left].isdisjoint(partitions[right])
        for left in partitions
        for right in partitions
        if left < right
    )


def _transactions(prefix: str, amount_factor: float = 1.0) -> pd.DataFrame:
    rows = []
    for customer in range(90):
        for purchase in range(3):
            rows.append(
                {
                    "customer_id": f"{prefix}-{customer}",
                    "ts": pd.Timestamp("2020-01-01")
                    + pd.Timedelta(days=customer + purchase * (customer % 3 + 1), hours=8 + purchase),
                    "amount": amount_factor * (10 + customer + purchase),
                }
            )
    return pd.DataFrame(rows).sort_values(["customer_id", "ts"]).reset_index(drop=True)


def test_calibration_and_feature_matrix_cannot_contain_evaluation_ids() -> None:
    real = _transactions("real")
    synthetic = _transactions("synthetic", 100)
    real_partitions = customer_partitions(real["customer_id"], seed=23, namespace="real-v2")
    synthetic_partitions = customer_partitions(
        synthetic["customer_id"], seed=23, namespace="synthetic-v2"
    )
    real_ids = real_partitions["calibration"]
    synthetic_ids = synthetic_partitions["calibration"]
    real_cal = real[real["customer_id"].isin(real_ids)]
    synthetic_cal = synthetic[synthetic["customer_id"].isin(synthetic_ids)]
    profile = fit_behavior_profile(real_cal, customer_ids_sha256="calibration-id-hash")
    real_transform, synthetic_transform, _ = fit_calibrations(real_cal, synthetic_cal)
    real_matrix = customer_features(real_cal, real_transform)
    synthetic_matrix = customer_features(synthetic_cal, synthetic_transform)
    assert set(real_matrix.index).isdisjoint(
        real_partitions["diagnostic"] | real_partitions["sealed_final"]
    )
    assert set(synthetic_matrix.index).isdisjoint(
        synthetic_partitions["diagnostic"] | synthetic_partitions["sealed_final"]
    )
    assert profile.source_partition == "calibration"
    assert profile.source_customer_ids_sha256 == "calibration-id-hash"
    assert profile.source_customer_count == len(real_ids)
    assert profile.as_dict()["source"]["transactions_sha256"] == profile.source_transactions_sha256
    profile_payload = profile.as_dict()
    assert profile_payload["schema_version"] == 3
    assert tuple(profile_payload["primitive_quantiles"]) == COPULA_PRIMITIVES
    assert all(len(values) == 32 for values in profile_payload["primitive_quantiles"].values())
    assert all(len(values) == 32 for values in profile_payload["singleton_quantiles"].values())
    assert np.linalg.eigvalsh(profile.copula_correlation).min() >= -1e-8
    assert not any(
        token in json.dumps(profile_payload).lower()
        for token in ("invoice_no", "evaluator_feature", '"rows"')
    )
    assert tuple(real_matrix.columns) == FEATURES
    assert tuple(synthetic_matrix.columns) == FEATURES
    assert not any(token in column for column in FEATURES for token in ("customer", "label", "split"))


def test_copula_draw_is_psd_deterministic_and_preserves_rank_correlation() -> None:
    real = _transactions("real")
    profile = fit_behavior_profile(real, customer_ids_sha256="a" * 64)
    quantiles = [list(values) for values in profile.primitive_quantiles]
    quantiles[0] = list(np.linspace(1, 30, 32))
    quantiles[3] = list(np.linspace(-2, 2, 32))
    correlation = np.eye(len(COPULA_PRIMITIVES))
    correlation[0, 3] = correlation[3, 0] = 0.8
    profile = replace(
        profile,
        primitive_quantiles=tuple(tuple(values) for values in quantiles),
        copula_correlation=tuple(tuple(row) for row in correlation),
    )
    first = profile.sample_primitives(5_000, seed=91)
    second = profile.sample_primitives(5_000, seed=91)
    counts = np.asarray([row["invoice_count"] for row in first])
    centers = np.asarray([row["relative_log_amount_center"] for row in first])
    repeated = counts > 1
    singleton_rows = [row for row in first if row["invoice_count"] == 1]

    assert first == second
    assert np.linalg.eigvalsh(profile.copula_correlation).min() >= -1e-8
    assert np.corrcoef(counts[repeated], centers[repeated])[0, 1] > 0.65
    assert all(row["active_duration_fraction"] == 0 for row in singleton_rows)
    repeated_rows = [row for row in first if row["invoice_count"] > 1]
    assert abs(
        sum(row["active_duration_fraction"] == 0 for row in repeated_rows)
        / len(repeated_rows)
        - profile.repeat_same_day_probability
    ) <= 1 / len(repeated_rows)
    assert all(
        min(
            abs(
                np.arctan2(
                    np.sin(
                        np.arctan2(row["circadian_phase_sin"], row["circadian_phase_cos"])
                        - 2 * np.pi * hour / 24
                    ),
                    np.cos(
                        np.arctan2(row["circadian_phase_sin"], row["circadian_phase_cos"])
                        - 2 * np.pi * hour / 24
                    ),
                )
            )
            for hour in range(24)
        )
        < 1e-10
        for row in singleton_rows
    )
    assert np.linalg.eigvalsh(_nearest_correlation_psd(correlation)).min() >= -1e-8


def test_empirical_bayes_likelihood_degrees_and_posterior_mode() -> None:
    groups = [
        (
            "singleton",
            pd.DataFrame(
                {"amount": [10.0], "ts": [pd.Timestamp("2020-01-01 08:00:00")]}
            ),
        ),
        (
            "repeat",
            pd.DataFrame(
                {
                    "amount": [5.0, 10.0, 20.0],
                    "ts": pd.to_datetime(
                        ["2020-01-01 08:00", "2020-01-01 09:00", "2020-01-03 10:00"]
                    ),
                }
            ),
        ),
    ]
    for index in range(20):
        groups.append(
            (
                f"extra-{index}",
                pd.DataFrame(
                    {
                        "amount": [8.0 + index, 9.5 + index, 12.0 + index],
                        "ts": pd.to_datetime(
                            [
                                "2020-01-01 08:00",
                                "2020-01-01 10:00"
                                if index % 2
                                else "2020-01-03 10:00",
                                "2020-01-05 10:00",
                            ]
                        ),
                    }
                ),
            )
        )
    scales, alpha, beta = _empirical_bayes_log_scales(groups)
    values = np.log(groups[1][1]["amount"].to_numpy())
    degrees = len(values) - 1
    residual_sum = float(np.sum((values - values.mean()) ** 2))
    assert scales["singleton"] == pytest.approx(np.sqrt(beta / (alpha + 1)))
    assert scales["repeat"] == pytest.approx(
        np.sqrt((beta + residual_sum / 2) / (alpha + degrees / 2 + 1))
    )
    probabilities, short_alpha, short_beta = _empirical_bayes_short_probabilities(groups)
    assert probabilities["singleton"] == pytest.approx(
        short_alpha / (short_alpha + short_beta)
    )
    assert probabilities["repeat"] == pytest.approx(
        (1 + short_alpha) / (2 + short_alpha + short_beta)
    )


def test_orientation_free_bootstrap_is_deterministic() -> None:
    seed_folds = {
        seed: [
            {
                "evaluation_real_customers": 20,
                "_real_predictions": np.linspace(0.1, 0.4, 20),
                "_synthetic_predictions": np.linspace(0.6, 0.9, 20),
            }
            for _ in range(5)
        ]
        for seed in (1, 2, 3)
    }
    first = _orientation_free_bootstrap(seed_folds, repeats=100, seed=7)
    second = _orientation_free_bootstrap(seed_folds, repeats=100, seed=7)
    assert first == second
    assert first[1] >= 0.99
    assert first[2] >= first[1]


def test_protocol_hash_excludes_only_its_hash_field() -> None:
    payload = {"schema_version": 3, "value": "frozen"}
    digest = _protocol_hash(payload)
    assert _protocol_hash({**payload, "protocol_sha256": digest}) == digest
    assert _protocol_hash({**payload, "value": "changed"}) != digest


def test_v3_development_failure_artifact_is_fail_closed() -> None:
    path = Path(__file__).resolve().parents[1] / "results/entity_fidelity_v3_development_failed.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert {
        "status": result["status"],
        "release_eligible": result["release_eligible"],
        "calibration_only": result["calibration_only"],
        "diagnostic_transactions_accessed": result["diagnostic_transactions_accessed"],
        "sealed_final_transactions_accessed": result["sealed_final_transactions_accessed"],
        "threshold_max_d_upper_95": result["threshold_max_d_upper_95"],
        "k2_implemented": result["evidence"]["read_only_k2_diagnostic"]["implemented"],
    } == {
        "status": "STOP_FAIL",
        "release_eligible": False,
        "calibration_only": True,
        "diagnostic_transactions_accessed": False,
        "sealed_final_transactions_accessed": False,
        "threshold_max_d_upper_95": 0.65,
        "k2_implemented": False,
    }


def test_verify_fails_closed_without_sealed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_hash = "frozen"
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "protocol_sha256": protocol_hash,
                "gate_passed": True,
                "calibration_only": True,
                "diagnostic_transactions_accessed": False,
                "sealed_final_transactions_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sim.entity_fidelity.load_frozen_protocol",
        lambda _: {"protocol_sha256": protocol_hash},
    )
    result = verify_v3_artifacts(
        protocol_path=tmp_path / "protocol.json",
        readiness_path=readiness,
        receipt_path=tmp_path / "receipt.json",
        final_path=tmp_path / "final.json",
    )
    assert result["verified"] is False
    assert result["checks"]["sealed_artifacts_complete"] is False


def test_sealed_final_requires_protocol_bound_confirmation_before_partition_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sim.entity_fidelity.load_frozen_protocol",
        lambda _: {"protocol_sha256": "frozen-protocol"},
    )
    monkeypatch.setattr(
        "sim.entity_fidelity.load_retail_partition",
        lambda *_args, **_kwargs: pytest.fail("no partition may be opened"),
    )
    with pytest.raises(ValueError, match="protocol-hash-bound"):
        run_entity_fidelity_v3_final(
            tmp_path / "must-not-be-opened.xlsx",
            tmp_path / "result.json",
            protocol_path=tmp_path / "protocol.json",
            readiness_path=tmp_path / "readiness.json",
            receipt_path=tmp_path / "receipt.json",
            sealed_final_confirmation="APPROVED_SINGLE_FINAL_SCORE:wrong",
        )


def test_one_shot_receipt_is_written_before_fake_sealed_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = _transactions("calibration")
    profile = fit_behavior_profile(
        calibration,
        customer_ids_sha256=_ids_hash(calibration["customer_id"].unique()),
    )
    protocol_hash = "frozen-protocol"
    protocol = {
        "protocol_sha256": protocol_hash,
        "generator": {"behavior_profile_sha256": profile.sha256},
    }
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(
        json.dumps(
            {
                "protocol_sha256": protocol_hash,
                "gate_passed": True,
                "calibration_only": True,
                "diagnostic_transactions_accessed": False,
                "sealed_final_transactions_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "receipt.json"
    calls: list[str] = []

    def fake_partition(_path: Path, partition: str) -> tuple[pd.DataFrame, dict[str, str]]:
        calls.append(partition)
        if partition == "calibration":
            return calibration, {}
        assert receipt_path.exists()
        raise RuntimeError("fake sealed boundary reached")

    monkeypatch.setattr("sim.entity_fidelity.load_frozen_protocol", lambda _: protocol)
    monkeypatch.setattr("sim.entity_fidelity.load_retail_partition", fake_partition)
    with pytest.raises(RuntimeError, match="fake sealed boundary"):
        run_entity_fidelity_v3_final(
            tmp_path / "reference.xlsx",
            tmp_path / "final.json",
            protocol_path=tmp_path / "protocol.json",
            readiness_path=readiness_path,
            receipt_path=receipt_path,
            sealed_final_confirmation=f"APPROVED_SINGLE_FINAL_SCORE:{protocol_hash}",
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert calls == ["calibration", "sealed_final"]
    assert receipt["status"] == "consumed"
    assert receipt["protocol_sha256"] == protocol_hash
    with pytest.raises(FileExistsError, match="receipt already exists"):
        run_entity_fidelity_v3_final(
            tmp_path / "reference.xlsx",
            tmp_path / "final.json",
            protocol_path=tmp_path / "protocol.json",
            readiness_path=readiness_path,
            receipt_path=receipt_path,
            sealed_final_confirmation=f"APPROVED_SINGLE_FINAL_SCORE:{protocol_hash}",
        )


def test_reference_provenance_rejects_changed_workbook(tmp_path: Path) -> None:
    workbook = tmp_path / "online_retail.xlsx"
    workbook.write_bytes(b"pinned workbook bytes")
    digest = hashlib.sha256(workbook.read_bytes()).hexdigest()
    manifest = {
        "dataset_id": DATASET_ID,
        "source_url": SOURCE_URL,
        "workbook_sha256": digest,
        "rows": EXPECTED_ROWS,
        "license": "CC BY 4.0",
    }
    workbook.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_reference_provenance(workbook, expected_sha256=digest)["workbook_sha256"] == digest
    workbook.write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance mismatch"):
        verify_reference_provenance(workbook, expected_sha256=digest)
