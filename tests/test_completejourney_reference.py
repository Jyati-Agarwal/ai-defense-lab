from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

import sim.completejourney_reference as reference


def _frame(rows: list[tuple[str, str, str, object]]) -> pd.DataFrame:
    timestamps = pd.Series([row[2] for row in rows], dtype="datetime64[ns]")
    count = len(rows)
    return pd.DataFrame(
        {
            "household_id": [row[0] for row in rows],
            "store_id": ["10"] * count,
            "basket_id": [row[1] for row in rows],
            "product_id": [str(100 + index) for index in range(count)],
            "quantity": [1] * count,
            "sales_value": [row[3] for row in rows],
            "retail_disc": [0.0] * count,
            "coupon_disc": [0.0] * count,
            "coupon_match_disc": [0.0] * count,
            "week": [1] * count,
            "transaction_timestamp": timestamps,
        },
        columns=reference.EXPECTED_COLUMNS,
    )


def _small_contract(monkeypatch: pytest.MonkeyPatch, *, rows: int, households: int = 3) -> None:
    monkeypatch.setattr(reference, "EXPECTED_ITEM_ROWS", rows)
    monkeypatch.setattr(reference, "EXPECTED_HOUSEHOLDS", households)
    monkeypatch.setattr(reference, "DEVELOPMENT_HOUSEHOLDS", 1)
    monkeypatch.setattr(reference, "QUALIFICATION_HOUSEHOLDS", households - 1)
    monkeypatch.setattr(reference.source, "RDS_SHA256", "a" * 64)


def _mrs_preregistration(tmp_path: Path) -> tuple[Path, str]:
    value = {
        "artifact": "entity_fidelity_mrs1_r2_preregistration",
        "completejourney": {"preprocessing_manifest_sha256_state": "unobserved"},
        "schema_version": 1,
        "study_revision": "mrs1.2",
    }
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path = tmp_path / "entity_fidelity_mrs1_r2_preregistration.json"
    path.write_bytes(raw)
    path.chmod(0o444)
    return path, hashlib.sha256(raw).hexdigest()


def test_uv_version_preflight_accepts_real_platform_suffix_and_rejects_other_versions() -> None:
    reference._require_uv_version("uv 0.11.8 (x86_64-unknown-linux-gnu)\n")
    reference._require_uv_version("uv 0.11.8")

    with pytest.raises(RuntimeError, match=r"requires uv 0\.11\.8"):
        reference._require_uv_version("uv 0.11.9 (x86_64-unknown-linux-gnu)")
    with pytest.raises(RuntimeError, match=r"requires uv 0\.11\.8"):
        reference._require_uv_version("uv 0.11.8 (x86_64-unknown-linux-gnu) unexpected")


def test_converter_restores_new_york_wall_clock_before_removing_timezone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert reference.converter_identity()["source_timezone"] == "America/New_York"
    frame = _frame([("1", "a", "2017-01-01 00:00:00", 1.0)])
    frame["transaction_timestamp"] = pd.to_datetime(["2018-01-01T04:30:00Z"], utc=True)
    monkeypatch.setattr(reference, "EXPECTED_ITEM_ROWS", 1)
    calls: list[tuple[str, str]] = []

    def read_r(path: str, *, timezone: str) -> dict[None, pd.DataFrame]:
        calls.append((path, timezone))
        localized = frame.copy()
        localized["transaction_timestamp"] = localized["transaction_timestamp"].dt.tz_convert(timezone)
        return {None: localized}

    pyreadr = types.ModuleType("pyreadr")
    pyreadr.read_r = read_r  # type: ignore[attr-defined]
    output = tmp_path / "converted.parquet"
    monkeypatch.setitem(sys.modules, "pyreadr", pyreadr)
    monkeypatch.setattr(sys, "argv", ["converter", "synthetic.rds", str(output)])

    exec(reference._converter_program(), {"__name__": "__main__"})

    converted = pd.read_parquet(output)
    assert calls == [("synthetic.rds", "America/New_York")]
    assert converted.loc[0, "transaction_timestamp"] == pd.Timestamp("2017-12-31 23:30:00")
    assert converted["transaction_timestamp"].dt.tz is None


@pytest.mark.parametrize(
    ("artifact", "study_revision"),
    [
        ("entity_fidelity_mrs1_r1_preregistration", "mrs1.2"),
        ("entity_fidelity_mrs1_r2_preregistration", "mrs1.1"),
    ],
)
def test_mrs_preregistration_accepts_only_r2_identity(
    tmp_path: Path, artifact: str, study_revision: str
) -> None:
    value = {"artifact": artifact, "schema_version": "mrs1", "study_revision": study_revision}
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path = tmp_path / "preregistration.json"
    path.write_bytes(raw)
    path.chmod(0o444)

    with pytest.raises(ValueError, match="canonical lifecycle artifact"):
        reference._verify_mrs_preregistration(path, hashlib.sha256(raw).hexdigest())


def test_aggregate_uses_decimal_half_even_preserves_baskets_and_excludes_only_nonpositive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        ("1", "b1", "2017-01-01 10:00:00", "1.005"),
        ("1", "b1", "2017-01-01 10:00:00", "1.000"),
        ("1", "b2", "2017-01-01 10:00:00", "0.005"),
        ("1", "b3", "2017-01-01 10:00:00", "2.675"),
        ("2", "c1", "2017-01-02 09:00:00", "3.335"),
        ("3", "d1", "2017-01-03 09:00:00", "-1.00"),
        ("3", "d2", "2017-01-03 09:00:00", "1.00"),
    ]
    _small_contract(monkeypatch, rows=len(rows))

    baskets, counts = reference.aggregate_transactions(_frame(rows))

    assert baskets[["basket_id", "sales_value_cents"]].values.tolist() == [
        ["b1", 200],
        ["b3", 268],
        ["c1", 334],
        ["d2", 100],
    ]
    assert counts == {
        "input_item_rows": 7,
        "input_households": 3,
        "baskets_before_exclusion": 6,
        "excluded_nonpositive_baskets": 2,
        "baskets_after_exclusion": 4,
    }
    assert baskets.loc[0, "transaction_timestamp"] == baskets.loc[1, "transaction_timestamp"]


def test_schema_timezone_and_basket_consistency_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        ("1", "same", "2017-01-01 10:00:00", 1.0),
        ("2", "same", "2017-01-01 10:00:00", 2.0),
        ("3", "other", "2017-01-02 10:00:00", 3.0),
    ]
    _small_contract(monkeypatch, rows=len(rows))
    frame = _frame(rows)

    with pytest.raises(ValueError, match="multiple households or timestamps"):
        reference.aggregate_transactions(frame)
    with pytest.raises(ValueError, match="columns changed"):
        reference.aggregate_transactions(frame.loc[:, list(reversed(reference.EXPECTED_COLUMNS))])
    aware = frame.copy()
    aware["transaction_timestamp"] = aware["transaction_timestamp"].dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone-naive"):
        reference.aggregate_transactions(aware)
    outside = _frame(
        [
            ("1", "a", "2017-01-01 00:00:00", 1.0),
            ("2", "b", "2017-12-31 23:59:59", 1.0),
            ("3", "c", "2018-01-01 00:00:00", 1.0),
        ]
    )
    with pytest.raises(ValueError, match="frozen observation window"):
        reference.aggregate_transactions(outside)


def test_household_split_is_exact_disjoint_and_input_order_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reference, "EXPECTED_HOUSEHOLDS", 5)
    monkeypatch.setattr(reference, "DEVELOPMENT_HOUSEHOLDS", 2)
    monkeypatch.setattr(reference, "QUALIFICATION_HOUSEHOLDS", 3)
    ids = ["100", "20", "3", "4000", "51"]

    first = reference.deterministic_household_partitions(ids)
    second = reference.deterministic_household_partitions(reversed(ids))
    expected = sorted(ids, key=lambda value: (hashlib.sha256(f"{reference.SPLIT_NAMESPACE}{value}".encode()).digest(), value))

    assert first == second
    assert first["development"] == frozenset(expected[:2])
    assert first["development"].isdisjoint(first["qualification"])
    assert first["development"] | first["qualification"] == frozenset(ids)


def test_preprocess_and_loader_never_open_the_other_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        ("1", "a", "2017-01-01 10:00:00", 1.0),
        ("2", "b", "2017-01-02 10:00:00", 2.0),
        ("3", "c", "2017-01-03 10:00:00", 3.0),
    ]
    _small_contract(monkeypatch, rows=len(rows))
    frame = _frame(rows)
    rds = tmp_path / "source" / "transactions.rds"
    rds.parent.mkdir()
    rds.write_bytes(b"synthetic; converter is replaced in this network-free test")
    rds.with_suffix(".manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        reference.source,
        "verify_existing",
        lambda _path, **_kwargs: {
            "rds_sha256": "a" * 64,
            "source_preregistration_sha256": reference.source.SOURCE_PREREGISTRATION_SHA256,
            "quarantine_receipt_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(reference, "convert_rds_isolated", lambda _rds, output: frame.to_parquet(output, index=False))
    output = tmp_path / "partitions"
    mrs_preregistration, mrs_preregistration_sha256 = _mrs_preregistration(tmp_path)

    manifest = reference.preprocess_completejourney_reference(
        rds,
        output,
        mrs_preregistration_path=mrs_preregistration,
        expected_mrs_preregistration_sha256=mrs_preregistration_sha256,
    )
    manifest_path = output / reference.MANIFEST_FILE
    manifest_sha256 = reference._sha256_file(manifest_path)

    assert manifest["mrs_preregistration_sha256"] == mrs_preregistration_sha256
    assert {
        metadata["mrs_preregistration_sha256"] for metadata in manifest["partitions"].values()
    } == {mrs_preregistration_sha256}
    assert manifest["household_partition_overlap"] == 0
    assert manifest["counts"]["input_item_rows"] == 3
    assert manifest["observation_window"] == {
        "start_inclusive": "2017-01-01T00:00:00",
        "end_exclusive": "2018-01-01T00:00:00",
        "timezone": None,
    }
    assert manifest["preprocessing"]["timestamp_semantics"] == reference.TIMESTAMP_SEMANTICS
    qualification = output / reference.QUALIFICATION_FILE
    hidden_qualification = output / "qualification.hidden"
    qualification.replace(hidden_qualification)
    development_frame, development_provenance = reference.load_completejourney_partition(
        output,
        "development",
        expected_manifest_sha256=manifest_sha256,
        expected_mrs_preregistration_sha256=mrs_preregistration_sha256,
    )
    assert len(development_frame) == development_provenance["baskets"]
    assert development_provenance["observation_start"] == "2017-01-01T00:00:00"
    assert development_provenance["observation_end_exclusive"] == "2018-01-01T00:00:00"
    assert development_provenance["mrs_preregistration_sha256"] == mrs_preregistration_sha256
    assert development_provenance["timestamp_semantics"] == reference.TIMESTAMP_SEMANTICS
    hidden_qualification.replace(qualification)

    development = output / reference.DEVELOPMENT_FILE
    hidden_development = output / "development.hidden"
    development.replace(hidden_development)
    qualification_frame, qualification_provenance = reference.load_completejourney_partition(
        output,
        "qualification",
        expected_manifest_sha256=manifest_sha256,
        expected_mrs_preregistration_sha256=mrs_preregistration_sha256,
    )
    assert len(qualification_frame) == qualification_provenance["baskets"]
    assert set(development_frame["household_id"]).isdisjoint(qualification_frame["household_id"])
    hidden_development.replace(development)

    assert '"household_ids":' not in json.dumps(manifest)
    with pytest.raises(ValueError, match="frozen protocol"):
        reference.load_completejourney_partition(
            output,
            "development",
            expected_manifest_sha256="0" * 64,
            expected_mrs_preregistration_sha256=mrs_preregistration_sha256,
        )

    with pytest.raises(ValueError, match="different MRS1 preregistration"):
        reference.load_completejourney_partition(
            output,
            "development",
            expected_manifest_sha256=manifest_sha256,
            expected_mrs_preregistration_sha256="0" * 64,
        )


def test_missing_or_mismatched_mrs_preregistration_prevents_source_and_converter_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(reference.source, "verify_existing", lambda *_args, **_kwargs: calls.append("source"))
    monkeypatch.setattr(reference, "convert_rds_isolated", lambda *_args, **_kwargs: calls.append("converter"))
    missing = tmp_path / "missing-preregistration.json"

    with pytest.raises(FileNotFoundError, match="preregistration is missing"):
        reference.preprocess_completejourney_reference(
            tmp_path / "transactions.rds",
            tmp_path / "output-missing",
            mrs_preregistration_path=missing,
            expected_mrs_preregistration_sha256="a" * 64,
        )
    assert calls == []

    preregistration, preregistration_sha256 = _mrs_preregistration(tmp_path)
    assert preregistration_sha256 != "b" * 64
    with pytest.raises(ValueError, match="preregistration SHA-256"):
        reference.preprocess_completejourney_reference(
            tmp_path / "transactions.rds",
            tmp_path / "output-mismatch",
            mrs_preregistration_path=preregistration,
            expected_mrs_preregistration_sha256="b" * 64,
        )
    assert calls == []


def test_duplicate_manifest_keys_and_partition_byte_tampering_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        reference._load_json(duplicate)

    rows = [
        ("1", "a", "2017-01-01 10:00:00", 1.0),
        ("2", "b", "2017-01-02 10:00:00", 2.0),
        ("3", "c", "2017-01-03 10:00:00", 3.0),
    ]
    _small_contract(monkeypatch, rows=len(rows))
    frame = _frame(rows)
    rds = tmp_path / "transactions.rds"
    rds.write_bytes(b"synthetic")
    rds.with_suffix(".manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        reference.source,
        "verify_existing",
        lambda _path, **_kwargs: {
            "rds_sha256": "a" * 64,
            "source_preregistration_sha256": reference.source.SOURCE_PREREGISTRATION_SHA256,
            "quarantine_receipt_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(reference, "convert_rds_isolated", lambda _rds, output: frame.to_parquet(output, index=False))
    output = tmp_path / "partitions"
    mrs_preregistration, mrs_preregistration_sha256 = _mrs_preregistration(tmp_path)
    reference.preprocess_completejourney_reference(
        rds,
        output,
        mrs_preregistration_path=mrs_preregistration,
        expected_mrs_preregistration_sha256=mrs_preregistration_sha256,
    )
    manifest_sha256 = reference._sha256_file(output / reference.MANIFEST_FILE)
    selected = output / reference.DEVELOPMENT_FILE
    selected.chmod(0o644)
    payload = selected.read_bytes()
    selected.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))

    with pytest.raises(ValueError, match="Parquet SHA-256"):
        reference.load_completejourney_partition(
            output,
            "development",
            expected_manifest_sha256=manifest_sha256,
            expected_mrs_preregistration_sha256=mrs_preregistration_sha256,
        )
