"""Trusted Complete Journey basket preprocessing and partition-scoped loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import download_completejourney_reference as source


EXPECTED_COLUMNS = (
    "household_id",
    "store_id",
    "basket_id",
    "product_id",
    "quantity",
    "sales_value",
    "retail_disc",
    "coupon_disc",
    "coupon_match_disc",
    "week",
    "transaction_timestamp",
)
USED_COLUMNS = ("household_id", "basket_id", "transaction_timestamp", "sales_value")
PARTITION_COLUMNS = ("household_id", "basket_id", "transaction_timestamp", "sales_value_cents")
EXPECTED_ITEM_ROWS = 1_469_307
EXPECTED_HOUSEHOLDS = 2_469
DEVELOPMENT_HOUSEHOLDS = 988
QUALIFICATION_HOUSEHOLDS = 1_481
SPLIT_NAMESPACE = "completejourney-v1.1.1:mrs1:"
DEVELOPMENT_FILE = "completejourney_development.parquet"
QUALIFICATION_FILE = "completejourney_qualification.parquet"
MANIFEST_FILE = "completejourney_reference.manifest.json"
DEFAULT_MRS_PREREGISTRATION = Path("results/entity_fidelity_mrs1_r2_preregistration.json")
MRS_PREREGISTRATION_ARTIFACT = "entity_fidelity_mrs1_r2_preregistration"
MRS_STUDY_REVISION = "mrs1.2"
OBSERVATION_START = "2017-01-01T00:00:00"
OBSERVATION_END_EXCLUSIVE = "2018-01-01T00:00:00"
SOURCE_TIMEZONE = "America/New_York"
TIMESTAMP_SEMANTICS = (
    "source timezone-aware POSIXct converted to America/New_York local wall-clock, "
    "then timezone removed"
)
UV_VERSION = "0.11.8"
CONVERTER_REQUIREMENTS = ("pyreadr==0.5.6", "pandas==3.0.5", "pyarrow==21.0.0")
_UV_VERSION_OUTPUT = re.compile(
    r"uv (?P<version>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))(?: \([^()\r\n]+\))?"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _verify_mrs_preregistration(path: Path, expected_sha256: str) -> str:
    if not _is_sha256(expected_sha256):
        raise ValueError("a lowercase MRS1 preregistration SHA-256 is required")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("canonical MRS1 preregistration is missing or symbolic")
    metadata = path.stat()
    if (metadata.st_mode & 0o7777) != 0o444:
        raise ValueError("canonical MRS1 preregistration must have mode 0444")
    if metadata.st_size > 1_000_000:
        raise ValueError("MRS1 preregistration is unexpectedly large")
    raw = path.read_bytes()
    if len(raw) != metadata.st_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("MRS1 preregistration SHA-256 differs from the frozen lifecycle")
    try:
        value = json.loads(raw)
        canonical = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        raise ValueError("MRS1 preregistration is not canonical JSON") from error
    if (
        not isinstance(value, dict)
        or value.get("artifact") != MRS_PREREGISTRATION_ARTIFACT
        or value.get("study_revision") != MRS_STUDY_REVISION
        or raw != canonical
    ):
        raise ValueError("MRS1 preregistration is not the canonical lifecycle artifact")
    return expected_sha256


def _converter_program() -> str:
    return f"""
import sys
import pandas as pd
import pyreadr

expected_columns = {EXPECTED_COLUMNS!r}
expected_rows = {EXPECTED_ITEM_ROWS}
objects = pyreadr.read_r(sys.argv[1], timezone={SOURCE_TIMEZONE!r})
if list(objects) != [None]:
    raise ValueError("transactions.rds must contain exactly one unnamed object")
frame = objects[None]
if not isinstance(frame, pd.DataFrame):
    raise TypeError("transactions.rds object is not a data frame")
if tuple(frame.columns) != expected_columns or len(frame) != expected_rows:
    raise ValueError("Complete Journey RDS schema or row count changed")
timestamps = frame["transaction_timestamp"]
if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
    frame["transaction_timestamp"] = timestamps.dt.tz_localize(None)
elif not pd.api.types.is_datetime64_any_dtype(timestamps.dtype):
    raise TypeError("transaction_timestamp is not an R datetime")
frame["transaction_timestamp"] = frame["transaction_timestamp"].astype("datetime64[ns]")
frame.to_parquet(sys.argv[2], engine="pyarrow", compression="zstd", index=False)
""".strip()


def converter_identity() -> dict[str, object]:
    program = _converter_program().encode()
    return {
        "runner": "uv",
        "uv_version": UV_VERSION,
        "python": "3.11",
        "requirements": list(CONVERTER_REQUIREMENTS),
        "source_timezone": SOURCE_TIMEZONE,
        "program_sha256": hashlib.sha256(program).hexdigest(),
    }


def _require_uv_version(version_output: str) -> None:
    reported = version_output.strip()
    match = _UV_VERSION_OUTPUT.fullmatch(reported)
    if match is None or match.group("version") != UV_VERSION:
        raise RuntimeError(f"Complete Journey converter requires uv {UV_VERSION}, found {reported!r}")


def convert_rds_isolated(rds_path: Path, parquet_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for the isolated Complete Journey converter")
    version_output = subprocess.run(
        [uv, "--version"], check=True, capture_output=True, text=True, timeout=10
    ).stdout
    _require_uv_version(version_output)
    command = [
        uv,
        "run",
        "--quiet",
        "--isolated",
        "--no-project",
        "--no-config",
        "--no-env-file",
        "--no-python-downloads",
        "--python",
        "3.11",
    ]
    for requirement in CONVERTER_REQUIREMENTS:
        command.extend(("--with", requirement))
    command.extend(("python", "-I", "-c", _converter_program(), str(rds_path), str(parquet_path)))
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    subprocess.run(command, check=True, env=environment, timeout=900)


def _validate_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or any(ord(character) < 32 for character in value):
        raise ValueError(f"{field} must contain nonempty canonical strings")
    return value


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("sales_value must be finite numeric data")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("sales_value must be finite numeric data") from error
    if not result.is_finite():
        raise ValueError("sales_value must be finite numeric data")
    return result


def _assert_observation_window(timestamps: pd.Series) -> None:
    if (
        (timestamps < pd.Timestamp(OBSERVATION_START))
        | (timestamps >= pd.Timestamp(OBSERVATION_END_EXCLUSIVE))
    ).any():
        raise ValueError("retained Complete Journey basket falls outside the frozen observation window")


def _aggregate_selected(selected: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if tuple(selected.columns) != USED_COLUMNS or len(selected) != EXPECTED_ITEM_ROWS:
        raise ValueError("Complete Journey selected source contract changed")
    if selected.isna().any().any():
        raise ValueError("Complete Journey basket source fields contain missing values")
    if not pd.api.types.is_datetime64_any_dtype(selected["transaction_timestamp"].dtype):
        raise TypeError("transaction_timestamp must be datetime data")
    if selected["transaction_timestamp"].dt.tz is not None:
        raise ValueError("transaction_timestamp must be timezone-naive local wall-clock data")
    if str(selected["transaction_timestamp"].dtype) != "datetime64[ns]":
        raise ValueError("transaction_timestamp must use nanosecond local wall-clock precision")

    baskets: dict[str, list[Any]] = {}
    households: set[str] = set()
    with localcontext() as context:
        context.prec = 50
        for household_value, basket_value, timestamp, sales_value in selected.itertuples(index=False, name=None):
            household_id = _validate_identifier(household_value, "household_id")
            basket_id = _validate_identifier(basket_value, "basket_id")
            households.add(household_id)
            amount = _decimal(sales_value)
            existing = baskets.get(basket_id)
            if existing is None:
                baskets[basket_id] = [household_id, timestamp, amount]
            else:
                if existing[0] != household_id or existing[1] != timestamp:
                    raise ValueError("basket_id spans multiple households or timestamps")
                existing[2] += amount

        if len(households) != EXPECTED_HOUSEHOLDS:
            raise ValueError("Complete Journey household count changed")
        records: list[tuple[str, str, Any, int]] = []
        excluded = 0
        for basket_id, (household_id, timestamp, total) in baskets.items():
            cents = int(total.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN) * 100)
            if cents <= 0:
                excluded += 1
            else:
                records.append((household_id, basket_id, timestamp, cents))

    records.sort(key=lambda row: (row[0], row[2], row[1]))
    result = pd.DataFrame.from_records(records, columns=PARTITION_COLUMNS)
    if result.empty or result["household_id"].nunique() != EXPECTED_HOUSEHOLDS:
        raise ValueError("every Complete Journey household must retain a positive basket")
    result["transaction_timestamp"] = result["transaction_timestamp"].astype("datetime64[ns]")
    result["sales_value_cents"] = result["sales_value_cents"].astype("int64")
    _assert_observation_window(result["transaction_timestamp"])
    return result, {
        "input_item_rows": len(selected),
        "input_households": len(households),
        "baskets_before_exclusion": len(baskets),
        "excluded_nonpositive_baskets": excluded,
        "baskets_after_exclusion": len(result),
    }


def aggregate_transactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate the official item contract and reduce it to positive basket totals."""
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError("Complete Journey transaction columns changed")
    if len(frame) != EXPECTED_ITEM_ROWS:
        raise ValueError("Complete Journey item row count changed")
    return _aggregate_selected(frame.loc[:, USED_COLUMNS])


def deterministic_household_partitions(household_ids: Iterable[str]) -> dict[str, frozenset[str]]:
    identifiers = {_validate_identifier(value, "household_id") for value in household_ids}
    if len(identifiers) != EXPECTED_HOUSEHOLDS:
        raise ValueError("Complete Journey household split requires exactly 2,469 households")
    ranked = sorted(
        identifiers,
        key=lambda household_id: (hashlib.sha256(f"{SPLIT_NAMESPACE}{household_id}".encode()).digest(), household_id),
    )
    development = frozenset(ranked[:DEVELOPMENT_HOUSEHOLDS])
    qualification = frozenset(ranked[DEVELOPMENT_HOUSEHOLDS:])
    if len(development) != DEVELOPMENT_HOUSEHOLDS or len(qualification) != QUALIFICATION_HOUSEHOLDS or development & qualification:
        raise ValueError("Complete Journey household partition contract failed")
    return {"development": development, "qualification": qualification}


def _household_ids_sha256(ids: Iterable[str]) -> str:
    payload = "completejourney-v1.1.1:mrs1:partition-households\n" + "\n".join(sorted(ids)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def _partition_frame(frame: pd.DataFrame, ids: frozenset[str]) -> pd.DataFrame:
    partition = frame.loc[frame["household_id"].isin(ids), PARTITION_COLUMNS].copy()
    partition.sort_values(["household_id", "transaction_timestamp", "basket_id"], kind="stable", inplace=True)
    partition.reset_index(drop=True, inplace=True)
    if set(partition["household_id"].unique()) != set(ids):
        raise ValueError("Complete Journey partition lost assigned households")
    return partition


def _write_partition(frame: pd.DataFrame, path: Path) -> None:
    schema = pa.schema(
        [
            pa.field("household_id", pa.string(), nullable=False),
            pa.field("basket_id", pa.string(), nullable=False),
            pa.field("transaction_timestamp", pa.timestamp("ns"), nullable=False),
            pa.field("sales_value_cents", pa.int64(), nullable=False),
        ]
    )
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=["household_id"],
        version="2.6",
        write_statistics=True,
    )


def _static_manifest_fields() -> dict[str, object]:
    return {
        "source_columns": list(EXPECTED_COLUMNS),
        "columns_used": list(USED_COLUMNS),
        "output_columns": list(PARTITION_COLUMNS),
        "amount_rule": "sum Decimal(str(sales_value)); quantize once to cents with ROUND_HALF_EVEN; exclude cents <= 0",
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "observation_window_rule": "frozen Complete Journey 2017 local calendar year; start inclusive, end exclusive",
        "split_namespace": SPLIT_NAMESPACE,
        "split_rule": "ascending full SHA256(namespace + household_id), household_id tie-break",
        "converter": converter_identity(),
    }


def _partition_metadata(file: str, frame: pd.DataFrame, path: Path) -> dict[str, object]:
    ids = frozenset(frame["household_id"].unique())
    return {
        "file": file,
        "households": len(ids),
        "baskets": len(frame),
        "household_ids_sha256": _household_ids_sha256(ids),
        "parquet_sha256": _sha256_file(path),
    }


def preprocess_completejourney_reference(
    rds_path: Path,
    output_dir: Path,
    *,
    mrs_preregistration_path: Path,
    expected_mrs_preregistration_sha256: str,
    source_preregistration_path: Path = source.DEFAULT_SOURCE_PREREGISTRATION,
    quarantine_receipt_path: Path = source.DEFAULT_QUARANTINE_RECEIPT,
) -> dict[str, object]:
    """Verify, convert, aggregate, split, and atomically publish both partitions once."""
    mrs_preregistration_sha256 = _verify_mrs_preregistration(
        mrs_preregistration_path,
        expected_mrs_preregistration_sha256,
    )
    source_manifest = source.verify_existing(
        rds_path,
        preregistration_path=source_preregistration_path,
        receipt_path=quarantine_receipt_path,
    )
    source_manifest_path = rds_path.with_suffix(".manifest.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = (
        output_dir / DEVELOPMENT_FILE,
        output_dir / QUALIFICATION_FILE,
        output_dir / MANIFEST_FILE,
    )
    if any(target.exists() or target.is_symlink() for target in targets):
        raise FileExistsError("Complete Journey partition artifacts are immutable and already exist")

    with tempfile.TemporaryDirectory(prefix=".completejourney-preprocess-", dir=output_dir) as directory:
        temporary = Path(directory)
        raw_parquet = temporary / "transactions.full.parquet"
        convert_rds_isolated(rds_path, raw_parquet)
        raw_file = pq.ParquetFile(raw_parquet)
        if tuple(raw_file.schema_arrow.names) != EXPECTED_COLUMNS or raw_file.metadata.num_rows != EXPECTED_ITEM_ROWS:
            raise ValueError("isolated converter output violates the Complete Journey raw contract")
        raw_frame = pd.read_parquet(raw_parquet, columns=list(USED_COLUMNS))
        baskets, counts = _aggregate_selected(raw_frame)
        split = deterministic_household_partitions(baskets["household_id"])
        development = _partition_frame(baskets, split["development"])
        qualification = _partition_frame(baskets, split["qualification"])
        if set(development["household_id"]) & set(qualification["household_id"]):
            raise ValueError("Complete Journey household partitions overlap")

        development_part = temporary / DEVELOPMENT_FILE
        qualification_part = temporary / QUALIFICATION_FILE
        manifest_part = temporary / MANIFEST_FILE
        _write_partition(development, development_part)
        _write_partition(qualification, qualification_part)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "dataset": "completejourney",
            "version": source.VERSION,
            "license": source.LICENSE,
            "mrs_preregistration_sha256": mrs_preregistration_sha256,
            "source": {
                "git_tag": source.GIT_TAG,
                "git_commit": source.GIT_COMMIT,
                "source_url": source.SOURCE_URL,
                "git_blob_sha1": source.GIT_BLOB_SHA1,
                "rds_size_bytes": source.RDS_SIZE_BYTES,
                "rds_sha256": source_manifest["rds_sha256"],
                "manifest_sha256": _sha256_file(source_manifest_path),
                "source_preregistration_sha256": source_manifest["source_preregistration_sha256"],
                "quarantine_receipt_sha256": source_manifest["quarantine_receipt_sha256"],
            },
            "preprocessing": _static_manifest_fields(),
            "observation_window": {
                "start_inclusive": OBSERVATION_START,
                "end_exclusive": OBSERVATION_END_EXCLUSIVE,
                "timezone": None,
            },
            "counts": counts,
            "household_partition_overlap": 0,
            "partitions": {
                "development": {
                    **_partition_metadata(DEVELOPMENT_FILE, development, development_part),
                    "mrs_preregistration_sha256": mrs_preregistration_sha256,
                },
                "qualification": {
                    **_partition_metadata(QUALIFICATION_FILE, qualification, qualification_part),
                    "mrs_preregistration_sha256": mrs_preregistration_sha256,
                },
            },
        }
        _validate_manifest(manifest)
        manifest_part.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for path in (development_part, qualification_part, manifest_part):
            path.chmod(0o444)
        os.replace(development_part, targets[0])
        os.replace(qualification_part, targets[1])
        os.replace(manifest_part, targets[2])
        return manifest


def _require_keys(value: object, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{where} keys violate the Complete Journey manifest contract")
    return value


def _validate_manifest(manifest: object) -> dict[str, Any]:
    value = _require_keys(
        manifest,
        {"schema_version", "dataset", "version", "license", "mrs_preregistration_sha256", "source", "preprocessing", "observation_window", "counts", "household_partition_overlap", "partitions"},
        "manifest",
    )
    if value["schema_version"] != 1 or value["dataset"] != "completejourney" or value["version"] != source.VERSION or value["license"] != source.LICENSE:
        raise ValueError("Complete Journey manifest identity changed")
    if not _is_sha256(value["mrs_preregistration_sha256"]):
        raise ValueError("Complete Journey MRS1 preregistration binding is missing")
    source_value = _require_keys(
        value["source"],
        {"git_tag", "git_commit", "source_url", "git_blob_sha1", "rds_size_bytes", "rds_sha256", "manifest_sha256", "source_preregistration_sha256", "quarantine_receipt_sha256"},
        "source",
    )
    expected_source = {
        "git_tag": source.GIT_TAG,
        "git_commit": source.GIT_COMMIT,
        "source_url": source.SOURCE_URL,
        "git_blob_sha1": source.GIT_BLOB_SHA1,
        "rds_size_bytes": source.RDS_SIZE_BYTES,
        "rds_sha256": source._frozen_sha256(),
    }
    if (
        any(source_value[key] != expected for key, expected in expected_source.items())
        or not _is_sha256(source_value["manifest_sha256"])
        or source_value["source_preregistration_sha256"] != source.SOURCE_PREREGISTRATION_SHA256
        or not _is_sha256(source_value["quarantine_receipt_sha256"])
    ):
        raise ValueError("Complete Journey source binding changed")
    if value["preprocessing"] != _static_manifest_fields():
        raise ValueError("Complete Journey preprocessing contract changed")
    if value["observation_window"] != {
        "start_inclusive": OBSERVATION_START,
        "end_exclusive": OBSERVATION_END_EXCLUSIVE,
        "timezone": None,
    }:
        raise ValueError("Complete Journey observation window changed")
    counts = _require_keys(
        value["counts"],
        {"input_item_rows", "input_households", "baskets_before_exclusion", "excluded_nonpositive_baskets", "baskets_after_exclusion"},
        "counts",
    )
    if (
        counts["input_item_rows"] != EXPECTED_ITEM_ROWS
        or counts["input_households"] != EXPECTED_HOUSEHOLDS
        or not all(isinstance(counts[key], int) and not isinstance(counts[key], bool) and counts[key] >= 0 for key in counts)
        or counts["baskets_before_exclusion"] - counts["excluded_nonpositive_baskets"] != counts["baskets_after_exclusion"]
        or value["household_partition_overlap"] != 0
    ):
        raise ValueError("Complete Journey aggregate counts changed")
    partitions = _require_keys(value["partitions"], {"development", "qualification"}, "partitions")
    total_baskets = 0
    for name, filename, households in (
        ("development", DEVELOPMENT_FILE, DEVELOPMENT_HOUSEHOLDS),
        ("qualification", QUALIFICATION_FILE, QUALIFICATION_HOUSEHOLDS),
    ):
        metadata = _require_keys(
            partitions[name],
            {"file", "households", "baskets", "household_ids_sha256", "parquet_sha256", "mrs_preregistration_sha256"},
            f"partition {name}",
        )
        if (
            metadata["file"] != filename
            or metadata["households"] != households
            or not isinstance(metadata["baskets"], int)
            or isinstance(metadata["baskets"], bool)
            or metadata["baskets"] <= 0
            or not _is_sha256(metadata["household_ids_sha256"])
            or not _is_sha256(metadata["parquet_sha256"])
            or metadata["mrs_preregistration_sha256"] != value["mrs_preregistration_sha256"]
        ):
            raise ValueError(f"Complete Journey {name} partition metadata changed")
        total_baskets += metadata["baskets"]
    if total_baskets != counts["baskets_after_exclusion"]:
        raise ValueError("Complete Journey partition basket counts are not exhaustive")
    return value


def _load_json(path: Path) -> object:
    if path.stat().st_size > 1_000_000:
        raise ValueError("Complete Journey manifest is unexpectedly large")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Complete Journey manifest contains duplicate JSON keys")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def _read_partition(path: Path) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    expected_schema = pa.schema(
        [
            pa.field("household_id", pa.string(), nullable=False),
            pa.field("basket_id", pa.string(), nullable=False),
            pa.field("transaction_timestamp", pa.timestamp("ns"), nullable=False),
            pa.field("sales_value_cents", pa.int64(), nullable=False),
        ]
    )
    if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
        raise ValueError("Complete Journey partition Parquet schema changed")
    return parquet.read().to_pandas()


def _validate_partition_frame(frame: pd.DataFrame, metadata: dict[str, Any]) -> None:
    if tuple(frame.columns) != PARTITION_COLUMNS or len(frame) != metadata["baskets"]:
        raise ValueError("Complete Journey partition dimensions changed")
    if frame.isna().any().any() or frame["basket_id"].duplicated().any() or (frame["sales_value_cents"] <= 0).any():
        raise ValueError("Complete Journey partition contains invalid baskets")
    _assert_observation_window(frame["transaction_timestamp"])
    ids = frozenset(_validate_identifier(value, "household_id") for value in frame["household_id"])
    if len(ids) != metadata["households"] or _household_ids_sha256(ids) != metadata["household_ids_sha256"]:
        raise ValueError("Complete Journey partition household binding changed")
    ordering = pd.MultiIndex.from_frame(frame.loc[:, ["household_id", "transaction_timestamp", "basket_id"]])
    if not ordering.is_monotonic_increasing:
        raise ValueError("Complete Journey partition baskets are not canonically ordered")


def load_completejourney_partition(
    output_dir: Path,
    partition: Literal["development", "qualification"],
    *,
    expected_manifest_sha256: str,
    expected_mrs_preregistration_sha256: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and validate only the explicitly requested, hash-bound partition."""
    if partition not in {"development", "qualification"}:
        raise ValueError("unknown Complete Journey partition")
    if not _is_sha256(expected_manifest_sha256):
        raise ValueError("a frozen Complete Journey manifest SHA-256 is required")
    if not _is_sha256(expected_mrs_preregistration_sha256):
        raise ValueError("a frozen MRS1 preregistration SHA-256 is required")
    manifest_path = output_dir / MANIFEST_FILE
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("trusted Complete Journey manifest is missing or symbolic")
    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Complete Journey manifest SHA-256 differs from the frozen protocol")
    manifest = _validate_manifest(_load_json(manifest_path))
    if manifest["mrs_preregistration_sha256"] != expected_mrs_preregistration_sha256:
        raise ValueError("Complete Journey manifest belongs to a different MRS1 preregistration")
    metadata = manifest["partitions"][partition]
    if metadata["mrs_preregistration_sha256"] != expected_mrs_preregistration_sha256:
        raise ValueError(f"Complete Journey {partition} partition belongs to a different MRS1 preregistration")
    filename = DEVELOPMENT_FILE if partition == "development" else QUALIFICATION_FILE
    path = output_dir / filename
    if metadata["file"] != filename or path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"trusted Complete Journey {partition} partition is missing or symbolic")
    if _sha256_file(path) != metadata["parquet_sha256"]:
        raise ValueError(f"Complete Journey {partition} Parquet SHA-256 changed")
    frame = _read_partition(path)
    _validate_partition_frame(frame, metadata)
    if _sha256_file(path) != metadata["parquet_sha256"]:
        raise ValueError(f"Complete Journey {partition} Parquet changed while it was read")
    provenance = {
        "dataset": "completejourney",
        "version": source.VERSION,
        "partition": partition,
        "manifest_sha256": manifest_sha256,
        "partition_sha256": metadata["parquet_sha256"],
        "household_ids_sha256": metadata["household_ids_sha256"],
        "households": metadata["households"],
        "baskets": metadata["baskets"],
        "source_rds_sha256": manifest["source"]["rds_sha256"],
        "mrs_preregistration_sha256": expected_mrs_preregistration_sha256,
        "timestamp_semantics": manifest["preprocessing"]["timestamp_semantics"],
        "observation_start": OBSERVATION_START,
        "observation_end_exclusive": OBSERVATION_END_EXCLUSIVE,
        "amount_unit": "USD cents",
    }
    return frame, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rds", type=Path, default=Path("data/reference/completejourney/transactions.rds"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/reference/completejourney"))
    parser.add_argument("--mrs-preregistration", type=Path, default=DEFAULT_MRS_PREREGISTRATION)
    parser.add_argument("--mrs-preregistration-sha256", required=True)
    parser.add_argument(
        "--source-preregistration",
        type=Path,
        default=source.DEFAULT_SOURCE_PREREGISTRATION,
    )
    parser.add_argument("--quarantine-receipt", type=Path, default=source.DEFAULT_QUARANTINE_RECEIPT)
    args = parser.parse_args()
    print(
        json.dumps(
            preprocess_completejourney_reference(
                args.rds,
                args.output_dir,
                mrs_preregistration_path=args.mrs_preregistration,
                expected_mrs_preregistration_sha256=args.mrs_preregistration_sha256,
                source_preregistration_path=args.source_preregistration,
                quarantine_receipt_path=args.quarantine_receipt,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
