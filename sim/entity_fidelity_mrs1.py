"""Fail-closed fitting and staged fidelity gates for the MRS1 model family."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol

from sim.entity_fidelity import (
    EXPECTED_PARTITION_HASHES,
    FEATURES,
    _classifier,
    _classifier_protocol,
    _dependencies,
    _ids_hash,
    _transactions_hash,
    customer_features,
    fit_calibrations,
)
from scripts.download_retail_reference import (
    DATASET_ID as UCI352_DATASET_ID,
    LICENSE as UCI352_LICENSE,
    SOURCE_URL as UCI352_SOURCE_URL,
    WORKBOOK_SHA256 as UCI352_WORKBOOK_SHA256,
)
from sim.entity_fidelity_v4 import (
    BOOTSTRAP_REPEATS,
    CROSS_FIT_FOLDS,
    EVALUATION_SEEDS,
    GATE_MAX_D_UPPER_95,
    UCI502_ARCHIVE_SHA256,
    UCI502_DATASET_ID,
    UCI502_LICENSE,
    UCI502_SHEET,
    UCI502_SOURCE_URL,
    UCI502_WINDOW_END,
    UCI502_WINDOW_START,
    UCI502_WORKBOOK_SHA256,
    _crossfit_max_seed_bootstrap,
)
from sim.population_mrs1 import (
    JEFFREYS_PSEUDOCOUNT,
    STUDENT_T_DF,
    CalendarHazardClock,
    MRS1Profile,
    generate_synthetic_transactions_mrs1,
    sequence_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
SUPERSEDED_PREREGISTRATION = (
    ROOT / "results" / "entity_fidelity_mrs1_preregistration.json"
)
DEFAULT_PREFLIGHT_FAILURE = (
    ROOT / "results" / "entity_fidelity_mrs1_preflight_failure.json"
)
R1_PREREGISTRATION = (
    ROOT / "results" / "entity_fidelity_mrs1_r1_preregistration.json"
)
R1_PREPROCESS_FAILURE = (
    ROOT / "results" / "entity_fidelity_mrs1_r1_preprocess_failure.json"
)
DEFAULT_PREREGISTRATION = (
    ROOT / "results" / "entity_fidelity_mrs1_r2_preregistration.json"
)
DEFAULT_PROTOCOL = ROOT / "results" / "entity_fidelity_mrs1_protocol.json"
DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR = ROOT / "data" / "reference" / "completejourney"
DEFAULT_QUALIFICATION_RECEIPT = ROOT / "results" / "entity_fidelity_mrs1_qualification_receipt.json"
DEFAULT_QUALIFICATION = ROOT / "results" / "entity_fidelity_mrs1_qualification.json"
DEFAULT_UCI352_TRANSFER = ROOT / "results" / "entity_fidelity_mrs1_uci352_transfer.json"
DEFAULT_UCI352_REFERENCE = ROOT / "data" / "reference" / "online_retail.xlsx"
DEFAULT_UCI502_REFERENCE = ROOT / "data" / "reference" / "online_retail_ii.xlsx"
DEFAULT_FINAL_RECEIPT = ROOT / "results" / "entity_fidelity_mrs1_final_receipt.json"
DEFAULT_FINAL = ROOT / "results" / "entity_fidelity_mrs1_final.json"
DEFAULT_COMPLETION_SEAL = ROOT / "results" / "entity_fidelity_mrs1_completion_seal.json"

SCHEMA_VERSION = "mrs1"
STUDY_REVISION = "mrs1.2"
SUPERSEDED_PREREGISTRATION_SHA256 = (
    "648c0d1d8020c06c6d184cf451c9caff90acd401e6db3b62393b91439ce3cb7a"
)
PREFLIGHT_FAILURE_SHA256 = (
    "10f4cba07c6d37f6012c04a81d0468409a38dc26334774bdfb34cb8223b2aca9"
)
SUPERSEDED_PREPROCESSOR_SHA256 = (
    "c5c1a45807be5e7a0e4ead43124027141290e59656539d4890ff1c82d9f29be0"
)
R1_PREREGISTRATION_SHA256 = (
    "24c080955926b7c4ea42ef8d590f3b4e8250bc755ddbb25b6a90385a95357b2e"
)
R1_PREPROCESS_FAILURE_SHA256 = (
    "9aa113dbaea1b72f8b1cd0cae8a6a167290fabcc1e008f4f2b6289e01c941099"
)
R1_PREPROCESSOR_SHA256 = (
    "46c69ab0de5fa5ded4bc2941745c7c676c2feade2739634a9e45eced8cdb3b57"
)
DEVELOPMENT_HOUSEHOLDS = 988
QUALIFICATION_HOUSEHOLDS = 1_481
COMPLETEJOURNEY_OBSERVATION_START = "2017-01-01T00:00:00"
COMPLETEJOURNEY_OBSERVATION_END = "2018-01-01T00:00:00"
COMPLETEJOURNEY_SOURCE_TIMEZONE = "America/New_York"
COMPLETEJOURNEY_TIMESTAMP_SEMANTICS = (
    "source timezone-aware POSIXct converted to America/New_York local wall-clock, "
    "then timezone removed"
)
UCI352_MANIFEST_SHA256 = "0fca9904171c1382bfdd8dc1369929a310817a63472474dc61dee4a4ae800cb9"
UCI502_MANIFEST_SHA256 = "8092c0b746d0c03d319570bc483e1f3085f057fc9e42f0de8c2b258524573f38"
UCI352_OBSERVATION_START = "2010-12-01T00:00:00"
UCI352_OBSERVATION_END = "2011-12-10T00:00:00"
UCI352_CALIBRATION_CUSTOMERS = 2_440
UCI352_CALIBRATION_TRANSACTIONS = 10_621
SEED = 20260827
FULL_BOOTSTRAP_SEED = 20260857
GAUSS_LAGUERRE_NODES = 48
OPTIMIZER_METHOD = "L-BFGS-B"
OPTIMIZER_MAXITER = 600
BOUNDARY_TOLERANCE = 1e-5
TIMING_PARAMETER_BOUNDS = (
    (math.log(0.25), math.log(10.0)),
    (math.log(0.05), math.log(365.0)),
    (math.log(0.25), math.log(20.0)),
)
TIMING_STARTS = (
    (math.log(1.0), math.log(14.0), math.log(1.0)),
    (math.log(2.0), math.log(30.0), math.log(2.0)),
    (math.log(0.6), math.log(7.0), math.log(0.5)),
)
AMOUNT_PARAMETER_BOUNDS = (
    (-20.0, 30.0),
    (-5.0, 5.0),
    (-5.0, 5.0),
    (-0.95, 0.95),
    (-6.0, 6.0),
)
AMOUNT_STARTS = (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (5.0, 0.0, 0.0, 0.3, 0.0),
    (10.0, 0.0, 0.0, -0.3, 1.0),
)
DEPENDENCY_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "lightgbm",
)
FINAL_CONFIRMATION = "APPROVED_SINGLE_FINAL_SCORE"
FORBIDDEN_ARTIFACT_KEYS = {
    "customer_id",
    "customer_ids",
    "_real_customer_ids",
    "raw_rows",
    "transaction_rows",
    "timestamps",
    "trajectories",
    "predictions",
    "prediction_vectors",
    "_real_predictions",
    "_synthetic_predictions",
    "posterior_weights",
    "posterior_log_frailty",
}


def _uci502_final_source() -> dict[str, Any]:
    return {
        "dataset_id": UCI502_DATASET_ID,
        "source_url": UCI502_SOURCE_URL,
        "archive_sha256": UCI502_ARCHIVE_SHA256,
        "workbook_sha256": UCI502_WORKBOOK_SHA256,
        "manifest_sha256": UCI502_MANIFEST_SHA256,
        "license": UCI502_LICENSE,
        "downloader_sha256": _file_sha256(
            ROOT / "scripts" / "download_retail_ii_reference.py"
        ),
        "sheet": UCI502_SHEET,
        "fixed_window": f"[{UCI502_WINDOW_START}, {UCI502_WINDOW_END})",
        "observation_start": UCI502_WINDOW_START,
        "observation_end": UCI502_WINDOW_END,
    }


def _uci352_transfer_source() -> dict[str, Any]:
    return {
        "dataset_id": UCI352_DATASET_ID,
        "source_url": UCI352_SOURCE_URL,
        "workbook_sha256": UCI352_WORKBOOK_SHA256,
        "manifest_sha256": UCI352_MANIFEST_SHA256,
        "license": UCI352_LICENSE,
        "partition": "calibration",
        "customer_ids_sha256": EXPECTED_PARTITION_HASHES["calibration"],
        "customers": UCI352_CALIBRATION_CUSTOMERS,
        "transactions": UCI352_CALIBRATION_TRANSACTIONS,
        "observation_start": UCI352_OBSERVATION_START,
        "observation_end": UCI352_OBSERVATION_END,
    }


class CompleteJourneyLoader(Protocol):
    """Exact callable shape of sim.completejourney_reference's trusted loader."""

    def __call__(
        self,
        output_dir: Path,
        partition: Literal["development", "qualification"],
        *,
        expected_manifest_sha256: str,
        expected_mrs_preregistration_sha256: str,
    ) -> tuple[Any, dict[str, object]]: ...


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_hashes() -> dict[str, str]:
    paths = (
        ROOT / "sim" / "population_mrs1.py",
        ROOT / "sim" / "entity_fidelity_mrs1.py",
        ROOT / "tests" / "test_population_mrs1.py",
        ROOT / "tests" / "test_entity_fidelity_mrs1.py",
        ROOT / "sim" / "entity_fidelity.py",
        ROOT / "sim" / "entity_fidelity_v4.py",
        ROOT / "sim" / "completejourney_reference.py",
        ROOT / "scripts" / "download_completejourney_reference.py",
        ROOT / "scripts" / "download_retail_reference.py",
        ROOT / "scripts" / "download_retail_ii_reference.py",
    )
    return {
        str(path.relative_to(ROOT)): _file_sha256(path)
        for path in paths
        if path.exists()
    }


def _dependency_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in DEPENDENCY_PACKAGES}


def _reference_source_chain() -> dict[str, str]:
    """Verify source custody artifacts without opening the retained RDS bytes."""
    from scripts import download_completejourney_reference as source

    preregistration_path = ROOT / source.DEFAULT_SOURCE_PREREGISTRATION
    receipt_path = ROOT / source.DEFAULT_QUARANTINE_RECEIPT
    manifest_path = ROOT / "data/reference/completejourney/transactions.manifest.json"
    preregistration_sha256 = source._verify_source_preregistration(
        preregistration_path
    )
    receipt, receipt_sha256 = source._verify_quarantine_receipt(
        receipt_path, preregistration_sha256
    )
    if receipt["observed_rds_sha256"] != source._frozen_sha256():
        raise ValueError("Complete Journey frozen RDS pin differs from custody receipt")
    manifest = json.loads(manifest_path.read_bytes())
    if (
        manifest.get("rds_sha256") != source._frozen_sha256()
        or manifest.get("source_preregistration_sha256")
        != preregistration_sha256
        or manifest.get("quarantine_receipt_sha256") != receipt_sha256
    ):
        raise ValueError("Complete Journey retained-source manifest lineage changed")
    return {
        "downloader_sha256": _file_sha256(
            ROOT / "scripts/download_completejourney_reference.py"
        ),
        "preprocessor_sha256": _file_sha256(
            ROOT / "sim/completejourney_reference.py"
        ),
        "source_preregistration_sha256": preregistration_sha256,
        "quarantine_receipt_sha256": receipt_sha256,
        "rds_manifest_sha256": _file_sha256(manifest_path),
        "rds_sha256": source._frozen_sha256(),
    }


def _assert_aggregate_only(value: Any, *, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_ARTIFACT_KEYS:
                raise ValueError(f"{path}.{key} is forbidden individual-level material")
            _assert_aggregate_only(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_aggregate_only(item, path=f"{path}[{index}]")


def _write_artifact(path: Path, value: Mapping[str, Any]) -> bytes:
    _assert_aggregate_only(value)
    encoded = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return encoded


def _read_artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{path.name} is missing or symbolic")
    if path.stat().st_mode & 0o222:
        raise ValueError(f"{path.name} must be immutable before it is consumed")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    if raw != _canonical_bytes(value):
        raise ValueError(f"{path.name} is not canonical JSON")
    _assert_aggregate_only(value)
    return value


def _as_local_wall(value: Any) -> datetime:
    timestamp = value.to_pydatetime() if hasattr(value, "to_pydatetime") else datetime.fromisoformat(str(value))
    if timestamp.tzinfo is not None:
        raise ValueError("MRS1 accepts explicit naive local-wall timestamps only")
    return timestamp


def _validate_frame(frame: Any, *, start: datetime, end: datetime) -> Any:
    _, pd, _, _ = _dependencies()
    if not {"customer_id", "ts", "amount"} <= set(frame.columns):
        raise ValueError("MRS1 input needs customer_id, ts, and amount")
    result = frame[["customer_id", "ts", "amount"]].copy()
    result["customer_id"] = result["customer_id"].astype(str)
    result["ts"] = pd.to_datetime(result["ts"], errors="raise")
    if result["ts"].dt.tz is not None:
        raise ValueError("MRS1 input timestamps must be naive local-wall values")
    result["amount"] = pd.to_numeric(result["amount"], errors="raise")
    if result.empty or result["customer_id"].eq("").any():
        raise ValueError("MRS1 input contains no usable households")
    if not result["amount"].map(lambda value: math.isfinite(float(value)) and value > 0).all():
        raise ValueError("MRS1 amount marks must be positive and finite")
    if not result["ts"].map(lambda value: start <= value.to_pydatetime() < end).all():
        raise ValueError("MRS1 input timestamp lies outside the declared window")
    return result.sort_values(["customer_id", "ts"], kind="stable").reset_index(drop=True)


def fit_calendar_hazard(
    frame: Any,
    *,
    observation_start: Any,
    observation_end: Any,
) -> dict[str, tuple[Any, ...]]:
    """Fit exposure-adjusted marginal 24x7x53 factors from unique epochs."""
    start, end = _as_local_wall(observation_start), _as_local_wall(observation_end)
    if start >= end:
        raise ValueError("calendar exposure window must be ordered")
    epochs = frame[["customer_id", "ts"]].drop_duplicates()
    if not epochs["ts"].map(lambda value: start <= value.to_pydatetime() < end).all():
        raise ValueError("calendar epoch lies outside its exposure window")
    hours = tuple(int((epochs["ts"].dt.hour == index).sum()) for index in range(24))
    weekdays = tuple(int((epochs["ts"].dt.weekday == index).sum()) for index in range(7))
    iso_weeks = epochs["ts"].dt.isocalendar().week.astype(int)
    weeks = tuple(int((iso_weeks == index).sum()) for index in range(1, 54))

    exposure = {
        "hour": [0.0] * 24,
        "weekday": [0.0] * 7,
        "week": [0.0] * 53,
    }
    current = start
    while current < end:
        boundary = min(
            current.replace(minute=0, second=0, microsecond=0)
            + timedelta(hours=1),
            end,
        )
        duration_hours = (boundary - current).total_seconds() / 3_600
        exposure["hour"][current.hour] += duration_hours
        exposure["weekday"][current.weekday()] += duration_hours
        exposure["week"][current.isocalendar().week - 1] += duration_hours
        current = boundary
    hour_exposure = tuple(exposure["hour"])
    weekday_exposure = tuple(exposure["weekday"])
    week_exposure = tuple(exposure["week"])

    def normalize(
        counts: tuple[int, ...], exposures: tuple[float, ...]
    ) -> tuple[float, ...]:
        exposed_rates = tuple(
            (count + JEFFREYS_PSEUDOCOUNT) / hours_exposed
            for count, hours_exposed in zip(counts, exposures, strict=True)
            if hours_exposed > 0
        )
        exposed_mean = sum(exposed_rates) / len(exposed_rates)
        adjusted_rates = tuple(
            (count + JEFFREYS_PSEUDOCOUNT) / hours_exposed
            if hours_exposed > 0
            else exposed_mean
            for count, hours_exposed in zip(counts, exposures, strict=True)
        )
        mean = sum(adjusted_rates) / len(adjusted_rates)
        return tuple(value / mean for value in adjusted_rates)

    return {
        "hour_counts": hours,
        "weekday_counts": weekdays,
        "week_counts": weeks,
        "hour_exposure_hours": hour_exposure,
        "weekday_exposure_hours": weekday_exposure,
        "week_exposure_hours": week_exposure,
        "hour_weights": normalize(hours, hour_exposure),
        "weekday_weights": normalize(weekdays, weekday_exposure),
        "week_weights": normalize(weeks, week_exposure),
    }


def _profile_shell(
    *,
    frame: Any,
    source_dataset: str,
    source_partition: str,
    customer_ids_sha256: str,
    transactions_sha256: str,
    start: datetime,
    end: datetime,
    calendar: Mapping[str, tuple[Any, ...]],
) -> MRS1Profile:
    """Build a valid clock carrier before timing/amount coefficients are fitted."""
    groups = frame.groupby(["customer_id", "ts"], sort=False).size()
    counts = tuple(int((groups.map(lambda value: min(int(value), 4)) == index).sum()) for index in range(1, 5))
    probabilities = tuple(value / sum(counts) for value in counts)
    return MRS1Profile(
        source_dataset=source_dataset,
        source_partition=source_partition,
        source_customer_ids_sha256=customer_ids_sha256,
        source_transactions_sha256=transactions_sha256,
        source_customer_count=int(frame["customer_id"].nunique()),
        source_transaction_count=len(frame),
        observation_start=start.isoformat(),
        observation_end=end.isoformat(),
        hour_counts=calendar["hour_counts"],
        weekday_counts=calendar["weekday_counts"],
        week_counts=calendar["week_counts"],
        hour_exposure_hours=calendar["hour_exposure_hours"],
        weekday_exposure_hours=calendar["weekday_exposure_hours"],
        week_exposure_hours=calendar["week_exposure_hours"],
        hour_weights=calendar["hour_weights"],
        weekday_weights=calendar["weekday_weights"],
        week_weights=calendar["week_weights"],
        renewal_shape=1.0,
        mean_operational_gap_days=14.0,
        frailty_shape=1.0,
        multiplicity_counts=counts,
        multiplicity_probabilities=probabilities,
        amount_intercept=1.0,
        amount_log_frailty=0.0,
        amount_log1p_gap_days=0.0,
        amount_ar1=0.0,
        amount_innovation_scale=1.0,
    )


def _operational_sequences(frame: Any, clock: CalendarHazardClock) -> list[tuple[str, Any]]:
    import numpy as np

    sequences: list[tuple[str, Any]] = []
    for customer_id, group in frame.groupby("customer_id", sort=True):
        timestamps = group["ts"].drop_duplicates().sort_values(kind="stable")
        coordinates = np.asarray(
            [clock.to_operational(timestamp.to_pydatetime()) for timestamp in timestamps],
            dtype=float,
        )
        if not len(coordinates) or coordinates[0] < 0 or coordinates[-1] >= clock.total or (
            len(coordinates) > 1 and not np.all(np.diff(coordinates) > 0)
        ):
            raise ValueError("MRS1 operational sequence is invalid")
        sequences.append((str(customer_id), coordinates))
    return sequences


def _quadrature_log_weights(frailty_shape: float) -> tuple[Any, Any]:
    import numpy as np
    from scipy.special import gammaln, logsumexp

    nodes, weights = np.polynomial.laguerre.laggauss(GAUSS_LAGUERRE_NODES)
    log_weights = (
        np.log(weights)
        + (frailty_shape - 1) * np.log(nodes)
        - gammaln(frailty_shape)
    )
    log_weights -= logsumexp(log_weights)
    return nodes / frailty_shape, log_weights


def _sequence_log_likelihoods(
    coordinates: Any,
    *,
    horizon: float,
    renewal_shape: float,
    mean_gap: float,
    frailties: Any,
) -> Any:
    import numpy as np
    from scipy.special import gammaincc, gammaln

    rates = renewal_shape * frailties / mean_gap
    first = float(coordinates[0])
    first_survival = gammaincc(renewal_shape, rates * first)
    with np.errstate(divide="ignore"):
        log_likelihood = np.where(
            first_survival > 0,
            np.log(first_survival) - math.log(mean_gap) + np.log(frailties),
            -np.inf,
        )
    if len(coordinates) > 1:
        gaps = np.diff(coordinates)
        log_likelihood += np.sum(
            renewal_shape * np.log(rates[:, None])
            - gammaln(renewal_shape)
            + (renewal_shape - 1) * np.log(gaps[None, :])
            - rates[:, None] * gaps[None, :],
            axis=1,
        )
    censor = horizon - float(coordinates[-1])
    censor_survival = gammaincc(renewal_shape, rates * censor)
    with np.errstate(divide="ignore"):
        log_likelihood += np.where(
            censor_survival > 0, np.log(censor_survival), -np.inf
        )
    return log_likelihood


def _log_observation_probability(
    *,
    horizon: float,
    renewal_shape: float,
    mean_gap: float,
    frailties: Any,
    log_weights: Any,
) -> float:
    import numpy as np
    from scipy.special import gammaincc, logsumexp

    z = renewal_shape * frailties * horizon / mean_gap
    residual_survival = gammaincc(renewal_shape + 1, z) - (
        z / renewal_shape
    ) * gammaincc(renewal_shape, z)
    if np.any(~np.isfinite(residual_survival)) or np.any(residual_survival < -1e-12) or np.any(
        residual_survival > 1 + 1e-12
    ):
        return -math.inf
    probabilities = 1 - residual_survival
    with np.errstate(divide="ignore"):
        terms = np.where(
            probabilities > 0, log_weights + np.log(probabilities), -np.inf
        )
    return float(logsumexp(terms))


def timing_negative_log_likelihood(
    log_parameters: Any, sequences: list[tuple[str, Any]], *, horizon: float
) -> float:
    import numpy as np
    from scipy.special import logsumexp

    renewal_shape, mean_gap, frailty_shape = map(float, np.exp(log_parameters))
    frailties, log_weights = _quadrature_log_weights(frailty_shape)
    log_observed = _log_observation_probability(
        horizon=horizon,
        renewal_shape=renewal_shape,
        mean_gap=mean_gap,
        frailties=frailties,
        log_weights=log_weights,
    )
    if not math.isfinite(log_observed):
        return math.inf
    value = 0.0
    for _, coordinates in sequences:
        conditional = _sequence_log_likelihoods(
            coordinates,
            horizon=horizon,
            renewal_shape=renewal_shape,
            mean_gap=mean_gap,
            frailties=frailties,
        )
        integrated = float(logsumexp(log_weights + conditional)) - log_observed
        if not math.isfinite(integrated):
            return math.inf
        value -= integrated
    return value


def _strict_minimize(objective: Any, starts: tuple[tuple[float, ...], ...], bounds: Any, *, label: str) -> Any:
    import numpy as np
    from scipy.optimize import minimize

    candidates = []
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method=OPTIMIZER_METHOD,
            bounds=bounds,
            options={"maxiter": OPTIMIZER_MAXITER, "ftol": 1e-11, "gtol": 1e-7},
        )
        if result.success and math.isfinite(float(result.fun)) and np.isfinite(result.x).all():
            candidates.append(result)
    if not candidates:
        raise RuntimeError(f"MRS1 {label} optimizer did not converge from any frozen start")
    best = min(candidates, key=lambda result: (float(result.fun), tuple(map(float, result.x))))
    if any(
        value - lower <= BOUNDARY_TOLERANCE or upper - value <= BOUNDARY_TOLERANCE
        for value, (lower, upper) in zip(best.x, bounds, strict=True)
    ):
        raise RuntimeError(f"MRS1 {label} optimum reached a frozen parameter boundary")
    return best


def fit_timing_model(
    sequences: list[tuple[str, Any]], *, horizon: float
) -> tuple[tuple[float, float, float], dict[str, float]]:
    import numpy as np
    from scipy.special import logsumexp

    result = _strict_minimize(
        lambda values: timing_negative_log_likelihood(values, sequences, horizon=horizon),
        TIMING_STARTS,
        TIMING_PARAMETER_BOUNDS,
        label="timing",
    )
    renewal_shape, mean_gap, frailty_shape = map(float, np.exp(result.x))
    frailties, log_weights = _quadrature_log_weights(frailty_shape)
    posterior_log_frailty: dict[str, float] = {}
    for customer_id, coordinates in sequences:
        likelihoods = _sequence_log_likelihoods(
            coordinates,
            horizon=horizon,
            renewal_shape=renewal_shape,
            mean_gap=mean_gap,
            frailties=frailties,
        )
        posterior = log_weights + likelihoods
        posterior -= logsumexp(posterior)
        posterior_log_frailty[customer_id] = float(
            np.sum(np.exp(posterior) * np.log(frailties))
        )
    return (renewal_shape, mean_gap, frailty_shape), posterior_log_frailty


def _amount_design(
    frame: Any,
    posterior_log_frailty: Mapping[str, float],
    *,
    observation_start: datetime,
) -> tuple[Any, ...]:
    import numpy as np

    y: list[float] = []
    log_u: list[float] = []
    log_gap: list[float] = []
    previous_y: list[float] = []
    previous_log_gap: list[float] = []
    repeated: list[bool] = []
    for customer_id, group in frame.groupby("customer_id", sort=True):
        prior_timestamp = observation_start
        prior_log_amount = None
        prior_gap = 0.0
        for row in group.sort_values("ts", kind="stable").itertuples(index=False):
            current_y = math.log(float(row.amount))
            gap = (row.ts - prior_timestamp).total_seconds() / 86_400
            y.append(current_y)
            log_u.append(posterior_log_frailty[str(customer_id)])
            log_gap.append(math.log1p(gap))
            previous_y.append(0.0 if prior_log_amount is None else prior_log_amount)
            previous_log_gap.append(prior_gap)
            repeated.append(prior_log_amount is not None)
            prior_timestamp, prior_log_amount, prior_gap = row.ts, current_y, math.log1p(gap)
    return tuple(
        np.asarray(values, dtype=bool if index == 5 else float)
        for index, values in enumerate(
            (y, log_u, log_gap, previous_y, previous_log_gap, repeated)
        )
    )


def amount_negative_log_likelihood(parameters: Any, design: tuple[Any, ...]) -> float:
    import numpy as np
    from scipy.special import gammaln

    intercept, beta_u, beta_gap, phi, log_scale = map(float, parameters)
    y, log_u, log_gap, previous_y, previous_log_gap, repeated = design
    baseline = intercept + beta_u * log_u + beta_gap * log_gap
    previous_baseline = intercept + beta_u * log_u + beta_gap * previous_log_gap
    mean = baseline + repeated * phi * (previous_y - previous_baseline)
    scale = math.exp(log_scale)
    standardized = (y - mean) / scale
    log_density = (
        gammaln((STUDENT_T_DF + 1) / 2)
        - gammaln(STUDENT_T_DF / 2)
        - 0.5 * math.log(STUDENT_T_DF * math.pi)
        - log_scale
        - ((STUDENT_T_DF + 1) / 2) * np.log1p(standardized**2 / STUDENT_T_DF)
    )
    value = -float(np.sum(log_density))
    return value if math.isfinite(value) else math.inf


def fit_amount_model(
    frame: Any,
    posterior_log_frailty: Mapping[str, float],
    *,
    observation_start: datetime,
) -> tuple[float, ...]:
    import numpy as np

    design = _amount_design(
        frame, posterior_log_frailty, observation_start=observation_start
    )
    result = _strict_minimize(
        lambda values: amount_negative_log_likelihood(values, design),
        AMOUNT_STARTS,
        AMOUNT_PARAMETER_BOUNDS,
        label="amount",
    )
    intercept, beta_u, beta_gap, phi, log_scale = map(float, result.x)
    return intercept, beta_u, beta_gap, phi, float(np.exp(log_scale))


def fit_mrs1_profile(
    frame: Any,
    *,
    source_dataset: str,
    source_partition: str,
    customer_ids_sha256: str,
    transactions_sha256: str,
    observation_start: Any,
    observation_end: Any,
) -> MRS1Profile:
    start, end = _as_local_wall(observation_start), _as_local_wall(observation_end)
    frame = _validate_frame(frame, start=start, end=end)
    if customer_ids_sha256 != _ids_hash(frame["customer_id"].unique()) or (
        transactions_sha256 != _transactions_hash(frame)
    ):
        raise ValueError("MRS1 source hashes do not match the supplied frame")
    calendar = fit_calendar_hazard(
        frame,
        observation_start=start,
        observation_end=end,
    )
    shell = _profile_shell(
        frame=frame,
        source_dataset=source_dataset,
        source_partition=source_partition,
        customer_ids_sha256=customer_ids_sha256,
        transactions_sha256=transactions_sha256,
        start=start,
        end=end,
        calendar=calendar,
    )
    clock = CalendarHazardClock(shell, start, end)
    timing, posterior_log_frailty = fit_timing_model(
        _operational_sequences(frame, clock), horizon=clock.total
    )
    amount = fit_amount_model(
        frame, posterior_log_frailty, observation_start=start
    )
    # posterior_log_frailty is deliberately ephemeral and leaves scope here.
    return MRS1Profile(
        **{
            **shell.__dict__,
            "renewal_shape": timing[0],
            "mean_operational_gap_days": timing[1],
            "frailty_shape": timing[2],
            "amount_intercept": amount[0],
            "amount_log_frailty": amount[1],
            "amount_log1p_gap_days": amount[2],
            "amount_ar1": amount[3],
            "amount_innovation_scale": amount[4],
        }
    )


def _source_sequence_hashes(frame: Any, *, start: datetime) -> set[str]:
    return {
        sequence_sha256(
            [(row.ts.to_pydatetime(), float(row.amount)) for row in group.itertuples(index=False)],
            start=start,
        )
        for _, group in frame.groupby("customer_id", sort=False)
    }


def validate_generator(
    profile: MRS1Profile,
    source_frame: Any,
    *,
    seed: int,
    customers: int | None = None,
) -> dict[str, Any]:
    start, end = datetime.fromisoformat(profile.observation_start), datetime.fromisoformat(
        profile.observation_end
    )
    source_ids = set(source_frame["customer_id"].astype(str))
    source_hashes = _source_sequence_hashes(source_frame, start=start)
    customers = customers or profile.source_customer_count
    arguments = {
        "customers": customers,
        "seed": seed,
        "forbidden_customer_ids": source_ids,
        "forbidden_sequence_hashes": source_hashes,
    }
    first = generate_synthetic_transactions_mrs1(profile, **arguments)
    second = generate_synthetic_transactions_mrs1(profile, **arguments)
    first_hash, second_hash = _transactions_hash(first), _transactions_hash(second)
    if first_hash != second_hash:
        raise RuntimeError("MRS1 determinism check failed")
    if first["customer_id"].nunique() != customers:
        raise RuntimeError("MRS1 exact-customer check failed")
    return {
        "seed": seed,
        "customers": customers,
        "transactions": len(first),
        "synthetic_transactions_sha256": first_hash,
        "exact_source_sequence_matches": 0,
        "deterministic": True,
    }


def load_completejourney_for_mrs1(
    output_dir: Path,
    partition: Literal["development", "qualification"],
    *,
    expected_manifest_sha256: str,
    expected_mrs_preregistration_sha256: str,
    loader: CompleteJourneyLoader | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Adapt the trusted basket schema to the evaluator's three-column schema."""
    if loader is None:
        from sim.completejourney_reference import load_completejourney_partition

        loader = load_completejourney_partition
    raw, source = loader(
        output_dir,
        partition,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_mrs_preregistration_sha256=(
            expected_mrs_preregistration_sha256
        ),
    )
    if tuple(raw.columns) != (
        "household_id",
        "basket_id",
        "transaction_timestamp",
        "sales_value_cents",
    ):
        raise ValueError("Complete Journey adapter received a changed basket schema")
    expected_households = (
        DEVELOPMENT_HOUSEHOLDS if partition == "development" else QUALIFICATION_HOUSEHOLDS
    )
    required = {
        "dataset",
        "version",
        "partition",
        "manifest_sha256",
        "partition_sha256",
        "household_ids_sha256",
        "households",
        "baskets",
        "source_rds_sha256",
        "mrs_preregistration_sha256",
        "timestamp_semantics",
        "observation_start",
        "observation_end_exclusive",
        "amount_unit",
    }
    if not required <= set(source) or source.get("dataset") != "completejourney" or source.get(
        "partition"
    ) != partition or source.get("manifest_sha256") != expected_manifest_sha256 or source.get(
        "mrs_preregistration_sha256"
    ) != expected_mrs_preregistration_sha256 or source.get(
        "households"
    ) != expected_households or source.get("baskets") != len(raw) or source.get(
        "timestamp_semantics"
    ) != COMPLETEJOURNEY_TIMESTAMP_SEMANTICS or source.get(
        "observation_start"
    ) != COMPLETEJOURNEY_OBSERVATION_START or source.get(
        "observation_end_exclusive"
    ) != COMPLETEJOURNEY_OBSERVATION_END or source.get(
        "amount_unit"
    ) != "USD cents":
        raise ValueError("Complete Journey provenance violates the MRS1 frozen contract")
    frame = raw.rename(
        columns={
            "household_id": "customer_id",
            "transaction_timestamp": "ts",
        }
    )[["customer_id", "ts"]].copy()
    frame["amount"] = raw["sales_value_cents"].to_numpy(dtype=float) / 100
    start, end = (
        datetime.fromisoformat(COMPLETEJOURNEY_OBSERVATION_START),
        datetime.fromisoformat(COMPLETEJOURNEY_OBSERVATION_END),
    )
    frame = _validate_frame(frame, start=start, end=end)
    if frame["customer_id"].nunique() != expected_households:
        raise ValueError("Complete Journey adapter lost a household")
    return frame, {
        **dict(source),
        "observation_start": COMPLETEJOURNEY_OBSERVATION_START,
        "observation_end": COMPLETEJOURNEY_OBSERVATION_END,
        "profile_customer_ids_sha256": _ids_hash(frame["customer_id"].unique()),
        "profile_transactions_sha256": _transactions_hash(frame),
    }


def _r1_preflight_lineage() -> dict[str, Any]:
    superseded = _read_artifact(SUPERSEDED_PREREGISTRATION)
    if (
        (SUPERSEDED_PREREGISTRATION.stat().st_mode & 0o7777) != 0o444
        or (DEFAULT_PREFLIGHT_FAILURE.stat().st_mode & 0o7777) != 0o444
        or hashlib.sha256(_canonical_bytes(superseded)).hexdigest()
        != SUPERSEDED_PREREGISTRATION_SHA256
        or superseded.get("artifact") != "entity_fidelity_mrs1_preregistration"
        or superseded.get("study_revision") != "mrs1.0"
        or superseded.get("completejourney", {})
        .get("source_chain", {})
        .get("preprocessor_sha256")
        != SUPERSEDED_PREPROCESSOR_SHA256
    ):
        raise ValueError("superseded MRS1 preregistration lineage changed")
    failure = _read_artifact(DEFAULT_PREFLIGHT_FAILURE)
    expected_failure = {
        "artifact": "entity_fidelity_mrs1_preprocessing_preflight_failure",
        "converter_subprocess_started": False,
        "failure": {
            "exception": "RuntimeError",
            "message": (
                "Complete Journey converter requires uv 0.11.8, found "
                "'uv 0.11.8 (x86_64-unknown-linux-gnu)'"
            ),
            "stage": "uv_version_preflight",
        },
        "mrs1_preregistration_sha256": SUPERSEDED_PREREGISTRATION_SHA256,
        "partition_artifacts_created": False,
        "preprocessor_sha256": SUPERSEDED_PREPROCESSOR_SHA256,
        "qualification_partition_materialized_or_used": False,
        "rds_bytes_read_for_sha256_verification": True,
        "schema_version": 1,
        "source_transaction_rows_parsed_or_materialized": False,
        "status": "STOP_SUPERSEDED_BY_IMPLEMENTATION_R1",
        "superseding_change_scope": (
            "parse the pinned uv semantic version from its canonical CLI output only"
        ),
        "uv_expected_version": "0.11.8",
        "uv_reported_version": "uv 0.11.8 (x86_64-unknown-linux-gnu)",
    }
    if (
        failure != expected_failure
        or hashlib.sha256(_canonical_bytes(failure)).hexdigest()
        != PREFLIGHT_FAILURE_SHA256
    ):
        raise ValueError("MRS1 preprocessing preflight-failure evidence changed")
    return {
        "revision": "r1",
        "change_scope": failure["superseding_change_scope"],
        "superseded_preregistration_path": str(
            SUPERSEDED_PREREGISTRATION.relative_to(ROOT)
        ),
        "superseded_preregistration_sha256": SUPERSEDED_PREREGISTRATION_SHA256,
        "preflight_failure_path": str(DEFAULT_PREFLIGHT_FAILURE.relative_to(ROOT)),
        "preflight_failure_sha256": PREFLIGHT_FAILURE_SHA256,
        "preflight_failure_status": failure["status"],
        "rds_bytes_read_for_sha256_verification": True,
        "source_transaction_rows_parsed_or_materialized": False,
        "partition_artifacts_created": False,
        "qualification_partition_materialized_or_used": False,
    }


def _r2_preprocess_lineage() -> dict[str, Any]:
    prior_lineage = _r1_preflight_lineage()
    r1_preregistration = _read_artifact(R1_PREREGISTRATION)
    if (
        (R1_PREREGISTRATION.stat().st_mode & 0o7777) != 0o444
        or hashlib.sha256(_canonical_bytes(r1_preregistration)).hexdigest()
        != R1_PREREGISTRATION_SHA256
        or r1_preregistration.get("artifact")
        != "entity_fidelity_mrs1_r1_preregistration"
        or r1_preregistration.get("study_revision") != "mrs1.1"
        or r1_preregistration.get("lineage") != prior_lineage
        or r1_preregistration.get("completejourney", {})
        .get("source_chain", {})
        .get("preprocessor_sha256")
        != R1_PREPROCESSOR_SHA256
        or r1_preregistration.get("implementation_sha256", {}).get(
            "sim/completejourney_reference.py"
        )
        != R1_PREPROCESSOR_SHA256
    ):
        raise ValueError("superseded MRS1 r1 preregistration lineage changed")
    failure = _read_artifact(R1_PREPROCESS_FAILURE)
    expected_source_evidence = {
        "calendar_mapping": "2017-01-01 + (day - 285)",
        "day_filter": "day >= 285 and day < 650",
        "git_commit": "5b5d06192b9856edd04e4d405787af2f2e4a1fef",
        "path": "data-raw/prep-data.R",
        "timezone": COMPLETEJOURNEY_SOURCE_TIMEZONE,
    }
    expected_failure = {
        "artifact": "entity_fidelity_mrs1_r1_preprocessing_failure",
        "basket_aggregation_completed_in_memory": True,
        "converter_subprocess_started": True,
        "deterministic_household_split_computed": False,
        "development_or_qualification_partition_created": False,
        "failure": {
            "exception": "ValueError",
            "message": (
                "retained Complete Journey basket falls outside the frozen "
                "observation window"
            ),
            "stage": (
                "post_conversion_observation_window_validation_before_partition_split"
            ),
        },
        "model_fit_or_metric_computed": False,
        "mrs1_r1_preregistration_sha256": R1_PREREGISTRATION_SHA256,
        "official_source_evidence": expected_source_evidence,
        "partition_files_created": False,
        "preprocessor_sha256": R1_PREPROCESSOR_SHA256,
        "qualification_partition_selected_or_scored": False,
        "raw_source_rows_transiently_parsed": True,
        "schema_version": 1,
        "status": "STOP_SUPERSEDED_BY_IMPLEMENTATION_R2",
        "superseding_change_scope": (
            "restore the official America/New_York local-wall timestamp before "
            "removing timezone metadata; no window, model, seed, feature, "
            "threshold, or split change"
        ),
    }
    if (
        (R1_PREPROCESS_FAILURE.stat().st_mode & 0o7777) != 0o444
        or failure != expected_failure
        or hashlib.sha256(_canonical_bytes(failure)).hexdigest()
        != R1_PREPROCESS_FAILURE_SHA256
    ):
        raise ValueError("MRS1 r1 preprocessing-failure evidence changed")
    return {
        "revision": "r2",
        "change_scope": failure["superseding_change_scope"],
        "r0_to_r1": prior_lineage,
        "r1_preregistration_path": str(R1_PREREGISTRATION.relative_to(ROOT)),
        "r1_preregistration_sha256": R1_PREREGISTRATION_SHA256,
        "r1_preprocess_failure_path": str(R1_PREPROCESS_FAILURE.relative_to(ROOT)),
        "r1_preprocess_failure_sha256": R1_PREPROCESS_FAILURE_SHA256,
        "r1_preprocess_failure_status": failure["status"],
        "r1_preprocessor_sha256": R1_PREPROCESSOR_SHA256,
        "official_source_evidence": expected_source_evidence,
        "source_access_before_r2_preregistration": {
            "raw_source_rows_transiently_parsed": True,
            "basket_aggregation_completed_in_memory": True,
            "deterministic_household_split_computed": False,
            "partition_files_created": False,
            "model_fit_or_metric_computed": False,
            "qualification_partition_selected_or_scored": False,
        },
    }


def _preregistration_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "entity_fidelity_mrs1_r2_preregistration",
        "study_revision": STUDY_REVISION,
        "lineage": _r2_preprocess_lineage(),
        "new_model_family": True,
        "not_v10_tuning": True,
        "source_bytes_downloaded": True,
        "source_rds_bytes_read_for_sha256_verification": True,
        "source_transaction_rows_transiently_parsed": True,
        "source_transaction_rows_persisted_or_retained": False,
        "basket_aggregation_completed_in_memory": True,
        "basket_aggregation_persisted_or_retained": False,
        "completejourney_partitions_materialized_or_loaded": False,
        "access_scope": {
            "canonical_mrs_lifecycle_is_single_use": True,
            "arbitrary_same_user_python_reads_cryptographically_prevented": False,
            "admin_readable_raw_partitions_require_external_worm_custody": True,
        },
        "completejourney": {
            "partition_manifest_state": (
                "not_created_after_stopped_r1_preprocessing"
            ),
            "partition_manifest_sha256": None,
            "development_households": DEVELOPMENT_HOUSEHOLDS,
            "qualification_households": QUALIFICATION_HOUSEHOLDS,
            "development_only_adaptation": True,
            "qualification_single_access": True,
            "observation_window": (
                f"[{COMPLETEJOURNEY_OBSERVATION_START},"
                f"{COMPLETEJOURNEY_OBSERVATION_END}) local wall"
            ),
            "source_timestamp_timezone": COMPLETEJOURNEY_SOURCE_TIMEZONE,
            "stored_timestamp_semantics": COMPLETEJOURNEY_TIMESTAMP_SEMANTICS,
            "source_chain": _reference_source_chain(),
        },
        "model": {
            "frailty": "one mean-one Gamma draw per customer",
            "renewal": "stationary Gamma gaps in operational time",
            "calendar_hazard_shape": [24, 7, 53],
            "calendar_pseudocount": JEFFREYS_PSEUDOCOUNT,
            "calendar_fit": (
                "Jeffreys exposure-adjusted marginal factors: "
                "(count+0.5)/exposure_hours for exposed bins; unexposed bins use "
                "the exposed-bin mean; arithmetic mean one; not joint Poisson MLE"
            ),
            "multiplicity_categories": ["1", "2", "3", "4+ rendered as 4"],
            "amount_distribution": "Student-t5 AR(1) log amount",
            "timing_to_amount_coupling": (
                "two-stage posterior-score approximation E[log U | timing]; "
                "not joint shared-latent MLE"
            ),
            "forbidden_postprocessing": [
                "independent count sampling",
                "independent end sampling",
                "clipping",
                "repair",
                "shuffle",
                "post-scaling",
            ],
        },
        "optimizer": {
            "method": OPTIMIZER_METHOD,
            "maximum_iterations": OPTIMIZER_MAXITER,
            "boundary_tolerance": BOUNDARY_TOLERANCE,
            "gauss_laguerre_nodes": GAUSS_LAGUERRE_NODES,
            "timing_bounds": [list(values) for values in TIMING_PARAMETER_BOUNDS],
            "timing_starts": [list(values) for values in TIMING_STARTS],
            "amount_bounds": [list(values) for values in AMOUNT_PARAMETER_BOUNDS],
            "amount_starts": [list(values) for values in AMOUNT_STARTS],
            "failure": "terminal STOP on nonconvergence or any active bound",
        },
        "evaluator": {
            "features": list(FEATURES),
            "classifier": _classifier_protocol(),
            "seeds": list(EVALUATION_SEEDS),
            "folds": CROSS_FIT_FOLDS,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": FULL_BOOTSTRAP_SEED,
            "orientation_free_threshold": GATE_MAX_D_UPPER_95,
            "development_fit_diagnostic": {
                "profile_scope": "all_development",
                "crossfit_scope": "discriminator/calibration_only",
                "out_of_sample": False,
                "optimistic": True,
                "not_release_evidence": True,
            },
            "qualification_is_first_independent_release_gate": True,
        },
        "gate_order": [
            "development_freeze",
            "one-receipt qualification",
            "frozen UCI352 transfer",
            "receipt-before-UCI502 one-shot final",
        ],
        "uci502_final_source": _uci502_final_source(),
        "uci352_transfer_source": _uci352_transfer_source(),
        "terminal_stop_conditions": [
            "provenance",
            "access chronology",
            "optimizer",
            "class or parameter drift",
            "novelty",
            "determinism",
            "gate failure",
        ],
        "implementation_sha256": _implementation_hashes(),
    }


def write_mrs1_preregistration() -> dict[str, Any]:
    path = DEFAULT_PREREGISTRATION
    forbidden_later_artifacts = (
        DEFAULT_PROTOCOL,
        DEFAULT_QUALIFICATION_RECEIPT,
        DEFAULT_QUALIFICATION,
        DEFAULT_UCI352_TRANSFER,
        DEFAULT_FINAL_RECEIPT,
        DEFAULT_FINAL,
        DEFAULT_COMPLETION_SEAL,
        DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR
        / "completejourney_reference.manifest.json",
        DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR / "completejourney_development.parquet",
        DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR
        / "completejourney_qualification.parquet",
    )
    if any(item.exists() or item.is_symlink() for item in forbidden_later_artifacts):
        raise FileExistsError(
            "MRS1 preregistration cannot be created after downstream materialization"
        )
    value = _preregistration_payload()
    _write_artifact(path, value)
    return value


def load_mrs1_preregistration(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    value = _read_artifact(path)
    if value != _preregistration_payload():
        raise ValueError("MRS1 preregistration differs from the frozen implementation")
    return value


def freeze_mrs1_protocol() -> dict[str, Any]:
    reference_dir = DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR
    preregistration_path = DEFAULT_PREREGISTRATION
    protocol_path = DEFAULT_PROTOCOL
    preregistration = load_mrs1_preregistration(preregistration_path)
    preregistration_bytes = _canonical_bytes(preregistration)
    mrs_preregistration_sha256 = hashlib.sha256(preregistration_bytes).hexdigest()
    from sim.completejourney_reference import MANIFEST_FILE, _validate_manifest

    manifest_path = reference_dir / MANIFEST_FILE
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("trusted Complete Journey manifest is missing or symbolic")
    if manifest_path.stat().st_mode & 0o222:
        raise ValueError("trusted Complete Journey manifest must be immutable")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = _validate_manifest(json.loads(manifest_bytes))
    except json.JSONDecodeError as error:
        raise ValueError("trusted Complete Journey manifest is invalid JSON") from error
    if manifest_bytes != _canonical_bytes(manifest):
        raise ValueError("trusted Complete Journey manifest is not canonical JSON")
    if not isinstance(manifest, dict) or manifest.get(
        "mrs_preregistration_sha256"
    ) != mrs_preregistration_sha256:
        raise ValueError("Complete Journey manifest is not bound to this MRS1 preregistration")
    manifest_source_chain = {
        "downloader_sha256": _file_sha256(
            ROOT / "scripts" / "download_completejourney_reference.py"
        ),
        "preprocessor_sha256": _file_sha256(
            ROOT / "sim" / "completejourney_reference.py"
        ),
        "source_preregistration_sha256": manifest["source"][
            "source_preregistration_sha256"
        ],
        "quarantine_receipt_sha256": manifest["source"][
            "quarantine_receipt_sha256"
        ],
        "rds_manifest_sha256": manifest["source"]["manifest_sha256"],
        "rds_sha256": manifest["source"]["rds_sha256"],
    }
    if manifest_source_chain != preregistration["completejourney"]["source_chain"]:
        raise ValueError("Complete Journey manifest source chain differs from preregistration")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    frame, reference = load_completejourney_for_mrs1(
        reference_dir,
        "development",
        expected_manifest_sha256=manifest_sha256,
        expected_mrs_preregistration_sha256=mrs_preregistration_sha256,
    )
    if reference["households"] != DEVELOPMENT_HOUSEHOLDS:
        raise ValueError("MRS1 development partition size changed")
    profile = fit_mrs1_profile(
        frame,
        source_dataset=f"completejourney-{reference['version']}",
        source_partition="development",
        customer_ids_sha256=reference["profile_customer_ids_sha256"],
        transactions_sha256=reference["profile_transactions_sha256"],
        observation_start=COMPLETEJOURNEY_OBSERVATION_START,
        observation_end=COMPLETEJOURNEY_OBSERVATION_END,
    )
    generator_check = validate_generator(profile, frame, seed=SEED)
    development_fit_diagnostic = {
        "profile_scope": "all_development",
        "crossfit_scope": "discriminator/calibration_only",
        "out_of_sample": False,
        "optimistic": True,
        "not_release_evidence": True,
        **evaluate_mrs1_gate(
            frame,
            reference,
            frozen_profile=profile,
        ),
    }
    protocol: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "entity_fidelity_mrs1_frozen_protocol",
        "mrs_preregistration_sha256": mrs_preregistration_sha256,
        "expected_completejourney_manifest_sha256": manifest_sha256,
        "reference_source_chain": preregistration["completejourney"]["source_chain"],
        "development_reference": {
            key: reference[key]
            for key in (
                "dataset",
                "version",
                "partition",
                "manifest_sha256",
                "partition_sha256",
                "household_ids_sha256",
                "households",
                "baskets",
                "source_rds_sha256",
                "mrs_preregistration_sha256",
                "timestamp_semantics",
                "amount_unit",
            )
        }
        | {
            "observation_start": reference["observation_start"],
            "observation_end": reference["observation_end"],
        },
        "development_profile": profile.as_dict(),
        "development_profile_sha256": profile.sha256,
        "generator_check": generator_check,
        "development_fit_diagnostic": development_fit_diagnostic,
        "development_diagnostic_status": (
            "SANITY_PASS"
            if development_fit_diagnostic["gate_passed"]
            else "STOP_FAIL"
        ),
        "implementation_sha256": _implementation_hashes(),
        "dependency_versions": _dependency_versions(),
        "dependency_lock_sha256": _file_sha256(ROOT / "requirements.lock"),
        "optimizer": preregistration["optimizer"],
        "evaluator": preregistration["evaluator"],
        "uci502_final_source": preregistration["uci502_final_source"],
        "uci352_transfer_source": preregistration["uci352_transfer_source"],
        "qualification_accessed": False,
        "uci352_accessed": False,
        "uci502_accessed": False,
    }
    protocol["protocol_sha256"] = _canonical_hash(protocol)
    _write_artifact(protocol_path, protocol)
    return protocol


def load_frozen_mrs1_protocol(
    path: Path = DEFAULT_PROTOCOL,
    *,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    value = _read_artifact(path)
    stored_hash = value.get("protocol_sha256")
    if stored_hash != _canonical_hash(
        {key: item for key, item in value.items() if key != "protocol_sha256"}
    ):
        raise ValueError("MRS1 protocol hash is invalid")
    if value.get("implementation_sha256") != _implementation_hashes() or value.get(
        "dependency_versions"
    ) != _dependency_versions() or value.get("dependency_lock_sha256") != _file_sha256(
        ROOT / "requirements.lock"
    ):
        raise ValueError("MRS1 frozen implementation or dependencies drifted")
    profile = MRS1Profile.from_dict(value["development_profile"])
    if profile.sha256 != value.get("development_profile_sha256"):
        raise ValueError("MRS1 development profile binding changed")
    frozen_preregistration = _preregistration_payload()
    if value.get("optimizer") != frozen_preregistration["optimizer"] or value.get(
        "evaluator"
    ) != frozen_preregistration["evaluator"]:
        raise ValueError("MRS1 frozen optimizer or evaluator changed")
    if value.get("uci502_final_source") != frozen_preregistration[
        "uci502_final_source"
    ]:
        raise ValueError("MRS1 frozen UCI502 source pins changed")
    if value.get("uci352_transfer_source") != _uci352_transfer_source():
        raise ValueError("MRS1 frozen UCI352 source pins changed")
    if value.get("reference_source_chain") != _reference_source_chain():
        raise ValueError("MRS1 Complete Journey source custody chain changed")
    diagnostic = value.get("development_fit_diagnostic", {})
    if (
        not _gate_evidence_valid(diagnostic)
        or {
            "profile_scope": diagnostic.get("profile_scope"),
            "crossfit_scope": diagnostic.get("crossfit_scope"),
            "out_of_sample": diagnostic.get("out_of_sample"),
            "optimistic": diagnostic.get("optimistic"),
            "not_release_evidence": diagnostic.get("not_release_evidence"),
        }
        != {
            "profile_scope": "all_development",
            "crossfit_scope": "discriminator/calibration_only",
            "out_of_sample": False,
            "optimistic": True,
            "not_release_evidence": True,
        }
        or value.get("development_diagnostic_status")
        != ("SANITY_PASS" if diagnostic.get("gate_passed") else "STOP_FAIL")
    ):
        raise ValueError("MRS1 development-fit diagnostic evidence changed")
    if value.get("mrs_preregistration_sha256") != value.get(
        "development_reference", {}
    ).get("mrs_preregistration_sha256"):
        raise ValueError("MRS1 preregistration binding changed")
    preregistration = load_mrs1_preregistration(preregistration_path)
    if value.get("mrs_preregistration_sha256") != hashlib.sha256(
        _canonical_bytes(preregistration)
    ).hexdigest():
        raise ValueError("MRS1 protocol is not bound to the canonical preregistration")
    return value


def _fold_assignments(customer_ids: Any, *, seed: int) -> dict[str, int]:
    return {
        str(customer_id): int.from_bytes(
            hashlib.sha256(
                f"{seed}:mrs1-frozen-crossfit:{customer_id}".encode()
            ).digest()[:8],
            "big",
        )
        % CROSS_FIT_FOLDS
        for customer_id in sorted(set(map(str, customer_ids)))
    }


def _crossfit_seed(
    frame: Any,
    reference: Mapping[str, Any],
    *,
    seed: int,
    frozen_profile: MRS1Profile,
) -> list[dict[str, Any]]:
    np, pd, _, roc_auc_score = _dependencies()
    assignments = _fold_assignments(frame["customer_id"], seed=seed)
    folds: list[dict[str, Any]] = []
    source_ids = set(frame["customer_id"].astype(str))
    source_hashes = _source_sequence_hashes(
        frame, start=_as_local_wall(reference["observation_start"])
    )
    for fold in range(CROSS_FIT_FOLDS):
        evaluation_ids = {
            customer_id for customer_id, assigned in assignments.items() if assigned == fold
        }
        training_ids = set(assignments) - evaluation_ids
        if not training_ids or not evaluation_ids:
            raise ValueError("MRS1 cross-fit produced an empty fold")
        real_training = frame.loc[frame["customer_id"].isin(training_ids)].copy()
        real_evaluation = frame.loc[frame["customer_id"].isin(evaluation_ids)].copy()
        profile = frozen_profile
        target_start = _as_local_wall(reference["observation_start"])
        target_end = _as_local_wall(reference["observation_end"])
        synthetic_training = generate_synthetic_transactions_mrs1(
            profile,
            customers=len(training_ids),
            seed=seed + 10_000 * fold + 101,
            start=target_start,
            end=target_end,
            forbidden_customer_ids=source_ids,
            forbidden_sequence_hashes=source_hashes,
        )
        synthetic_evaluation = generate_synthetic_transactions_mrs1(
            profile,
            customers=len(evaluation_ids),
            seed=seed + 10_000 * fold + 202,
            start=target_start,
            end=target_end,
            forbidden_customer_ids=source_ids,
            forbidden_sequence_hashes=source_hashes,
        )
        real_transform, synthetic_transform, _ = fit_calibrations(
            real_training, synthetic_training
        )
        real_training_features = customer_features(real_training, real_transform)
        synthetic_training_features = customer_features(
            synthetic_training, synthetic_transform
        )
        classifier = _classifier(seed + fold)
        classifier.fit(
            pd.concat(
                (real_training_features, synthetic_training_features),
                ignore_index=True,
            ),
            np.concatenate(
                (
                    np.zeros(len(real_training_features), dtype="int8"),
                    np.ones(len(synthetic_training_features), dtype="int8"),
                )
            ),
        )
        real_evaluation_features = customer_features(real_evaluation, real_transform)
        real_predictions = classifier.predict_proba(real_evaluation_features)[:, 1]
        synthetic_predictions = classifier.predict_proba(
            customer_features(synthetic_evaluation, synthetic_transform)
        )[:, 1]
        auc = float(
            roc_auc_score(
                np.concatenate(
                    (np.zeros(len(real_predictions)), np.ones(len(synthetic_predictions)))
                ),
                np.concatenate((real_predictions, synthetic_predictions)),
            )
        )
        folds.append(
            {
                "fold": fold,
                "training_real_customers": len(training_ids),
                "evaluation_real_customers": len(evaluation_ids),
                "evaluation_synthetic_customers": len(synthetic_predictions),
                "training_customer_ids_sha256": _ids_hash(training_ids),
                "evaluation_customer_ids_sha256": _ids_hash(evaluation_ids),
                "profile_sha256": profile.sha256,
                "auc": auc,
                "d": max(auc, 1 - auc),
                "_real_customer_ids": tuple(real_evaluation_features.index),
                "_real_predictions": real_predictions,
                "_synthetic_predictions": synthetic_predictions,
            }
        )
    return folds


def evaluate_mrs1_gate(
    frame: Any,
    reference: Mapping[str, Any],
    *,
    frozen_profile: MRS1Profile,
) -> dict[str, Any]:
    import numpy as np

    seed_folds = {
        seed: _crossfit_seed(
            frame, reference, seed=seed, frozen_profile=frozen_profile
        )
        for seed in EVALUATION_SEEDS
    }
    seed_scores = {
        str(seed): sum(
            fold["d"] * fold["evaluation_real_customers"] for fold in folds
        )
        / sum(fold["evaluation_real_customers"] for fold in folds)
        for seed, folds in seed_folds.items()
    }
    bootstrap, low, high = _crossfit_max_seed_bootstrap(
        seed_folds, repeats=BOOTSTRAP_REPEATS, seed=FULL_BOOTSTRAP_SEED
    )
    return {
        "seeds": {
            str(seed): {
                "fold_weighted_d": seed_scores[str(seed)],
                "folds": [
                    {key: value for key, value in fold.items() if not key.startswith("_")}
                    for fold in folds
                ],
            }
            for seed, folds in seed_folds.items()
        },
        "metrics": {
            "orientation_free_max_seed_d": max(seed_scores.values()),
            "bootstrap_max_seed_d_mean": float(np.mean(bootstrap)),
            "bootstrap_max_seed_d_ci_95": [low, high],
            "threshold_max_d_upper_95": GATE_MAX_D_UPPER_95,
        },
        "gate_passed": high <= GATE_MAX_D_UPPER_95,
    }


def _gate_evidence_valid(value: Mapping[str, Any]) -> bool:
    try:
        seeds, metrics = value["seeds"], value["metrics"]
        if (
            type(seeds) is not dict
            or type(metrics) is not dict
            or set(seeds) != set(map(str, EVALUATION_SEEDS))
            or set(metrics)
            != {
                "orientation_free_max_seed_d",
                "bootstrap_max_seed_d_mean",
                "bootstrap_max_seed_d_ci_95",
                "threshold_max_d_upper_95",
            }
            or type(value.get("gate_passed")) is not bool
        ):
            return False
        weighted_scores: list[float] = []
        for seed in EVALUATION_SEEDS:
            entry = seeds[str(seed)]
            folds = entry["folds"]
            if (
                type(entry) is not dict
                or set(entry) != {"fold_weighted_d", "folds"}
                or type(entry["fold_weighted_d"]) is not float
                or type(folds) is not list
                or len(folds) != CROSS_FIT_FOLDS
                or {
                fold["fold"] for fold in folds
                }
                != set(range(CROSS_FIT_FOLDS))
            ):
                return False
            for fold in folds:
                if (
                    type(fold) is not dict
                    or set(fold)
                    != {
                        "fold",
                        "training_real_customers",
                        "evaluation_real_customers",
                        "evaluation_synthetic_customers",
                        "training_customer_ids_sha256",
                        "evaluation_customer_ids_sha256",
                        "profile_sha256",
                        "auc",
                        "d",
                    }
                    or any(
                        type(fold[key]) is not int
                        for key in (
                            "fold",
                            "training_real_customers",
                            "evaluation_real_customers",
                            "evaluation_synthetic_customers",
                        )
                    )
                    or type(fold["auc"]) is not float
                    or type(fold["d"]) is not float
                    or not all(
                        _is_sha256(fold[key])
                        for key in (
                            "training_customer_ids_sha256",
                            "evaluation_customer_ids_sha256",
                            "profile_sha256",
                        )
                    )
                ):
                    return False
                auc, d = fold["auc"], fold["d"]
                if (
                    not 0 <= auc <= 1
                    or not math.isclose(d, max(auc, 1 - auc), abs_tol=1e-12)
                    or fold["training_real_customers"] < 1
                    or fold["evaluation_real_customers"] < 1
                    or fold["evaluation_synthetic_customers"]
                    != fold["evaluation_real_customers"]
                ):
                    return False
            weighted = sum(
                fold["d"] * fold["evaluation_real_customers"]
                for fold in folds
            ) / sum(fold["evaluation_real_customers"] for fold in folds)
            if not math.isclose(weighted, entry["fold_weighted_d"], abs_tol=1e-12):
                return False
            weighted_scores.append(weighted)
        ci = metrics["bootstrap_max_seed_d_ci_95"]
        if (
            type(ci) is not list
            or len(ci) != 2
            or any(type(item) is not float for item in ci)
            or any(
                type(metrics[key]) is not float
                for key in (
                    "orientation_free_max_seed_d",
                    "bootstrap_max_seed_d_mean",
                    "threshold_max_d_upper_95",
                )
            )
        ):
            return False
        low, high = ci
        mean = metrics["bootstrap_max_seed_d_mean"]
        if not (
            0.5 <= low <= high <= 1
            and 0.5 <= mean <= 1
            and metrics["threshold_max_d_upper_95"] == GATE_MAX_D_UPPER_95
            and math.isclose(
                float(metrics["orientation_free_max_seed_d"]),
                max(weighted_scores),
                abs_tol=1e-12,
            )
        ):
            return False
        return value.get("gate_passed") is (high <= GATE_MAX_D_UPPER_95)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def _bind_artifact(value: dict[str, Any]) -> dict[str, Any]:
    value["artifact_sha256"] = _canonical_hash(value)
    return value


def _artifact_hash_valid(value: Mapping[str, Any]) -> bool:
    return value.get("artifact_sha256") == _canonical_hash(
        {key: item for key, item in value.items() if key != "artifact_sha256"}
    )


def _receipt(
    *, artifact: str, protocol_sha256: str, prerequisite_sha256: str
) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "artifact": artifact,
        "status": "consumed",
        "protocol_sha256": protocol_sha256,
        "prerequisite_artifact_sha256": prerequisite_sha256,
        "receipt_before_qualification_partition_load_or_use": True,
    }
    value["receipt_sha256"] = _canonical_hash(value)
    return value


def run_mrs1_qualification() -> dict[str, Any]:
    reference_dir = DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR
    protocol_path = DEFAULT_PROTOCOL
    receipt_path = DEFAULT_QUALIFICATION_RECEIPT
    output_path = DEFAULT_QUALIFICATION
    if receipt_path.exists() or output_path.exists():
        raise FileExistsError("MRS1 qualification attempt is already consumed")
    protocol = load_frozen_mrs1_protocol(protocol_path)
    protocol_bytes = _canonical_bytes(protocol)
    if protocol["development_fit_diagnostic"]["gate_passed"] is not True:
        raise RuntimeError("terminal STOP: MRS1 development-fit diagnostic did not pass")
    receipt = _receipt(
        artifact="entity_fidelity_mrs1_qualification_receipt",
        protocol_sha256=protocol["protocol_sha256"],
        prerequisite_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
    )
    receipt_bytes = _write_artifact(receipt_path, receipt)
    # This is intentionally the first qualification loader call.
    frame, reference = load_completejourney_for_mrs1(
        reference_dir,
        "qualification",
        expected_manifest_sha256=protocol[
            "expected_completejourney_manifest_sha256"
        ],
        expected_mrs_preregistration_sha256=protocol[
            "mrs_preregistration_sha256"
        ],
    )
    profile = MRS1Profile.from_dict(protocol["development_profile"])
    gate = evaluate_mrs1_gate(frame, reference, frozen_profile=profile)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "entity_fidelity_mrs1_qualification",
        "protocol_sha256": protocol["protocol_sha256"],
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "reference": {
            key: reference[key]
            for key in (
                "dataset",
                "version",
                "partition",
                "manifest_sha256",
                "partition_sha256",
                "household_ids_sha256",
                "households",
                "baskets",
                "source_rds_sha256",
                "mrs_preregistration_sha256",
                "timestamp_semantics",
                "amount_unit",
                "observation_start",
                "observation_end",
            )
        },
        **gate,
        "status": "PASS" if gate["gate_passed"] else "STOP_FAIL",
    }
    if gate["gate_passed"]:
        result["qualified_profile"] = profile.as_dict()
        result["qualified_profile_sha256"] = profile.sha256
        result["generator_check"] = validate_generator(
            profile,
            frame,
            seed=SEED,
            customers=QUALIFICATION_HOUSEHOLDS,
        )
    _bind_artifact(result)
    _write_artifact(output_path, result)
    return result


def load_passed_qualification(
    path: Path = DEFAULT_QUALIFICATION,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    receipt_path: Path = DEFAULT_QUALIFICATION_RECEIPT,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    value = _read_artifact(path)
    protocol = load_frozen_mrs1_protocol(
        protocol_path,
        preregistration_path=preregistration_path,
    )
    receipt = _read_artifact(receipt_path)
    protocol_bytes = _canonical_bytes(protocol)
    receipt_bytes = _canonical_bytes(receipt)
    if (
        not _artifact_hash_valid(value)
        or not _gate_evidence_valid(value)
        or value.get("gate_passed") is not True
        or value.get("status") != "PASS"
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("artifact") != "entity_fidelity_mrs1_qualification"
        or value.get("reference", {}).get("partition") != "qualification"
        or value.get("reference", {}).get("households") != QUALIFICATION_HOUSEHOLDS
        or value.get("reference", {}).get("manifest_sha256")
        != protocol.get("expected_completejourney_manifest_sha256")
        or value.get("reference", {}).get("mrs_preregistration_sha256")
        != protocol.get("mrs_preregistration_sha256")
        or value.get("protocol_sha256") != protocol.get("protocol_sha256")
        or value.get("receipt_sha256")
        != hashlib.sha256(receipt_bytes).hexdigest()
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("artifact")
        != "entity_fidelity_mrs1_qualification_receipt"
        or receipt.get("status") != "consumed"
        or receipt.get("protocol_sha256") != protocol.get("protocol_sha256")
        or receipt.get("prerequisite_artifact_sha256")
        != hashlib.sha256(protocol_bytes).hexdigest()
        or receipt.get("receipt_before_qualification_partition_load_or_use")
        is not True
        or receipt.get("receipt_sha256")
        != _canonical_hash(
            {key: item for key, item in receipt.items() if key != "receipt_sha256"}
        )
    ):
        raise ValueError("MRS1 qualification did not pass")
    profile = MRS1Profile.from_dict(value["qualified_profile"])
    generator = value.get("generator_check", {})
    if (
        profile.sha256 != value.get("qualified_profile_sha256")
        or profile.as_dict() != protocol.get("development_profile")
        or profile.sha256 != protocol.get("development_profile_sha256")
        or profile.source_partition != "development"
        or profile.source_customer_count != DEVELOPMENT_HOUSEHOLDS
        or generator.get("customers") != QUALIFICATION_HOUSEHOLDS
        or generator.get("exact_source_sequence_matches") != 0
        or generator.get("deterministic") is not True
    ):
        raise ValueError("MRS1 qualified profile binding changed")
    return value


def load_uci352_for_mrs1(
    workbook_path: Path = DEFAULT_UCI352_REFERENCE,
) -> tuple[Any, dict[str, Any]]:
    """Load only the predeclared UCI352 calibration partition for transfer."""
    from sim.entity_fidelity import load_retail_partition

    frame, source = load_retail_partition(workbook_path, "calibration")
    reference = {
        "dataset": "UCI Online Retail 352",
        "dataset_id": source["dataset_id"],
        "source_url": source["source_url"],
        "workbook_sha256": source["workbook_sha256"],
        "manifest_sha256": source["manifest_sha256"],
        "license": source["license"],
        "partition": source["selected_partition"],
        "customer_ids_sha256": source["selected_customer_ids_sha256"],
        "customers": int(frame["customer_id"].nunique()),
        "transactions": len(frame),
        "observation_start": UCI352_OBSERVATION_START,
        "observation_end": UCI352_OBSERVATION_END,
    }
    return frame, reference


def run_mrs1_frozen_transfer(
    *,
    workbook_path: Path = DEFAULT_UCI352_REFERENCE,
) -> dict[str, Any]:
    qualification_path = DEFAULT_QUALIFICATION
    output_path = DEFAULT_UCI352_TRANSFER
    if output_path.exists():
        raise FileExistsError("MRS1 UCI352 transfer attempt is already consumed")
    qualification = load_passed_qualification(qualification_path)
    qualification_bytes = _canonical_bytes(qualification)
    profile = MRS1Profile.from_dict(qualification["qualified_profile"])
    # UCI352 is opened only after the qualification artifact has passed validation.
    frame, reference = load_uci352_for_mrs1(workbook_path)
    expected_reference = _uci352_transfer_source()
    if any(reference.get(key) != value for key, value in expected_reference.items()):
        raise ValueError("UCI352 transfer provenance or frozen partition changed")
    start, end = _as_local_wall(reference["observation_start"]), _as_local_wall(
        reference["observation_end"]
    )
    frame = _validate_frame(frame, start=start, end=end)
    normalized_reference = {
        **dict(reference),
        "partition": reference.get("partition", "uci352-transfer"),
        "observation_start": start.isoformat(),
        "observation_end": end.isoformat(),
    }
    gate = evaluate_mrs1_gate(
        frame, normalized_reference, frozen_profile=profile
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "entity_fidelity_mrs1_frozen_uci352_transfer",
        "qualification_sha256": hashlib.sha256(qualification_bytes).hexdigest(),
        "qualified_profile_sha256": profile.sha256,
        "reference": {
            key: normalized_reference[key] for key in expected_reference
        },
        **gate,
        "status": "PASS" if gate["gate_passed"] else "STOP_FAIL",
    }
    _bind_artifact(result)
    _write_artifact(output_path, result)
    return result


def _validate_passed_transfer_value(
    value: dict[str, Any], qualification: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        not _artifact_hash_valid(value)
        or not _gate_evidence_valid(value)
        or value.get("gate_passed") is not True
        or value.get("status") != "PASS"
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("artifact") != "entity_fidelity_mrs1_frozen_uci352_transfer"
        or value.get("qualification_sha256")
        != hashlib.sha256(_canonical_bytes(qualification)).hexdigest()
        or value.get("qualified_profile_sha256")
        != qualification.get("qualified_profile_sha256")
        or any(
            value.get("reference", {}).get(key) != expected
            for key, expected in _uci352_transfer_source().items()
        )
    ):
        raise ValueError("MRS1 UCI352 transfer did not pass")
    return value


def load_passed_transfer(
    path: Path = DEFAULT_UCI352_TRANSFER,
    *,
    qualification_path: Path = DEFAULT_QUALIFICATION,
    protocol_path: Path = DEFAULT_PROTOCOL,
    qualification_receipt_path: Path = DEFAULT_QUALIFICATION_RECEIPT,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    value = _read_artifact(path)
    qualification = load_passed_qualification(
        qualification_path,
        protocol_path=protocol_path,
        receipt_path=qualification_receipt_path,
        preregistration_path=preregistration_path,
    )
    return _validate_passed_transfer_value(value, qualification)


def _load_uci502_final_for_mrs1() -> tuple[Any, dict[str, Any]]:
    """Open the fixed UCI502 slice only after validating the consumed receipt."""
    receipt = _read_artifact(DEFAULT_FINAL_RECEIPT)
    qualification = load_passed_qualification(DEFAULT_QUALIFICATION)
    transfer = _validate_passed_transfer_value(
        _read_artifact(DEFAULT_UCI352_TRANSFER), qualification
    )
    expected_confirmation = (
        f"{FINAL_CONFIRMATION}:{qualification['protocol_sha256']}"
    )
    if (
        receipt.get("artifact")
        != "entity_fidelity_mrs1_one_shot_final_receipt"
        or receipt.get("status") != "consumed"
        or receipt.get("receipt_before_uci502_workbook_load_or_use") is not True
        or receipt.get("final_source") != _uci502_final_source()
        or receipt.get("protocol_sha256")
        != qualification.get("protocol_sha256")
        or receipt.get("prerequisite_artifact_sha256")
        != hashlib.sha256(_canonical_bytes(transfer)).hexdigest()
        or receipt.get("confirmation_sha256")
        != hashlib.sha256(expected_confirmation.encode()).hexdigest()
        or receipt.get("receipt_sha256")
        != _canonical_hash(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
    ):
        raise ValueError("canonical MRS1 final receipt is invalid before UCI502 access")

    # Deliberately import the sole production loader only after the receipt check.
    from sim.entity_fidelity_v4 import load_retail_ii_readiness

    frame, source = load_retail_ii_readiness(
        DEFAULT_UCI502_REFERENCE,
        excluded_customer_ids=set(),
    )
    expected = _uci502_final_source()
    for key in (
        "dataset_id",
        "source_url",
        "archive_sha256",
        "workbook_sha256",
        "manifest_sha256",
        "license",
        "sheet",
    ):
        if source.get(key) != expected[key]:
            raise ValueError("UCI502 final loader provenance changed")
    if (
        source.get("second_worksheet_rows_iterated") is not False
        or source.get("nonselected_transactions_materialized") is not False
    ):
        raise ValueError("UCI502 final loader exceeded its frozen slice")
    return frame, {
        "dataset": "UCI Online Retail II 502",
        **expected,
        "partition": "one-shot-final-fixed-window",
        "customer_ids_sha256": source["selected_customer_ids_sha256"],
        "customers": int(frame["customer_id"].nunique()),
        "transactions": len(frame),
    }


def run_mrs1_one_shot_final(
    *,
    confirmation: str | None,
) -> dict[str, Any]:
    qualification_path = DEFAULT_QUALIFICATION
    transfer_path = DEFAULT_UCI352_TRANSFER
    receipt_path = DEFAULT_FINAL_RECEIPT
    output_path = DEFAULT_FINAL
    completion_seal_path = DEFAULT_COMPLETION_SEAL
    if receipt_path.exists() or output_path.exists() or completion_seal_path.exists():
        raise FileExistsError("MRS1 one-shot final attempt is already consumed")
    qualification = load_passed_qualification(qualification_path)
    qualification_bytes = _canonical_bytes(qualification)
    transfer = _validate_passed_transfer_value(
        _read_artifact(transfer_path), qualification
    )
    transfer_bytes = _canonical_bytes(transfer)
    if (
        transfer.get("qualification_sha256")
        != hashlib.sha256(qualification_bytes).hexdigest()
        or transfer.get("qualified_profile_sha256")
        != qualification.get("qualified_profile_sha256")
    ):
        raise ValueError("MRS1 transfer is not bound to the passed qualification")
    protocol_sha256 = qualification["protocol_sha256"]
    if confirmation != f"{FINAL_CONFIRMATION}:{protocol_sha256}":
        raise ValueError("MRS1 one-shot final confirmation is absent or incorrect")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "entity_fidelity_mrs1_one_shot_final_receipt",
        "status": "consumed",
        "protocol_sha256": protocol_sha256,
        "prerequisite_artifact_sha256": hashlib.sha256(transfer_bytes).hexdigest(),
        "confirmation_sha256": hashlib.sha256(confirmation.encode()).hexdigest(),
        "final_source": _uci502_final_source(),
        "receipt_before_uci502_workbook_load_or_use": True,
    }
    receipt["receipt_sha256"] = _canonical_hash(receipt)
    receipt_bytes = _write_artifact(receipt_path, receipt)
    frame, reference = _load_uci502_final_for_mrs1()
    expected_source = receipt["final_source"]
    if (
        reference.get("dataset_id") != expected_source["dataset_id"]
        or reference.get("source_url") != expected_source["source_url"]
        or reference.get("archive_sha256") != expected_source["archive_sha256"]
        or reference.get("workbook_sha256") != expected_source["workbook_sha256"]
        or reference.get("manifest_sha256") != expected_source["manifest_sha256"]
        or reference.get("license") != expected_source["license"]
        or reference.get("downloader_sha256")
        != expected_source["downloader_sha256"]
        or reference.get("sheet") != expected_source["sheet"]
        or reference.get("observation_start") != UCI502_WINDOW_START
        or reference.get("observation_end") != UCI502_WINDOW_END
    ):
        raise ValueError("UCI502 final provenance or fixed window changed")
    profile = MRS1Profile.from_dict(qualification["qualified_profile"])
    start, end = _as_local_wall(reference["observation_start"]), _as_local_wall(
        reference["observation_end"]
    )
    frame = _validate_frame(frame, start=start, end=end)
    gate = evaluate_mrs1_gate(
        frame,
        {
            **dict(reference),
            "partition": reference.get("partition", "uci502-one-shot-final"),
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
        },
        frozen_profile=profile,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "entity_fidelity_mrs1_one_shot_final",
        "protocol_sha256": protocol_sha256,
        "transfer_sha256": hashlib.sha256(transfer_bytes).hexdigest(),
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "qualified_profile_sha256": profile.sha256,
        "reference": {
            key: value
            for key, value in reference.items()
            if key
            in {
                "dataset",
                "version",
                "partition",
                "manifest_sha256",
                "workbook_sha256",
                "customer_ids_sha256",
                "customers",
                "transactions",
                "observation_start",
                "observation_end",
            }
        },
        **gate,
        "status": "PASS" if gate["gate_passed"] else "STOP_FAIL",
    }
    _bind_artifact(result)
    final_bytes = _write_artifact(output_path, result)
    if gate["gate_passed"]:
        seal = {
            "schema_version": SCHEMA_VERSION,
            "artifact": "entity_fidelity_mrs1_completion_seal",
            "status": "PASS",
            "protocol_sha256": protocol_sha256,
            "final_sha256": hashlib.sha256(final_bytes).hexdigest(),
            "qualified_profile_sha256": profile.sha256,
        }
        seal["seal_sha256"] = _canonical_hash(seal)
        _write_artifact(completion_seal_path, seal)
    return result


def load_passed_final(
    path: Path = DEFAULT_FINAL,
    *,
    receipt_path: Path = DEFAULT_FINAL_RECEIPT,
    transfer_path: Path = DEFAULT_UCI352_TRANSFER,
    completion_seal_path: Path = DEFAULT_COMPLETION_SEAL,
) -> dict[str, Any]:
    value = _read_artifact(path)
    receipt = _read_artifact(receipt_path)
    transfer = load_passed_transfer(transfer_path)
    seal = _read_artifact(completion_seal_path)
    if (
        not _artifact_hash_valid(value)
        or not _gate_evidence_valid(value)
        or value.get("gate_passed") is not True
        or value.get("status") != "PASS"
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("artifact") != "entity_fidelity_mrs1_one_shot_final"
        or value.get("receipt_sha256") != _file_sha256(receipt_path)
        or value.get("transfer_sha256") != _file_sha256(transfer_path)
        or receipt.get("receipt_sha256")
        != _canonical_hash(
            {key: item for key, item in receipt.items() if key != "receipt_sha256"}
        )
        or receipt.get("artifact")
        != "entity_fidelity_mrs1_one_shot_final_receipt"
        or receipt.get("status") != "consumed"
        or receipt.get("receipt_before_uci502_workbook_load_or_use") is not True
        or receipt.get("final_source") != _uci502_final_source()
        or receipt.get("prerequisite_artifact_sha256") != _file_sha256(transfer_path)
        or receipt.get("protocol_sha256") != value.get("protocol_sha256")
        or transfer.get("qualified_profile_sha256")
        != value.get("qualified_profile_sha256")
        or seal.get("seal_sha256")
        != _canonical_hash(
            {key: item for key, item in seal.items() if key != "seal_sha256"}
        )
        or seal.get("artifact") != "entity_fidelity_mrs1_completion_seal"
        or seal.get("status") != "PASS"
        or seal.get("protocol_sha256") != value.get("protocol_sha256")
        or seal.get("qualified_profile_sha256")
        != value.get("qualified_profile_sha256")
        or seal.get("final_sha256") != _file_sha256(path)
    ):
        raise ValueError("MRS1 one-shot final chain did not pass validation")
    return value


def verify_mrs1_lifecycle(
    stage: Literal["preregistration", "protocol", "qualification", "transfer", "final"],
    *,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
    protocol_path: Path = DEFAULT_PROTOCOL,
    qualification_path: Path = DEFAULT_QUALIFICATION,
    transfer_path: Path = DEFAULT_UCI352_TRANSFER,
    final_receipt_path: Path = DEFAULT_FINAL_RECEIPT,
    final_path: Path = DEFAULT_FINAL,
    completion_seal_path: Path = DEFAULT_COMPLETION_SEAL,
) -> dict[str, Any]:
    """Read-only verification of the frozen chain through an explicit stage."""
    stages = ("preregistration", "protocol", "qualification", "transfer", "final")
    if stage not in stages:
        raise ValueError("unknown MRS1 verification stage")
    verified: dict[str, str] = {}
    preregistration = load_mrs1_preregistration(preregistration_path)
    preregistration_sha256 = hashlib.sha256(
        _canonical_bytes(preregistration)
    ).hexdigest()
    verified["preregistration_sha256"] = preregistration_sha256
    if stage == "preregistration":
        return {"verified_through": stage, **verified}

    protocol = load_frozen_mrs1_protocol(
        protocol_path,
        preregistration_path=preregistration_path,
    )
    if protocol.get("mrs_preregistration_sha256") != preregistration_sha256:
        raise ValueError("MRS1 protocol is not bound to the preregistration")
    verified["protocol_sha256"] = _file_sha256(protocol_path)
    if stage == "protocol":
        return {"verified_through": stage, **verified}

    qualification = load_passed_qualification(
        qualification_path,
        protocol_path=protocol_path,
        receipt_path=DEFAULT_QUALIFICATION_RECEIPT,
        preregistration_path=preregistration_path,
    )
    if (
        qualification.get("protocol_sha256") != protocol.get("protocol_sha256")
        or qualification.get("qualified_profile_sha256")
        != protocol.get("development_profile_sha256")
        or qualification.get("reference", {}).get("manifest_sha256")
        != protocol.get("expected_completejourney_manifest_sha256")
        or qualification.get("reference", {}).get("mrs_preregistration_sha256")
        != preregistration_sha256
    ):
        raise ValueError("MRS1 qualification chain binding changed")
    verified["qualification_sha256"] = _file_sha256(qualification_path)
    if stage == "qualification":
        return {"verified_through": stage, **verified}

    transfer = load_passed_transfer(
        transfer_path,
        qualification_path=qualification_path,
        protocol_path=protocol_path,
        qualification_receipt_path=DEFAULT_QUALIFICATION_RECEIPT,
        preregistration_path=preregistration_path,
    )
    if (
        transfer.get("qualification_sha256") != _file_sha256(qualification_path)
        or transfer.get("qualified_profile_sha256")
        != qualification.get("qualified_profile_sha256")
    ):
        raise ValueError("MRS1 transfer chain binding changed")
    verified["transfer_sha256"] = _file_sha256(transfer_path)
    if stage == "transfer":
        return {"verified_through": stage, **verified}

    final = load_passed_final(
        final_path,
        receipt_path=final_receipt_path,
        transfer_path=transfer_path,
        completion_seal_path=completion_seal_path,
    )
    if (
        final.get("protocol_sha256") != protocol.get("protocol_sha256")
        or final.get("qualified_profile_sha256")
        != qualification.get("qualified_profile_sha256")
    ):
        raise ValueError("MRS1 final chain binding changed")
    verified["final_sha256"] = _file_sha256(final_path)
    verified["completion_seal_sha256"] = _file_sha256(completion_seal_path)
    return {"verified_through": stage, **verified}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    preregister = subcommands.add_parser("preregister")
    freeze = subcommands.add_parser("freeze")
    qualify = subcommands.add_parser("qualify")
    transfer = subcommands.add_parser("transfer")
    transfer.add_argument(
        "--reference-xlsx", type=Path, default=DEFAULT_UCI352_REFERENCE
    )
    final = subcommands.add_parser("final")
    final.add_argument("--confirmation", required=True)
    verify = subcommands.add_parser("verify")
    verify.add_argument(
        "--stage",
        required=True,
        choices=("preregistration", "protocol", "qualification", "transfer", "final"),
    )
    args = parser.parse_args()
    if args.command == "preregister":
        result = write_mrs1_preregistration()
    elif args.command == "freeze":
        result = freeze_mrs1_protocol()
    elif args.command == "qualify":
        result = run_mrs1_qualification()
    elif args.command == "transfer":
        result = run_mrs1_frozen_transfer(workbook_path=args.reference_xlsx)
    elif args.command == "final":
        result = run_mrs1_one_shot_final(
            confirmation=args.confirmation,
        )
    else:
        result = verify_mrs1_lifecycle(args.stage)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
