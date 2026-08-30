"""MRS2 fitting primitives and one-shot preregistered DEVELOPMENT lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from sim.entity_fidelity import (
    FEATURES,
    _classifier,
    _classifier_protocol,
    _dependencies,
    _ids_hash,
    _transactions_hash,
    customer_features,
    fit_calibrations,
)
from sim.entity_fidelity_mrs1 import (
    COMPLETEJOURNEY_OBSERVATION_END,
    COMPLETEJOURNEY_OBSERVATION_START,
    COMPLETEJOURNEY_TIMESTAMP_SEMANTICS,
    _gate_evidence_valid as _mrs1_gate_evidence_valid,
    fit_calendar_hazard,
)
from sim.entity_fidelity_v4 import (
    BOOTSTRAP_REPEATS,
    CROSS_FIT_FOLDS,
    EVALUATION_SEEDS,
    GATE_MAX_D_UPPER_95,
    _crossfit_max_seed_bootstrap,
)
from sim.population_mrs2 import (
    MRS1_STOP_PROTOCOL_SHA256,
    CalendarHazardClock,
    MRS2Profile,
    frailty_score,
    generate_synthetic_transactions_mrs2,
    pooled_hour_probabilities,
    preferred_hour_multiplier,
    sequence_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = ROOT / "results/entity_fidelity_mrs2_preregistration.json"
DEFAULT_DEVELOPMENT_ATTEMPT = (
    ROOT / "results/entity_fidelity_mrs2_development_attempt.json"
)
DEFAULT_DEVELOPMENT = ROOT / "results/entity_fidelity_mrs2_development.json"
MRS1_STOP_PROTOCOL = ROOT / "results/entity_fidelity_mrs1_protocol.json"
MRS1_R2_PREREGISTRATION = (
    ROOT / "results/entity_fidelity_mrs1_r2_preregistration.json"
)
DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR = ROOT / "data/reference/completejourney"
DEFAULT_COMPLETEJOURNEY_MANIFEST = (
    DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR
    / "completejourney_reference.manifest.json"
)
DEFAULT_DEVELOPMENT_PARQUET = (
    DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR / "completejourney_development.parquet"
)
SCHEMA_VERSION = "mrs2"
STUDY_REVISION = "mrs2.0"
DEVELOPMENT_HOUSEHOLDS = 988
QUALIFICATION_HOUSEHOLDS = 1_481
SEED = 20260827
FULL_BOOTSTRAP_SEED = 20260857
GAUSS_LAGUERRE_NODES = 32
OPTIMIZER_METHOD = "L-BFGS-B"
OPTIMIZER_MAXITER = 600
BOUNDARY_TOLERANCE = 1e-5
TIMING_PARAMETER_BOUNDS = (
    (math.log(0.25), math.log(10.0)),
    (math.log(0.05), math.log(365.0)),
    (math.log(0.25), math.log(20.0)),
    (0.0, 0.95),
)
TIMING_STARTS = (
    (math.log(1.0), math.log(14.0), math.log(1.0), 0.2),
    (math.log(2.0), math.log(30.0), math.log(2.0), 0.5),
    (math.log(0.6), math.log(7.0), math.log(0.5), 0.8),
)
AMOUNT_PARAMETER_BOUNDS = (
    (-20.0, 30.0),
    (-5.0, 5.0),
    (-5.0, 5.0),
    (-0.95, 0.95),
    (-6.0, 6.0),
    (-6.0, 6.0),
)
AMOUNT_STARTS = (
    (6.0, 0.0, 0.0, 0.0, -1.0, -1.0),
    (5.0, 0.5, 0.2, 0.3, -0.5, 0.0),
    (7.0, -0.5, -0.2, -0.3, 0.0, -0.5),
)
DEPENDENCY_PACKAGES = (
    "cryptography",
    "lightgbm",
    "numpy",
    "openpyxl",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
)
_ALLOWED_LIFECYCLE_MODES = frozenset(
    {"structural", "synthetic", "preregister", "development", "verify"}
)


def mrs2_lifecycle_contract() -> dict[str, Any]:
    """Return the code-level lifecycle surface without touching artifacts."""
    return {
        "schema_version": SCHEMA_VERSION,
        "study_revision": STUDY_REVISION,
        "predecessor": {
            "model": "mrs1",
            "terminal_state": "STOP",
            "protocol_sha256": MRS1_STOP_PROTOCOL_SHA256,
        },
        "allowed_modes": sorted(_ALLOWED_LIFECYCLE_MODES),
        "implemented_stages": ["preregistration", "development"],
        "development_one_shot": True,
        "qualification_partition_sealed": True,
        "uci_sources_sealed": True,
    }


def require_mrs2_lifecycle_mode(mode: str) -> None:
    if mode not in _ALLOWED_LIFECYCLE_MODES:
        raise RuntimeError(
            "MRS2 implements preregistered DEVELOPMENT only; qualification, "
            "transfer, final, and UCI sources remain sealed"
        )


def _as_local_wall(value: Any) -> datetime:
    timestamp = (
        value.to_pydatetime()
        if hasattr(value, "to_pydatetime")
        else datetime.fromisoformat(str(value))
    )
    if timestamp.tzinfo is not None:
        raise ValueError("MRS2 accepts explicit naive local-wall timestamps only")
    return timestamp


def _validate_frame(frame: Any, *, start: datetime, end: datetime) -> Any:
    import pandas as pd

    if not {"customer_id", "ts", "amount"} <= set(frame.columns):
        raise ValueError("MRS2 input needs customer_id, ts, and integer-cent amount")
    result = frame[["customer_id", "ts", "amount"]].copy()
    result["customer_id"] = result["customer_id"].astype(str)
    result["ts"] = pd.to_datetime(result["ts"], errors="raise")
    if result["ts"].dt.tz is not None:
        raise ValueError("MRS2 input timestamps must be naive local-wall values")
    numeric = pd.to_numeric(result["amount"], errors="raise")
    if result.empty or result["customer_id"].eq("").any():
        raise ValueError("MRS2 input contains no usable households")
    if not numeric.map(
        lambda value: math.isfinite(float(value))
        and float(value) > 0
        and float(value).is_integer()
    ).all():
        raise ValueError("MRS2 amount marks must be positive integer cents")
    result["amount"] = numeric.astype("int64")
    if not result["ts"].map(
        lambda value: start <= value.to_pydatetime() < end
    ).all():
        raise ValueError("MRS2 input timestamp lies outside the declared window")
    return result.sort_values(["customer_id", "ts"], kind="stable").reset_index(
        drop=True
    )


def _quadrature_log_weights(frailty_shape: float) -> tuple[Any, Any]:
    """Generalized Gauss-Laguerre rule for U~Gamma(a, rate=a)."""
    import numpy as np
    from scipy.special import gammaln, logsumexp, roots_genlaguerre

    if not math.isfinite(frailty_shape) or frailty_shape <= 0:
        raise ValueError("frailty_shape must be positive and finite")
    nodes, weights = roots_genlaguerre(
        GAUSS_LAGUERRE_NODES, frailty_shape - 1.0
    )
    log_weights = np.log(weights) - gammaln(frailty_shape)
    log_weights -= logsumexp(log_weights)
    return nodes / frailty_shape, log_weights


def _observation_probabilities(
    *,
    horizon: float,
    renewal_shape: float,
    mean_gap: float,
    frailties: Any,
) -> Any:
    import numpy as np
    from scipy.special import gammaincc

    z = renewal_shape * frailties * horizon / mean_gap
    residual_survival = gammaincc(renewal_shape + 1, z) - (
        z / renewal_shape
    ) * gammaincc(renewal_shape, z)
    probabilities = 1.0 - residual_survival
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities < -1e-12)
        or np.any(probabilities > 1 + 1e-12)
    ):
        raise ValueError("MRS2 zero-truncation probability is invalid")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    if not np.any(probabilities > 0):
        raise ValueError("MRS2 zero-truncation probability vanished numerically")
    return probabilities


def _calendar_sequences(frame: Any) -> list[tuple[datetime, ...]]:
    sequences: list[tuple[datetime, ...]] = []
    for _, group in frame.groupby("customer_id", sort=True):
        timestamps = group["ts"].drop_duplicates().sort_values(kind="stable")
        values = tuple(timestamp.to_pydatetime() for timestamp in timestamps)
        if not values or any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError("MRS2 calendar sequence is invalid")
        sequences.append(values)
    return sequences


def _timing_bases(
    sequences: list[tuple[datetime, ...]], clock: CalendarHazardClock
) -> list[tuple[Any, Any, Any, float]]:
    """Cache all lambda-invariant clock coordinates and Jacobian counts."""
    import numpy as np

    bases = []
    probabilities = np.asarray(
        clock.profile.preferred_hour_probabilities, dtype=float
    )
    for timestamps in sequences:
        base = np.asarray(
            [clock.base_operational(timestamp) for timestamp in timestamps],
            dtype=float,
        )
        hour_coordinates = np.asarray(
            [
                [clock.hour_operational(timestamp, hour) for timestamp in timestamps]
                for hour in range(24)
            ],
            dtype=float,
        )
        delta = hour_coordinates / probabilities[:, None] - base[None, :]
        hour_counts = np.bincount(
            [timestamp.hour for timestamp in timestamps], minlength=24
        ).astype(float)
        log_base_jacobian = sum(
            math.log(clock.base_jacobian(timestamp)) for timestamp in timestamps
        )
        bases.append((base, delta, hour_counts, log_base_jacobian))
    return bases


def _timing_log_components_from_basis(
    basis: tuple[Any, Any, Any, float],
    *,
    clock: CalendarHazardClock,
    renewal_shape: float,
    mean_gap: float,
    frailties: Any,
    log_weights: Any,
    preference_strength: float,
) -> Any:
    import numpy as np
    from scipy.special import gammaincc, gammaln

    base, delta, hour_counts, log_base_jacobian = basis
    probabilities = np.asarray(clock.profile.preferred_hour_probabilities)
    coordinates = base[None, :] + preference_strength * delta
    rates = renewal_shape * frailties / mean_gap
    first_survival = gammaincc(
        renewal_shape, coordinates[:, :1] * rates[None, :]
    )
    with np.errstate(divide="ignore"):
        likelihood = np.where(
            first_survival > 0,
            np.log(first_survival)
            - math.log(mean_gap)
            + np.log(frailties)[None, :],
            -np.inf,
        )
    if coordinates.shape[1] > 1:
        gaps = np.diff(coordinates, axis=1)
        likelihood += (
            (coordinates.shape[1] - 1)
            * (
                renewal_shape * np.log(rates)[None, :]
                - gammaln(renewal_shape)
            )
            + (renewal_shape - 1) * np.sum(np.log(gaps), axis=1)[:, None]
            - np.sum(gaps, axis=1)[:, None] * rates[None, :]
        )
    censor_survival = gammaincc(
        renewal_shape,
        (clock.total - coordinates[:, -1:]) * rates[None, :],
    )
    with np.errstate(divide="ignore"):
        likelihood += np.where(
            censor_survival > 0, np.log(censor_survival), -np.inf
        )
    multipliers = np.full((24, 24), 1.0 - preference_strength)
    multipliers[np.arange(24), np.arange(24)] += (
        preference_strength / probabilities
    )
    log_jacobian = log_base_jacobian + np.log(multipliers) @ hour_counts
    return (
        np.log(probabilities)[:, None]
        + log_weights[None, :]
        + likelihood
        + log_jacobian[:, None]
    ).ravel()


def _timing_log_components(
    timestamps: tuple[datetime, ...],
    *,
    clock: CalendarHazardClock,
    renewal_shape: float,
    mean_gap: float,
    frailties: Any,
    log_weights: Any,
    preference_strength: float,
) -> Any:
    basis = _timing_bases([timestamps], clock)[0]
    return _timing_log_components_from_basis(
        basis,
        clock=clock,
        renewal_shape=renewal_shape,
        mean_gap=mean_gap,
        frailties=frailties,
        log_weights=log_weights,
        preference_strength=preference_strength,
    )


def _log_observation_probability(
    *,
    clock: CalendarHazardClock,
    renewal_shape: float,
    mean_gap: float,
    frailties: Any,
    log_weights: Any,
) -> float:
    import numpy as np
    from scipy.special import logsumexp

    observed = _observation_probabilities(
        horizon=clock.total,
        renewal_shape=renewal_shape,
        mean_gap=mean_gap,
        frailties=frailties,
    )
    with np.errstate(divide="ignore"):
        log_observed = np.where(observed > 0, np.log(observed), -np.inf)
    components = [
        math.log(probability) + log_weights + log_observed
        for probability in clock.profile.preferred_hour_probabilities
    ]
    return float(logsumexp(np.concatenate(components)))


def timing_negative_log_likelihood(
    parameters: Any,
    sequences: list[tuple[datetime, ...]],
    *,
    profile: MRS2Profile,
    observation_start: datetime,
    observation_end: datetime,
) -> float:
    import numpy as np
    from scipy.special import logsumexp

    clock = CalendarHazardClock(profile, observation_start, observation_end)
    bases = _timing_bases(sequences, clock)
    return _timing_negative_log_likelihood_from_bases(
        parameters, bases, clock=clock
    )


def _timing_negative_log_likelihood_from_bases(
    parameters: Any,
    bases: list[tuple[Any, Any, Any, float]],
    *,
    clock: CalendarHazardClock,
) -> float:
    import numpy as np
    from scipy.special import logsumexp

    renewal_shape, mean_gap, frailty_shape = map(float, np.exp(parameters[:3]))
    preference_strength = float(parameters[3])
    if not 0 <= preference_strength < 1:
        return math.inf
    frailties, log_weights = _quadrature_log_weights(frailty_shape)
    try:
        log_observed = _log_observation_probability(
            clock=clock,
            renewal_shape=renewal_shape,
            mean_gap=mean_gap,
            frailties=frailties,
            log_weights=log_weights,
        )
        value = 0.0
        for basis in bases:
            components = _timing_log_components_from_basis(
                basis,
                clock=clock,
                renewal_shape=renewal_shape,
                mean_gap=mean_gap,
                frailties=frailties,
                log_weights=log_weights,
                preference_strength=preference_strength,
            )
            integrated = float(logsumexp(components)) - log_observed
            if not math.isfinite(integrated):
                return math.inf
            value -= integrated
        return value
    except (ValueError, FloatingPointError):
        return math.inf


def _strict_minimize(
    objective: Any,
    starts: tuple[tuple[float, ...], ...],
    bounds: Any,
    *,
    label: str,
) -> Any:
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
        if result.success and math.isfinite(float(result.fun)) and np.isfinite(
            result.x
        ).all():
            candidates.append(result)
    if not candidates:
        raise RuntimeError(f"MRS2 {label} optimizer did not converge from any frozen start")
    best = min(
        candidates, key=lambda result: (float(result.fun), tuple(map(float, result.x)))
    )
    if any(
        value - lower <= BOUNDARY_TOLERANCE
        or upper - value <= BOUNDARY_TOLERANCE
        for value, (lower, upper) in zip(best.x, bounds, strict=True)
    ):
        raise RuntimeError(f"MRS2 {label} optimum reached a frozen parameter boundary")
    return best


def _timing_posteriors(
    sequences: list[tuple[datetime, ...]],
    *,
    profile: MRS2Profile,
    observation_start: datetime,
    observation_end: datetime,
    renewal_shape: float,
    mean_gap: float,
    frailty_shape: float,
    preference_strength: float,
    clock: CalendarHazardClock | None = None,
    bases: list[tuple[Any, Any, Any, float]] | None = None,
) -> list[tuple[Any, Any]]:
    import numpy as np
    from scipy.special import logsumexp

    frailties, log_weights = _quadrature_log_weights(frailty_shape)
    component_frailties = np.tile(frailties, 24)
    clock = clock or CalendarHazardClock(profile, observation_start, observation_end)
    bases = bases or _timing_bases(sequences, clock)
    posteriors = []
    for basis in bases:
        log_posterior = _timing_log_components_from_basis(
            basis,
            clock=clock,
            renewal_shape=renewal_shape,
            mean_gap=mean_gap,
            frailties=frailties,
            log_weights=log_weights,
            preference_strength=preference_strength,
        )
        log_posterior -= logsumexp(log_posterior)
        posteriors.append((component_frailties, log_posterior))
    return posteriors


def fit_timing_model(
    sequences: list[tuple[datetime, ...]],
    *,
    profile: MRS2Profile,
    observation_start: datetime,
    observation_end: datetime,
) -> tuple[tuple[float, float, float, float], list[tuple[Any, Any]]]:
    import numpy as np

    clock = CalendarHazardClock(profile, observation_start, observation_end)
    bases = _timing_bases(sequences, clock)
    result = _strict_minimize(
        lambda values: _timing_negative_log_likelihood_from_bases(
            values, bases, clock=clock
        ),
        TIMING_STARTS,
        TIMING_PARAMETER_BOUNDS,
        label="timing",
    )
    renewal_shape, mean_gap, frailty_shape = map(float, np.exp(result.x[:3]))
    strength = float(result.x[3])
    timing = (renewal_shape, mean_gap, frailty_shape, strength)
    return timing, _timing_posteriors(
        sequences,
        profile=profile,
        observation_start=observation_start,
        observation_end=observation_end,
        renewal_shape=renewal_shape,
        mean_gap=mean_gap,
        frailty_shape=frailty_shape,
        preference_strength=strength,
        clock=clock,
        bases=bases,
    )


def _amount_design(frame: Any, *, observation_start: datetime) -> list[tuple[Any, Any]]:
    import numpy as np

    design = []
    for _, group in frame.groupby("customer_id", sort=True):
        y = []
        log_gap = []
        previous = observation_start
        for row in group.sort_values("ts", kind="stable").itertuples(index=False):
            y.append(math.log(int(row.amount)))
            gap = (row.ts.to_pydatetime() - previous).total_seconds() / 86_400
            log_gap.append(math.log1p(gap))
            previous = row.ts.to_pydatetime()
        design.append((np.asarray(y, dtype=float), np.asarray(log_gap, dtype=float)))
    return design


def gaussian_ar1_covariance(
    length: int,
    *,
    ar1: float,
    innovation_scale: float,
    household_scale: float = 0.0,
) -> Any:
    """Stationary AR(1) covariance plus analytically integrated B covariance."""
    import numpy as np

    if type(length) is not int or length < 1:
        raise ValueError("length must be a positive integer")
    if not -1 < ar1 < 1 or innovation_scale <= 0 or household_scale < 0:
        raise ValueError("invalid Gaussian AR1 covariance parameters")
    distance = np.abs(np.subtract.outer(np.arange(length), np.arange(length)))
    stationary_variance = innovation_scale**2 / (1.0 - ar1**2)
    return stationary_variance * np.power(ar1, distance) + household_scale**2


def _collapse_timing_posterior(
    frailties: Any, log_posterior: Any
) -> tuple[Any, Any]:
    """Exactly sum C because amount marks depend on U but not C."""
    import numpy as np
    from scipy.special import logsumexp

    frailties = np.asarray(frailties, dtype=float)
    log_posterior = np.asarray(log_posterior, dtype=float)
    if len(frailties) == 24 * GAUSS_LAGUERRE_NODES:
        grid = frailties.reshape(24, GAUSS_LAGUERRE_NODES)
        if np.array_equal(grid, np.broadcast_to(grid[0], grid.shape)):
            return grid[0], logsumexp(
                log_posterior.reshape(24, GAUSS_LAGUERRE_NODES), axis=0
            )
    return frailties, log_posterior


def _ar1_rank1_log_density(
    residuals: Any,
    *,
    ar1: float,
    innovation_scale: float,
    household_scale: float,
) -> Any:
    """Exact O(n) Normal log density for R_AR1 + tau^2 11' covariance."""
    import numpy as np

    residuals = np.asarray(residuals, dtype=float)
    length = residuals.shape[1]
    inverse_innovation_variance = 1.0 / innovation_scale**2
    if length == 1:
        base_quadratic = (
            (1.0 - ar1**2)
            * residuals[:, 0] ** 2
            * inverse_innovation_variance
        )
        one_precision_residual = (
            (1.0 - ar1**2)
            * residuals[:, 0]
            * inverse_innovation_variance
        )
        one_precision_one = (1.0 - ar1**2) * inverse_innovation_variance
    else:
        base_quadratic = (
            residuals[:, 0] ** 2
            + residuals[:, -1] ** 2
            + (1.0 + ar1**2) * np.sum(residuals[:, 1:-1] ** 2, axis=1)
            - 2.0 * ar1 * np.sum(residuals[:, :-1] * residuals[:, 1:], axis=1)
        ) * inverse_innovation_variance
        precision_one = np.full(length, (1.0 - ar1) ** 2)
        precision_one[[0, -1]] = 1.0 - ar1
        precision_one *= inverse_innovation_variance
        one_precision_residual = residuals @ precision_one
        one_precision_one = float(np.sum(precision_one))
    denominator = 1.0 + household_scale**2 * one_precision_one
    quadratic = base_quadratic - (
        household_scale**2 * one_precision_residual**2 / denominator
    )
    log_determinant = (
        2.0 * length * math.log(innovation_scale)
        - math.log1p(-(ar1**2))
        + math.log(denominator)
    )
    return -0.5 * (
        length * math.log(2.0 * math.pi) + log_determinant + quadratic
    )


def amount_negative_log_likelihood(
    parameters: Any,
    design: list[tuple[Any, Any]],
    timing_posteriors: list[tuple[Any, Any]],
    *,
    frailty_shape: float,
) -> float:
    import numpy as np
    from scipy.special import logsumexp

    if len(design) != len(timing_posteriors) or not design:
        raise ValueError("amount design and timing posteriors must align by household")
    intercept, beta_score, beta_gap, ar1, log_innovation, log_household = map(
        float, parameters
    )
    innovation_scale = math.exp(log_innovation)
    household_scale = math.exp(log_household)
    value = 0.0
    for (y, log_gap), (frailties, log_posterior) in zip(
        design, timing_posteriors, strict=True
    ):
        frailties, log_posterior = _collapse_timing_posterior(
            frailties, log_posterior
        )
        scores = frailty_score(frailties, frailty_shape)
        means = (
            intercept
            + beta_score * scores[:, None]
            + beta_gap * log_gap[None, :]
        )
        residuals = y[None, :] - means
        log_density = _ar1_rank1_log_density(
            residuals,
            ar1=ar1,
            innovation_scale=innovation_scale,
            household_scale=household_scale,
        )
        integrated = float(logsumexp(log_posterior + log_density))
        if not math.isfinite(integrated):
            return math.inf
        value -= integrated
    return value


def fit_amount_model(
    frame: Any,
    timing_posteriors: list[tuple[Any, Any]],
    *,
    observation_start: datetime,
    frailty_shape: float,
) -> tuple[float, ...]:
    import numpy as np

    design = _amount_design(frame, observation_start=observation_start)
    result = _strict_minimize(
        lambda values: amount_negative_log_likelihood(
            values, design, timing_posteriors, frailty_shape=frailty_shape
        ),
        AMOUNT_STARTS,
        AMOUNT_PARAMETER_BOUNDS,
        label="amount",
    )
    intercept, beta_score, beta_gap, ar1, log_innovation, log_household = map(
        float, result.x
    )
    return (
        intercept,
        beta_score,
        beta_gap,
        float(math.exp(log_household)),
        ar1,
        float(math.exp(log_innovation)),
    )


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
) -> MRS2Profile:
    groups = frame.groupby(["customer_id", "ts"], sort=False).size()
    value_counts = groups.value_counts().sort_index()
    values = tuple(map(int, value_counts.index))
    counts = tuple(map(int, value_counts.values))
    epochs = sum(counts)
    probabilities = tuple(value / epochs for value in counts)
    pooled = pooled_hour_probabilities(
        start=start,
        end=end,
        hour_weights=calendar["hour_weights"],
        weekday_weights=calendar["weekday_weights"],
        week_weights=calendar["week_weights"],
    )
    return MRS2Profile(
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
        preferred_hour_probabilities=pooled,
        preferred_hour_strength=0.2,
        renewal_shape=1.0,
        mean_operational_gap_days=14.0,
        frailty_shape=1.0,
        multiplicity_values=values,
        multiplicity_counts=counts,
        multiplicity_probabilities=probabilities,
        amount_intercept=6.0,
        amount_frailty_score=0.0,
        amount_log1p_gap_days=0.0,
        amount_household_scale=1.0,
        amount_ar1=0.0,
        amount_innovation_scale=1.0,
    )


def fit_mrs2_profile(
    frame: Any,
    *,
    source_dataset: str,
    source_partition: str,
    customer_ids_sha256: str,
    transactions_sha256: str,
    observation_start: Any,
    observation_end: Any,
) -> MRS2Profile:
    start = _as_local_wall(observation_start)
    end = _as_local_wall(observation_end)
    frame = _validate_frame(frame, start=start, end=end)
    if customer_ids_sha256 != _ids_hash(frame["customer_id"].unique()) or (
        transactions_sha256 != _transactions_hash(frame)
    ):
        raise ValueError("MRS2 source hashes do not match the supplied frame")
    calendar = fit_calendar_hazard(
        frame, observation_start=start, observation_end=end
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
    sequences = _calendar_sequences(frame)
    timing, timing_posteriors = fit_timing_model(
        sequences,
        profile=shell,
        observation_start=start,
        observation_end=end,
    )
    amount = fit_amount_model(
        frame,
        timing_posteriors,
        observation_start=start,
        frailty_shape=timing[2],
    )
    # timing_posteriors are deliberately ephemeral and leave scope here.
    return MRS2Profile(
        **{
            **shell.__dict__,
            "preferred_hour_strength": timing[3],
            "renewal_shape": timing[0],
            "mean_operational_gap_days": timing[1],
            "frailty_shape": timing[2],
            "amount_intercept": amount[0],
            "amount_frailty_score": amount[1],
            "amount_log1p_gap_days": amount[2],
            "amount_household_scale": amount[3],
            "amount_ar1": amount[4],
            "amount_innovation_scale": amount[5],
        }
    )


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


def _write_artifact(path: Path, value: Mapping[str, Any]) -> bytes:
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
        raise ValueError(f"{path.name} must be immutable")
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or raw != _canonical_bytes(value):
        raise ValueError(f"{path.name} is not canonical JSON")
    return value


def _implementation_hashes() -> dict[str, str]:
    paths = (
        ROOT / "sim/population_mrs2.py",
        ROOT / "sim/entity_fidelity_mrs2.py",
        ROOT / "tests/test_population_mrs2.py",
        ROOT / "tests/test_entity_fidelity_mrs2.py",
    )
    if not all(path.is_file() and not path.is_symlink() for path in paths):
        raise FileNotFoundError("the four canonical MRS2 source/test files are required")
    return {str(path.relative_to(ROOT)): _file_sha256(path) for path in paths}


def _runtime_dependency_hashes() -> dict[str, str]:
    paths = (
        ROOT / "sim/mandate_twin.py",
        ROOT / "sim/population.py",
        ROOT / "sim/population_v4.py",
        ROOT / "sim/population_mrs1.py",
        ROOT / "sim/entity_fidelity_mrs1.py",
        ROOT / "sim/entity_fidelity.py",
        ROOT / "sim/entity_fidelity_v4.py",
        ROOT / "sim/completejourney_reference.py",
        ROOT / "scripts/download_completejourney_reference.py",
        ROOT / "scripts/download_retail_reference.py",
        ROOT / "scripts/download_retail_ii_reference.py",
    )
    if not all(path.is_file() and not path.is_symlink() for path in paths):
        raise FileNotFoundError("an MRS2 runtime source dependency is missing")
    return {str(path.relative_to(ROOT)): _file_sha256(path) for path in paths}


def _dependency_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in DEPENDENCY_PACKAGES}


def _mrs1_stop_binding() -> dict[str, Any]:
    value = _read_artifact(MRS1_STOP_PROTOCOL)
    file_sha256 = _file_sha256(MRS1_STOP_PROTOCOL)
    internal_hash = value.get("protocol_sha256")
    expected_manifest_sha256 = value.get(
        "expected_completejourney_manifest_sha256"
    )
    development_reference = value.get("development_reference", {})
    if (
        file_sha256 != MRS1_STOP_PROTOCOL_SHA256
        or value.get("artifact") != "entity_fidelity_mrs1_frozen_protocol"
        or value.get("development_diagnostic_status") != "STOP_FAIL"
        or value.get("development_fit_diagnostic", {}).get("gate_passed") is not False
        or value.get("qualification_accessed") is not False
        or value.get("uci352_accessed") is not False
        or value.get("uci502_accessed") is not False
        or internal_hash
        != _canonical_hash(
            {key: item for key, item in value.items() if key != "protocol_sha256"}
        )
        or not _is_sha256(value.get("mrs_preregistration_sha256"))
        or not _is_sha256(expected_manifest_sha256)
        or development_reference.get("partition") != "development"
        or development_reference.get("households") != DEVELOPMENT_HOUSEHOLDS
        or development_reference.get("manifest_sha256")
        != expected_manifest_sha256
        or not _is_sha256(development_reference.get("partition_sha256"))
        or not _is_sha256(development_reference.get("household_ids_sha256"))
        or development_reference.get("mrs_preregistration_sha256")
        != value.get("mrs_preregistration_sha256")
    ):
        raise ValueError("canonical MRS1 protocol is not the exact terminal STOP prerequisite")
    return {
        "path": "results/entity_fidelity_mrs1_protocol.json",
        "file_sha256": file_sha256,
        "protocol_sha256": internal_hash,
        "mrs1_preregistration_sha256": value["mrs_preregistration_sha256"],
        "expected_completejourney_manifest_sha256": expected_manifest_sha256,
        "development_partition_sha256": development_reference[
            "partition_sha256"
        ],
        "development_household_ids_sha256": development_reference[
            "household_ids_sha256"
        ],
        "development_diagnostic_status": "STOP_FAIL",
        "development_gate_passed": False,
        "qualification_accessed": False,
        "uci352_accessed": False,
        "uci502_accessed": False,
    }


def _completejourney_bindings(mrs1: Mapping[str, Any]) -> dict[str, Any]:
    r2 = _read_artifact(MRS1_R2_PREREGISTRATION)
    r2_sha256 = _file_sha256(MRS1_R2_PREREGISTRATION)
    if (
        r2.get("artifact") != "entity_fidelity_mrs1_r2_preregistration"
        or r2.get("study_revision") != "mrs1.2"
        or r2_sha256 != mrs1["mrs1_preregistration_sha256"]
    ):
        raise ValueError("Complete Journey MRS1 r2 preregistration binding changed")
    manifest = _read_artifact(DEFAULT_COMPLETEJOURNEY_MANIFEST)
    manifest_sha256 = _file_sha256(DEFAULT_COMPLETEJOURNEY_MANIFEST)
    partitions = manifest.get("partitions", {})
    development = partitions.get("development", {})
    qualification = partitions.get("qualification", {})
    source = manifest.get("source", {})
    preprocessing = manifest.get("preprocessing", {})
    observation_window = manifest.get("observation_window", {})
    if (
        manifest.get("dataset") != "completejourney"
        or manifest.get("version") != "1.1.1"
        or manifest_sha256 != mrs1["expected_completejourney_manifest_sha256"]
        or manifest.get("mrs_preregistration_sha256") != r2_sha256
        or observation_window
        != {
            "start_inclusive": COMPLETEJOURNEY_OBSERVATION_START,
            "end_exclusive": COMPLETEJOURNEY_OBSERVATION_END,
            "timezone": None,
        }
        or preprocessing.get("timestamp_semantics")
        != COMPLETEJOURNEY_TIMESTAMP_SEMANTICS
        or not _is_sha256(source.get("rds_sha256"))
        or development.get("file") != "completejourney_development.parquet"
        or development.get("households") != DEVELOPMENT_HOUSEHOLDS
        or type(development.get("baskets")) is not int
        or development["baskets"] < DEVELOPMENT_HOUSEHOLDS
        or development.get("parquet_sha256")
        != mrs1["development_partition_sha256"]
        or development.get("household_ids_sha256")
        != mrs1["development_household_ids_sha256"]
        or development.get("mrs_preregistration_sha256") != r2_sha256
        or qualification.get("file") != "completejourney_qualification.parquet"
        or qualification.get("households") != QUALIFICATION_HOUSEHOLDS
        or qualification.get("mrs_preregistration_sha256") != r2_sha256
    ):
        raise ValueError("Complete Journey manifest or partition binding changed")
    return {
        "mrs1_r2_preregistration_path": (
            "results/entity_fidelity_mrs1_r2_preregistration.json"
        ),
        "mrs1_r2_preregistration_sha256": r2_sha256,
        "manifest_path": (
            "data/reference/completejourney/completejourney_reference.manifest.json"
        ),
        "manifest_sha256": manifest_sha256,
        "development_parquet_path": (
            "data/reference/completejourney/completejourney_development.parquet"
        ),
        "development_parquet_sha256": development["parquet_sha256"],
        "development_parquet_hash_source": "immutable Complete Journey manifest",
        "development_parquet_opened_before_preregistration": False,
        "dataset": "completejourney",
        "version": manifest["version"],
        "development_households": DEVELOPMENT_HOUSEHOLDS,
        "development_baskets": development["baskets"],
        "development_household_ids_sha256": development[
            "household_ids_sha256"
        ],
        "source_rds_sha256": source["rds_sha256"],
        "timestamp_semantics": preprocessing["timestamp_semantics"],
        "observation_start": observation_window["start_inclusive"],
        "observation_end_exclusive": observation_window["end_exclusive"],
        "amount_unit": "USD cents",
        "qualification_households": QUALIFICATION_HOUSEHOLDS,
        "qualification_parquet_opened": False,
    }


def _preregistration_payload() -> dict[str, Any]:
    mrs1 = _mrs1_stop_binding()
    source = _completejourney_bindings(mrs1)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "entity_fidelity_mrs2_preregistration",
        "study_revision": STUDY_REVISION,
        "lineage": mrs1,
        "completejourney": source,
        "model": {
            "renewal": "MRS1 stationary Gamma renewal with right censoring and N>0 conditioning",
            "global_calendar": "unchanged MRS1 hour*weekday*ISO-week operational clock",
            "preferred_hour_class": "C~Categorical(g)",
            "pooled_hour_mass": "g_h=integral(base_hazard*1[hour=h])/integral(base_hazard)",
            "preferred_hour_multiplier": "r_c(h)=(1-lambda)+lambda*1[h=c]/g_c",
            "timing_likelihood": "full C,U mixture with calendar Jacobian and zero-truncation",
            "frailty": "U~Gamma(a,rate=a)",
            "frailty_score": "sqrt(3)*(2*GammaCDF(U;a,rate=a)-1)",
            "amount": "Gaussian stationary AR1 on log integer cents plus B~Normal(0,tau^2)",
            "amount_likelihood": "full timing posterior over C,U; B analytically integrated",
            "multiplicity": "exact observed positive integer PMF",
            "rounding": "positive half-even integer cents",
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
            "failure": "terminal STOP; no retry under this preregistration",
        },
        "evaluator": {
            "features": list(FEATURES),
            "classifier": _classifier_protocol(),
            "seeds": list(EVALUATION_SEEDS),
            "folds_per_seed": CROSS_FIT_FOLDS,
            "total_folds": len(EVALUATION_SEEDS) * CROSS_FIT_FOLDS,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": FULL_BOOTSTRAP_SEED,
            "orientation_free_threshold": GATE_MAX_D_UPPER_95,
            "profile_scope": "all development; discriminator/calibration cross-fit only",
        },
        "one_shot": {
            "attempt_consumed_before_loader_import_or_call": True,
            "base_exception_after_attempt_is_terminal": True,
            "retry_permitted": False,
            "pass_status": "PASS",
            "failure_status": "STOP_FAIL",
            "qualification_implemented": False,
            "transfer_implemented": False,
            "final_implemented": False,
        },
        "privacy": {
            "aggregate_only_artifacts": True,
            "contains_ids": False,
            "contains_rows": False,
            "contains_posteriors": False,
        },
        "implementation_sha256": _implementation_hashes(),
        "runtime_dependency_sha256": _runtime_dependency_hashes(),
        "dependency_versions": _dependency_versions(),
        "dependency_lock_sha256": _file_sha256(ROOT / "requirements.lock"),
    }


def write_mrs2_preregistration() -> dict[str, Any]:
    if any(
        path.exists() or path.is_symlink()
        for path in (DEFAULT_DEVELOPMENT_ATTEMPT, DEFAULT_DEVELOPMENT)
    ):
        raise FileExistsError("MRS2 preregistration cannot follow a development attempt")
    value = _preregistration_payload()
    _write_artifact(DEFAULT_PREREGISTRATION, value)
    return value


def load_mrs2_preregistration() -> dict[str, Any]:
    value = _read_artifact(DEFAULT_PREREGISTRATION)
    if value != _preregistration_payload():
        raise ValueError("MRS2 preregistration differs from its exact frozen bindings")
    return value


def _source_sequence_hashes(frame: Any, *, start: datetime) -> set[str]:
    return {
        sequence_sha256(
            [
                (row.ts.to_pydatetime(), int(row.amount))
                for row in group.itertuples(index=False)
            ],
            start=start,
        )
        for _, group in frame.groupby("customer_id", sort=False)
    }


def validate_generator(
    profile: MRS2Profile,
    source_frame: Any,
    *,
    seed: int,
    customers: int | None = None,
) -> dict[str, Any]:
    start = datetime.fromisoformat(profile.observation_start)
    source_ids = set(source_frame["customer_id"].astype(str))
    arguments = {
        "customers": customers or profile.source_customer_count,
        "seed": seed,
        "forbidden_customer_ids": source_ids,
        "forbidden_sequence_hashes": _source_sequence_hashes(
            source_frame, start=start
        ),
    }
    first = generate_synthetic_transactions_mrs2(profile, **arguments)
    second = generate_synthetic_transactions_mrs2(profile, **arguments)
    first_hash = _transactions_hash(first)
    if first_hash != _transactions_hash(second):
        raise RuntimeError("MRS2 deterministic generator check failed")
    expected_customers = int(arguments["customers"])
    if first["customer_id"].nunique() != expected_customers or first["amount"].dtype.kind not in "iu":
        raise RuntimeError("MRS2 generator structural check failed")
    return {
        "seed": seed,
        "customers": expected_customers,
        "transactions": len(first),
        "synthetic_transactions_sha256": first_hash,
        "exact_source_sequence_matches": 0,
        "deterministic": True,
        "positive_integer_cents": bool(first["amount"].ge(1).all()),
    }


def validate_mrs2_structure(profile: MRS2Profile) -> dict[str, Any]:
    import numpy as np

    start = datetime.fromisoformat(profile.observation_start)
    end = datetime.fromisoformat(profile.observation_end)
    base = CalendarHazardClock(profile, start, end)
    pooled_rate_error = max(
        abs(
            sum(
                probability
                * preferred_hour_multiplier(
                    profile.preferred_hour_probabilities,
                    profile.preferred_hour_strength,
                    preferred_hour,
                    observed_hour,
                )
                for preferred_hour, probability in enumerate(
                    profile.preferred_hour_probabilities
                )
            )
            - 1.0
        )
        for observed_hour in range(24)
    )
    class_time_error = max(
        abs(
            base.class_operational(
                end, preferred_hour, profile.preferred_hour_strength
            )
            - base.total
        )
        for preferred_hour in range(24)
    )
    frailties, log_weights = _quadrature_log_weights(profile.frailty_shape)
    weights = np.exp(log_weights)
    covariance = gaussian_ar1_covariance(
        4,
        ar1=profile.amount_ar1,
        innovation_scale=profile.amount_innovation_scale,
        household_scale=profile.amount_household_scale,
    )
    checks = {
        "preferred_hour_classes": 24,
        "pooled_rate_max_abs_error": float(pooled_rate_error),
        "class_horizon_max_abs_error": float(class_time_error),
        "quadrature_nodes": len(frailties),
        "quadrature_mass": float(weights.sum()),
        "quadrature_mean_frailty": float(np.sum(weights * frailties)),
        "frailty_score_bounded": bool(
            np.max(np.abs(frailty_score(frailties, profile.frailty_shape)))
            <= math.sqrt(3)
        ),
        "amount_covariance_positive_definite": bool(
            np.linalg.eigvalsh(covariance).min() > 0
        ),
        "multiplicity_epoch_count": sum(profile.multiplicity_counts),
        "multiplicity_transaction_count": sum(
            value * count
            for value, count in zip(
                profile.multiplicity_values,
                profile.multiplicity_counts,
                strict=True,
            )
        ),
    }
    if not (
        checks["pooled_rate_max_abs_error"] <= 1e-12
        and checks["class_horizon_max_abs_error"] <= 1e-9
        and checks["quadrature_nodes"] == GAUSS_LAGUERRE_NODES
        and math.isclose(checks["quadrature_mass"], 1.0, abs_tol=2e-12)
        and math.isclose(
            checks["quadrature_mean_frailty"], 1.0, abs_tol=2e-12
        )
        and checks["frailty_score_bounded"] is True
        and checks["amount_covariance_positive_definite"] is True
        and checks["multiplicity_transaction_count"]
        == profile.source_transaction_count
    ):
        raise RuntimeError("MRS2 structural validation failed")
    return checks


def _fold_assignments(customer_ids: Any, *, seed: int) -> dict[str, int]:
    return {
        str(customer_id): int.from_bytes(
            hashlib.sha256(
                f"{seed}:mrs2-frozen-development-crossfit:{customer_id}".encode()
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
    frozen_profile: MRS2Profile,
) -> list[dict[str, Any]]:
    import numpy as np
    import pandas as pd

    _, _, _, roc_auc_score = _dependencies()
    assignments = _fold_assignments(frame["customer_id"], seed=seed)
    source_ids = set(frame["customer_id"].astype(str))
    source_hashes = _source_sequence_hashes(
        frame, start=_as_local_wall(reference["observation_start"])
    )
    target_start = _as_local_wall(reference["observation_start"])
    target_end = _as_local_wall(reference["observation_end"])
    folds = []
    for fold in range(CROSS_FIT_FOLDS):
        evaluation_ids = {
            customer_id
            for customer_id, assigned in assignments.items()
            if assigned == fold
        }
        training_ids = set(assignments) - evaluation_ids
        if not training_ids or not evaluation_ids:
            raise ValueError("MRS2 cross-fit produced an empty fold")
        real_training = frame.loc[frame["customer_id"].isin(training_ids)].copy()
        real_evaluation = frame.loc[frame["customer_id"].isin(evaluation_ids)].copy()
        synthetic_training = generate_synthetic_transactions_mrs2(
            frozen_profile,
            customers=len(training_ids),
            seed=seed + 10_000 * fold + 101,
            start=target_start,
            end=target_end,
            forbidden_customer_ids=source_ids,
            forbidden_sequence_hashes=source_hashes,
        )
        synthetic_evaluation = generate_synthetic_transactions_mrs2(
            frozen_profile,
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
        real_evaluation_features = customer_features(
            real_evaluation, real_transform
        )
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
                "profile_sha256": frozen_profile.sha256,
                "auc": auc,
                "d": max(auc, 1 - auc),
                "_real_customer_ids": tuple(real_evaluation_features.index),
                "_real_predictions": real_predictions,
                "_synthetic_predictions": synthetic_predictions,
            }
        )
    return folds


def evaluate_mrs2_gate(
    frame: Any,
    reference: Mapping[str, Any],
    *,
    frozen_profile: MRS2Profile,
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
                    {
                        key: value
                        for key, value in fold.items()
                        if not key.startswith("_")
                    }
                    for fold in folds
                ],
            }
            for seed, folds in seed_folds.items()
        },
        "metrics": {
            "orientation_free_max_seed_d": max(seed_scores.values()),
            "bootstrap_max_seed_d_mean": float(np.mean(bootstrap)),
            "bootstrap_max_seed_d_ci_95": [float(low), float(high)],
            "threshold_max_d_upper_95": GATE_MAX_D_UPPER_95,
        },
        "gate_passed": bool(high <= GATE_MAX_D_UPPER_95),
    }


def _gate_evidence_valid(value: Mapping[str, Any]) -> bool:
    return _mrs1_gate_evidence_valid(value)


def _load_canonical_development(
    preregistration: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Import and call the trusted loader only after the attempt is consumed."""
    from sim.completejourney_reference import load_completejourney_partition

    source = preregistration["completejourney"]
    raw, reference = load_completejourney_partition(
        DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR,
        "development",
        expected_manifest_sha256=source["manifest_sha256"],
        expected_mrs_preregistration_sha256=source[
            "mrs1_r2_preregistration_sha256"
        ],
    )
    return raw, dict(reference)


def _expected_development_loader_reference(
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    source = preregistration["completejourney"]
    return {
        "dataset": source["dataset"],
        "version": source["version"],
        "partition": "development",
        "manifest_sha256": source["manifest_sha256"],
        "partition_sha256": source["development_parquet_sha256"],
        "household_ids_sha256": source["development_household_ids_sha256"],
        "households": source["development_households"],
        "baskets": source["development_baskets"],
        "source_rds_sha256": source["source_rds_sha256"],
        "mrs_preregistration_sha256": source[
            "mrs1_r2_preregistration_sha256"
        ],
        "timestamp_semantics": source["timestamp_semantics"],
        "observation_start": source["observation_start"],
        "observation_end_exclusive": source["observation_end_exclusive"],
        "amount_unit": source["amount_unit"],
    }


def _adapt_development(
    raw: Any,
    reference: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    if tuple(raw.columns) != (
        "household_id",
        "basket_id",
        "transaction_timestamp",
        "sales_value_cents",
    ):
        raise ValueError("Complete Journey adapter received a changed basket schema")
    expected_reference = _expected_development_loader_reference(preregistration)
    if (
        dict(reference) != expected_reference
        or reference.get("baskets") != len(raw)
    ):
        raise ValueError("Complete Journey provenance violates the MRS2 preregistration")
    frame = raw.rename(
        columns={
            "household_id": "customer_id",
            "transaction_timestamp": "ts",
            "sales_value_cents": "amount",
        }
    )[["customer_id", "ts", "amount"]].copy()
    start = datetime.fromisoformat(COMPLETEJOURNEY_OBSERVATION_START)
    end = datetime.fromisoformat(COMPLETEJOURNEY_OBSERVATION_END)
    frame = _validate_frame(frame, start=start, end=end)
    if frame["customer_id"].nunique() != DEVELOPMENT_HOUSEHOLDS:
        raise ValueError("Complete Journey adapter lost a development household")
    return frame, {
        **dict(reference),
        "observation_end": COMPLETEJOURNEY_OBSERVATION_END,
        "profile_customer_ids_sha256": _ids_hash(frame["customer_id"].unique()),
        "profile_transactions_sha256": _transactions_hash(frame),
    }


def _attempt_payload(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "entity_fidelity_mrs2_development_attempt",
        "status": "CONSUMED_ONE_SHOT",
        "mrs2_preregistration_sha256": _file_sha256(DEFAULT_PREREGISTRATION),
        "mrs1_stop_protocol_sha256": MRS1_STOP_PROTOCOL_SHA256,
        "completejourney": preregistration["completejourney"],
        "implementation_sha256": _implementation_hashes(),
        "runtime_dependency_sha256": _runtime_dependency_hashes(),
        "dependency_versions": _dependency_versions(),
        "dependency_lock_sha256": _file_sha256(ROOT / "requirements.lock"),
        "development_loader_imported_or_called": False,
        "qualification_accessed": False,
        "uci_sources_accessed": False,
        "retry_permitted": False,
    }


def _artifact_hash(value: Mapping[str, Any]) -> str:
    return _canonical_hash(
        {key: item for key, item in value.items() if key != "artifact_sha256"}
    )


def run_mrs2_development() -> dict[str, Any]:
    preregistration = load_mrs2_preregistration()
    if DEFAULT_DEVELOPMENT.exists() or DEFAULT_DEVELOPMENT.is_symlink():
        raise FileExistsError("MRS2 development output already exists")
    attempt = _attempt_payload(preregistration)
    _write_artifact(DEFAULT_DEVELOPMENT_ATTEMPT, attempt)
    attempt_sha256 = _file_sha256(DEFAULT_DEVELOPMENT_ATTEMPT)

    # Everything below this line is terminal on BaseException: the attempt stays consumed.
    raw, loaded_reference = _load_canonical_development(preregistration)
    frame, reference = _adapt_development(
        raw, loaded_reference, preregistration
    )
    profile = fit_mrs2_profile(
        frame,
        source_dataset=f"completejourney-{reference['version']}",
        source_partition="development",
        customer_ids_sha256=reference["profile_customer_ids_sha256"],
        transactions_sha256=reference["profile_transactions_sha256"],
        observation_start=COMPLETEJOURNEY_OBSERVATION_START,
        observation_end=COMPLETEJOURNEY_OBSERVATION_END,
    )
    structural_checks = validate_mrs2_structure(profile)
    generator_check = validate_generator(profile, frame, seed=SEED)
    gate = evaluate_mrs2_gate(frame, reference, frozen_profile=profile)
    if not _gate_evidence_valid(gate):
        raise RuntimeError("MRS2 development gate returned invalid evidence")
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "entity_fidelity_mrs2_development",
        "study_revision": STUDY_REVISION,
        "status": "PASS" if gate["gate_passed"] else "STOP_FAIL",
        "mrs2_preregistration_sha256": _file_sha256(DEFAULT_PREREGISTRATION),
        "development_attempt_sha256": attempt_sha256,
        "mrs1_stop_protocol_sha256": MRS1_STOP_PROTOCOL_SHA256,
        "completejourney": preregistration["completejourney"],
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
                "observation_start",
                "observation_end_exclusive",
                "observation_end",
                "amount_unit",
                "profile_customer_ids_sha256",
                "profile_transactions_sha256",
            )
        },
        "development_profile": profile.as_dict(),
        "development_profile_sha256": profile.sha256,
        "structural_checks": structural_checks,
        "generator_check": generator_check,
        "gate": gate,
        "implementation_sha256": _implementation_hashes(),
        "runtime_dependency_sha256": _runtime_dependency_hashes(),
        "dependency_versions": _dependency_versions(),
        "dependency_lock_sha256": _file_sha256(ROOT / "requirements.lock"),
        "development_partition_accessed": True,
        "qualification_accessed": False,
        "qualification_artifact_created": False,
        "uci_sources_accessed": False,
        "retry_permitted": False,
    }
    value["artifact_sha256"] = _artifact_hash(value)
    _write_artifact(DEFAULT_DEVELOPMENT, value)
    return value


def _generator_evidence_valid(value: object, *, customers: int) -> bool:
    return (
        type(value) is dict
        and set(value)
        == {
            "seed",
            "customers",
            "transactions",
            "synthetic_transactions_sha256",
            "exact_source_sequence_matches",
            "deterministic",
            "positive_integer_cents",
        }
        and value["seed"] == SEED
        and value["customers"] == customers
        and type(value["transactions"]) is int
        and value["transactions"] >= customers
        and _is_sha256(value["synthetic_transactions_sha256"])
        and value["exact_source_sequence_matches"] == 0
        and value["deterministic"] is True
        and value["positive_integer_cents"] is True
    )


def verify_mrs2_lifecycle(stage: str) -> dict[str, Any]:
    if stage == "preregistration":
        return load_mrs2_preregistration()
    if stage != "development":
        raise ValueError("MRS2 verification stage must be preregistration or development")
    preregistration = load_mrs2_preregistration()
    attempt = _read_artifact(DEFAULT_DEVELOPMENT_ATTEMPT)
    if attempt != _attempt_payload(preregistration):
        raise ValueError("MRS2 development attempt binding changed")
    value = _read_artifact(DEFAULT_DEVELOPMENT)
    expected_keys = {
        "schema_version",
        "artifact",
        "study_revision",
        "status",
        "mrs2_preregistration_sha256",
        "development_attempt_sha256",
        "mrs1_stop_protocol_sha256",
        "completejourney",
        "development_reference",
        "development_profile",
        "development_profile_sha256",
        "structural_checks",
        "generator_check",
        "gate",
        "implementation_sha256",
        "runtime_dependency_sha256",
        "dependency_versions",
        "dependency_lock_sha256",
        "development_partition_accessed",
        "qualification_accessed",
        "qualification_artifact_created",
        "uci_sources_accessed",
        "retry_permitted",
        "artifact_sha256",
    }
    profile = MRS2Profile.from_dict(value.get("development_profile", {}))
    gate = value.get("gate", {})
    expected_status = "PASS" if gate.get("gate_passed") is True else "STOP_FAIL"
    if (
        set(value) != expected_keys
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("artifact") != "entity_fidelity_mrs2_development"
        or value.get("study_revision") != STUDY_REVISION
        or value.get("artifact_sha256") != _artifact_hash(value)
        or value.get("mrs2_preregistration_sha256")
        != _file_sha256(DEFAULT_PREREGISTRATION)
        or value.get("development_attempt_sha256")
        != _file_sha256(DEFAULT_DEVELOPMENT_ATTEMPT)
        or value.get("mrs1_stop_protocol_sha256") != MRS1_STOP_PROTOCOL_SHA256
        or value.get("completejourney") != preregistration["completejourney"]
        or value.get("development_profile_sha256") != profile.sha256
        or value.get("structural_checks") != validate_mrs2_structure(profile)
        or not _generator_evidence_valid(
            value.get("generator_check"), customers=DEVELOPMENT_HOUSEHOLDS
        )
        or not _gate_evidence_valid(gate)
        or value.get("status") != expected_status
        or value.get("implementation_sha256") != _implementation_hashes()
        or value.get("runtime_dependency_sha256")
        != _runtime_dependency_hashes()
        or value.get("dependency_versions") != _dependency_versions()
        or value.get("dependency_lock_sha256")
        != _file_sha256(ROOT / "requirements.lock")
        or value.get("development_partition_accessed") is not True
        or value.get("qualification_accessed") is not False
        or value.get("qualification_artifact_created") is not False
        or value.get("uci_sources_accessed") is not False
        or value.get("retry_permitted") is not False
    ):
        raise ValueError("MRS2 development artifact failed exact verification")
    reference = value.get("development_reference", {})
    expected_reference = {
        **_expected_development_loader_reference(preregistration),
        "observation_end": preregistration["completejourney"][
            "observation_end_exclusive"
        ],
        "profile_customer_ids_sha256": profile.source_customer_ids_sha256,
        "profile_transactions_sha256": profile.source_transactions_sha256,
    }
    if (
        reference != expected_reference
        or profile.source_dataset
        != f"completejourney-{preregistration['completejourney']['version']}"
        or profile.source_customer_count != DEVELOPMENT_HOUSEHOLDS
        or profile.source_transaction_count != reference.get("baskets")
        or profile.source_partition != "development"
        or profile.observation_start != reference.get("observation_start")
        or profile.observation_end != reference.get("observation_end")
    ):
        raise ValueError("MRS2 development reference binding changed")
    if any(
        fold.get("profile_sha256") != profile.sha256
        for seed in gate["seeds"].values()
        for fold in seed["folds"]
    ):
        raise ValueError("MRS2 gate fold profile binding changed")
    raw, loaded_reference = _load_canonical_development(preregistration)
    frame, recomputed_reference = _adapt_development(
        raw, loaded_reference, preregistration
    )
    if recomputed_reference != reference:
        raise ValueError("MRS2 development source recomputation changed")
    recomputed_gate = evaluate_mrs2_gate(
        frame, recomputed_reference, frozen_profile=profile
    )
    if recomputed_gate != gate:
        raise ValueError("MRS2 development gate recomputation changed")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preregister")
    commands.add_parser("development")
    verify = commands.add_parser("verify")
    verify.add_argument(
        "--stage", choices=("preregistration", "development"), required=True
    )
    arguments = parser.parse_args()
    if arguments.command == "preregister":
        value = write_mrs2_preregistration()
    elif arguments.command == "development":
        value = run_mrs2_development()
    else:
        value = verify_mrs2_lifecycle(arguments.stage)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
