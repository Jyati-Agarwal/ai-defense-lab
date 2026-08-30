"""Schema-6 support-safe checkerboard population and canonical generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from sim.population import ROOT
from sim.population_v4 import (
    CHECKERBOARD_BINS,
    QUANTILE_PROBABILITIES,
    PopulationSimulatorV4,
    _canonical_hash,
    _largest_remainder_counts,
    _residual_cell_counts,
    checkerboard_marginals,
)


CHECKERBOARD_STRATA = ("singleton", "multiday_repeat")
SINGLETON_CONTINUOUS_PRIMITIVES = (
    "active_day_position",
    "relative_log_amount_center",
)
REPEAT_PRIMITIVES = (
    "invoice_count",
    "active_end_fraction",
    "span_within_end_fraction",
    "relative_log_amount_center",
    "eb_log_amount_scale",
    "eb_short_session_probability",
    "circadian_phase_latent",
    "circadian_resultant",
)
STRATUM_PRIMITIVES = {
    "singleton": SINGLETON_CONTINUOUS_PRIMITIVES + ("fixed_hour",),
    "multiday_repeat": REPEAT_PRIMITIVES,
}
STRATUM_CELL_SHAPES = {
    "singleton": (CHECKERBOARD_BINS, CHECKERBOARD_BINS, 24),
    "multiday_repeat": (CHECKERBOARD_BINS,) * len(REPEAT_PRIMITIVES),
}
STRATUM_CELL_COUNTS = {
    name: math.prod(shape) for name, shape in STRATUM_CELL_SHAPES.items()
}
MIN_CHECKERBOARD_CUSTOMERS = 32
ATOM_SELECTION_NAMESPACE = "entity-fidelity-v6-same-day-atom"
ATOM_SELECTION_RULE = (
    "select exactly round(q*repeat_count) repeat positions by ascending "
    "SHA256(seed:entity-fidelity-v6-same-day-atom:position), ties by position"
)
LOCALIZATION_FACTOR_DEFAULT = 1.0
SAMPLED_PRIMITIVE_FIELDS = frozenset(
    {
        "invoice_count",
        "active_day_position",
        "active_end_fraction",
        "span_within_end_fraction",
        "relative_log_amount_center",
        "eb_log_amount_scale",
        "eb_short_session_probability",
        "circadian_phase_latent",
        "circadian_phase_sin",
        "circadian_phase_cos",
        "circadian_resultant",
        "eb_circadian_concentration",
        "fixed_hour",
        "first_day",
        "last_day",
        "active_start_fraction",
        "active_duration_fraction",
        "same_day_atom",
    }
)


def _require_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} fields differ from the frozen schema")


def _continuous_primitives(name: str) -> tuple[str, ...]:
    return (
        SINGLETON_CONTINUOUS_PRIMITIVES
        if name == "singleton"
        else REPEAT_PRIMITIVES
    )


@dataclass(frozen=True)
class CheckerboardStratumV6:
    """One fixed-size V6 mixed categorical/continuous checkerboard."""

    name: str
    customer_count: int
    primitive_quantiles: tuple[tuple[float, ...], ...]
    cell_masses: tuple[float, ...]

    def __post_init__(self) -> None:
        import numpy as np

        if self.name not in CHECKERBOARD_STRATA:
            raise ValueError("unknown v6 checkerboard stratum")
        primitives = _continuous_primitives(self.name)
        if self.customer_count < MIN_CHECKERBOARD_CUSTOMERS:
            raise ValueError("every v6 checkerboard needs at least 32 customers")
        if (
            len(self.primitive_quantiles) != len(primitives)
            or any(len(values) != 32 for values in self.primitive_quantiles)
            or any(
                not math.isfinite(value)
                for values in self.primitive_quantiles
                for value in values
            )
            or any(
                left > right
                for values in self.primitive_quantiles
                for left, right in zip(values, values[1:])
            )
        ):
            raise ValueError("v6 checkerboard quantiles are invalid")
        quantiles_by_name = dict(
            zip(primitives, self.primitive_quantiles, strict=True)
        )
        if self.name == "singleton":
            if any(
                not 0 <= value < 1
                for value in quantiles_by_name["active_day_position"]
            ):
                raise ValueError("v6 singleton day quantiles left their support")
        elif (
            any(
                value < 2 or not float(value).is_integer()
                for value in quantiles_by_name["invoice_count"]
            )
            or any(
                not 0 <= value < 1
                for name in ("active_end_fraction", "span_within_end_fraction")
                for value in quantiles_by_name[name]
            )
            or any(value <= 0 for value in quantiles_by_name["eb_log_amount_scale"])
            or any(
                not 0 <= value <= 1
                for name in (
                    "eb_short_session_probability",
                    "circadian_resultant",
                )
                for value in quantiles_by_name[name]
            )
            or any(
                not 0 <= value < 2 * math.pi
                for value in quantiles_by_name["circadian_phase_latent"]
            )
        ):
            raise ValueError("v6 repeat quantiles left their support")
        masses = np.asarray(self.cell_masses, dtype=float)
        if (
            masses.shape != (STRATUM_CELL_COUNTS[self.name],)
            or not np.isfinite(masses).all()
            or (masses < 0).any()
            or not math.isclose(float(masses.sum()), 1.0, abs_tol=2e-10)
        ):
            raise ValueError("v6 checkerboard masses are invalid")
        if self.name == "singleton":
            tensor = masses.reshape(STRATUM_CELL_SHAPES[self.name])
            marginals = (tensor.sum(axis=(1, 2)), tensor.sum(axis=(0, 2)))
        else:
            marginals = checkerboard_marginals(masses, len(REPEAT_PRIMITIVES))
        if any(
            not math.isclose(float(value), 1 / CHECKERBOARD_BINS, abs_tol=2e-10)
            for marginal in marginals
            for value in marginal
        ):
            raise ValueError("v6 continuous checkerboard marginals must be thirds")

    @property
    def primitives(self) -> tuple[str, ...]:
        return STRATUM_PRIMITIVES[self.name]

    def as_dict(self) -> dict[str, Any]:
        continuous = _continuous_primitives(self.name)
        return {
            "customers": self.customer_count,
            "primitives": list(self.primitives),
            "cell_shape": list(STRATUM_CELL_SHAPES[self.name]),
            "primitive_quantiles": {
                name: list(values)
                for name, values in zip(
                    continuous, self.primitive_quantiles, strict=True
                )
            },
            "cell_masses": list(self.cell_masses),
        }

    @classmethod
    def from_dict(
        cls, name: str, value: dict[str, Any]
    ) -> "CheckerboardStratumV6":
        _require_keys(
            value,
            {
                "customers",
                "primitives",
                "cell_shape",
                "primitive_quantiles",
                "cell_masses",
            },
            f"v6 {name} stratum",
        )
        if (
            name not in CHECKERBOARD_STRATA
            or tuple(value["primitives"]) != STRATUM_PRIMITIVES[name]
            or tuple(value["cell_shape"]) != STRATUM_CELL_SHAPES[name]
        ):
            raise ValueError("v6 checkerboard structure changed")
        continuous = _continuous_primitives(name)
        quantiles = value["primitive_quantiles"]
        if set(quantiles) != set(continuous):
            raise ValueError("v6 checkerboard quantile fields changed")
        return cls(
            name=name,
            customer_count=int(value["customers"]),
            primitive_quantiles=tuple(
                tuple(map(float, quantiles[primitive]))
                for primitive in continuous
            ),
            cell_masses=tuple(map(float, value["cell_masses"])),
        )


@dataclass(frozen=True)
class BehaviorProfileV6:
    """Fixed aggregates for support-safe day and hour rendering."""

    source_dataset: str
    source_partition: str
    source_customer_ids_sha256: str
    source_transactions_sha256: str
    source_customer_count: int
    source_transaction_count: int
    repeat_customer_count: int
    same_day_repeat_probability: float
    quantile_probabilities: tuple[float, ...]
    strata: tuple[CheckerboardStratumV6, ...]
    hour_weights: tuple[float, ...]
    calendar_weights: tuple[float, ...]
    observation_days: int
    session_gap_log_mean: float
    session_gap_log_std: float
    circadian_phase_cut: float
    global_log_amount_center: float
    localization_factor: float
    amount_variance_prior_alpha: float
    amount_variance_prior_beta: float
    short_session_prior_alpha: float
    short_session_prior_beta: float

    def __post_init__(self) -> None:
        if self.source_partition != "calibration":
            raise ValueError("v6 profiles may only be fitted on calibration customers")
        if self.quantile_probabilities != QUANTILE_PROBABILITIES:
            raise ValueError("v6 profile must use the frozen 32 knots")
        if tuple(item.name for item in self.strata) != CHECKERBOARD_STRATA:
            raise ValueError("v6 checkerboard order changed")
        singleton_count, multiday_count = (item.customer_count for item in self.strata)
        if (
            singleton_count + self.repeat_customer_count != self.source_customer_count
            or self.repeat_customer_count < multiday_count
            or self.source_transaction_count < self.source_customer_count
            or len(self.source_customer_ids_sha256) != 64
            or len(self.source_transactions_sha256) != 64
        ):
            raise ValueError("v6 source or structural counts are invalid")
        expected_q = (self.repeat_customer_count - multiday_count) / self.repeat_customer_count
        if (
            not math.isfinite(self.same_day_repeat_probability)
            or not math.isclose(
                self.same_day_repeat_probability, expected_q, abs_tol=1e-15
            )
        ):
            raise ValueError("v6 same-day repeat atom is inconsistent")
        if (
            len(self.hour_weights) != 24
            or any(weight < 0 or not math.isfinite(weight) for weight in self.hour_weights)
            or not math.isclose(sum(self.hour_weights), 1.0, abs_tol=1e-9)
            or not any(weight > 0 for weight in self.hour_weights)
        ):
            raise ValueError("v6 profile needs 24 normalized hour weights")
        if (
            len(self.calendar_weights) != 12
            or any(weight < 0 or not math.isfinite(weight) for weight in self.calendar_weights)
            or not math.isclose(sum(self.calendar_weights), 1.0, abs_tol=1e-9)
        ):
            raise ValueError("v6 profile needs 12 normalized calendar weights")
        scalars = (
            self.session_gap_log_mean,
            self.session_gap_log_std,
            self.circadian_phase_cut,
            self.global_log_amount_center,
            self.localization_factor,
            self.amount_variance_prior_alpha,
            self.amount_variance_prior_beta,
            self.short_session_prior_alpha,
            self.short_session_prior_beta,
        )
        if (
            isinstance(self.observation_days, bool)
            or not isinstance(self.observation_days, int)
            or self.observation_days < 2
            or not all(map(math.isfinite, scalars))
            or self.session_gap_log_std < 0
            or not 0 <= self.circadian_phase_cut < 2 * math.pi
            or self.localization_factor <= 0
            or min(
                self.amount_variance_prior_alpha,
                self.amount_variance_prior_beta,
                self.short_session_prior_alpha,
                self.short_session_prior_beta,
            ) <= 0
        ):
            raise ValueError("v6 renderer or empirical-Bayes values are invalid")
        amount_log_floor = math.log(100 * self.localization_factor)
        if any(
            amount_log_floor + self.global_log_amount_center + relative_center < 0
            for stratum in self.strata
            for relative_center in stratum.primitive_quantiles[
                _continuous_primitives(stratum.name).index(
                    "relative_log_amount_center"
                )
            ]
        ):
            raise ValueError("v6 relative amount centers render below one minor unit")

    @property
    def activity_mean_transactions(self) -> float:
        return self.source_transaction_count / self.source_customer_count

    @property
    def sha256(self) -> str:
        return _canonical_hash(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 6,
            "source": {
                "dataset": self.source_dataset,
                "partition": self.source_partition,
                "customer_ids_sha256": self.source_customer_ids_sha256,
                "transactions_sha256": self.source_transactions_sha256,
                "customers": self.source_customer_count,
                "transactions": self.source_transaction_count,
            },
            "checkerboard": {
                "bins": CHECKERBOARD_BINS,
                "quantile_probabilities": list(self.quantile_probabilities),
                "strata": {item.name: item.as_dict() for item in self.strata},
                "smoothing": False,
                "pseudocounts": False,
                "stored_rank_rows": False,
            },
            "same_day_repeat_atom": {
                "condition": "invoice_count>=2",
                "repeat_customers": self.repeat_customer_count,
                "probability": self.same_day_repeat_probability,
                "allocation": "exact round(probability*sampled_repeat_count)",
                "selection": ATOM_SELECTION_RULE,
                "projection": "retain sampled end day and set first day equal to last day",
            },
            "renderer": {
                "hour_weights": list(self.hour_weights),
                "calendar_weights": list(self.calendar_weights),
                "observation_days": self.observation_days,
                "session_gap_log_mean": self.session_gap_log_mean,
                "session_gap_log_std": self.session_gap_log_std,
                "circadian_phase_cut": self.circadian_phase_cut,
                "global_log_amount_center": self.global_log_amount_center,
                "source_minor_units_per_major": 100,
                "localization_factor": self.localization_factor,
            },
            "empirical_bayes": {
                "amount_variance_prior_alpha": self.amount_variance_prior_alpha,
                "amount_variance_prior_beta": self.amount_variance_prior_beta,
                "short_session_prior_alpha": self.short_session_prior_alpha,
                "short_session_prior_beta": self.short_session_prior_beta,
            },
            "privacy": {
                "fixed_size_aggregate": True,
                "contains_customer_ids": False,
                "contains_raw_rows": False,
                "contains_rank_rows": False,
                "contains_evaluator_feature_rows": False,
                "contains_prediction_vectors": False,
                "differential_privacy_claimed": False,
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BehaviorProfileV6":
        _require_keys(
            value,
            {
                "schema_version",
                "source",
                "checkerboard",
                "same_day_repeat_atom",
                "renderer",
                "empirical_bayes",
                "privacy",
            },
            "v6 profile",
        )
        if value["schema_version"] != 6:
            raise ValueError("only schema-6 behavior profiles are accepted")
        source = value["source"]
        checkerboard = value["checkerboard"]
        atom = value["same_day_repeat_atom"]
        renderer = value["renderer"]
        eb = value["empirical_bayes"]
        _require_keys(source, {"dataset", "partition", "customer_ids_sha256", "transactions_sha256", "customers", "transactions"}, "v6 source")
        _require_keys(checkerboard, {"bins", "quantile_probabilities", "strata", "smoothing", "pseudocounts", "stored_rank_rows"}, "v6 checkerboard")
        _require_keys(atom, {"condition", "repeat_customers", "probability", "allocation", "selection", "projection"}, "v6 atom")
        _require_keys(renderer, {"hour_weights", "calendar_weights", "observation_days", "session_gap_log_mean", "session_gap_log_std", "circadian_phase_cut", "global_log_amount_center", "source_minor_units_per_major", "localization_factor"}, "v6 renderer")
        _require_keys(eb, {"amount_variance_prior_alpha", "amount_variance_prior_beta", "short_session_prior_alpha", "short_session_prior_beta"}, "v6 empirical-Bayes")
        if value["privacy"] != {
            "fixed_size_aggregate": True,
            "contains_customer_ids": False,
            "contains_raw_rows": False,
            "contains_rank_rows": False,
            "contains_evaluator_feature_rows": False,
            "contains_prediction_vectors": False,
            "differential_privacy_claimed": False,
        }:
            raise ValueError("v6 privacy declaration changed")
        if (
            checkerboard["bins"] != CHECKERBOARD_BINS
            or checkerboard["smoothing"] is not False
            or checkerboard["pseudocounts"] is not False
            or checkerboard["stored_rank_rows"] is not False
            or set(checkerboard["strata"]) != set(CHECKERBOARD_STRATA)
            or atom != {
                "condition": "invoice_count>=2",
                "repeat_customers": atom["repeat_customers"],
                "probability": atom["probability"],
                "allocation": "exact round(probability*sampled_repeat_count)",
                "selection": ATOM_SELECTION_RULE,
                "projection": "retain sampled end day and set first day equal to last day",
            }
            or renderer["source_minor_units_per_major"] != 100
        ):
            raise ValueError("v6 structural contract changed")
        return cls(
            source_dataset=str(source["dataset"]),
            source_partition=str(source["partition"]),
            source_customer_ids_sha256=str(source["customer_ids_sha256"]),
            source_transactions_sha256=str(source["transactions_sha256"]),
            source_customer_count=int(source["customers"]),
            source_transaction_count=int(source["transactions"]),
            repeat_customer_count=int(atom["repeat_customers"]),
            same_day_repeat_probability=float(atom["probability"]),
            quantile_probabilities=tuple(map(float, checkerboard["quantile_probabilities"])),
            strata=tuple(
                CheckerboardStratumV6.from_dict(name, checkerboard["strata"][name])
                for name in CHECKERBOARD_STRATA
            ),
            hour_weights=tuple(map(float, renderer["hour_weights"])),
            calendar_weights=tuple(map(float, renderer["calendar_weights"])),
            observation_days=int(renderer["observation_days"]),
            session_gap_log_mean=float(renderer["session_gap_log_mean"]),
            session_gap_log_std=float(renderer["session_gap_log_std"]),
            circadian_phase_cut=float(renderer["circadian_phase_cut"]),
            global_log_amount_center=float(renderer["global_log_amount_center"]),
            localization_factor=float(renderer["localization_factor"]),
            amount_variance_prior_alpha=float(eb["amount_variance_prior_alpha"]),
            amount_variance_prior_beta=float(eb["amount_variance_prior_beta"]),
            short_session_prior_alpha=float(eb["short_session_prior_alpha"]),
            short_session_prior_beta=float(eb["short_session_prior_beta"]),
        )

    def _sample_singletons(self, count: int, rng: Any) -> list[dict[str, float]]:
        import numpy as np

        if count == 0:
            return []
        stratum = self.strata[0]
        allocated = _residual_cell_counts(stratum.cell_masses, count, rng)
        indices = np.repeat(np.arange(len(allocated)), allocated)
        rng.shuffle(indices)
        cells = np.column_stack(np.unravel_index(indices, STRATUM_CELL_SHAPES["singleton"]))
        uniforms = (cells[:, :2] + rng.random((count, 2))) / CHECKERBOARD_BINS
        probabilities = np.asarray(self.quantile_probabilities, dtype=float)
        values = np.column_stack([
            np.interp(uniforms[:, index], probabilities, quantiles)
            for index, quantiles in enumerate(stratum.primitive_quantiles)
        ])
        output = []
        for row_values, hour in zip(values, cells[:, 2], strict=True):
            u_day, amount_center = map(float, row_values)
            if not 0 <= u_day < 1:
                raise RuntimeError("v6 singleton day coordinate left its support")
            day = math.floor(u_day * self.observation_days)
            fixed_hour = int(hour)
            theta = 2 * math.pi * fixed_hour / 24
            output.append({
                "invoice_count": 1.0,
                "active_day_position": u_day,
                "active_end_fraction": 0.0,
                "span_within_end_fraction": 0.0,
                "relative_log_amount_center": amount_center,
                "eb_log_amount_scale": 0.0,
                "eb_short_session_probability": 0.0,
                "circadian_phase_latent": (theta - self.circadian_phase_cut) % (2 * math.pi),
                "circadian_phase_sin": math.sin(theta),
                "circadian_phase_cos": math.cos(theta),
                "circadian_resultant": 1.0,
                "eb_circadian_concentration": 0.0,
                "fixed_hour": float(fixed_hour),
                "first_day": float(day),
                "last_day": float(day),
                "active_start_fraction": day / self.observation_days,
                "active_duration_fraction": 0.0,
                "same_day_atom": 0.0,
            })
        return output

    def _sample_repeats(self, count: int, rng: Any) -> list[dict[str, float]]:
        import numpy as np

        if count == 0:
            return []
        stratum = self.strata[1]
        allocated = _residual_cell_counts(stratum.cell_masses, count, rng)
        indices = np.repeat(np.arange(len(allocated)), allocated)
        rng.shuffle(indices)
        cells = np.column_stack(np.unravel_index(indices, STRATUM_CELL_SHAPES["multiday_repeat"]))
        uniforms = (cells + rng.random(cells.shape)) / CHECKERBOARD_BINS
        probabilities = np.asarray(self.quantile_probabilities, dtype=float)
        sampled_columns = []
        for index, quantiles in enumerate(stratum.primitive_quantiles):
            if index == 0:
                positions = np.searchsorted(probabilities, uniforms[:, index], side="left")
                positions = np.minimum(positions, len(probabilities) - 1)
                sampled_columns.append(np.asarray(quantiles, dtype=float)[positions])
            else:
                sampled_columns.append(np.interp(uniforms[:, index], probabilities, quantiles))
        sampled = np.column_stack(sampled_columns)
        output = []
        for values in sampled:
            row = dict(zip(REPEAT_PRIMITIVES, map(float, values), strict=True))
            row["invoice_count"] = float(max(2, math.floor(row["invoice_count"] + 0.5)))
            u_end = row["active_end_fraction"]
            u_span = row["span_within_end_fraction"]
            if (
                not 0 <= u_end < 1
                or not 0 <= u_span < 1
                or row["eb_log_amount_scale"] <= 0
                or not 0 <= row["eb_short_session_probability"] <= 1
                or not 0 <= row["circadian_resultant"] <= 1
            ):
                raise RuntimeError("v6 repeat primitive left its fitted support")
            last_day = 1 + math.floor(u_end * (self.observation_days - 1))
            span_days = 1 + math.floor(u_span * last_day)
            first_day = last_day - span_days
            if not 0 <= first_day < last_day <= self.observation_days - 1:
                raise RuntimeError("v6 decoded repeat days are invalid")
            phase = (row["circadian_phase_latent"] + self.circadian_phase_cut) % (2 * math.pi)
            row.update({
                "active_day_position": 0.0,
                "circadian_phase_sin": math.sin(phase),
                "circadian_phase_cos": math.cos(phase),
                "eb_circadian_concentration": 0.0,
                "fixed_hour": -1.0,
                "first_day": float(first_day),
                "last_day": float(last_day),
                "active_start_fraction": first_day / self.observation_days,
                "active_duration_fraction": (last_day - first_day) / self.observation_days,
                "same_day_atom": 0.0,
            })
            output.append(row)
        return output

    def sample_primitives(self, count: int, *, seed: int) -> list[dict[str, float]]:
        """Sample exact hurdle/cells and project only the declared repeat atom."""
        import numpy as np

        if count < 0:
            raise ValueError("sample count must be nonnegative")
        if count == 0:
            return []
        rng = np.random.default_rng(seed)
        singleton_count, repeat_count = _largest_remainder_counts(
            count, (self.strata[0].customer_count, self.repeat_customer_count)
        )
        singleton_rows = self._sample_singletons(singleton_count, rng)
        repeat_rows = self._sample_repeats(repeat_count, rng)
        atom_count = round(self.same_day_repeat_probability * repeat_count)
        atom_order = sorted(
            range(repeat_count),
            key=lambda index: (
                hashlib.sha256(f"{seed}:{ATOM_SELECTION_NAMESPACE}:{index}".encode()).digest(),
                index,
            ),
        )
        for index in atom_order[:atom_count]:
            row = repeat_rows[index]
            row["first_day"] = row["last_day"]
            row["active_start_fraction"] = row["last_day"] / self.observation_days
            row["active_duration_fraction"] = 0.0
            row["same_day_atom"] = 1.0
        output = singleton_rows + repeat_rows
        if len(output) != count or sum(row["invoice_count"] == 1 for row in output) != singleton_count or sum(row["same_day_atom"] == 1 for row in output) != atom_count:
            raise RuntimeError("v6 exact customer or atom allocation changed")
        permutation = rng.permutation(count)
        return [output[int(index)] for index in permutation]


def _hour_points(hours: Iterable[int]) -> tuple[tuple[float, float], ...]:
    return tuple(
        (math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24))
        for hour in hours
    )


def moment_matched_hour_counts(
    hour_weights: Iterable[float], *, phase: float, resultant: float, count: int
) -> tuple[tuple[int, ...], tuple[float, float]]:
    """Empirical-base exponential tilt followed by deterministic integer repair."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial import ConvexHull
    from scipy.special import logsumexp

    base = np.asarray(tuple(hour_weights), dtype=float)
    if base.shape != (24,) or count < 1 or not 0 <= resultant <= 1:
        raise ValueError("invalid hour-moment inputs")
    supported = np.flatnonzero(base > 0)
    if len(supported) == 0:
        raise ValueError("hour base has empty support")
    if len(supported) == 1:
        full = [0] * 24
        full[int(supported[0])] = count
        point = _hour_points((int(supported[0]),))[0]
        return tuple(full), point
    points = np.asarray(_hour_points(map(int, supported)), dtype=float)
    weights = base[supported] / base[supported].sum()
    target = resultant * np.asarray((math.sin(phase), math.cos(phase)))

    def segment_projection(start: Any, end: Any) -> tuple[Any, float]:
        edge = end - start
        fraction = float(np.dot(target - start, edge) / np.dot(edge, edge))
        fraction = min(1.0, max(0.0, fraction))
        return start + fraction * edge, fraction

    def face_distribution(left: int, right: int, fraction: float) -> Any:
        probabilities = np.zeros(len(supported), dtype=float)
        probabilities[left] = 1.0 - fraction
        probabilities[right] += fraction
        return probabilities

    limiting_probabilities = None
    if np.linalg.matrix_rank(points - points[0], tol=1e-12) < 2:
        distances = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
        left, right = np.unravel_index(int(np.argmax(distances)), distances.shape)
        projected, fraction = segment_projection(points[left], points[right])
        limiting_probabilities = face_distribution(left, right, fraction)
    else:
        hull = ConvexHull(points)
        hull_values = hull.equations[:, :2] @ target + hull.equations[:, 2]
        if (hull_values <= 1e-12).all():
            projected = target
            boundary_edges = [
                (
                    min(int(supported[left]), int(supported[right])),
                    max(int(supported[left]), int(supported[right])),
                    equation_index,
                    left,
                    right,
                )
                for equation_index, (left, right) in enumerate(hull.simplices)
                if hull_values[equation_index] >= -1e-12
            ]
            if boundary_edges:
                _, _, _, left, right = min(boundary_edges)
                projected, fraction = segment_projection(points[left], points[right])
                limiting_probabilities = face_distribution(left, right, fraction)
        else:
            candidates = []
            for left, right in hull.simplices:
                candidate, fraction = segment_projection(points[left], points[right])
                candidates.append(
                    (
                        float(np.sum((candidate - target) ** 2)),
                        min(int(supported[left]), int(supported[right])),
                        max(int(supported[left]), int(supported[right])),
                        candidate,
                        left,
                        right,
                        fraction,
                    )
                )
            selected = min(candidates, key=lambda item: item[:3])
            projected = selected[3]
            limiting_probabilities = face_distribution(
                selected[4], selected[5], selected[6]
            )
    if not np.isfinite(projected).all():
        raise RuntimeError("supported-hour convex-hull projection is non-finite")

    def objective(lam: Any) -> float:
        logits = np.log(weights) + points @ lam
        return float(logsumexp(logits) - lam @ projected)

    def gradient(lam: Any) -> Any:
        logits = np.log(weights) + points @ lam
        probabilities = np.exp(logits - logsumexp(logits))
        return probabilities @ points - projected

    if limiting_probabilities is None:
        solved = minimize(
            objective,
            np.zeros(2),
            jac=gradient,
            method="L-BFGS-B",
            bounds=[(-500.0, 500.0)] * 2,
            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 5_000, "maxls": 100},
        )
        logits = np.log(weights) + points @ solved.x
        probabilities = np.exp(logits - logsumexp(logits))
        optimizer_converged = solved.success
    else:
        probabilities = limiting_probabilities
        optimizer_converged = True
    residual = probabilities @ points - projected
    if (
        not optimizer_converged
        or not np.isfinite(probabilities).all()
        or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-12)
        or float(np.linalg.norm(residual)) > 1e-7
    ):
        raise RuntimeError("hour exponential tilt did not converge to its projected moment")
    expected = count * probabilities
    integer = np.floor(expected).astype(int)
    remaining = count - int(integer.sum())
    order = sorted(range(len(integer)), key=lambda i: (-(expected[i] - integer[i]), int(supported[i])))
    for index in order[:remaining]:
        integer[index] += 1

    def error(values: Any) -> float:
        moment = values @ points / count
        return float(np.sum((moment - projected) ** 2))

    def improves(candidate: float, current: float) -> bool:
        return candidate < current - 32 * math.ulp(max(1.0, current))

    def best_one_event_exchange() -> tuple[float, int, int, int, int] | None:
        current = error(integer)
        best: tuple[float, int, int, int, int] | None = None
        for source in range(len(integer)):
            if integer[source] == 0:
                continue
            for destination in range(len(integer)):
                if source == destination:
                    continue
                candidate = integer.copy()
                candidate[source] -= 1
                candidate[destination] += 1
                candidate_error = error(candidate)
                if improves(candidate_error, current):
                    choice = (
                        candidate_error,
                        int(supported[source]),
                        int(supported[destination]),
                        source,
                        destination,
                    )
                    if best is None or choice < best:
                        best = choice
        return best

    def settle_one_event_exchanges() -> None:
        visited: set[tuple[int, ...]] = set()
        while True:
            state = tuple(map(int, integer))
            if state in visited:
                raise RuntimeError("one-event hour repair entered a cycle")
            visited.add(state)
            best = best_one_event_exchange()
            if best is None:
                return
            integer[best[3]] -= 1
            integer[best[4]] += 1

    addition_pairs = [
        (left, right)
        for left in range(len(integer))
        for right in range(left, len(integer))
    ]
    addition_vectors = np.asarray(
        [points[left] + points[right] for left, right in addition_pairs]
    )

    def best_two_event_exchange() -> tuple[float, int, int, int, int, int, int, int, int] | None:
        current = error(integer)
        current_sum = integer @ points
        removals = [
            (left, right)
            for left in range(len(integer))
            for right in range(left, len(integer))
            if integer[left] > 0
            and integer[right] > 0
            and (left != right or integer[left] >= 2)
        ]
        best: tuple[float, int, int, int, int, int, int, int, int] | None = None
        for source_left, source_right in removals:
            removed = current_sum - points[source_left] - points[source_right]
            moments = (removed + addition_vectors) / count
            deltas = moments - projected
            errors = np.einsum("ij,ij->i", deltas, deltas)
            addition_index = int(np.argmin(errors))
            candidate_error = float(errors[addition_index])
            if improves(candidate_error, current):
                destination_left, destination_right = addition_pairs[addition_index]
                choice = (
                    candidate_error,
                    int(supported[source_left]),
                    int(supported[source_right]),
                    int(supported[destination_left]),
                    int(supported[destination_right]),
                    source_left,
                    source_right,
                    destination_left,
                    destination_right,
                )
                if best is None or choice < best:
                    best = choice
        return best

    settle_one_event_exchanges()
    visited: set[tuple[int, ...]] = set()
    while True:
        state = tuple(map(int, integer))
        if state in visited:
            raise RuntimeError("two-event hour repair entered a cycle")
        visited.add(state)
        best = best_two_event_exchange()
        if best is None:
            break
        integer[best[5]] -= 1
        integer[best[6]] -= 1
        integer[best[7]] += 1
        integer[best[8]] += 1
        settle_one_event_exchanges()
    full = np.zeros(24, dtype=int)
    full[supported] = integer
    if int(full.sum()) != count or (full < 0).any():
        raise RuntimeError("integer hour allocation changed its exact total")
    moment = full @ np.asarray(_hour_points(range(24)), dtype=float) / count
    return tuple(map(int, full)), (float(moment[0]), float(moment[1]))


class PopulationSimulatorV6(PopulationSimulatorV4):
    """Stateful event builder with support-safe V6 timestamps and hours."""

    behavior_profile: BehaviorProfileV6

    def _build_customers(self, count: int) -> dict[str, Any]:
        sampled = (
            self.behavior_profile.sample_primitives(count, seed=self.seed + 101)
            if self.behavior_profile
            else None
        )
        customers = super()._build_customers(count)
        if sampled is None:
            return customers
        for customer, row in zip(customers.values(), sampled, strict=True):
            if set(row) != SAMPLED_PRIMITIVE_FIELDS:
                raise RuntimeError("v6 sampled primitive fields changed")
            customer.avg_ticket_minor = (
                100
                * self.behavior_profile.localization_factor
                * math.exp(
                    self.behavior_profile.global_log_amount_center
                    + row["relative_log_amount_center"]
                )
            )
            customer.fixed_hour = int(row["fixed_hour"]) if row["invoice_count"] == 1 else None
            customer._v6_first_day = int(row["first_day"])
            customer._v6_last_day = int(row["last_day"])
            customer._v6_phase = math.atan2(row["circadian_phase_sin"], row["circadian_phase_cos"])
            customer._v6_resultant = row["circadian_resultant"]
            customer._v6_same_day_atom = bool(row["same_day_atom"])
        return customers

    def _calendar_day(self, first_day: int, last_day: int) -> int:
        days = list(range(first_day, last_day + 1))
        weights = [
            self.behavior_profile.calendar_weights[
                min(11, int(((day + 0.5) / self.behavior_profile.observation_days) * 12))
            ]
            for day in days
        ]
        return self.rng.choices(days, weights=weights if sum(weights) else None)[0]

    def _session_sizes(self, quota: int, *, multiday: bool) -> list[int]:
        sizes: list[int] = []
        remaining = quota
        while remaining:
            size = 1
            remaining -= 1
            while remaining and self.rng.random() < self._v6_customer.short_session_probability:
                size += 1
                remaining -= 1
            sizes.append(size)
        if multiday and len(sizes) == 1:
            sizes = [quota - 1, 1]
        return sizes

    @staticmethod
    def _merge_sessions(sizes: list[int], maximum: int) -> list[int]:
        while len(sizes) > maximum:
            index = min(
                range(len(sizes) - 1),
                key=lambda candidate: (sizes[candidate] + sizes[candidate + 1], candidate),
            )
            sizes[index : index + 2] = [sizes[index] + sizes[index + 1]]
        return sizes

    def _session_days(
        self, first_day: int, last_day: int, sessions: int
    ) -> list[int]:
        if sessions == 1:
            return [first_day]
        days = [first_day]
        for index in range(1, sessions - 1):
            remaining = sessions - 1 - index
            lower = days[-1] + 2
            upper = last_day - 2 * remaining
            days.append(self._calendar_day(lower, upper))
        days.append(last_day)
        if any(right - left < 2 for left, right in zip(days, days[1:])):
            raise RuntimeError("v6 inter-session calendar separation failed")
        return days

    def _seconds_within_hour(
        self,
        size: int,
        *,
        anchor: str | None = None,
        lower_second: int = 0,
        upper_second: int = 3_599,
    ) -> list[int]:
        capacity = upper_second - lower_second + 1
        if not 0 <= lower_second <= upper_second < 3_600 or size > capacity:
            raise ValueError("an hour window cannot contain the requested unique events")
        if size == 1:
            return [
                lower_second
                if anchor == "early"
                else upper_second
                if anchor == "late"
                else self.rng.randrange(lower_second, upper_second + 1)
            ]
        cumulative = [0.0]
        for _ in range(size - 1):
            cumulative.append(
                cumulative[-1]
                + min(
                    3_599.0,
                    max(
                        1.0,
                        self.rng.lognormvariate(
                            self.behavior_profile.session_gap_log_mean,
                            self.behavior_profile.session_gap_log_std,
                        ),
                    ),
                )
            )
        span = capacity - 1
        scale = min(1.0, span / cumulative[-1])
        offsets: list[int] = []
        for index, value in enumerate(cumulative):
            lower = offsets[-1] + 1 if offsets else 0
            upper = span - (size - 1 - index)
            offsets.append(min(upper, max(lower, round(value * scale))))
        available = span - offsets[-1]
        start = (
            0
            if anchor == "early"
            else available
            if anchor == "late"
            else self.rng.randrange(available + 1)
        )
        return [lower_second + start + offset for offset in offsets]

    def _customer_schedule(self, customer: Any, quota: int) -> list[tuple[Any, str]]:
        if not self.behavior_profile:
            return super()._customer_schedule(customer, quota)
        if quota < 1:
            raise ValueError("v6 customer quota must be positive")
        first_day = customer._v6_first_day
        last_day = customer._v6_last_day
        multiday = first_day < last_day
        if quota == 1 and multiday:
            raise RuntimeError("a singleton cannot have a multiday support window")
        if not 0 <= first_day <= last_day < self.behavior_profile.observation_days:
            raise RuntimeError("v6 customer support window is invalid")
        if not multiday and quota > 86_400:
            raise ValueError("same-day repeat quota exceeds strict one-second capacity")
        self._v6_customer = customer
        session_sizes = self._session_sizes(quota, multiday=multiday)
        if not multiday:
            session_sizes = [quota]
        else:
            maximum_sessions = max(2, (last_day - first_day) // 2 + 1)
            session_sizes = self._merge_sessions(
                session_sizes, maximum_sessions
            )
        if multiday and len(session_sizes) < 2:
            raise RuntimeError("v6 multiday repeat needs at least two sessions")
        if multiday and last_day - first_day == 1:
            session_sizes = self._merge_sessions(session_sizes, 2)
            days = [first_day, last_day]
        else:
            days = self._session_days(first_day, last_day, len(session_sizes))
        if customer.fixed_hour is not None:
            hour_counts = tuple(
                quota if hour == customer.fixed_hour else 0 for hour in range(24)
            )
            allocated_moment = (
                math.sin(2 * math.pi * customer.fixed_hour / 24),
                math.cos(2 * math.pi * customer.fixed_hour / 24),
            )
        else:
            hour_counts, allocated_moment = moment_matched_hour_counts(
                self.behavior_profile.hour_weights,
                phase=customer._v6_phase,
                resultant=customer._v6_resultant,
                count=quota,
            )
        hours = [hour for hour, count in enumerate(hour_counts) for _ in range(count)]
        self.rng.shuffle(hours)
        adjacent_two_session_window = multiday and last_day - first_day == 1
        if adjacent_two_session_window:
            # The one-day support edge needs an ordered partition so the two
            # distinct sessions remain more than 24 hours apart.
            hours.sort()
        session_assignments: list[tuple[int, list[int]]] = []
        cursor = 0
        for day, size in zip(days, session_sizes, strict=True):
            session_hours = sorted(hours[cursor : cursor + size])
            cursor += size
            session_assignments.append((day, session_hours))
        if cursor != quota:
            raise RuntimeError("v6 hour/session partition changed the quota")
        boundary_windows: dict[tuple[int, int], tuple[int, int]] = {}
        if adjacent_two_session_window:
            left_hours = session_assignments[0][1]
            right_hours = session_assignments[1][1]
            if left_hours[-1] == right_hours[0]:
                boundary_hour = left_hours[-1]
                left_count = left_hours.count(boundary_hour)
                right_count = right_hours.count(boundary_hour)
                if left_count + right_count > 3_600:
                    raise ValueError(
                        "adjacent-day boundary hour lacks strict one-second capacity"
                    )
                split = min(
                    3_600 - right_count,
                    max(
                        left_count,
                        round(3_600 * left_count / (left_count + right_count)),
                    ),
                )
                boundary_windows[(0, boundary_hour)] = (0, split - 1)
                boundary_windows[(1, boundary_hour)] = (split, 3_599)
        session_timestamps: list[list[Any]] = []
        for session_index, (day, session_hours) in enumerate(session_assignments):
            rendered: list[Any] = []
            for hour, size in sorted(Counter(session_hours).items()):
                anchor = None
                if adjacent_two_session_window:
                    anchor = "early" if session_index == 0 else "late"
                lower_second, upper_second = boundary_windows.get(
                    (session_index, hour), (0, 3_599)
                )
                rendered.extend(
                    self._profile_start
                    + timedelta(days=day, hours=hour, seconds=second)
                    for second in self._seconds_within_hour(
                        size,
                        anchor=anchor,
                        lower_second=lower_second,
                        upper_second=upper_second,
                    )
                )
            session_timestamps.append(sorted(rendered))
        if any(
            (right[0] - left[-1]).total_seconds() <= 86_400
            for left, right in zip(session_timestamps, session_timestamps[1:])
        ):
            raise RuntimeError("v6 distinct sessions must be separated by more than 24 hours")
        timestamps = sorted(
            timestamp for session in session_timestamps for timestamp in session
        )
        rendered_counts = tuple(Counter(timestamp.hour for timestamp in timestamps).get(hour, 0) for hour in range(24))
        points = _hour_points(range(24))
        rendered_moment = (
            sum(rendered_counts[h] * points[h][0] for h in range(24)) / quota,
            sum(rendered_counts[h] * points[h][1] for h in range(24)) / quota,
        )
        window_start = self._profile_start + timedelta(days=first_day)
        window_end = self._profile_start + timedelta(days=last_day + 1)
        if (
            len(timestamps) != quota
            or len(set(timestamps)) != quota
            or any(right <= left for left, right in zip(timestamps, timestamps[1:]))
            or timestamps[0].date() != window_start.date()
            or timestamps[-1].date() != (window_end - timedelta(days=1)).date()
            or any(not window_start <= timestamp < window_end for timestamp in timestamps)
            or rendered_counts != hour_counts
            or not math.isclose(rendered_moment[0], allocated_moment[0], abs_tol=1e-12)
            or not math.isclose(rendered_moment[1], allocated_moment[1], abs_tol=1e-12)
            or (multiday and len({timestamp.date() for timestamp in timestamps}) < 2)
        ):
            raise RuntimeError("v6 rendered schedule violated its frozen assertions")
        return [(timestamp, customer.customer_id) for timestamp in timestamps]

    def _prepare_activity_quotas(self, events: int) -> None:
        super()._prepare_activity_quotas(events)
        if self.behavior_profile and self._quota_prepared_for == events:
            if len(self._schedule) != events:
                raise RuntimeError("v6 canonical allocator changed the exact target")
            if any(
                (round(customer.activity_weight) == 1 and customer.remaining_quota != 1)
                or (round(customer.activity_weight) >= 2 and customer.remaining_quota < 2)
                for customer in self.customers.values()
            ):
                raise RuntimeError("v6 canonical allocator changed a structural stratum")


def generate_population_dataset_v6(
    output: Path,
    *,
    events: int = 50_000,
    seed: int = 20260827,
    fraud_rate: float = 0.0055,
    benign_adversarial_rate: float = 0.02,
    behavior_profile: BehaviorProfileV6 | None = None,
) -> dict[str, Any]:
    """Generate the canonical exact-size event file through the V6 scheduler."""
    from jsonschema import Draft202012Validator, FormatChecker

    from sim.attacks import all_attacks, apply_attack
    from sim.mandate_twin import DETERMINISTIC_SIGNING_AVAILABLE

    if events < 1 or not 0 <= fraud_rate <= 1 or not 0 <= benign_adversarial_rate <= 1:
        raise ValueError("invalid event count or rate")
    if fraud_rate + benign_adversarial_rate > 1:
        raise ValueError("fraud and benign-adversarial rates exceed one")
    if not DETERMINISTIC_SIGNING_AVAILABLE:
        raise RuntimeError("deterministic population output requires cryptography >= 43")
    fraud_count = round(events * fraud_rate)
    benign_count = round(events * benign_adversarial_rate)
    attacks = all_attacks(include_heldout=True)
    assigned = [attacks[index % len(attacks)] for index in range(fraud_count)]
    early_fraud: list[Any] = []
    late_fraud: list[Any] = []
    frozen_fraud: list[Any] = []
    for attack in attacks:
        occurrences = [item for item in assigned if item is attack]
        if attack.split in {"C", "D"}:
            frozen_fraud.extend(occurrences)
        else:
            cut = max(1, round(len(occurrences) * 0.75))
            early_fraud.extend(occurrences[:cut])
            late_fraud.extend(occurrences[cut:])
    legitimate_count = events - fraud_count - benign_count
    early_legitimate = round(legitimate_count * 0.75)
    early = [("fraud", attack) for attack in early_fraud] + [("legitimate", None)] * early_legitimate
    late = [("fraud", attack) for attack in late_fraud] + [("legitimate", None)] * (legitimate_count - early_legitimate)
    frozen = [("fraud", attack) for attack in frozen_fraud]
    simulator = PopulationSimulatorV6(
        seed=seed,
        customers=max(1, round(events / behavior_profile.activity_mean_transactions)) if behavior_profile else 800,
        behavior_profile=behavior_profile,
    )
    simulator._prepare_activity_quotas(events)
    for phase in (early, late, frozen):
        simulator.rng.shuffle(phase)
    schedule = early + late + frozen + [("E", None)] * benign_count
    schema = json.loads((ROOT / "contracts" / "event.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    digest = hashlib.sha256()
    split_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    generated_events: list[dict[str, Any]] = []
    previous_clock = simulator.clock
    for kind, attack in schedule:
        if kind == "fraud":
            event = apply_attack(
                simulator.generate_legitimate(force_channel=attack.base_channel),
                attack,
                registry=simulator.registry,
            )
        elif kind == "E":
            event = simulator.generate_legitimate(weird=True)
        else:
            event = simulator.generate_legitimate()
        if simulator.clock < previous_clock:
            raise RuntimeError("v6 simulator clock regressed during canonical generation")
        previous_clock = simulator.clock
        generated_events.append(event)
    generated_events.sort(
        key=lambda event: datetime.fromisoformat(
            str(event["ts"]).replace("Z", "+00:00")
        )
    )
    with temporary.open("w", encoding="utf-8") as handle:
        previous_timestamp: datetime | None = None
        for event in generated_events:
            timestamp = datetime.fromisoformat(
                str(event["ts"]).replace("Z", "+00:00")
            )
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise RuntimeError("v6 canonical output timestamps are not monotonic")
            previous_timestamp = timestamp
            validator.validate(event)
            line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode())
            split_counts[event["label"]["split"]] += 1
            if event["label"]["attack_family"]:
                family_counts[event["label"]["attack_family"]] += 1
    os.replace(temporary, output)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "events": events,
        "fraud_events": fraud_count,
        "fraud_rate": fraud_count / events,
        "benign_adversarial_events": benign_count,
        "split_counts": dict(sorted(split_counts.items())),
        "attack_family_counts": dict(sorted(family_counts.items())),
        "sha256": digest.hexdigest(),
        "synthetic_pii": True,
        "api_calls": 0,
        "behavior_profile_schema_version": 6 if behavior_profile else None,
        "behavior_profile_sha256": behavior_profile.sha256 if behavior_profile else None,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "events.jsonl")
    parser.add_argument("--events", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--fraud-rate", type=float, default=0.0055)
    parser.add_argument("--benign-adversarial-rate", type=float, default=0.02)
    parser.add_argument("--behavior-profile", type=Path)
    args = parser.parse_args()
    profile = None
    if args.behavior_profile:
        payload = json.loads(args.behavior_profile.read_text(encoding="utf-8"))
        profile = BehaviorProfileV6.from_dict(payload.get("behavior_profile", payload))
    print(json.dumps(generate_population_dataset_v6(args.output, events=args.events, seed=args.seed, fraud_rate=args.fraud_rate, benign_adversarial_rate=args.benign_adversarial_rate, behavior_profile=profile), sort_keys=True))


if __name__ == "__main__":
    main()
