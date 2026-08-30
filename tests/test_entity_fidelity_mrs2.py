from __future__ import annotations

import inspect
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import gamma

import sim.entity_fidelity_mrs2 as mrs2
import sim.population_mrs2 as population_mrs2
from sim.population_mrs2 import (
    MRS1_STOP_PROTOCOL_SHA256,
    CalendarHazardClock,
    frailty_score,
    generate_synthetic_transactions_mrs2,
)
from tests.test_population_mrs2 import profile_fixture_mrs2


def test_lifecycle_is_versioned_stop_bound_and_real_sources_stay_sealed() -> None:
    contract = mrs2.mrs2_lifecycle_contract()

    assert contract["schema_version"] == "mrs2"
    assert contract["study_revision"] == "mrs2.0"
    assert contract["predecessor"] == {
        "model": "mrs1",
        "terminal_state": "STOP",
        "protocol_sha256": MRS1_STOP_PROTOCOL_SHA256,
    }
    assert contract["implemented_stages"] == ["preregistration", "development"]
    assert contract["development_one_shot"] is True
    assert contract["qualification_partition_sealed"] is True
    assert contract["uci_sources_sealed"] is True
    for allowed in ("structural", "synthetic", "preregister", "development", "verify"):
        mrs2.require_mrs2_lifecycle_mode(allowed)
    for forbidden in ("qualification", "transfer", "uci352", "final"):
        with pytest.raises(RuntimeError, match="remain sealed"):
            mrs2.require_mrs2_lifecycle_mode(forbidden)


@pytest.mark.parametrize("shape", (0.251, 0.5, 1.0, 2.5, 10.0, 19.9))
def test_fixed_generalized_gauss_laguerre_preserves_gamma_mass_and_mean(
    shape: float,
) -> None:
    frailties, log_weights = mrs2._quadrature_log_weights(shape)
    weights = np.exp(log_weights)

    assert len(frailties) == mrs2.GAUSS_LAGUERRE_NODES == 32
    assert weights.sum() == pytest.approx(1, abs=2e-14)
    assert np.sum(weights * frailties) == pytest.approx(1, abs=2e-13)


@pytest.mark.parametrize("shape", (0.251, 0.5, 1.0, 2.5, 10.0, 19.9))
def test_bounded_gamma_cdf_score_has_exact_continuous_prior_moments(shape: float) -> None:
    probabilities = (np.arange(20_000) + 0.5) / 20_000
    frailties = gamma.ppf(probabilities, a=shape, scale=1 / shape)
    scores = frailty_score(frailties, shape)

    assert np.max(np.abs(scores)) < math.sqrt(3)
    assert np.mean(scores) == pytest.approx(0, abs=2e-12)
    assert np.var(scores) == pytest.approx(1, abs=2e-8)


def test_timing_likelihood_is_finite_zero_truncated_and_uses_full_clock_mixture() -> None:
    profile = profile_fixture_mrs2(preference_strength=0.75)
    frame = generate_synthetic_transactions_mrs2(profile, customers=20, seed=20260828)
    sequences = mrs2._calendar_sequences(frame)
    start = datetime.fromisoformat(profile.observation_start)
    end = datetime.fromisoformat(profile.observation_end)
    parameters = np.asarray(
        [
            math.log(profile.renewal_shape),
            math.log(profile.mean_operational_gap_days),
            math.log(profile.frailty_shape),
            profile.preferred_hour_strength,
        ]
    )

    value = mrs2.timing_negative_log_likelihood(
        parameters,
        sequences,
        profile=profile,
        observation_start=start,
        observation_end=end,
    )
    no_preference = mrs2.timing_negative_log_likelihood(
        np.asarray([*parameters[:3], 0.0]),
        sequences,
        profile=profile,
        observation_start=start,
        observation_end=end,
    )
    posteriors = mrs2._timing_posteriors(
        sequences,
        profile=profile,
        observation_start=start,
        observation_end=end,
        renewal_shape=profile.renewal_shape,
        mean_gap=profile.mean_operational_gap_days,
        frailty_shape=profile.frailty_shape,
        preference_strength=profile.preferred_hour_strength,
    )

    assert math.isfinite(value) and math.isfinite(no_preference)
    assert value != pytest.approx(no_preference)
    assert all(len(frailties) == 24 * 32 for frailties, _ in posteriors)
    assert all(np.exp(log_weights).sum() == pytest.approx(1) for _, log_weights in posteriors)
    clock = CalendarHazardClock(profile, start, end)
    frailties, log_weights = mrs2._quadrature_log_weights(profile.frailty_shape)
    observed = mrs2._log_observation_probability(
        clock=clock,
        renewal_shape=profile.renewal_shape,
        mean_gap=profile.mean_operational_gap_days,
        frailties=frailties,
        log_weights=log_weights,
    )
    assert math.isfinite(observed) and observed < 0


def test_cached_vectorized_timing_components_equal_the_class_clock_reference() -> None:
    from sim.entity_fidelity_mrs1 import _sequence_log_likelihoods

    profile = profile_fixture_mrs2(preference_strength=0.63)
    start = datetime.fromisoformat(profile.observation_start)
    end = datetime.fromisoformat(profile.observation_end)
    timestamps = (
        start.replace(hour=3, minute=11),
        start.replace(day=2, hour=12, minute=7),
        start.replace(day=3, hour=22, minute=43),
    )
    base_clock = CalendarHazardClock(profile, start, end)
    frailties, log_weights = mrs2._quadrature_log_weights(profile.frailty_shape)
    optimized = mrs2._timing_log_components_from_basis(
        mrs2._timing_bases([timestamps], base_clock)[0],
        clock=base_clock,
        renewal_shape=profile.renewal_shape,
        mean_gap=profile.mean_operational_gap_days,
        frailties=frailties,
        log_weights=log_weights,
        preference_strength=profile.preferred_hour_strength,
    )
    reference = []
    for preferred_hour, probability in enumerate(
        profile.preferred_hour_probabilities
    ):
        clock = CalendarHazardClock(
            profile, start, end, preferred_hour=preferred_hour
        )
        coordinates = np.asarray(
            [clock.to_operational(timestamp) for timestamp in timestamps]
        )
        likelihood = _sequence_log_likelihoods(
            coordinates,
            horizon=clock.total,
            renewal_shape=profile.renewal_shape,
            mean_gap=profile.mean_operational_gap_days,
            frailties=frailties,
        )
        reference.extend(
            math.log(probability)
            + log_weights
            + likelihood
            + sum(math.log(clock.jacobian(value)) for value in timestamps)
        )

    assert optimized == pytest.approx(reference, abs=2e-12)


def test_preferred_hour_strength_is_recovered_from_synthetic_timing() -> None:
    profile = profile_fixture_mrs2(preference_strength=0.75)
    frame = generate_synthetic_transactions_mrs2(profile, customers=50, seed=20260829)
    sequences = mrs2._calendar_sequences(frame)
    start = datetime.fromisoformat(profile.observation_start)
    end = datetime.fromisoformat(profile.observation_end)
    fixed = np.log(
        [
            profile.renewal_shape,
            profile.mean_operational_gap_days,
            profile.frailty_shape,
        ]
    )

    result = minimize_scalar(
        lambda strength: mrs2.timing_negative_log_likelihood(
            np.asarray([*fixed, strength]),
            sequences,
            profile=profile,
            observation_start=start,
            observation_end=end,
        ),
        bounds=(0.02, 0.94),
        method="bounded",
        options={"xatol": 0.01},
    )

    assert result.success
    assert result.x == pytest.approx(profile.preferred_hour_strength, abs=0.08)


def test_gaussian_ar1_variance_and_analytic_household_intercept_covariance() -> None:
    covariance = mrs2.gaussian_ar1_covariance(
        4, ar1=0.6, innovation_scale=0.8, household_scale=0.3
    )
    stationary = 0.8**2 / (1 - 0.6**2)

    assert np.diag(covariance) == pytest.approx([stationary + 0.3**2] * 4)
    assert covariance[0, 1] == pytest.approx(stationary * 0.6 + 0.3**2)
    assert covariance[0, 3] == pytest.approx(stationary * 0.6**3 + 0.3**2)
    assert np.linalg.eigvalsh(covariance).min() > 0


@pytest.mark.parametrize("length", (1, 2, 9))
def test_linear_ar1_rank1_density_matches_dense_gaussian(length: int) -> None:
    rng = np.random.default_rng(20 + length)
    residuals = rng.normal(size=(7, length))
    covariance = mrs2.gaussian_ar1_covariance(
        length, ar1=-0.35, innovation_scale=0.7, household_scale=0.4
    )
    sign, log_determinant = np.linalg.slogdet(covariance)
    dense = -0.5 * (
        length * math.log(2 * math.pi)
        + log_determinant
        + np.einsum(
            "ij,ij->i",
            residuals,
            np.linalg.solve(covariance, residuals.T).T,
        )
    )

    linear = mrs2._ar1_rank1_log_density(
        residuals,
        ar1=-0.35,
        innovation_scale=0.7,
        household_scale=0.4,
    )

    assert sign > 0
    assert linear == pytest.approx(dense, abs=2e-12)


def test_amount_fit_recovers_coefficients_without_posterior_mean_plugin() -> None:
    rng = np.random.default_rng(20260827)
    shape = 2.5
    true_intercept, true_beta = 6.2, 0.55
    fixed_ar1, fixed_innovation, fixed_household = 0.35, 0.16, 0.22
    design = []
    posteriors = []
    for quantile in np.linspace(0.03, 0.97, 180):
        frailty = gamma.ppf(quantile, a=shape, scale=1 / shape)
        score = frailty_score(frailty, shape)
        covariance = mrs2.gaussian_ar1_covariance(
            4,
            ar1=fixed_ar1,
            innovation_scale=fixed_innovation,
            household_scale=fixed_household,
        )
        y = true_intercept + true_beta * score + rng.multivariate_normal(
            np.zeros(4), covariance
        )
        design.append((y, np.zeros(4)))
        # A one-point timing posterior is a limiting exact posterior, not its mean.
        posteriors.append((np.asarray([frailty]), np.asarray([0.0])))

    fixed = np.asarray(
        [0.0, fixed_ar1, math.log(fixed_innovation), math.log(fixed_household)]
    )

    def objective(values: Any) -> float:
        return mrs2.amount_negative_log_likelihood(
            np.asarray([values[0], values[1], fixed[0], fixed[1], fixed[2], fixed[3]]),
            design,
            posteriors,
            frailty_shape=shape,
        )

    result = minimize(objective, np.asarray([6.0, 0.3]), method="L-BFGS-B")

    assert result.success
    assert result.x[0] == pytest.approx(true_intercept, abs=0.06)
    assert result.x[1] == pytest.approx(true_beta, abs=0.06)


def test_amount_likelihood_marginalizes_all_timing_components() -> None:
    design = [(np.asarray([6.0, 6.1]), np.zeros(2))]
    frailties = np.asarray([0.4, 2.2])
    mixed = [(frailties, np.log([0.5, 0.5]))]
    first_only = [(frailties[:1], np.asarray([0.0]))]
    parameters = np.asarray([6.0, 1.0, 0.0, 0.2, -1.0, -1.0])

    mixed_value = mrs2.amount_negative_log_likelihood(
        parameters, design, mixed, frailty_shape=2.5
    )
    plugin_like_value = mrs2.amount_negative_log_likelihood(
        parameters, design, first_only, frailty_shape=2.5
    )

    assert math.isfinite(mixed_value)
    assert mixed_value != pytest.approx(plugin_like_value)


def _isolate_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    preregistration_path = tmp_path / "entity_fidelity_mrs2_preregistration.json"
    attempt_path = tmp_path / "entity_fidelity_mrs2_development_attempt.json"
    development_path = tmp_path / "entity_fidelity_mrs2_development.json"
    mrs1_r2_path = tmp_path / "entity_fidelity_mrs1_r2_preregistration.json"
    manifest_path = tmp_path / "completejourney_reference.manifest.json"
    mrs1_stop_path = tmp_path / "entity_fidelity_mrs1_protocol.json"
    qualification_path = tmp_path / "completejourney_qualification.parquet"
    monkeypatch.setattr(mrs2, "DEFAULT_PREREGISTRATION", preregistration_path)
    monkeypatch.setattr(mrs2, "DEFAULT_DEVELOPMENT_ATTEMPT", attempt_path)
    monkeypatch.setattr(mrs2, "DEFAULT_DEVELOPMENT", development_path)
    monkeypatch.setattr(mrs2, "MRS1_R2_PREREGISTRATION", mrs1_r2_path)
    monkeypatch.setattr(mrs2, "DEFAULT_COMPLETEJOURNEY_MANIFEST", manifest_path)
    monkeypatch.setattr(mrs2, "MRS1_STOP_PROTOCOL", mrs1_stop_path)
    monkeypatch.setattr(mrs2, "DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR", tmp_path)

    mrs1_r2 = {
        "artifact": "entity_fidelity_mrs1_r2_preregistration",
        "study_revision": "mrs1.2",
    }
    mrs2._write_artifact(mrs1_r2_path, mrs1_r2)
    r2_sha256 = mrs2._file_sha256(mrs1_r2_path)
    manifest = {
        "dataset": "completejourney",
        "version": "1.1.1",
        "mrs_preregistration_sha256": r2_sha256,
        "source": {"rds_sha256": "f" * 64},
        "preprocessing": {
            "timestamp_semantics": mrs2.COMPLETEJOURNEY_TIMESTAMP_SEMANTICS
        },
        "observation_window": {
            "start_inclusive": mrs2.COMPLETEJOURNEY_OBSERVATION_START,
            "end_exclusive": mrs2.COMPLETEJOURNEY_OBSERVATION_END,
            "timezone": None,
        },
        "partitions": {
            "development": {
                "file": "completejourney_development.parquet",
                "households": mrs2.DEVELOPMENT_HOUSEHOLDS,
                "baskets": mrs2.DEVELOPMENT_HOUSEHOLDS,
                "household_ids_sha256": "d" * 64,
                "parquet_sha256": "e" * 64,
                "mrs_preregistration_sha256": r2_sha256,
            },
            "qualification": {
                "file": "completejourney_qualification.parquet",
                "households": mrs2.QUALIFICATION_HOUSEHOLDS,
                "mrs_preregistration_sha256": r2_sha256,
            },
        },
    }
    mrs2._write_artifact(manifest_path, manifest)
    manifest_sha256 = mrs2._file_sha256(manifest_path)
    stop = {
        "artifact": "entity_fidelity_mrs1_frozen_protocol",
        "mrs_preregistration_sha256": r2_sha256,
        "expected_completejourney_manifest_sha256": manifest_sha256,
        "development_reference": {
            "partition": "development",
            "households": mrs2.DEVELOPMENT_HOUSEHOLDS,
            "manifest_sha256": manifest_sha256,
            "partition_sha256": "e" * 64,
            "household_ids_sha256": "d" * 64,
            "mrs_preregistration_sha256": r2_sha256,
        },
        "development_diagnostic_status": "STOP_FAIL",
        "development_fit_diagnostic": {"gate_passed": False},
        "qualification_accessed": False,
        "uci352_accessed": False,
        "uci502_accessed": False,
    }
    stop["protocol_sha256"] = mrs2._canonical_hash(stop)
    mrs2._write_artifact(mrs1_stop_path, stop)
    stop_sha256 = mrs2._file_sha256(mrs1_stop_path)
    monkeypatch.setattr(mrs2, "MRS1_STOP_PROTOCOL_SHA256", stop_sha256)
    monkeypatch.setattr(
        population_mrs2, "MRS1_STOP_PROTOCOL_SHA256", stop_sha256
    )
    qualification_path.write_bytes(b"sealed qualification sentinel")
    return {
        "r2_sha256": r2_sha256,
        "manifest_sha256": manifest_sha256,
        "stop_sha256": stop_sha256,
        "qualification_path": str(qualification_path),
    }


def _development_fixture() -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.DataFrame(
        {
            "household_id": [
                f"household-{index:04d}" for index in range(mrs2.DEVELOPMENT_HOUSEHOLDS)
            ],
            "basket_id": [
                f"basket-{index:04d}" for index in range(mrs2.DEVELOPMENT_HOUSEHOLDS)
            ],
            "transaction_timestamp": pd.to_datetime(
                ["2017-06-01 12:00:00"] * mrs2.DEVELOPMENT_HOUSEHOLDS
            ),
            "sales_value_cents": np.arange(
                500, 500 + mrs2.DEVELOPMENT_HOUSEHOLDS, dtype="int64"
            ),
        }
    )
    reference = {
        "dataset": "completejourney",
        "version": "1.1.1",
        "partition": "development",
        "manifest_sha256": "placeholder",
        "partition_sha256": "e" * 64,
        "household_ids_sha256": "d" * 64,
        "households": mrs2.DEVELOPMENT_HOUSEHOLDS,
        "baskets": len(raw),
        "source_rds_sha256": "f" * 64,
        "mrs_preregistration_sha256": "placeholder",
        "timestamp_semantics": mrs2.COMPLETEJOURNEY_TIMESTAMP_SEMANTICS,
        "observation_start": mrs2.COMPLETEJOURNEY_OBSERVATION_START,
        "observation_end_exclusive": mrs2.COMPLETEJOURNEY_OBSERVATION_END,
        "amount_unit": "USD cents",
    }
    return raw, reference


def _gate_fixture(profile_sha256: str, *, passed: bool = True) -> dict[str, object]:
    auc = 0.55 if passed else 0.75
    upper = 0.60 if passed else 0.80
    counts = (198, 198, 198, 197, 197)
    seeds = {}
    for seed in mrs2.EVALUATION_SEEDS:
        folds = [
            {
                "fold": fold,
                "training_real_customers": mrs2.DEVELOPMENT_HOUSEHOLDS - count,
                "evaluation_real_customers": count,
                "evaluation_synthetic_customers": count,
                "training_customer_ids_sha256": f"{seed + fold:064x}"[-64:],
                "evaluation_customer_ids_sha256": f"{seed + 100 + fold:064x}"[-64:],
                "profile_sha256": profile_sha256,
                "auc": float(auc),
                "d": float(auc),
            }
            for fold, count in enumerate(counts)
        ]
        seeds[str(seed)] = {"fold_weighted_d": float(auc), "folds": folds}
    return {
        "seeds": seeds,
        "metrics": {
            "orientation_free_max_seed_d": float(auc),
            "bootstrap_max_seed_d_mean": float(auc),
            "bootstrap_max_seed_d_ci_95": [float(auc - 0.01), float(upper)],
            "threshold_max_d_upper_95": mrs2.GATE_MAX_D_UPPER_95,
        },
        "gate_passed": passed,
    }


def _mock_successful_development(
    monkeypatch: pytest.MonkeyPatch,
    *,
    passed: bool = True,
) -> tuple[object, dict[str, int]]:
    raw, reference = _development_fixture()
    preregistration = mrs2.load_mrs2_preregistration()
    reference["manifest_sha256"] = preregistration["completejourney"][
        "manifest_sha256"
    ]
    reference["mrs_preregistration_sha256"] = preregistration[
        "completejourney"
    ]["mrs1_r2_preregistration_sha256"]
    frame, adapted = mrs2._adapt_development(raw, reference, preregistration)
    calendar = mrs2.fit_calendar_hazard(
        frame,
        observation_start=mrs2.COMPLETEJOURNEY_OBSERVATION_START,
        observation_end=mrs2.COMPLETEJOURNEY_OBSERVATION_END,
    )
    profile = mrs2._profile_shell(
        frame=frame,
        source_dataset="completejourney-1.1.1",
        source_partition="development",
        customer_ids_sha256=adapted["profile_customer_ids_sha256"],
        transactions_sha256=adapted["profile_transactions_sha256"],
        start=datetime.fromisoformat(mrs2.COMPLETEJOURNEY_OBSERVATION_START),
        end=datetime.fromisoformat(mrs2.COMPLETEJOURNEY_OBSERVATION_END),
        calendar=calendar,
    )
    calls = {"loader": 0, "gate": 0}

    def loader(value: object) -> tuple[pd.DataFrame, dict[str, object]]:
        calls["loader"] += 1
        assert mrs2.DEFAULT_DEVELOPMENT_ATTEMPT.is_file()
        if calls["loader"] == 1:
            assert not mrs2.DEFAULT_DEVELOPMENT.exists()
        return raw, reference

    def gate(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls["gate"] += 1
        return _gate_fixture(profile.sha256, passed=passed)

    monkeypatch.setattr(mrs2, "_load_canonical_development", loader)
    monkeypatch.setattr(mrs2, "fit_mrs2_profile", lambda *_args, **_kwargs: profile)
    monkeypatch.setattr(
        mrs2,
        "validate_generator",
        lambda *_args, **_kwargs: {
            "seed": mrs2.SEED,
            "customers": mrs2.DEVELOPMENT_HOUSEHOLDS,
            "transactions": mrs2.DEVELOPMENT_HOUSEHOLDS,
            "synthetic_transactions_sha256": "a" * 64,
            "exact_source_sequence_matches": 0,
            "deterministic": True,
            "positive_integer_cents": True,
        },
    )
    monkeypatch.setattr(mrs2, "evaluate_mrs2_gate", gate)
    return profile, calls


def test_preregistration_binds_stop_source_code_dependencies_and_opens_no_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _isolate_lifecycle(tmp_path, monkeypatch)
    value = mrs2.write_mrs2_preregistration()

    assert value["lineage"]["file_sha256"] == bindings["stop_sha256"]
    assert value["lineage"]["development_gate_passed"] is False
    assert value["lineage"]["qualification_accessed"] is False
    assert value["completejourney"]["manifest_sha256"] == bindings["manifest_sha256"]
    assert value["completejourney"]["development_parquet_sha256"] == "e" * 64
    assert value["completejourney"][
        "development_parquet_opened_before_preregistration"
    ] is False
    assert value["completejourney"]["development_households"] == 988
    assert value["completejourney"]["qualification_households"] == 1_481
    assert len(value["implementation_sha256"]) == 4
    assert "sim/entity_fidelity_mrs1.py" in value["runtime_dependency_sha256"]
    assert "sim/mandate_twin.py" in value["runtime_dependency_sha256"]
    assert "cryptography" in value["dependency_versions"]
    assert "openpyxl" in value["dependency_versions"]
    assert len(value["evaluator"]["features"]) == 8
    assert value["evaluator"]["total_folds"] == 15
    assert value["optimizer"]["gauss_laguerre_nodes"] == 32
    assert oct(mrs2.DEFAULT_PREREGISTRATION.stat().st_mode & 0o777) == "0o444"
    assert mrs2.verify_mrs2_lifecycle("preregistration") == value
    assert Path(bindings["qualification_path"]).read_bytes() == b"sealed qualification sentinel"
    with pytest.raises(FileExistsError):
        mrs2.write_mrs2_preregistration()


def test_development_consumes_attempt_before_loader_runs_gate_once_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _isolate_lifecycle(tmp_path, monkeypatch)
    mrs2.write_mrs2_preregistration()
    profile, calls = _mock_successful_development(monkeypatch)

    value = mrs2.run_mrs2_development()

    assert calls == {"loader": 1, "gate": 1}
    assert value["status"] == "PASS"
    assert value["development_profile_sha256"] == profile.sha256
    assert value["development_partition_accessed"] is True
    assert value["qualification_accessed"] is False
    assert value["uci_sources_accessed"] is False
    assert value["development_attempt_sha256"] == mrs2._file_sha256(
        mrs2.DEFAULT_DEVELOPMENT_ATTEMPT
    )
    assert oct(mrs2.DEFAULT_DEVELOPMENT.stat().st_mode & 0o777) == "0o444"
    assert mrs2.verify_mrs2_lifecycle("development") == value
    assert Path(bindings["qualification_path"]).read_bytes() == b"sealed qualification sentinel"


def test_development_persists_terminal_stop_fail_when_the_single_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _isolate_lifecycle(tmp_path, monkeypatch)
    mrs2.write_mrs2_preregistration()
    _, calls = _mock_successful_development(monkeypatch, passed=False)

    value = mrs2.run_mrs2_development()

    assert calls == {"loader": 1, "gate": 1}
    assert value["status"] == "STOP_FAIL"
    assert value["gate"]["gate_passed"] is False
    assert value["retry_permitted"] is False
    assert mrs2.verify_mrs2_lifecycle("development") == value
    assert Path(bindings["qualification_path"]).read_bytes() == b"sealed qualification sentinel"


def test_base_exception_burns_attempt_and_cannot_retry_or_touch_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _isolate_lifecycle(tmp_path, monkeypatch)
    mrs2.write_mrs2_preregistration()
    calls = 0

    def failed_loader(_value: object) -> object:
        nonlocal calls
        calls += 1
        assert mrs2.DEFAULT_DEVELOPMENT_ATTEMPT.is_file()
        raise KeyboardInterrupt("synthetic terminal loader failure")

    monkeypatch.setattr(mrs2, "_load_canonical_development", failed_loader)
    with pytest.raises(KeyboardInterrupt, match="terminal"):
        mrs2.run_mrs2_development()

    assert calls == 1
    assert mrs2.DEFAULT_DEVELOPMENT_ATTEMPT.is_file()
    assert not mrs2.DEFAULT_DEVELOPMENT.exists()
    with pytest.raises(FileExistsError):
        mrs2.run_mrs2_development()
    assert calls == 1
    assert Path(bindings["qualification_path"]).read_bytes() == b"sealed qualification sentinel"


def test_development_verifier_rejects_gate_profile_hash_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_lifecycle(tmp_path, monkeypatch)
    mrs2.write_mrs2_preregistration()
    _mock_successful_development(monkeypatch)
    value = mrs2.run_mrs2_development()
    changed = json.loads(json.dumps(value))
    changed["gate"]["seeds"][str(mrs2.EVALUATION_SEEDS[0])]["folds"][0][
        "profile_sha256"
    ] = "0" * 64
    changed["artifact_sha256"] = mrs2._artifact_hash(changed)
    changed_path = tmp_path / "changed-development.json"
    mrs2._write_artifact(changed_path, changed)
    monkeypatch.setattr(mrs2, "DEFAULT_DEVELOPMENT", changed_path)

    with pytest.raises(ValueError, match="fold profile binding"):
        mrs2.verify_mrs2_lifecycle("development")


def test_development_verifier_rejects_loaded_source_provenance_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_lifecycle(tmp_path, monkeypatch)
    mrs2.write_mrs2_preregistration()
    _mock_successful_development(monkeypatch)
    value = mrs2.run_mrs2_development()
    changed = json.loads(json.dumps(value))
    changed["development_reference"]["source_rds_sha256"] = "0" * 64
    changed["artifact_sha256"] = mrs2._artifact_hash(changed)
    changed_path = tmp_path / "changed-source-development.json"
    mrs2._write_artifact(changed_path, changed)
    monkeypatch.setattr(mrs2, "DEFAULT_DEVELOPMENT", changed_path)

    with pytest.raises(ValueError, match="reference binding"):
        mrs2.verify_mrs2_lifecycle("development")


def test_development_verifier_recomputes_and_rejects_forged_gate_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_lifecycle(tmp_path, monkeypatch)
    mrs2.write_mrs2_preregistration()
    _, calls = _mock_successful_development(monkeypatch, passed=False)
    value = mrs2.run_mrs2_development()
    forged = json.loads(json.dumps(value))
    forged["gate"]["metrics"]["bootstrap_max_seed_d_mean"] = 0.58
    forged["gate"]["metrics"]["bootstrap_max_seed_d_ci_95"] = [0.55, 0.60]
    forged["gate"]["gate_passed"] = True
    forged["status"] = "PASS"
    forged["artifact_sha256"] = mrs2._artifact_hash(forged)
    forged_path = tmp_path / "forged-pass-development.json"
    mrs2._write_artifact(forged_path, forged)
    monkeypatch.setattr(mrs2, "DEFAULT_DEVELOPMENT", forged_path)

    with pytest.raises(ValueError, match="gate recomputation"):
        mrs2.verify_mrs2_lifecycle("development")
    assert calls == {"loader": 2, "gate": 2}


def test_gate_evaluator_is_deterministic_and_emits_exactly_fifteen_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = profile_fixture_mrs2()
    counts = (198, 198, 198, 197, 197)

    def crossfit(
        _frame: object,
        _reference: object,
        *,
        seed: int,
        frozen_profile: object,
    ) -> list[dict[str, object]]:
        assert frozen_profile is profile
        return [
            {
                "fold": fold,
                "training_real_customers": mrs2.DEVELOPMENT_HOUSEHOLDS - count,
                "evaluation_real_customers": count,
                "evaluation_synthetic_customers": count,
                "training_customer_ids_sha256": f"{seed + fold:064x}"[-64:],
                "evaluation_customer_ids_sha256": f"{seed + fold + 100:064x}"[-64:],
                "profile_sha256": profile.sha256,
                "auc": 0.55,
                "d": 0.55,
                "_real_customer_ids": tuple(range(count)),
                "_real_predictions": np.full(count, 0.45),
                "_synthetic_predictions": np.full(count, 0.55),
            }
            for fold, count in enumerate(counts)
        ]

    def bootstrap(
        seed_folds: object, *, repeats: int, seed: int
    ) -> tuple[np.ndarray, float, float]:
        assert len(seed_folds) == 3
        assert repeats == 10_000
        assert seed == mrs2.FULL_BOOTSTRAP_SEED
        return np.asarray([0.55, 0.55]), 0.54, 0.60

    monkeypatch.setattr(mrs2, "_crossfit_seed", crossfit)
    monkeypatch.setattr(mrs2, "_crossfit_max_seed_bootstrap", bootstrap)

    first = mrs2.evaluate_mrs2_gate(None, {}, frozen_profile=profile)
    second = mrs2.evaluate_mrs2_gate(None, {}, frozen_profile=profile)

    assert first == second
    assert sum(len(value["folds"]) for value in first["seeds"].values()) == 15
    assert first["gate_passed"] is True
    assert mrs2._gate_evidence_valid(first)


def test_lifecycle_entrypoints_have_no_path_or_loader_bypass() -> None:
    assert tuple(inspect.signature(mrs2.write_mrs2_preregistration).parameters) == ()
    assert tuple(inspect.signature(mrs2.run_mrs2_development).parameters) == ()
    assert tuple(inspect.signature(mrs2.verify_mrs2_lifecycle).parameters) == (
        "stage",
    )
