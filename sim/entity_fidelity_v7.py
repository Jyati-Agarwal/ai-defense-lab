"""Fail-closed schema-7 entity fidelity after the V6.1 calibration diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.download_retail_ii_reference import (
    ARCHIVE_SHA256 as UCI502_ARCHIVE_SHA256,
    DATASET_ID as UCI502_DATASET_ID,
    LICENSE as UCI502_LICENSE,
    SOURCE_URL as UCI502_SOURCE_URL,
    WORKBOOK_SHA256 as UCI502_WORKBOOK_SHA256,
)
from scripts.download_retail_reference import (
    DATASET_ID as UCI352_DATASET_ID,
    LICENSE as UCI352_LICENSE,
    SOURCE_URL as UCI352_SOURCE_URL,
    WORKBOOK_SHA256 as UCI352_WORKBOOK_SHA256,
    sha256_file,
)
from sim.entity_fidelity import (
    EXPECTED_PARTITION_HASHES,
    FEATURES,
    DomainCalibration,
    _classifier,
    _classifier_protocol,
    _dependencies,
    _empirical_bayes_log_scales,
    _empirical_bayes_short_probabilities,
    _ids_hash,
    _positive_gaps_seconds,
    _transactions_hash,
    _write_exclusive_json,
    customer_features,
    fit_calibrations,
    load_retail_partition,
)
from sim.entity_fidelity_v4 import (
    AGGREGATE_CONTAINER_ALLOWLIST,
    BOOTSTRAP_REPEATS,
    CHECKERBOARD_CELL_ALLOCATION,
    CHECKERBOARD_TIE_RULE,
    CROSS_FIT_FOLDS,
    EVALUATION_SEEDS,
    FORBIDDEN_ARTIFACT_KEYS,
    GATE_MAX_D_UPPER_95,
    SEED,
    UCI502_MIN_SPLIT_CUSTOMERS,
    UCI502_SHEET,
    UCI502_SPLIT_ASSIGNMENT,
    UCI502_WINDOW_END,
    UCI502_WINDOW_START,
    _crossfit_max_seed_bootstrap,
    _max_seed_bootstrap,
    fractional_checkerboard_masses,
    fractional_tie_weights,
    uci502_readiness_split as uci502_final_split,
)
from sim.population_v7 import (
    ATOM_SELECTION_RULE,
    CHECKERBOARD_BINS,
    CHECKERBOARD_STRATA,
    LOCALIZATION_FACTOR_DEFAULT,
    MIN_CHECKERBOARD_CUSTOMERS,
    QUANTILE_PROBABILITIES,
    REPEAT_PRIMITIVES,
    SINGLETON_CONTINUOUS_PRIMITIVES,
    STRATUM_CELL_COUNTS,
    STRATUM_CELL_SHAPES,
    STRATUM_PRIMITIVES,
    REPEAT_INVOICE_COUNT_MAX,
    BehaviorProfileV7,
    CheckerboardStratumV6,
    PopulationSimulatorV7,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "data" / "reference" / "online_retail.xlsx"
DEFAULT_FINAL_REFERENCE = ROOT / "data" / "reference" / "online_retail_ii.xlsx"
DEFAULT_PREREGISTRATION = (
    ROOT / "results" / "entity_fidelity_v7_preregistration.json"
)
DEFAULT_PROTOCOL = ROOT / "results" / "entity_fidelity_v7_protocol.json"
DEFAULT_EXPLORATORY = ROOT / "results" / "entity_fidelity_v7_exploratory.json"
DEFAULT_DEVELOPMENT_ATTEMPT = (
    ROOT / "results" / "entity_fidelity_v7_development_attempt.json"
)
DEFAULT_SEAL_RECEIPT = ROOT / "results" / "entity_fidelity_v7_final_receipt.json"
DEFAULT_OUTPUT = ROOT / "results" / "entity_fidelity_v7_final.json"
DEFAULT_COMPLETION_SEAL = (
    ROOT / "results" / "entity_fidelity_v7_completion_seal.json"
)
UCI352_MANIFEST_SHA256 = (
    "0fca9904171c1382bfdd8dc1369929a310817a63472474dc61dee4a4ae800cb9"
)
UCI352_CALIBRATION_TRANSACTIONS_SHA256 = (
    "3614fdd252819b132b5900ac085372e89a148f7f51784bc3d14149d09b22abc1"
)
UCI352_CALIBRATION_CUSTOMERS = 2_440
UCI352_CALIBRATION_TRANSACTIONS = 10_621
UCI502_MANIFEST_SHA256 = (
    "8092c0b746d0c03d319570bc483e1f3085f057fc9e42f0de8c2b258524573f38"
)
STUDY_REVISION = "7.0"
PARENT_V6_FAILURE_SHA256 = (
    "e0ae36824751d50394a2406e155f8fd51ae456a61e63a8bcc32d53e26e0e5fc5"
)
PARENT_V6_ATTEMPT_SHA256 = (
    "ac319a34dcdfdd9efab64b6687066cbf73ec890e7387d14c62953f5142b43ff4"
)
PARENT_V6_ENTITY_SOURCE_SHA256 = (
    "c6dd095a6e5290855a15d82de2d46addbf7c2e580f119dbc34b54ea4d9efdcf6"
)
PARENT_V6_POPULATION_SOURCE_SHA256 = (
    "aa5c90f9132b3c4c3a98aa288a4b0378bd89f6889e5c9e5edb08df37ae08397e"
)
PARENT_V6_ENTITY_TEST_SHA256 = (
    "f7a514ba427ef7e49664b053dfe7532c36c7b133fc8f6223aeb128ded57e8c6f"
)
PARENT_V6_POPULATION_TEST_SHA256 = (
    "92c0fc06c4a2a66f40edfabae827f4399b67e3be107ace782f9144575ab4d3b8"
)
PARENT_V61_PREREGISTRATION_SHA256 = (
    "72764883c640fe1f69f430ace1c98fd33aa5d6de36ca0be299bcee2c06119fba"
)
PARENT_V61_PROTOCOL_SHA256 = (
    "2d9208b37926e324d1d987414ce259bca40e22bd8bd81422c9e181a8b131fa28"
)
PARENT_V61_ATTEMPT_SHA256 = (
    "855470de6484490e6071e376f47261f1e513488d445fd1f5abfc8e04762b354f"
)
PARENT_V61_EXPLORATORY_SHA256 = (
    "fbc6f639d6ebab488a48e9396e4bfeb2dd6992d296233f42173ee1da14b498fe"
)
V7_CORRECTION_RULE = (
    "post-V6.1 calibration-diagnostic correction: exact repeat invoice-count "
    "PMF on support 0..256, realized synthetic total, and deterministic balanced "
    "multiday session anchors; all other V6.1 model choices remain unchanged"
)
FUTILITY_SEED = 20260828
FUTILITY_FOLD = 2
FUTILITY_BOOTSTRAP_REPEATS = 10_000
FUTILITY_BOOTSTRAP_SEED = 20260897
FUTILITY_QUANTILE = 0.025
NOVELTY_DRAWS = 1_000_000
DEPENDENCY_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "openpyxl",
    "cryptography",
)
TIME_COORDINATES = {
    "singleton_encode": "u_day=(day+0.5)/O",
    "singleton_decode": "day=floor(u_day*O)",
    "repeat_end_encode": "u_end=(last_day-0.5)/(O-1)",
    "repeat_end_decode": "last_day=1+floor(u_end*(O-1))",
    "repeat_span_encode": "u_span=(span_days-0.5)/last_day",
    "repeat_span_decode": "span_days=1+floor(u_span*last_day); first_day=last_day-span_days",
    "window": "integer UTC days with midnight bounds [first_day,last_day+1); no clipping",
}
HOUR_ALLOCATION = {
    "singleton": "joint 3x3x24 category; rendered hour equals sampled category",
    "repeat_probability": "for convex-hull interior targets p_h is proportional to empirical w_h*exp(lambda_s*sin(theta_h)+lambda_c*cos(theta_h)); projected boundary targets use the deterministic exact edge mixture or vertex delta of the limiting exponential family",
    "repeat_integerization": "largest remainder, deterministic improving one-event exchanges, then deterministic improving two-event exchanges; lexicographic ties",
    "repeat_repair_termination": "squared-error descent exceeding a 32-ulp numerical guard over a finite integer state space with visited-state cycle assertions",
    "render": "seeded shuffle across sessions; sort hours within day; duplicate-hour positive truncated-lognormal gaps rescaled within 3600 seconds",
    "session_boundaries": "same-day repeats use one session; multiday sessions are merged to feasible capacity and separated by more than 24 hours; adjacent-day equal boundary hours jointly partition integer seconds 0..3599",
}
CANONICAL_ALLOCATION = (
    "singleton minimum 1, repeat minimum 2, extras by sampled-count residual; "
    "target exact and strata immutable"
)
AMOUNT_RENDERING = (
    "customer center minor=100*localization_factor*exp(global_log_amount_center+relative_log_amount_center), constrained >=1; per-event lognormal draw has the inherited event-schema minimum of 1 minor"
)
CHECKERBOARD_SAMPLING_V6 = (
    "singleton continuous U=(bin+V)/3 plus exact hour category; repeat U=(bin+V)/3, "
    "linear 32-knot inverse except inverted-CDF/right-step discrete invoice count"
)
EXPLORATORY_ASSIGNMENT = (
    "SHA256(seed:uci-v6-crossfit:customer_id) first64 modulo 5"
)
ONE_SHOT_FINAL_CONFIRMATION = "APPROVED_SINGLE_FINAL_SCORE"
V7_CONTAINER_ALLOWLIST = (
    AGGREGATE_CONTAINER_ALLOWLIST
    - {"training_and_sealed", "temporal_readiness", "sealed_final"}
) | {
    "same_day_repeat_atom",
    "development_training",
    "one_shot_final",
    "cell_shape",
    "time_coordinates",
    "hour_allocation",
    "failure",
    "development_gate",
    "futility",
    "repeat_invoice_count_pmf",
}
V7_PRIMITIVE_NAMES = frozenset(
    primitive
    for primitives in STRATUM_PRIMITIVES.values()
    for primitive in primitives
)
V7_FORBIDDEN_ARTIFACT_KEYS = FORBIDDEN_ARTIFACT_KEYS | {"customer_id"}

PASSED_EXPLORATORY_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "protocol_sha256",
        "calibration_only",
        "uci352_diagnostic_transactions_materialized_or_used",
        "uci502_transactions_accessed",
        "uci352_prior_holdout_transactions_materialized_or_used",
        "reference",
        "futility",
        "seeds",
        "metrics",
        "gate_passed",
    }
)
FAILURE_EXPLORATORY_KEYS = PASSED_EXPLORATORY_KEYS | {
    "status",
    "calibration_access_attempted",
    "threshold_changed",
    "failure",
}
DEVELOPMENT_ATTEMPT_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "protocol_sha256",
        "status",
        "calibration_access_reserved",
        "maximum_calibration_loader_calls",
        "uci502_transactions_accessed",
    }
)
FINAL_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "protocol_sha256",
        "status",
        "confirmation_sha256",
        "exploratory_artifact_sha256",
        "recomputed_exploratory_sha256",
        "behavior_profile_sha256",
        "final_dataset_id",
        "final_source_url",
        "final_archive_sha256",
        "final_workbook_sha256",
        "final_manifest_sha256",
        "final_license",
        "final_downloader_sha256",
        "final_sheet",
        "final_fixed_window",
        "final_split_assignment",
        "excluded_uci352_calibration_customer_ids_sha256",
        "receipt_before_final_transaction_sheet_parse_or_score",
        "receipt_sha256",
    }
)
FINAL_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "protocol_sha256",
        "protocol",
        "bindings",
        "behavior_profile",
        "reference",
        "seeds",
        "metrics",
    }
)
FINAL_PROTOCOL_KEYS = frozenset(
    {
        "protocol_sha256",
        "evaluation_partition",
        "one_shot_final_scored",
        "development_gate_recomputed_and_exactly_matched",
        "orientation_free_statistic",
        "bootstrap_repeats",
        "shared_real_bootstrap_indices_across_seeds",
        "independent_synthetic_bootstrap_indices_across_seeds",
    }
)
FINAL_BINDING_KEYS = frozenset(
    {
        "exploratory_artifact_sha256",
        "final_receipt_sha256",
        "behavior_profile_sha256",
    }
)
FINAL_REFERENCE_KEYS = frozenset(
    {"development_training", "one_shot_final"}
)
FINAL_DEVELOPMENT_REFERENCE_KEYS = frozenset(
    {
        "dataset_id",
        "source_url",
        "workbook_sha256",
        "manifest_sha256",
        "license",
        "downloader_sha256",
        "calibration_customer_ids_sha256",
        "calibration_customers",
        "full_workbook_customer_ids_scanned_for_routing",
        "noncalibration_transaction_values_materialized_or_used",
        "uci352_diagnostic_transactions_materialized_or_used",
        "uci352_prior_holdout_transactions_materialized_or_used",
    }
)
FINAL_ONE_SHOT_REFERENCE_KEYS = frozenset(
    {
        "dataset_id",
        "source_url",
        "archive_sha256",
        "workbook_sha256",
        "manifest_sha256",
        "license",
        "downloader_sha256",
        "sheet",
        "fixed_window",
        "split_assignment",
        "excluded_uci352_calibration_customer_ids_sha256",
        "transaction_time_disjoint",
        "customer_disjoint_from_profile_and_classifier_training",
        "same_retailer",
        "source_independent",
        "temporal_direction",
        "prospective_validation",
        "transaction_sheet_parsed_or_scored_before_receipt",
        "first_worksheet_opened_after_receipt",
        "second_worksheet_opened",
        "nonselected_transactions_materialized",
        "eligible_customers",
        "eligible_customer_ids_sha256",
        "transform_customers",
        "score_customers",
        "transform_customer_ids_sha256",
        "score_customer_ids_sha256",
        "transform_amount_ecdf_fit_on_transform_only",
        "transform_used_for_profile_generator_classifier_threshold_or_abort",
        "score_half_is_only_real_final_evaluation",
    }
)
COMPLETION_SEAL_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "protocol_sha256",
        "status",
        "final_artifact_sha256",
        "final_receipt_sha256",
    }
)
ARTIFACT_TOP_LEVEL_KEYS = {
    "entity_fidelity_v7_calibration_exploratory": PASSED_EXPLORATORY_KEYS,
    "entity_fidelity_v7_calibration_exploratory_failure": FAILURE_EXPLORATORY_KEYS,
    "entity_fidelity_v7_development_attempt": DEVELOPMENT_ATTEMPT_KEYS,
    "entity_fidelity_v7_one_shot_final_receipt": FINAL_RECEIPT_KEYS,
    "entity_fidelity_v7_uci502_one_shot_final": FINAL_RESULT_KEYS,
    "entity_fidelity_v7_completion_seal": COMPLETION_SEAL_KEYS,
}


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _protocol_hash(value: Mapping[str, Any]) -> str:
    return _canonical_hash(
        {key: item for key, item in value.items() if key != "protocol_sha256"}
    )


def _implementation_hashes() -> dict[str, str]:
    paths = (
        ROOT / "sim" / "population.py",
        ROOT / "sim" / "mandate_twin.py",
        ROOT / "sim" / "entity_fidelity.py",
        ROOT / "sim" / "population_v4.py",
        ROOT / "sim" / "entity_fidelity_v4.py",
        ROOT / "sim" / "population_v6.py",
        ROOT / "sim" / "population_v7.py",
        ROOT / "sim" / "entity_fidelity_v7.py",
        ROOT / "tests" / "test_population_v7.py",
        ROOT / "tests" / "test_entity_fidelity_v7.py",
        ROOT / "scripts" / "download_retail_reference.py",
        ROOT / "scripts" / "download_retail_ii_reference.py",
        ROOT / "data" / "market_prices.csv",
    )
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def _assert_aggregate_only(value: Any, *, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in V7_FORBIDDEN_ARTIFACT_KEYS:
                raise ValueError(f"{path}.{key} is not allowed in a v7 artifact")
            if isinstance(item, (Mapping, list, tuple)) and not (
                normalized in V7_CONTAINER_ALLOWLIST
                or normalized in CHECKERBOARD_STRATA
                or normalized in V7_PRIMITIVE_NAMES
                or normalized.isdigit()
            ):
                raise ValueError(
                    f"{path}.{key} is not an allowlisted aggregate container"
                )
            _assert_aggregate_only(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_aggregate_only(item, path=f"{path}[{index}]")


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _same_absolute_path(candidate: Path, canonical: Path) -> bool:
    return Path(os.path.abspath(candidate)) == Path(os.path.abspath(canonical))


def _read_regular_file_bytes(path: Path) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("V7 artifact reads require O_NOFOLLOW support")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError("V7 artifact must be a readable regular non-symlink file") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("V7 artifact path is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


PREREGISTRATION_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "study_revision",
        "status",
        "parent_v6_failure_sha256",
        "parent_v6_attempt_sha256",
        "parent_v6_entity_source_sha256",
        "parent_v6_population_source_sha256",
        "parent_v6_entity_test_sha256",
        "parent_v6_population_test_sha256",
        "parent_v61_preregistration_sha256",
        "parent_v61_protocol_sha256",
        "parent_v61_attempt_sha256",
        "parent_v61_exploratory_sha256",
        "implementation_tree_sha256",
        "correction",
        "interior_quantiles_unchanged",
        "futility_seed",
        "futility_fold",
        "futility_bootstrap_repeats",
        "futility_bootstrap_seed",
        "futility_quantile",
        "evaluation_seeds_sha256",
        "threshold_max_d_upper_95",
        "uci502_transactions_accessed",
        "preregistration_sha256",
    }
)


def _preregistration_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 7,
        "artifact": "entity_fidelity_v7_preregistration",
        "study_revision": STUDY_REVISION,
        "status": "frozen",
        "parent_v6_failure_sha256": PARENT_V6_FAILURE_SHA256,
        "parent_v6_attempt_sha256": PARENT_V6_ATTEMPT_SHA256,
        "parent_v61_preregistration_sha256": PARENT_V61_PREREGISTRATION_SHA256,
        "parent_v61_protocol_sha256": PARENT_V61_PROTOCOL_SHA256,
        "parent_v61_attempt_sha256": PARENT_V61_ATTEMPT_SHA256,
        "parent_v61_exploratory_sha256": PARENT_V61_EXPLORATORY_SHA256,
        "parent_v6_entity_source_sha256": PARENT_V6_ENTITY_SOURCE_SHA256,
        "parent_v6_population_source_sha256": PARENT_V6_POPULATION_SOURCE_SHA256,
        "parent_v6_entity_test_sha256": PARENT_V6_ENTITY_TEST_SHA256,
        "parent_v6_population_test_sha256": PARENT_V6_POPULATION_TEST_SHA256,
        "implementation_tree_sha256": _canonical_hash(_implementation_hashes()),
        "correction": V7_CORRECTION_RULE,
        "interior_quantiles_unchanged": True,
        "futility_seed": FUTILITY_SEED,
        "futility_fold": FUTILITY_FOLD,
        "futility_bootstrap_repeats": FUTILITY_BOOTSTRAP_REPEATS,
        "futility_bootstrap_seed": FUTILITY_BOOTSTRAP_SEED,
        "futility_quantile": FUTILITY_QUANTILE,
        "evaluation_seeds_sha256": _canonical_hash(
            {"seeds": list(EVALUATION_SEEDS)}
        ),
        "threshold_max_d_upper_95": GATE_MAX_D_UPPER_95,
        "uci502_transactions_accessed": False,
    }
    payload["preregistration_sha256"] = _canonical_hash(payload)
    return payload


def write_v7_preregistration(
    path: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    payload = _preregistration_payload()
    if path.exists():
        existing = json.loads(_read_regular_file_bytes(path))
        if existing != payload:
            raise FileExistsError("V7 preregistration exists with different content")
        return existing
    _write_exclusive_json(path, payload)
    path.chmod(0o444)
    return payload


def load_v7_preregistration(
    path: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    payload = json.loads(_read_regular_file_bytes(path))
    if not isinstance(payload, dict):
        raise ValueError("V7 preregistration must be a JSON object")
    expected_hash = _canonical_hash(
        {key: value for key, value in payload.items() if key != "preregistration_sha256"}
    )
    if (
        set(payload) != PREREGISTRATION_KEYS
        or payload != _preregistration_payload()
        or payload.get("preregistration_sha256") != expected_hash
        or sha256_file(ROOT / "results" / "entity_fidelity_v6_freeze_failure.json")
        != PARENT_V6_FAILURE_SHA256
        or sha256_file(ROOT / "results" / "entity_fidelity_v6_development_attempt.json")
        != PARENT_V6_ATTEMPT_SHA256
        or sha256_file(ROOT / "results" / "entity_fidelity_v61_preregistration.json")
        != PARENT_V61_PREREGISTRATION_SHA256
        or sha256_file(ROOT / "results" / "entity_fidelity_v61_protocol.json")
        != PARENT_V61_PROTOCOL_SHA256
        or sha256_file(ROOT / "results" / "entity_fidelity_v61_development_attempt.json")
        != PARENT_V61_ATTEMPT_SHA256
        or sha256_file(ROOT / "results" / "entity_fidelity_v61_exploratory.json")
        != PARENT_V61_EXPLORATORY_SHA256
        or sha256_file(ROOT / "sim" / "entity_fidelity_v6.py")
        != PARENT_V6_ENTITY_SOURCE_SHA256
        or sha256_file(ROOT / "sim" / "population_v6.py")
        != PARENT_V6_POPULATION_SOURCE_SHA256
        or sha256_file(ROOT / "tests" / "test_entity_fidelity_v6.py")
        != PARENT_V6_ENTITY_TEST_SHA256
        or sha256_file(ROOT / "tests" / "test_population_v6.py")
        != PARENT_V6_POPULATION_TEST_SHA256
    ):
        raise ValueError("V7 preregistration or parent lineage is invalid")
    return payload


def _artifact_semantics_valid(value: Mapping[str, Any]) -> bool:
    artifact = value.get("artifact")
    protocol_sha256 = value.get("protocol_sha256")
    if not _sha256_value(protocol_sha256):
        return False
    if artifact == "entity_fidelity_v7_development_attempt":
        return _development_attempt_valid(
            value, {"protocol_sha256": protocol_sha256}
        )
    if artifact == "entity_fidelity_v7_calibration_exploratory":
        return (
            value.get("calibration_only") is True
            and value.get("uci352_diagnostic_transactions_materialized_or_used")
            is False
            and value.get("uci502_transactions_accessed") is False
            and value.get(
                "uci352_prior_holdout_transactions_materialized_or_used"
            )
            is False
            and value.get("gate_passed") is True
            and _exploratory_evidence_valid(value)
        )
    if artifact == "entity_fidelity_v7_calibration_exploratory_failure":
        failure = value.get("failure")
        common = (
            value.get("status") == "STOP_FAIL"
            and value.get("calibration_only") is True
            and value.get("calibration_access_attempted") is True
            and value.get("uci352_diagnostic_transactions_materialized_or_used")
            is False
            and value.get("uci502_transactions_accessed") is False
            and value.get(
                "uci352_prior_holdout_transactions_materialized_or_used"
            )
            is False
            and value.get("threshold_changed") is False
            and value.get("gate_passed") is False
            and isinstance(value.get("reference"), Mapping)
            and isinstance(value.get("futility"), Mapping)
            and isinstance(value.get("seeds"), Mapping)
            and isinstance(value.get("metrics"), Mapping)
            and isinstance(failure, Mapping)
            and set(failure) == {"error_type", "message"}
        )
        if not common:
            return False
        if failure == {
            "error_type": "futility_threshold_exceeded",
            "message": "preregistered futility lower bound exceeded the frozen threshold",
        }:
            return (
                _futility_evidence_valid(value, require_pass=False)
                and _calibration_reference_valid(value)
                and not value["seeds"]
                and not value["metrics"]
            )
        if failure == {
            "error_type": "gate_threshold_exceeded",
            "message": "development gate exceeded its frozen threshold",
        }:
            return (
                _futility_evidence_valid(value, require_pass=True)
                and _exploratory_evidence_valid(value)
            )
        return (
            failure
            in (
                {
                    "error_type": "interrupted",
                    "message": "development gate aborted after calibration access attempt",
                },
                {
                    "error_type": "calibration_failure",
                    "message": "development gate aborted after calibration access attempt",
                },
            )
            and not value["reference"]
            and not value["futility"]
            and not value["seeds"]
            and not value["metrics"]
        )
    if artifact == "entity_fidelity_v7_one_shot_final_receipt":
        return _final_receipt_valid(
            value,
            {"protocol_sha256": protocol_sha256},
            exploratory_sha256=value.get("exploratory_artifact_sha256"),
            recomputed_exploratory_sha256=value.get(
                "recomputed_exploratory_sha256"
            ),
            behavior_profile_sha256=value.get("behavior_profile_sha256"),
        )
    if artifact == "entity_fidelity_v7_uci502_one_shot_final":
        bindings = value.get("bindings")
        if not isinstance(bindings, Mapping):
            return False
        return _final_result_valid(
            value,
            expected_protocol_sha256=protocol_sha256,
            expected_exploratory_sha256=bindings.get(
                "exploratory_artifact_sha256"
            ),
            expected_receipt_sha256=bindings.get("final_receipt_sha256"),
            expected_behavior_profile_sha256=bindings.get(
                "behavior_profile_sha256"
            ),
            require_pass=False,
        )
    if artifact == "entity_fidelity_v7_completion_seal":
        return (
            value.get("schema_version") == 7
            and value.get("status") == "complete"
            and _sha256_value(value.get("final_artifact_sha256"))
            and _sha256_value(value.get("final_receipt_sha256"))
        )
    return False


def _write_artifact(path: Path, value: Mapping[str, Any]) -> bytes:
    artifact = value.get("artifact")
    expected_keys = (
        ARTIFACT_TOP_LEVEL_KEYS.get(artifact)
        if isinstance(artifact, str)
        else None
    )
    if expected_keys is None:
        raise ValueError("unknown v7 artifact name")
    if set(value) != expected_keys or value.get("schema_version") != 7:
        raise ValueError(f"{artifact} top-level fields differ from the frozen schema")
    if not _artifact_semantics_valid(value):
        raise ValueError(f"{artifact} values differ from the frozen schema")
    _assert_aggregate_only(value)
    encoded = _pretty_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("v7 artifact write made no progress")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_descriptor = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return encoded


def _finite_number(value: Any, *, lower: float, upper: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and lower <= float(value) <= upper
    )


def _development_attempt_valid(
    attempt: Mapping[str, Any], protocol: Mapping[str, Any]
) -> bool:
    try:
        _assert_aggregate_only(attempt)
    except (TypeError, ValueError):
        return False
    return (
        set(attempt) == DEVELOPMENT_ATTEMPT_KEYS
        and attempt.get("schema_version") == 7
        and attempt.get("artifact") == "entity_fidelity_v7_development_attempt"
        and attempt.get("protocol_sha256") == protocol.get("protocol_sha256")
        and attempt.get("status") == "consumed"
        and attempt.get("calibration_access_reserved") is True
        and isinstance(attempt.get("maximum_calibration_loader_calls"), int)
        and not isinstance(attempt.get("maximum_calibration_loader_calls"), bool)
        and attempt.get("maximum_calibration_loader_calls") == 1
        and attempt.get("uci502_transactions_accessed") is False
    )


def _sha256_value(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _orientation_row_valid(value: Mapping[str, Any]) -> bool:
    auc = value.get("auc")
    d = value.get("d")
    return (
        _finite_number(auc, lower=0.0, upper=1.0)
        and _finite_number(d, lower=0.5, upper=1.0)
        and math.isclose(float(d), max(float(auc), 1 - float(auc)), abs_tol=1e-12)
    )


def _futility_evidence_valid(
    value: Mapping[str, Any], *, require_pass: bool | None = None
) -> bool:
    futility = value.get("futility")
    if not isinstance(futility, Mapping) or set(futility) != {
        "seed",
        "fold",
        "real_customers",
        "synthetic_customers",
        "auc",
        "d",
        "bootstrap_repeats",
        "bootstrap_rng_seed",
        "bootstrap_quantile",
        "bootstrap_d_mean",
        "bootstrap_d_q_025",
        "threshold_max_d_upper_95",
        "passed_to_full_gate",
    }:
        return False
    passed = futility.get("passed_to_full_gate")
    valid = (
        futility.get("seed") == FUTILITY_SEED
        and futility.get("fold") == FUTILITY_FOLD
        and all(
            isinstance(futility.get(key), int)
            and not isinstance(futility.get(key), bool)
            and futility[key] > 0
            for key in ("real_customers", "synthetic_customers")
        )
        and _orientation_row_valid(futility)
        and futility.get("bootstrap_repeats") == FUTILITY_BOOTSTRAP_REPEATS
        and futility.get("bootstrap_rng_seed") == FUTILITY_BOOTSTRAP_SEED
        and futility.get("bootstrap_quantile") == FUTILITY_QUANTILE
        and _finite_number(futility.get("bootstrap_d_mean"), lower=0.5, upper=1.0)
        and _finite_number(futility.get("bootstrap_d_q_025"), lower=0.5, upper=1.0)
        and futility.get("threshold_max_d_upper_95") == GATE_MAX_D_UPPER_95
        and isinstance(passed, bool)
        and passed is (float(futility.get("bootstrap_d_q_025", math.nan)) <= GATE_MAX_D_UPPER_95)
    )
    return valid and (require_pass is None or passed is require_pass)


def _calibration_reference_valid(value: Mapping[str, Any]) -> bool:
    return value.get("reference") == {
        "workbook_sha256": UCI352_WORKBOOK_SHA256,
        "partition": "calibration",
        "customer_ids_sha256": EXPECTED_PARTITION_HASHES["calibration"],
        "customers": UCI352_CALIBRATION_CUSTOMERS,
        "transactions": UCI352_CALIBRATION_TRANSACTIONS,
    }


def _representative_futility(fold: Mapping[str, Any]) -> dict[str, Any]:
    """Bootstrap real and synthetic customers separately before the full gate."""
    np, _, _, roc_auc_score = _dependencies()
    real_source = np.asarray(fold["_real_predictions"], dtype=float)
    synthetic_source = np.asarray(fold["_synthetic_predictions"], dtype=float)
    if fold.get("fold") != FUTILITY_FOLD or not len(real_source) or not len(synthetic_source):
        raise ValueError("futility requires the preregistered nonempty representative fold")
    rng = np.random.default_rng(FUTILITY_BOOTSTRAP_SEED)
    bootstraps = []
    labels = np.concatenate(
        (np.zeros(len(real_source)), np.ones(len(synthetic_source)))
    )
    for _ in range(FUTILITY_BOOTSTRAP_REPEATS):
        real = real_source[rng.integers(0, len(real_source), len(real_source))]
        synthetic = synthetic_source[
            rng.integers(0, len(synthetic_source), len(synthetic_source))
        ]
        auc = float(roc_auc_score(labels, np.concatenate((real, synthetic))))
        bootstraps.append(max(auc, 1 - auc))
    q_025 = float(np.quantile(bootstraps, FUTILITY_QUANTILE))
    result = {
        "seed": FUTILITY_SEED,
        "fold": FUTILITY_FOLD,
        "real_customers": len(real_source),
        "synthetic_customers": len(synthetic_source),
        "auc": float(fold["auc"]),
        "d": float(fold["d"]),
        "bootstrap_repeats": FUTILITY_BOOTSTRAP_REPEATS,
        "bootstrap_rng_seed": FUTILITY_BOOTSTRAP_SEED,
        "bootstrap_quantile": FUTILITY_QUANTILE,
        "bootstrap_d_mean": float(np.mean(bootstraps)),
        "bootstrap_d_q_025": q_025,
        "threshold_max_d_upper_95": GATE_MAX_D_UPPER_95,
        "passed_to_full_gate": q_025 <= GATE_MAX_D_UPPER_95,
    }
    if not _futility_evidence_valid({"futility": result}):
        raise RuntimeError("computed futility evidence is invalid")
    return result


def _gate_metrics_valid(
    value: Mapping[str, Any], *, point_values: Iterable[float]
) -> bool:
    points = tuple(point_values)
    metrics = value.get("metrics", {})
    interval = metrics.get("bootstrap_max_seed_d_ci_95")
    if (
        not points
        or not all(_finite_number(point, lower=0.5, upper=1.0) for point in points)
        or set(metrics)
        != {
            "orientation_free_max_seed_d",
            "bootstrap_max_seed_d_mean",
            "bootstrap_max_seed_d_ci_95",
            "threshold_max_d_upper_95",
        }
        or not isinstance(interval, list)
        or len(interval) != 2
        or not all(
            _finite_number(item, lower=0.5, upper=1.0) for item in interval
        )
        or float(interval[0]) > float(interval[1])
        or not _finite_number(
            metrics.get("bootstrap_max_seed_d_mean"), lower=0.5, upper=1.0
        )
        or not _finite_number(
            metrics.get("orientation_free_max_seed_d"), lower=0.5, upper=1.0
        )
        or metrics.get("threshold_max_d_upper_95") != GATE_MAX_D_UPPER_95
        or not math.isclose(
            float(metrics.get("orientation_free_max_seed_d", math.nan)),
            max(map(float, points)),
            abs_tol=1e-12,
        )
        or not isinstance(value.get("gate_passed"), bool)
    ):
        return False
    return value["gate_passed"] is (
        float(interval[1]) <= GATE_MAX_D_UPPER_95
    )


def fractional_singleton_masses(rows: Any) -> tuple[float, ...]:
    """Fit the 3x3x24 joint grid without turning hour into a rank coordinate."""
    np, _, _, _ = _dependencies()
    values = np.asarray(rows, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or len(values) == 0
        or not np.isfinite(values).all()
        or any(int(hour) != hour or not 0 <= hour < 24 for hour in values[:, 2])
    ):
        raise ValueError("singleton joint-grid fit needs day, amount, and exact hour")
    day_weights = fractional_tie_weights(values[:, 0])
    amount_weights = fractional_tie_weights(values[:, 1])
    masses = np.zeros(STRATUM_CELL_SHAPES["singleton"], dtype=float)
    for index, hour in enumerate(values[:, 2].astype(int)):
        masses[:, :, hour] += np.multiply.outer(
            day_weights[index], amount_weights[index]
        )
    masses /= len(values)
    if not math.isclose(float(masses.sum()), 1.0, abs_tol=1e-12):
        raise RuntimeError("singleton joint-grid masses do not sum to one")
    return tuple(map(float, masses.reshape(-1)))


def _fit_behavior_profile_and_rows(
    real_calibration: Any, *, customer_ids_sha256: str
) -> tuple[BehaviorProfileV7, dict[str, Any]]:
    np, _, _, _ = _dependencies()
    if real_calibration.empty:
        raise ValueError("calibration partition is empty")
    frame = real_calibration.sort_values(["customer_id", "ts"], kind="stable")
    groups = [
        (str(customer_id), group.copy())
        for customer_id, group in frame.groupby("customer_id", sort=True)
    ]
    eb_log_scales, amount_prior_alpha, amount_prior_beta = _empirical_bayes_log_scales(
        groups
    )
    (
        eb_short_probabilities,
        short_prior_alpha,
        short_prior_beta,
    ) = _empirical_bayes_short_probabilities(groups)
    timestamps = frame["ts"]
    window_start = timestamps.min().normalize()
    window_end = timestamps.max().normalize()
    observation_days = int((window_end - window_start).days + 1)
    if observation_days < 2:
        raise ValueError("v7 calibration window needs at least two calendar days")
    hour_counts = timestamps.dt.hour.value_counts()
    hour_weights = tuple(
        float(hour_counts.get(hour, 0) / len(frame)) for hour in range(24)
    )
    elapsed_fraction = (
        (timestamps.dt.normalize() - window_start).dt.days + 0.5
    ) / observation_days
    calendar_bins = np.minimum((elapsed_fraction * 12).astype(int), 11)
    calendar_counts = calendar_bins.value_counts()
    calendar_weights = tuple(
        float(calendar_counts.get(index, 0) / len(frame)) for index in range(12)
    )
    all_gaps = _positive_gaps_seconds(frame)
    short_gaps = all_gaps[all_gaps <= 86_400]
    if len(short_gaps) == 0:
        short_gaps = np.asarray(
            [min(86_400.0, float(np.median(all_gaps))) if len(all_gaps) else 3_600.0]
        )
    session_gap_logs = np.log(short_gaps)
    log_amounts = np.log(frame["amount"].to_numpy(dtype=float))
    global_log_center = float(np.mean(log_amounts))
    global_angles = 2 * np.pi * timestamps.dt.hour.to_numpy(dtype=float) / 24
    global_sin = float(np.mean(np.sin(global_angles)))
    global_cos = float(np.mean(np.cos(global_angles)))
    global_resultant = math.hypot(global_sin, global_cos)
    phase_cut = (math.atan2(global_sin, global_cos) + math.pi) % (2 * math.pi)
    primitive_rows: dict[str, list[list[float]]] = {
        name: [] for name in CHECKERBOARD_STRATA
    }
    repeat_customers = 0
    same_day_repeats = 0
    repeat_invoice_counts: list[int] = []
    for customer_id, group in groups:
        group = group.sort_values("ts", kind="stable")
        count = len(group)
        group_logs = np.log(group["amount"].to_numpy(dtype=float))
        angles = 2 * np.pi * group["ts"].dt.hour.to_numpy(dtype=float) / 24
        mean_sin = float(np.mean(np.sin(angles)))
        mean_cos = float(np.mean(np.cos(angles)))
        resultant = math.hypot(mean_sin, mean_cos)
        if resultant > 1e-12:
            phase_sin, phase_cos = mean_sin / resultant, mean_cos / resultant
        else:
            phase_norm = max(1e-12, global_resultant)
            phase_sin, phase_cos = global_sin / phase_norm, global_cos / phase_norm
        phase_latent = (math.atan2(phase_sin, phase_cos) - phase_cut) % (2 * math.pi)
        first_day = int((group["ts"].iloc[0].normalize() - window_start).days)
        last_day = int((group["ts"].iloc[-1].normalize() - window_start).days)
        base = {
            "invoice_count": float(count),
            "active_end_fraction": (
                (last_day - 0.5) / (observation_days - 1)
                if last_day > 0
                else math.nan
            ),
            "span_within_end_fraction": (
                (last_day - first_day - 0.5) / last_day
                if last_day > first_day
                else math.nan
            ),
            "relative_log_amount_center": float(group_logs.mean() - global_log_center),
            "eb_log_amount_scale": eb_log_scales[customer_id],
            "eb_short_session_probability": eb_short_probabilities[customer_id],
            "circadian_phase_latent": phase_latent,
            "circadian_resultant": resultant,
        }
        if count == 1:
            primitive_rows["singleton"].append(
                [
                    (first_day + 0.5) / observation_days,
                    base["relative_log_amount_center"],
                    float(group["ts"].iloc[0].hour),
                ]
            )
            continue
        repeat_customers += 1
        if count > REPEAT_INVOICE_COUNT_MAX:
            raise ValueError(
                f"observed repeat invoice count {count} exceeds fixed V7 support "
                f"maximum {REPEAT_INVOICE_COUNT_MAX}"
            )
        repeat_invoice_counts.append(count)
        if first_day == last_day:
            same_day_repeats += 1
            continue
        primitive_rows["multiday_repeat"].append(
            [base[name] for name in STRATUM_PRIMITIVES["multiday_repeat"]]
        )
    arrays = {
        name: np.asarray(primitive_rows[name], dtype=float)
        for name in CHECKERBOARD_STRATA
    }
    for name, rows in arrays.items():
        if len(rows) < MIN_CHECKERBOARD_CUSTOMERS:
            raise ValueError(
                f"{name} has {len(rows)} customers; at least "
                f"{MIN_CHECKERBOARD_CUSTOMERS} are required"
            )
    strata = []
    for name in CHECKERBOARD_STRATA:
        rows = arrays[name]
        continuous = rows[:, :2] if name == "singleton" else rows
        if name == "singleton":
            quantiles = np.quantile(
                continuous, QUANTILE_PROBABILITIES, axis=0, method="linear"
            ).T
        else:
            quantiles = np.vstack(
                [
                    np.quantile(
                        continuous[:, index],
                        QUANTILE_PROBABILITIES,
                        method="inverted_cdf" if index == 0 else "linear",
                    )
                    for index in range(continuous.shape[1])
                ]
            )
            observed_counts = continuous[:, 0]
            quantiles[0, 0] = observed_counts.min()
            quantiles[0, -1] = observed_counts.max()
            if (
                quantiles[0, 0] != observed_counts.min()
                or quantiles[0, -1] != observed_counts.max()
                or not all(float(value).is_integer() for value in quantiles[0])
            ):
                raise RuntimeError("discrete invoice knots lost observed support")
        strata.append(
            CheckerboardStratumV6(
                name=name,
                customer_count=len(rows),
                primitive_quantiles=tuple(
                    tuple(map(float, values)) for values in quantiles
                ),
                cell_masses=(
                    fractional_singleton_masses(rows)
                    if name == "singleton"
                    else fractional_checkerboard_masses(rows)
                ),
            )
        )
    profile = BehaviorProfileV7(
        source_dataset="UCI Online Retail 352",
        source_partition="calibration",
        source_customer_ids_sha256=customer_ids_sha256,
        source_transactions_sha256=_transactions_hash(frame),
        source_customer_count=len(groups),
        source_transaction_count=len(frame),
        repeat_customer_count=repeat_customers,
        same_day_repeat_probability=same_day_repeats / repeat_customers,
        quantile_probabilities=QUANTILE_PROBABILITIES,
        strata=tuple(strata),
        hour_weights=hour_weights,
        calendar_weights=calendar_weights,
        observation_days=observation_days,
        session_gap_log_mean=float(np.mean(session_gap_logs)),
        session_gap_log_std=float(np.std(session_gap_logs)),
        circadian_phase_cut=phase_cut,
        global_log_amount_center=global_log_center,
        localization_factor=LOCALIZATION_FACTOR_DEFAULT,
        amount_variance_prior_alpha=amount_prior_alpha,
        amount_variance_prior_beta=amount_prior_beta,
        short_session_prior_alpha=short_prior_alpha,
        short_session_prior_beta=short_prior_beta,
        repeat_invoice_count_pmf=tuple(
            repeat_invoice_counts.count(value) / repeat_customers
            for value in range(REPEAT_INVOICE_COUNT_MAX + 1)
        ),
    )
    return profile, arrays


def fit_behavior_profile_v7(
    real_calibration: Any, *, customer_ids_sha256: str
) -> BehaviorProfileV7:
    profile, _ = _fit_behavior_profile_and_rows(
        real_calibration, customer_ids_sha256=customer_ids_sha256
    )
    return profile


def novelty_check(
    profile: BehaviorProfileV7,
    calibration_primitive_rows: Mapping[str, Any],
    *,
    draws: int = NOVELTY_DRAWS,
    seed: int = SEED + 500,
    batch_size: int = 10_000,
) -> dict[str, Any]:
    if draws < 1 or batch_size < 1:
        raise ValueError("novelty draw counts must be positive")
    calibration = {
        name: {
            tuple(float(row[index]) for index in range(len(STRATUM_PRIMITIVES[name])))
            for row in rows
        }
        for name, rows in calibration_primitive_rows.items()
    }
    matches = 0
    completed = 0
    batch = 0
    while completed < draws:
        current = min(batch_size, draws - completed)
        for row in profile.sample_primitives(current, seed=seed + batch):
            name = "singleton" if row["invoice_count"] == 1 else "multiday_repeat"
            key = tuple(float(row[primitive]) for primitive in STRATUM_PRIMITIVES[name])
            matches += key in calibration[name]
        completed += current
        batch += 1
    return {
        "draws": draws,
        "exact_full_continuous_primitive_tuple_matches": matches,
        "passed": matches == 0,
    }


def generate_synthetic_transactions_v7(
    *, profile: BehaviorProfileV7, customers: int, seed: int
) -> Any:
    _, pd, _, _ = _dependencies()
    simulator = PopulationSimulatorV7(
        seed=seed,
        customers=customers,
        merchants_per_category=1,
        agentic_rate=0,
        behavior_profile=profile,
    )
    events = sum(
        round(customer.activity_weight) for customer in simulator.customers.values()
    )
    rows = (
        {
            "customer_id": event["txn"]["customer_id"],
            "ts": pd.Timestamp(event["ts"]).tz_convert("UTC").tz_localize(None),
            "amount": event["txn"]["amount_minor"] / 100,
        }
        for event in simulator.iter_legitimate(events)
    )
    return pd.DataFrame(rows).sort_values(
        ["customer_id", "ts"], kind="stable"
    ).reset_index(drop=True)


def freeze_v7_protocol(
    reference_xlsx: Path = DEFAULT_REFERENCE,
    protocol_path: Path = DEFAULT_PROTOCOL,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    preregistration = load_v7_preregistration(preregistration_path)
    real_calibration, reference = load_retail_partition(reference_xlsx, "calibration")
    profile, primitive_rows = _fit_behavior_profile_and_rows(
        real_calibration,
        customer_ids_sha256=_ids_hash(real_calibration["customer_id"].unique()),
    )
    novelty = novelty_check(profile, primitive_rows, draws=NOVELTY_DRAWS)
    if novelty["passed"] is not True:
        raise ValueError("v7 novelty check failed before protocol freeze")
    protocol: dict[str, Any] = {
        "schema_version": 7,
        "artifact": "entity_fidelity_v7_frozen_protocol",
        "study_revision": STUDY_REVISION,
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "parent_v6_failure_sha256": PARENT_V6_FAILURE_SHA256,
        "parent_v6_attempt_sha256": PARENT_V6_ATTEMPT_SHA256,
        "parent_v61_preregistration_sha256": PARENT_V61_PREREGISTRATION_SHA256,
        "parent_v61_protocol_sha256": PARENT_V61_PROTOCOL_SHA256,
        "parent_v61_attempt_sha256": PARENT_V61_ATTEMPT_SHA256,
        "parent_v61_exploratory_sha256": PARENT_V61_EXPLORATORY_SHA256,
        "implementation_sha256": _implementation_hashes(),
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in DEPENDENCY_PACKAGES
        },
        "dependency_lock_sha256": sha256_file(ROOT / "requirements.lock"),
        "sources": {
            "development_training": {
                "dataset_id": UCI352_DATASET_ID,
                "source_url": UCI352_SOURCE_URL,
                "workbook_sha256": reference["workbook_sha256"],
                "manifest_sha256": reference["manifest_sha256"],
                "license": UCI352_LICENSE,
                "downloader_sha256": sha256_file(
                    ROOT / "scripts" / "download_retail_reference.py"
                ),
                "calibration_customer_ids_sha256": EXPECTED_PARTITION_HASHES[
                    "calibration"
                ],
                "diagnostic_permitted": False,
                "full_workbook_customer_ids_scanned_for_routing": True,
                "nonselected_transactions_materialized": False,
                "nonselected_transaction_values_used": False,
            },
            "one_shot_final": {
                "dataset_id": UCI502_DATASET_ID,
                "source_url": UCI502_SOURCE_URL,
                "archive_sha256": UCI502_ARCHIVE_SHA256,
                "workbook_sha256": UCI502_WORKBOOK_SHA256,
                "manifest_sha256": UCI502_MANIFEST_SHA256,
                "license": UCI502_LICENSE,
                "downloader_sha256": sha256_file(
                    ROOT / "scripts" / "download_retail_ii_reference.py"
                ),
                "sheet": UCI502_SHEET,
                "fixed_window": f"[{UCI502_WINDOW_START}, {UCI502_WINDOW_END})",
                "exclude_uci352_calibration_customers": True,
                "split_assignment": UCI502_SPLIT_ASSIGNMENT,
                "minimum_customers_per_half": UCI502_MIN_SPLIT_CUSTOMERS,
                "transaction_time_disjoint": True,
                "customer_disjoint_from_profile_and_classifier_training": True,
                "same_retailer": True,
                "source_independent": False,
                "temporal_direction": "historical_before_training_period",
                "prospective_validation": False,
                "second_worksheet_opened": False,
                "transaction_sheet_parsed_or_scored_before_receipt": False,
                "transform_half_use": "amount ECDF and fixed-window transform only",
                "transform_used_for_profile_generator_classifier_threshold_or_abort": False,
            },
        },
        "generator": {
            "behavior_profile": profile.as_dict(),
            "behavior_profile_sha256": profile.sha256,
            "strata": {
                name: {
                    "primitives": list(STRATUM_PRIMITIVES[name]),
                    "continuous_quantiles": len(
                        SINGLETON_CONTINUOUS_PRIMITIVES
                        if name == "singleton"
                        else REPEAT_PRIMITIVES
                    ),
                    "cell_shape": list(STRATUM_CELL_SHAPES[name]),
                    "cell_masses": STRATUM_CELL_COUNTS[name],
                    "minimum_customers": MIN_CHECKERBOARD_CUSTOMERS,
                }
                for name in CHECKERBOARD_STRATA
            },
            "same_day_repeat_atom": {
                "probability": "P(first_day=last_day | invoice_count>=2), fitted on each profile/fold",
                "allocation": "exact round(probability*sampled_repeat_count)",
                "selection": ATOM_SELECTION_RULE,
                "projection": "retain sampled end day and set first day equal to last day",
            },
            "time_coordinates": TIME_COORDINATES,
            "hour_allocation": HOUR_ALLOCATION,
            "canonical_allocator": CANONICAL_ALLOCATION,
            "amount_rendering": AMOUNT_RENDERING,
            "checkerboard_bins": CHECKERBOARD_BINS,
            "quantile_probabilities": list(QUANTILE_PROBABILITIES),
            "tie_rule": CHECKERBOARD_TIE_RULE,
            "cell_allocation": CHECKERBOARD_CELL_ALLOCATION,
            "sampling": CHECKERBOARD_SAMPLING_V6,
            "smoothing": False,
            "pseudocounts": False,
            "stored_rank_rows": False,
            "repeat_invoice_count_allocation": (
                "exact residual allocation from fixed calibration PMF on integer "
                "support 0..256, rank-reordered by inherited invoice_count then index"
            ),
            "synthetic_event_target": (
                "sum(round(customer.activity_weight) for realized customers)"
            ),
            "multiday_session_anchors": (
                "first_day + (i*(last_day-first_day)+(sessions-1)//2)//"
                "(sessions-1), preserving endpoints and >=2-day gaps"
            ),
            "novelty": novelty,
        },
        "evaluation": {
            "seeds": list(EVALUATION_SEEDS),
            "features": list(FEATURES),
            "classifier": _classifier_protocol(),
            "exploratory_crossfit_folds": CROSS_FIT_FOLDS,
            "exploratory_assignment": EXPLORATORY_ASSIGNMENT,
            "orientation_free_statistic": "D=max(AUC,1-AUC)",
            "max_seed_statistic": True,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_quantile": 0.975,
            "shared_real_bootstrap_indices_across_seeds": True,
            "independent_synthetic_bootstrap_indices_across_seeds": True,
            "threshold_max_d_upper_95": GATE_MAX_D_UPPER_95,
            "futility": {
                "seed": FUTILITY_SEED,
                "fold": FUTILITY_FOLD,
                "bootstrap_repeats": FUTILITY_BOOTSTRAP_REPEATS,
                "bootstrap_rng_seed": FUTILITY_BOOTSTRAP_SEED,
                "bootstrap_quantile": FUTILITY_QUANTILE,
                "orientation_free_statistic": "D=max(AUC,1-AUC)",
                "stop_if": "bootstrap q0.025(D)>0.65",
            },
            "development_gate_recomputed_and_exactly_matched_before_final": True,
            "final_transform_and_score_in_one_noninteractive_run": True,
        },
        "development_gate": {
            "canonical_attempt_ledger_path": str(
                DEFAULT_DEVELOPMENT_ATTEMPT.relative_to(ROOT)
            ),
            "canonical_exploratory_path": str(DEFAULT_EXPLORATORY.relative_to(ROOT)),
            "attempt_reserved_before_calibration_access": True,
            "existing_attempt_blocks_retry": True,
            "base_exception_burns_attempt": True,
        },
        "one_shot_final": {
            "confirmation": f"{ONE_SHOT_FINAL_CONFIRMATION}:<protocol_sha256>",
            "canonical_receipt_path": str(DEFAULT_SEAL_RECEIPT.relative_to(ROOT)),
            "canonical_output_path": str(DEFAULT_OUTPUT.relative_to(ROOT)),
            "canonical_completion_seal_path": str(
                DEFAULT_COMPLETION_SEAL.relative_to(ROOT)
            ),
            "canonical_source_path": str(DEFAULT_FINAL_REFERENCE.relative_to(ROOT)),
            "receipt_before_final_transaction_sheet_parse_or_score": True,
            "receipt_bytes_revalidated_immediately_before_final_loader": True,
            "module_runtime_boundary_not_process_wide_capability_security": True,
            "final_written_before_completion_seal": True,
            "completion_seal_binds_exact_final_bytes": True,
            "completion_seal_is_tamper_evidence_not_signature_or_worm": True,
            "external_signed_or_worm_custody_is_deployment_responsibility": True,
            "external_signed_or_worm_custody_is_release_prerequisite": True,
            "crash_burns_attempt": True,
            "second_invocation_fails": True,
        },
        "privacy": {
            "artifact_policy": "fixed aggregates only",
            "contains_customer_ids": False,
            "contains_raw_rows": False,
            "contains_rank_rows": False,
            "contains_evaluator_feature_rows": False,
            "contains_prediction_vectors": False,
            "differential_privacy_claimed": False,
        },
    }
    protocol["protocol_sha256"] = _protocol_hash(protocol)
    _assert_aggregate_only(protocol)
    if protocol_path.exists():
        existing = json.loads(_read_regular_file_bytes(protocol_path))
        if existing != protocol:
            raise FileExistsError("frozen v7 protocol exists with different content")
        return existing
    _write_exclusive_json(protocol_path, protocol)
    protocol_path.chmod(0o444)
    return protocol


def load_frozen_v7_protocol(
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    preregistration = load_v7_preregistration()
    protocol = json.loads(_read_regular_file_bytes(protocol_path))
    if (
        set(protocol)
        != {
            "schema_version",
            "artifact",
            "study_revision",
            "preregistration_sha256",
            "parent_v6_failure_sha256",
            "parent_v6_attempt_sha256",
            "parent_v61_preregistration_sha256",
            "parent_v61_protocol_sha256",
            "parent_v61_attempt_sha256",
            "parent_v61_exploratory_sha256",
            "implementation_sha256",
            "dependency_versions",
            "dependency_lock_sha256",
            "sources",
            "generator",
            "evaluation",
            "development_gate",
            "one_shot_final",
            "privacy",
            "protocol_sha256",
        }
        or
        protocol.get("schema_version") != 7
        or protocol.get("artifact") != "entity_fidelity_v7_frozen_protocol"
        or protocol.get("study_revision") != STUDY_REVISION
        or protocol.get("preregistration_sha256")
        != preregistration["preregistration_sha256"]
        or protocol.get("parent_v6_failure_sha256")
        != PARENT_V6_FAILURE_SHA256
        or protocol.get("parent_v6_attempt_sha256")
        != PARENT_V6_ATTEMPT_SHA256
        or protocol.get("parent_v61_preregistration_sha256")
        != PARENT_V61_PREREGISTRATION_SHA256
        or protocol.get("parent_v61_protocol_sha256")
        != PARENT_V61_PROTOCOL_SHA256
        or protocol.get("parent_v61_attempt_sha256")
        != PARENT_V61_ATTEMPT_SHA256
        or protocol.get("parent_v61_exploratory_sha256")
        != PARENT_V61_EXPLORATORY_SHA256
        or protocol.get("protocol_sha256") != _protocol_hash(protocol)
    ):
        raise ValueError("not a valid frozen v7 protocol")
    if protocol.get("implementation_sha256") != _implementation_hashes():
        raise ValueError("implementation changed after the v7 protocol was frozen")
    dependency_versions = {
        name: importlib.metadata.version(name)
        for name in DEPENDENCY_PACKAGES
    }
    if (
        protocol.get("dependency_versions") != dependency_versions
        or protocol.get("dependency_lock_sha256")
        != sha256_file(ROOT / "requirements.lock")
    ):
        raise ValueError("v7 dependency bindings changed")
    sources = protocol.get("sources", {})
    training = sources.get("development_training", {})
    final_source = sources.get("one_shot_final", {})
    if (
        set(sources) != {"development_training", "one_shot_final"}
        or set(training)
        != {
            "dataset_id",
            "source_url",
            "workbook_sha256",
            "manifest_sha256",
            "license",
            "downloader_sha256",
            "calibration_customer_ids_sha256",
            "diagnostic_permitted",
            "full_workbook_customer_ids_scanned_for_routing",
            "nonselected_transactions_materialized",
            "nonselected_transaction_values_used",
        }
        or set(final_source)
        != {
            "dataset_id",
            "source_url",
            "archive_sha256",
            "workbook_sha256",
            "manifest_sha256",
            "license",
            "downloader_sha256",
            "sheet",
            "fixed_window",
            "exclude_uci352_calibration_customers",
            "split_assignment",
            "minimum_customers_per_half",
            "transaction_time_disjoint",
            "customer_disjoint_from_profile_and_classifier_training",
            "same_retailer",
            "source_independent",
            "temporal_direction",
            "prospective_validation",
            "second_worksheet_opened",
            "transaction_sheet_parsed_or_scored_before_receipt",
            "transform_half_use",
            "transform_used_for_profile_generator_classifier_threshold_or_abort",
        }
        or
        training.get("dataset_id") != UCI352_DATASET_ID
        or training.get("source_url") != UCI352_SOURCE_URL
        or training.get("workbook_sha256") != UCI352_WORKBOOK_SHA256
        or training.get("manifest_sha256") != UCI352_MANIFEST_SHA256
        or training.get("license") != UCI352_LICENSE
        or training.get("downloader_sha256")
        != sha256_file(ROOT / "scripts" / "download_retail_reference.py")
        or training.get("calibration_customer_ids_sha256")
        != EXPECTED_PARTITION_HASHES["calibration"]
        or training.get("diagnostic_permitted") is not False
        or training.get("full_workbook_customer_ids_scanned_for_routing") is not True
        or training.get("nonselected_transactions_materialized") is not False
        or training.get("nonselected_transaction_values_used") is not False
        or final_source.get("dataset_id") != UCI502_DATASET_ID
        or final_source.get("source_url") != UCI502_SOURCE_URL
        or final_source.get("archive_sha256") != UCI502_ARCHIVE_SHA256
        or final_source.get("workbook_sha256") != UCI502_WORKBOOK_SHA256
        or final_source.get("manifest_sha256") != UCI502_MANIFEST_SHA256
        or final_source.get("license") != UCI502_LICENSE
        or final_source.get("downloader_sha256")
        != sha256_file(ROOT / "scripts" / "download_retail_ii_reference.py")
        or final_source.get("sheet") != UCI502_SHEET
        or final_source.get("fixed_window")
        != f"[{UCI502_WINDOW_START}, {UCI502_WINDOW_END})"
        or final_source.get("split_assignment") != UCI502_SPLIT_ASSIGNMENT
        or final_source.get("minimum_customers_per_half")
        != UCI502_MIN_SPLIT_CUSTOMERS
        or final_source.get("exclude_uci352_calibration_customers") is not True
        or final_source.get("transaction_time_disjoint") is not True
        or final_source.get(
            "customer_disjoint_from_profile_and_classifier_training"
        )
        is not True
        or final_source.get("same_retailer") is not True
        or final_source.get("source_independent") is not False
        or final_source.get("temporal_direction")
        != "historical_before_training_period"
        or final_source.get("prospective_validation") is not False
        or final_source.get("second_worksheet_opened") is not False
        or final_source.get("transaction_sheet_parsed_or_scored_before_receipt")
        is not False
        or final_source.get("transform_half_use")
        != "amount ECDF and fixed-window transform only"
        or final_source.get(
            "transform_used_for_profile_generator_classifier_threshold_or_abort"
        )
        is not False
    ):
        raise ValueError("v7 source pins changed")
    generator = protocol.get("generator", {})
    profile = BehaviorProfileV7.from_dict(generator["behavior_profile"])
    expected_strata = {
        name: {
            "primitives": list(STRATUM_PRIMITIVES[name]),
            "continuous_quantiles": len(
                SINGLETON_CONTINUOUS_PRIMITIVES
                if name == "singleton"
                else REPEAT_PRIMITIVES
            ),
            "cell_shape": list(STRATUM_CELL_SHAPES[name]),
            "cell_masses": STRATUM_CELL_COUNTS[name],
            "minimum_customers": MIN_CHECKERBOARD_CUSTOMERS,
        }
        for name in CHECKERBOARD_STRATA
    }
    expected_atom = {
        "probability": "P(first_day=last_day | invoice_count>=2), fitted on each profile/fold",
        "allocation": "exact round(probability*sampled_repeat_count)",
        "selection": ATOM_SELECTION_RULE,
        "projection": "retain sampled end day and set first day equal to last day",
    }
    if (
        set(generator)
        != {
            "behavior_profile",
            "behavior_profile_sha256",
            "strata",
            "same_day_repeat_atom",
            "time_coordinates",
            "hour_allocation",
            "canonical_allocator",
            "amount_rendering",
            "checkerboard_bins",
            "quantile_probabilities",
            "tie_rule",
            "cell_allocation",
            "sampling",
            "smoothing",
            "pseudocounts",
            "stored_rank_rows",
            "repeat_invoice_count_allocation",
            "synthetic_event_target",
            "multiday_session_anchors",
            "novelty",
        }
        or profile.sha256 != generator.get("behavior_profile_sha256")
        or profile.as_dict() != generator.get("behavior_profile")
        or profile.source_dataset != "UCI Online Retail 352"
        or profile.source_customer_ids_sha256
        != EXPECTED_PARTITION_HASHES["calibration"]
        or profile.source_transactions_sha256
        != UCI352_CALIBRATION_TRANSACTIONS_SHA256
        or profile.source_customer_count != UCI352_CALIBRATION_CUSTOMERS
        or profile.source_transaction_count != UCI352_CALIBRATION_TRANSACTIONS
        or generator.get("strata") != expected_strata
        or generator.get("same_day_repeat_atom") != expected_atom
        or generator.get("time_coordinates") != TIME_COORDINATES
        or generator.get("hour_allocation") != HOUR_ALLOCATION
        or generator.get("canonical_allocator") != CANONICAL_ALLOCATION
        or generator.get("amount_rendering") != AMOUNT_RENDERING
        or generator.get("checkerboard_bins") != CHECKERBOARD_BINS
        or tuple(generator.get("quantile_probabilities", ()))
        != QUANTILE_PROBABILITIES
        or generator.get("tie_rule") != CHECKERBOARD_TIE_RULE
        or generator.get("cell_allocation") != CHECKERBOARD_CELL_ALLOCATION
        or generator.get("sampling") != CHECKERBOARD_SAMPLING_V6
        or generator.get("smoothing") is not False
        or generator.get("pseudocounts") is not False
        or generator.get("stored_rank_rows") is not False
        or generator.get("repeat_invoice_count_allocation")
        != "exact residual allocation from fixed calibration PMF on integer support 0..256, rank-reordered by inherited invoice_count then index"
        or generator.get("synthetic_event_target")
        != "sum(round(customer.activity_weight) for realized customers)"
        or generator.get("multiday_session_anchors")
        != "first_day + (i*(last_day-first_day)+(sessions-1)//2)//(sessions-1), preserving endpoints and >=2-day gaps"
    ):
        raise ValueError("v7 frozen generator contract changed")
    novelty = generator.get("novelty", {})
    if (
        set(novelty)
        != {
            "draws",
            "exact_full_continuous_primitive_tuple_matches",
            "passed",
        }
        or not isinstance(novelty.get("draws"), int)
        or isinstance(novelty.get("draws"), bool)
        or novelty.get("draws") != NOVELTY_DRAWS
        or novelty.get("passed") is not True
        or not isinstance(
            novelty.get("exact_full_continuous_primitive_tuple_matches"), int
        )
        or isinstance(
            novelty.get("exact_full_continuous_primitive_tuple_matches"), bool
        )
        or novelty.get("exact_full_continuous_primitive_tuple_matches") != 0
    ):
        raise ValueError("frozen v7 novelty gate is absent or failed")
    evaluation = protocol.get("evaluation", {})
    if (
        set(evaluation)
        != {
            "seeds",
            "features",
            "classifier",
            "exploratory_crossfit_folds",
            "exploratory_assignment",
            "orientation_free_statistic",
            "max_seed_statistic",
            "bootstrap_repeats",
            "bootstrap_quantile",
            "shared_real_bootstrap_indices_across_seeds",
            "independent_synthetic_bootstrap_indices_across_seeds",
            "threshold_max_d_upper_95",
            "futility",
            "development_gate_recomputed_and_exactly_matched_before_final",
            "final_transform_and_score_in_one_noninteractive_run",
        }
        or tuple(evaluation.get("seeds", ())) != EVALUATION_SEEDS
        or tuple(evaluation.get("features", ())) != FEATURES
        or evaluation.get("classifier") != _classifier_protocol()
        or evaluation.get("exploratory_crossfit_folds") != CROSS_FIT_FOLDS
        or evaluation.get("exploratory_assignment") != EXPLORATORY_ASSIGNMENT
        or evaluation.get("orientation_free_statistic") != "D=max(AUC,1-AUC)"
        or evaluation.get("max_seed_statistic") is not True
        or evaluation.get("bootstrap_repeats") != BOOTSTRAP_REPEATS
        or evaluation.get("bootstrap_quantile") != 0.975
        or evaluation.get("shared_real_bootstrap_indices_across_seeds") is not True
        or evaluation.get("independent_synthetic_bootstrap_indices_across_seeds")
        is not True
        or evaluation.get("threshold_max_d_upper_95") != GATE_MAX_D_UPPER_95
        or evaluation.get("futility")
        != {
            "seed": FUTILITY_SEED,
            "fold": FUTILITY_FOLD,
            "bootstrap_repeats": FUTILITY_BOOTSTRAP_REPEATS,
            "bootstrap_rng_seed": FUTILITY_BOOTSTRAP_SEED,
            "bootstrap_quantile": FUTILITY_QUANTILE,
            "orientation_free_statistic": "D=max(AUC,1-AUC)",
            "stop_if": "bootstrap q0.025(D)>0.65",
        }
        or evaluation.get(
            "development_gate_recomputed_and_exactly_matched_before_final"
        )
        is not True
        or evaluation.get("final_transform_and_score_in_one_noninteractive_run")
        is not True
    ):
        raise ValueError("v7 evaluation contract changed")
    if protocol.get("development_gate") != {
        "canonical_attempt_ledger_path": str(
            DEFAULT_DEVELOPMENT_ATTEMPT.relative_to(ROOT)
        ),
        "canonical_exploratory_path": str(DEFAULT_EXPLORATORY.relative_to(ROOT)),
        "attempt_reserved_before_calibration_access": True,
        "existing_attempt_blocks_retry": True,
        "base_exception_burns_attempt": True,
    }:
        raise ValueError("v7 development-gate contract changed")
    if protocol.get("one_shot_final") != {
        "confirmation": f"{ONE_SHOT_FINAL_CONFIRMATION}:<protocol_sha256>",
        "canonical_receipt_path": str(DEFAULT_SEAL_RECEIPT.relative_to(ROOT)),
        "canonical_output_path": str(DEFAULT_OUTPUT.relative_to(ROOT)),
        "canonical_completion_seal_path": str(
            DEFAULT_COMPLETION_SEAL.relative_to(ROOT)
        ),
        "canonical_source_path": str(DEFAULT_FINAL_REFERENCE.relative_to(ROOT)),
        "receipt_before_final_transaction_sheet_parse_or_score": True,
        "receipt_bytes_revalidated_immediately_before_final_loader": True,
        "module_runtime_boundary_not_process_wide_capability_security": True,
        "final_written_before_completion_seal": True,
        "completion_seal_binds_exact_final_bytes": True,
        "completion_seal_is_tamper_evidence_not_signature_or_worm": True,
        "external_signed_or_worm_custody_is_deployment_responsibility": True,
        "external_signed_or_worm_custody_is_release_prerequisite": True,
        "crash_burns_attempt": True,
        "second_invocation_fails": True,
    }:
        raise ValueError("v7 one-shot-final contract changed")
    expected_privacy = {
        "artifact_policy": "fixed aggregates only",
        "contains_customer_ids": False,
        "contains_raw_rows": False,
        "contains_rank_rows": False,
        "contains_evaluator_feature_rows": False,
        "contains_prediction_vectors": False,
        "differential_privacy_claimed": False,
    }
    if protocol.get("privacy") != expected_privacy:
        raise ValueError("v7 privacy contract changed")
    _assert_aggregate_only(protocol)
    return protocol


def _synthetic_group(
    profile: BehaviorProfileV7, *, customers: int, seed: int, prefix: str
) -> Any:
    frame = generate_synthetic_transactions_v7(
        profile=profile, customers=customers, seed=seed
    )
    frame["customer_id"] = prefix + frame["customer_id"].astype(str)
    return frame


def _crossfit_seed(
    real_calibration: Any, *, seed: int, fold_numbers: Iterable[int] | None = None
) -> list[dict[str, Any]]:
    """Fit the V7 profile, including q, independently inside every training fold."""
    np, pd, _, roc_auc_score = _dependencies()
    customer_ids = sorted(set(real_calibration["customer_id"]))
    fold_by_id = {
        customer_id: int.from_bytes(
            hashlib.sha256(
                f"{seed}:uci-v6-crossfit:{customer_id}".encode()
            ).digest()[:8],
            "big",
        )
        % CROSS_FIT_FOLDS
        for customer_id in customer_ids
    }
    folds = []
    selected_folds = fold_numbers if fold_numbers is not None else range(CROSS_FIT_FOLDS)
    for fold in selected_folds:
        evaluation_ids = {
            customer_id
            for customer_id, assigned in fold_by_id.items()
            if assigned == fold
        }
        training_ids = set(customer_ids) - evaluation_ids
        if not training_ids or not evaluation_ids:
            raise ValueError("v7 exploratory cross-fit fold is empty")
        real_training = real_calibration.loc[
            real_calibration["customer_id"].isin(training_ids)
        ].copy()
        real_evaluation = real_calibration.loc[
            real_calibration["customer_id"].isin(evaluation_ids)
        ].copy()
        profile = fit_behavior_profile_v7(
            real_training, customer_ids_sha256=_ids_hash(training_ids)
        )
        synthetic_training = _synthetic_group(
            profile,
            customers=len(training_ids),
            seed=seed + 10_000 * fold + 101,
            prefix=f"v7-train-{seed}-{fold}-",
        )
        synthetic_evaluation = _synthetic_group(
            profile,
            customers=len(evaluation_ids),
            seed=seed + 10_000 * fold + 202,
            prefix=f"v7-eval-{seed}-{fold}-",
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
        labels = np.concatenate(
            (np.zeros(len(real_predictions)), np.ones(len(synthetic_predictions)))
        )
        auc = float(
            roc_auc_score(
                labels, np.concatenate((real_predictions, synthetic_predictions))
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
                "training_repeat_customers": profile.repeat_customer_count,
                "training_multiday_repeat_customers": profile.strata[1].customer_count,
                "training_same_day_repeat_probability": (
                    profile.same_day_repeat_probability
                ),
                "auc": auc,
                "d": max(auc, 1 - auc),
                "_real_customer_ids": tuple(real_evaluation_features.index),
                "_real_predictions": real_predictions,
                "_synthetic_predictions": synthetic_predictions,
            }
        )
    return folds


def _compute_calibration_exploratory(
    real_calibration: Any,
    reference: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    np, _, _, _ = _dependencies()
    profile = fit_behavior_profile_v7(
        real_calibration,
        customer_ids_sha256=_ids_hash(real_calibration["customer_id"].unique()),
    )
    if profile.sha256 != protocol["generator"]["behavior_profile_sha256"]:
        raise ValueError("all-calibration v7 profile differs from the frozen protocol")
    representative = _crossfit_seed(
        real_calibration, seed=FUTILITY_SEED, fold_numbers=(FUTILITY_FOLD,)
    )[0]
    futility = _representative_futility(representative)
    reference_result = {
        "workbook_sha256": reference["workbook_sha256"],
        "partition": "calibration",
        "customer_ids_sha256": EXPECTED_PARTITION_HASHES["calibration"],
        "customers": int(real_calibration["customer_id"].nunique()),
        "transactions": len(real_calibration),
    }
    if futility["passed_to_full_gate"] is not True:
        return {
            "schema_version": 7,
            "artifact": "entity_fidelity_v7_calibration_exploratory",
            "protocol_sha256": protocol["protocol_sha256"],
            "calibration_only": True,
            "uci352_diagnostic_transactions_materialized_or_used": False,
            "uci502_transactions_accessed": False,
            "uci352_prior_holdout_transactions_materialized_or_used": False,
            "reference": reference_result,
            "futility": futility,
            "seeds": {},
            "metrics": {},
            "gate_passed": False,
        }
    seed_folds = {
        seed: (
            sorted(
                [
                    representative,
                    *_crossfit_seed(
                        real_calibration,
                        seed=seed,
                        fold_numbers=(0, 1, 3, 4),
                    ),
                ],
                key=lambda fold: fold["fold"],
            )
            if seed == FUTILITY_SEED
            else _crossfit_seed(real_calibration, seed=seed)
        )
        for seed in EVALUATION_SEEDS
    }
    seed_point = {
        str(seed): sum(
            fold["d"] * fold["evaluation_real_customers"] for fold in folds
        )
        / sum(fold["evaluation_real_customers"] for fold in folds)
        for seed, folds in seed_folds.items()
    }
    bootstrap, ci_low, ci_high = _crossfit_max_seed_bootstrap(
        seed_folds, repeats=BOOTSTRAP_REPEATS, seed=SEED + 30
    )
    result = {
        "schema_version": 7,
        "artifact": "entity_fidelity_v7_calibration_exploratory",
        "protocol_sha256": protocol["protocol_sha256"],
        "calibration_only": True,
        "uci352_diagnostic_transactions_materialized_or_used": False,
        "uci502_transactions_accessed": False,
        "uci352_prior_holdout_transactions_materialized_or_used": False,
        "reference": reference_result,
        "futility": futility,
        "seeds": {
            str(seed): {
                "fold_weighted_d": seed_point[str(seed)],
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
            "orientation_free_max_seed_d": max(seed_point.values()),
            "bootstrap_max_seed_d_mean": float(np.mean(bootstrap)),
            "bootstrap_max_seed_d_ci_95": [ci_low, ci_high],
            "threshold_max_d_upper_95": GATE_MAX_D_UPPER_95,
        },
        "gate_passed": ci_high <= GATE_MAX_D_UPPER_95,
    }
    return result


def run_calibration_exploratory(
    reference_xlsx: Path = DEFAULT_REFERENCE,
    output: Path = DEFAULT_EXPLORATORY,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    if (
        not _same_absolute_path(reference_xlsx, DEFAULT_REFERENCE)
        or not _same_absolute_path(output, DEFAULT_EXPLORATORY)
        or not _same_absolute_path(protocol_path, DEFAULT_PROTOCOL)
    ):
        raise ValueError("v7 development gate paths are canonical")
    reference_xlsx = DEFAULT_REFERENCE
    output = DEFAULT_EXPLORATORY
    protocol_path = DEFAULT_PROTOCOL
    if DEFAULT_DEVELOPMENT_ATTEMPT.exists() or output.exists():
        raise FileExistsError("canonical V7 development gate was already consumed")
    protocol = load_frozen_v7_protocol(protocol_path)
    attempt = {
        "schema_version": 7,
        "artifact": "entity_fidelity_v7_development_attempt",
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "consumed",
        "calibration_access_reserved": True,
        "maximum_calibration_loader_calls": 1,
        "uci502_transactions_accessed": False,
    }
    try:
        _write_artifact(DEFAULT_DEVELOPMENT_ATTEMPT, attempt)
    except FileExistsError:
        raise FileExistsError(
            "canonical V7 development gate was already consumed"
        ) from None
    try:
        real_calibration, reference = load_retail_partition(
            reference_xlsx, "calibration"
        )
        result = _compute_calibration_exploratory(
            real_calibration, reference, protocol
        )
        if result.get("gate_passed") is not True:
            futility_failed = result["futility"]["passed_to_full_gate"] is False
            result = {
                **result,
                "artifact": "entity_fidelity_v7_calibration_exploratory_failure",
                "status": "STOP_FAIL",
                "calibration_access_attempted": True,
                "threshold_changed": False,
                "failure": {
                    "error_type": (
                        "futility_threshold_exceeded"
                        if futility_failed
                        else "gate_threshold_exceeded"
                    ),
                    "message": (
                        "preregistered futility lower bound exceeded the frozen threshold"
                        if futility_failed
                        else "development gate exceeded its frozen threshold"
                    ),
                },
            }
        _write_artifact(output, result)
        return result
    except BaseException as error:
        failure = {
            "schema_version": 7,
            "artifact": "entity_fidelity_v7_calibration_exploratory_failure",
            "protocol_sha256": protocol["protocol_sha256"],
            "status": "STOP_FAIL",
            "calibration_only": True,
            "calibration_access_attempted": True,
            "uci352_diagnostic_transactions_materialized_or_used": False,
            "uci502_transactions_accessed": False,
            "uci352_prior_holdout_transactions_materialized_or_used": False,
            "threshold_changed": False,
            "reference": {},
            "futility": {},
            "seeds": {},
            "metrics": {},
            "gate_passed": False,
            "failure": {
                "error_type": (
                    "interrupted"
                    if isinstance(error, (KeyboardInterrupt, SystemExit))
                    else "calibration_failure"
                ),
                "message": "development gate aborted after calibration access attempt",
            },
        }
        try:
            _write_artifact(output, failure)
        except BaseException:
            pass
        raise


def _exploratory_evidence_valid(result: Mapping[str, Any]) -> bool:
    seeds = result.get("seeds")
    if (
        not _futility_evidence_valid(result, require_pass=True)
        or not _calibration_reference_valid(result)
        or not isinstance(seeds, Mapping)
        or set(seeds) != {str(seed) for seed in EVALUATION_SEEDS}
    ):
        return False
    seed_points: list[float] = []
    representative_fold: Mapping[str, Any] | None = None
    fold_keys = {
        "fold",
        "training_real_customers",
        "evaluation_real_customers",
        "evaluation_synthetic_customers",
        "training_customer_ids_sha256",
        "evaluation_customer_ids_sha256",
        "profile_sha256",
        "training_repeat_customers",
        "training_multiday_repeat_customers",
        "training_same_day_repeat_probability",
        "auc",
        "d",
    }
    for seed in EVALUATION_SEEDS:
        seed_result = seeds.get(str(seed))
        if not isinstance(seed_result, Mapping) or set(seed_result) != {
            "fold_weighted_d",
            "folds",
        }:
            return False
        folds = seed_result.get("folds")
        if not isinstance(folds, list) or len(folds) != CROSS_FIT_FOLDS:
            return False
        weighted = 0.0
        evaluation_total = 0
        observed_folds: set[int] = set()
        for fold in folds:
            if not isinstance(fold, Mapping) or set(fold) != fold_keys:
                return False
            fold_number = fold.get("fold")
            training_count = fold.get("training_real_customers")
            evaluation_count = fold.get("evaluation_real_customers")
            synthetic_count = fold.get("evaluation_synthetic_customers")
            repeat_count = fold.get("training_repeat_customers")
            multiday_count = fold.get("training_multiday_repeat_customers")
            q = fold.get("training_same_day_repeat_probability")
            if (
                not isinstance(fold_number, int)
                or isinstance(fold_number, bool)
                or not 0 <= fold_number < CROSS_FIT_FOLDS
                or not all(
                    isinstance(item, int)
                    and not isinstance(item, bool)
                    and item > 0
                    for item in (training_count, evaluation_count, synthetic_count)
                )
                or training_count + evaluation_count
                != UCI352_CALIBRATION_CUSTOMERS
                or synthetic_count != evaluation_count
                or not isinstance(repeat_count, int)
                or isinstance(repeat_count, bool)
                or not isinstance(multiday_count, int)
                or isinstance(multiday_count, bool)
                or multiday_count < MIN_CHECKERBOARD_CUSTOMERS
                or repeat_count < multiday_count
                or training_count - repeat_count < MIN_CHECKERBOARD_CUSTOMERS
                or not _finite_number(q, lower=0.0, upper=1.0)
                or not math.isclose(
                    float(q),
                    (repeat_count - multiday_count) / repeat_count,
                    abs_tol=1e-12,
                )
                or not all(
                    _sha256_value(fold.get(key))
                    for key in (
                        "training_customer_ids_sha256",
                        "evaluation_customer_ids_sha256",
                        "profile_sha256",
                    )
                )
                or fold.get("training_customer_ids_sha256")
                == fold.get("evaluation_customer_ids_sha256")
                or not _orientation_row_valid(fold)
            ):
                return False
            observed_folds.add(fold_number)
            if seed == FUTILITY_SEED and fold_number == FUTILITY_FOLD:
                representative_fold = fold
            weighted += evaluation_count * float(fold["d"])
            evaluation_total += evaluation_count
        point = seed_result.get("fold_weighted_d")
        if (
            observed_folds != set(range(CROSS_FIT_FOLDS))
            or evaluation_total != UCI352_CALIBRATION_CUSTOMERS
            or not _finite_number(point, lower=0.5, upper=1.0)
            or not math.isclose(
                float(point), weighted / evaluation_total, abs_tol=1e-12
            )
        ):
            return False
        seed_points.append(float(point))
    futility = result["futility"]
    if representative_fold is None or any(
        futility[key] != representative_fold[fold_key]
        for key, fold_key in (
            ("real_customers", "evaluation_real_customers"),
            ("synthetic_customers", "evaluation_synthetic_customers"),
            ("auc", "auc"),
            ("d", "d"),
        )
    ):
        return False
    return _gate_metrics_valid(result, point_values=seed_points)


def _parse_passed_exploratory(
    raw_bytes: bytes, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    result = json.loads(raw_bytes)
    if (
        not isinstance(result, Mapping)
        or set(result) != PASSED_EXPLORATORY_KEYS
        or result.get("schema_version") != 7
        or result.get("artifact")
        != "entity_fidelity_v7_calibration_exploratory"
        or result.get("protocol_sha256") != protocol["protocol_sha256"]
        or not _exploratory_evidence_valid(result)
        or result.get("gate_passed") is not True
        or result.get("calibration_only") is not True
        or result.get("uci352_diagnostic_transactions_materialized_or_used")
        is not False
        or result.get("uci502_transactions_accessed") is not False
        or result.get("uci352_prior_holdout_transactions_materialized_or_used")
        is not False
    ):
        raise ValueError("v7 exploratory gate is absent, failed, or not protocol-bound")
    _assert_aggregate_only(result)
    return dict(result)


def _load_passed_exploratory(
    path: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    return _parse_passed_exploratory(_read_regular_file_bytes(path), protocol)


def _final_seed_predictions(
    real_calibration: Any,
    historical_transform: Any,
    historical_score: Any,
    profile: BehaviorProfileV7,
    *,
    seed: int,
) -> dict[str, Any]:
    np, pd, _, roc_auc_score = _dependencies()
    training_customers = int(real_calibration["customer_id"].nunique())
    evaluation_customers = int(historical_score["customer_id"].nunique())
    synthetic_training = _synthetic_group(
        profile,
        customers=training_customers,
        seed=seed + 500_001,
        prefix=f"v7-final-train-{seed}-",
    )
    synthetic_evaluation = _synthetic_group(
        profile,
        customers=evaluation_customers,
        seed=seed + 500_002,
        prefix=f"v7-final-eval-{seed}-",
    )
    real_transform, synthetic_transform, _ = fit_calibrations(
        real_calibration, synthetic_training
    )
    real_training_features = customer_features(real_calibration, real_transform)
    synthetic_training_features = customer_features(
        synthetic_training, synthetic_transform
    )
    classifier = _classifier(seed)
    classifier.fit(
        pd.concat(
            (real_training_features, synthetic_training_features), ignore_index=True
        ),
        np.concatenate(
            (
                np.zeros(len(real_training_features), dtype="int8"),
                np.ones(len(synthetic_training_features), dtype="int8"),
            )
        ),
    )
    real_predictions = classifier.predict_proba(
        customer_features(historical_score, historical_transform)
    )[:, 1]
    synthetic_predictions = classifier.predict_proba(
        customer_features(synthetic_evaluation, synthetic_transform)
    )[:, 1]
    labels = np.concatenate(
        (np.zeros(len(real_predictions)), np.ones(len(synthetic_predictions)))
    )
    auc = float(
        roc_auc_score(labels, np.concatenate((real_predictions, synthetic_predictions)))
    )
    return {
        "auc": auc,
        "d": max(auc, 1 - auc),
        "real_customers": len(real_predictions),
        "synthetic_customers": len(synthetic_predictions),
        "_real_predictions": real_predictions,
        "_synthetic_predictions": synthetic_predictions,
    }


def _recompute_and_match_exploratory(
    protocol: Mapping[str, Any], exploratory_path: Path
) -> tuple[Any, dict[str, Any], dict[str, Any], str]:
    stored_bytes = _read_regular_file_bytes(exploratory_path)
    stored = _parse_passed_exploratory(stored_bytes, protocol)
    real_calibration, reference = load_retail_partition(
        DEFAULT_REFERENCE, "calibration"
    )
    recomputed = _compute_calibration_exploratory(
        real_calibration, reference, protocol
    )
    expected_bytes = _pretty_json_bytes(recomputed)
    if recomputed != stored or stored_bytes != expected_bytes:
        raise ValueError(
            "canonical exploratory artifact bytes differ from exact frozen recomputation"
        )
    return (
        real_calibration,
        reference,
        recomputed,
        hashlib.sha256(stored_bytes).hexdigest(),
    )


def _final_receipt_valid(
    receipt: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    exploratory_sha256: str,
    recomputed_exploratory_sha256: str,
    behavior_profile_sha256: str,
) -> bool:
    try:
        _assert_aggregate_only(receipt)
    except (TypeError, ValueError):
        return False
    expected_receipt_hash = _canonical_hash(
        {
            key: item
            for key, item in receipt.items()
            if key != "receipt_sha256"
        }
    )
    expected_confirmation_hash = hashlib.sha256(
        f"{ONE_SHOT_FINAL_CONFIRMATION}:{protocol['protocol_sha256']}".encode()
    ).hexdigest()
    return (
        set(receipt) == FINAL_RECEIPT_KEYS
        and all(
            _sha256_value(receipt.get(key))
            for key in FINAL_RECEIPT_KEYS
            if key.endswith("_sha256")
        )
        and receipt.get("schema_version") == 7
        and receipt.get("artifact")
        == "entity_fidelity_v7_one_shot_final_receipt"
        and receipt.get("protocol_sha256") == protocol.get("protocol_sha256")
        and receipt.get("status") == "consumed"
        and receipt.get("confirmation_sha256") == expected_confirmation_hash
        and receipt.get("exploratory_artifact_sha256") == exploratory_sha256
        and receipt.get("recomputed_exploratory_sha256")
        == recomputed_exploratory_sha256
        and receipt.get("behavior_profile_sha256") == behavior_profile_sha256
        and receipt.get("final_dataset_id") == UCI502_DATASET_ID
        and receipt.get("final_source_url") == UCI502_SOURCE_URL
        and receipt.get("final_archive_sha256") == UCI502_ARCHIVE_SHA256
        and receipt.get("final_workbook_sha256") == UCI502_WORKBOOK_SHA256
        and receipt.get("final_manifest_sha256") == UCI502_MANIFEST_SHA256
        and receipt.get("final_license") == UCI502_LICENSE
        and receipt.get("final_downloader_sha256")
        == sha256_file(ROOT / "scripts" / "download_retail_ii_reference.py")
        and receipt.get("final_sheet") == UCI502_SHEET
        and receipt.get("final_fixed_window")
        == f"[{UCI502_WINDOW_START}, {UCI502_WINDOW_END})"
        and receipt.get("final_split_assignment") == UCI502_SPLIT_ASSIGNMENT
        and receipt.get("excluded_uci352_calibration_customer_ids_sha256")
        == EXPECTED_PARTITION_HASHES["calibration"]
        and receipt.get(
            "receipt_before_final_transaction_sheet_parse_or_score"
        )
        is True
        and receipt.get("receipt_sha256") == expected_receipt_hash
    )


def run_entity_fidelity_v7_final(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    exploratory_path: Path = DEFAULT_EXPLORATORY,
    one_shot_final_confirmation: str | None = None,
) -> dict[str, Any]:
    np, pd, _, _ = _dependencies()
    if (
        not _same_absolute_path(protocol_path, DEFAULT_PROTOCOL)
        or not _same_absolute_path(exploratory_path, DEFAULT_EXPLORATORY)
    ):
        raise ValueError("one-shot final protocol and exploratory paths are canonical")
    protocol_path = DEFAULT_PROTOCOL
    exploratory_path = DEFAULT_EXPLORATORY
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError("canonical final entity-fidelity output already exists")
    if DEFAULT_SEAL_RECEIPT.exists():
        raise FileExistsError("canonical V7 one-shot final receipt already exists")
    if DEFAULT_COMPLETION_SEAL.exists():
        raise FileExistsError("canonical V7 completion seal already exists")
    protocol = load_frozen_v7_protocol(protocol_path)
    try:
        development_attempt = json.loads(
            _read_regular_file_bytes(DEFAULT_DEVELOPMENT_ATTEMPT)
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("canonical V7 development attempt is absent or invalid") from error
    if not isinstance(development_attempt, Mapping) or not _development_attempt_valid(
        development_attempt, protocol
    ):
        raise ValueError("canonical V7 development attempt is absent or invalid")
    expected_confirmation = (
        f"{ONE_SHOT_FINAL_CONFIRMATION}:{protocol['protocol_sha256']}"
    )
    if one_shot_final_confirmation != expected_confirmation:
        raise ValueError("one-shot final requires protocol-hash-bound approval")
    (
        real_calibration,
        _,
        recomputed_exploratory,
        exploratory_artifact_sha256,
    ) = (
        _recompute_and_match_exploratory(protocol, exploratory_path)
    )
    profile = fit_behavior_profile_v7(
        real_calibration,
        customer_ids_sha256=_ids_hash(real_calibration["customer_id"].unique()),
    )
    if profile.sha256 != protocol["generator"]["behavior_profile_sha256"]:
        raise ValueError("all-calibration V7 profile differs from frozen protocol")
    receipt = {
        "schema_version": 7,
        "artifact": "entity_fidelity_v7_one_shot_final_receipt",
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "consumed",
        "confirmation_sha256": hashlib.sha256(
            one_shot_final_confirmation.encode()
        ).hexdigest(),
        "exploratory_artifact_sha256": exploratory_artifact_sha256,
        "recomputed_exploratory_sha256": _canonical_hash(
            recomputed_exploratory
        ),
        "behavior_profile_sha256": profile.sha256,
        "final_dataset_id": UCI502_DATASET_ID,
        "final_source_url": UCI502_SOURCE_URL,
        "final_archive_sha256": UCI502_ARCHIVE_SHA256,
        "final_workbook_sha256": UCI502_WORKBOOK_SHA256,
        "final_manifest_sha256": UCI502_MANIFEST_SHA256,
        "final_license": UCI502_LICENSE,
        "final_downloader_sha256": sha256_file(
            ROOT / "scripts" / "download_retail_ii_reference.py"
        ),
        "final_sheet": UCI502_SHEET,
        "final_fixed_window": f"[{UCI502_WINDOW_START}, {UCI502_WINDOW_END})",
        "final_split_assignment": UCI502_SPLIT_ASSIGNMENT,
        "excluded_uci352_calibration_customer_ids_sha256": (
            EXPECTED_PARTITION_HASHES["calibration"]
        ),
        "receipt_before_final_transaction_sheet_parse_or_score": True,
    }
    receipt["receipt_sha256"] = _canonical_hash(receipt)
    receipt_bytes = _write_artifact(DEFAULT_SEAL_RECEIPT, receipt)

    calibration_ids = set(real_calibration["customer_id"].astype(str))
    if _ids_hash(calibration_ids) != EXPECTED_PARTITION_HASHES["calibration"]:
        raise ValueError("V7 calibration exclusion set differs from its pinned hash")
    persisted_receipt_bytes = _read_regular_file_bytes(DEFAULT_SEAL_RECEIPT)
    persisted_receipt = json.loads(persisted_receipt_bytes)
    if (
        persisted_receipt_bytes != receipt_bytes
        or not isinstance(persisted_receipt, Mapping)
        or dict(persisted_receipt) != receipt
        or not _final_receipt_valid(
            persisted_receipt,
            protocol,
            exploratory_sha256=exploratory_artifact_sha256,
            recomputed_exploratory_sha256=_canonical_hash(recomputed_exploratory),
            behavior_profile_sha256=profile.sha256,
        )
    ):
        raise ValueError("canonical V7 receipt bytes changed before final access")
    from sim.entity_fidelity_v4 import load_retail_ii_readiness

    historical, provenance = load_retail_ii_readiness(
        DEFAULT_FINAL_REFERENCE,
        excluded_customer_ids=calibration_ids,
    )
    if calibration_ids & set(historical["customer_id"].astype(str)):
        raise ValueError("UCI502 final contains a UCI352 calibration customer")
    if (
        provenance.get("workbook_sha256") != UCI502_WORKBOOK_SHA256
        or provenance.get("manifest_sha256") != UCI502_MANIFEST_SHA256
        or provenance.get("sheet") != UCI502_SHEET
        or provenance.get("second_worksheet_rows_iterated") is not False
        or provenance.get("nonselected_transactions_materialized") is not False
    ):
        raise ValueError("UCI502 one-shot final provenance changed")
    transform_ids, score_ids = uci502_final_split(
        historical["customer_id"].unique()
    )
    eligible_ids = set(historical["customer_id"].astype(str))
    if transform_ids & score_ids or transform_ids | score_ids != eligible_ids:
        raise ValueError("UCI502 deterministic final split is not an exact partition")
    transform = historical.loc[
        historical["customer_id"].isin(transform_ids)
    ].copy()
    score = historical.loc[historical["customer_id"].isin(score_ids)].copy()
    historical_transform = DomainCalibration(
        amount_log_reference=np.sort(
            np.log1p(transform["amount"].to_numpy(dtype=float))
        ),
        window_start=pd.Timestamp(UCI502_WINDOW_START),
        window_end=pd.Timestamp(UCI502_WINDOW_END),
        time_scale=1.0,
    )
    seed_predictions = {
        seed: _final_seed_predictions(
            real_calibration,
            historical_transform,
            score,
            profile,
            seed=seed,
        )
        for seed in EVALUATION_SEEDS
    }
    bootstrap, ci_low, ci_high = _max_seed_bootstrap(
        seed_predictions, repeats=BOOTSTRAP_REPEATS, seed=SEED + 32
    )
    result = {
        "schema_version": 7,
        "artifact": "entity_fidelity_v7_uci502_one_shot_final",
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol": {
            "protocol_sha256": protocol["protocol_sha256"],
            "evaluation_partition": "uci502_score_half",
            "one_shot_final_scored": True,
            "development_gate_recomputed_and_exactly_matched": True,
            "orientation_free_statistic": "D=max(AUC,1-AUC)",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "shared_real_bootstrap_indices_across_seeds": True,
            "independent_synthetic_bootstrap_indices_across_seeds": True,
        },
        "bindings": {
            "exploratory_artifact_sha256": exploratory_artifact_sha256,
            "final_receipt_sha256": receipt["receipt_sha256"],
            "behavior_profile_sha256": profile.sha256,
        },
        "behavior_profile": profile.as_dict(),
        "reference": {
            "development_training": {
                "dataset_id": UCI352_DATASET_ID,
                "source_url": UCI352_SOURCE_URL,
                "workbook_sha256": UCI352_WORKBOOK_SHA256,
                "manifest_sha256": UCI352_MANIFEST_SHA256,
                "license": UCI352_LICENSE,
                "downloader_sha256": sha256_file(
                    ROOT / "scripts" / "download_retail_reference.py"
                ),
                "calibration_customer_ids_sha256": EXPECTED_PARTITION_HASHES[
                    "calibration"
                ],
                "calibration_customers": int(
                    real_calibration["customer_id"].nunique()
                ),
                "full_workbook_customer_ids_scanned_for_routing": True,
                "noncalibration_transaction_values_materialized_or_used": False,
                "uci352_diagnostic_transactions_materialized_or_used": False,
                "uci352_prior_holdout_transactions_materialized_or_used": False,
            },
            "one_shot_final": {
                "dataset_id": UCI502_DATASET_ID,
                "source_url": UCI502_SOURCE_URL,
                "archive_sha256": UCI502_ARCHIVE_SHA256,
                "workbook_sha256": provenance["workbook_sha256"],
                "manifest_sha256": provenance["manifest_sha256"],
                "license": UCI502_LICENSE,
                "downloader_sha256": sha256_file(
                    ROOT / "scripts" / "download_retail_ii_reference.py"
                ),
                "sheet": UCI502_SHEET,
                "fixed_window": f"[{UCI502_WINDOW_START}, {UCI502_WINDOW_END})",
                "split_assignment": UCI502_SPLIT_ASSIGNMENT,
                "excluded_uci352_calibration_customer_ids_sha256": (
                    EXPECTED_PARTITION_HASHES["calibration"]
                ),
                "transaction_time_disjoint": True,
                "customer_disjoint_from_profile_and_classifier_training": True,
                "same_retailer": True,
                "source_independent": False,
                "temporal_direction": "historical_before_training_period",
                "prospective_validation": False,
                "transaction_sheet_parsed_or_scored_before_receipt": False,
                "first_worksheet_opened_after_receipt": True,
                "second_worksheet_opened": False,
                "nonselected_transactions_materialized": False,
                "eligible_customers": len(eligible_ids),
                "eligible_customer_ids_sha256": _ids_hash(eligible_ids),
                "transform_customers": len(transform_ids),
                "score_customers": len(score_ids),
                "transform_customer_ids_sha256": _ids_hash(transform_ids),
                "score_customer_ids_sha256": _ids_hash(score_ids),
                "transform_amount_ecdf_fit_on_transform_only": True,
                "transform_used_for_profile_generator_classifier_threshold_or_abort": False,
                "score_half_is_only_real_final_evaluation": True,
            },
        },
        "seeds": {
            str(seed): {
                key: value
                for key, value in prediction.items()
                if not key.startswith("_")
            }
            for seed, prediction in seed_predictions.items()
        },
        "metrics": {
            "orientation_free_max_seed_d": max(
                prediction["d"] for prediction in seed_predictions.values()
            ),
            "bootstrap_max_seed_d_mean": float(np.mean(bootstrap)),
            "bootstrap_max_seed_d_ci_95": [ci_low, ci_high],
            "threshold_max_d_upper_95": GATE_MAX_D_UPPER_95,
            "gate_passed": ci_high <= GATE_MAX_D_UPPER_95,
        },
    }
    final_bytes = _write_artifact(DEFAULT_OUTPUT, result)
    persisted_final_bytes = _read_regular_file_bytes(DEFAULT_OUTPUT)
    if persisted_final_bytes != final_bytes:
        raise ValueError("canonical V7 final bytes changed before completion sealing")
    completion = {
        "schema_version": 7,
        "artifact": "entity_fidelity_v7_completion_seal",
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "complete",
        "final_artifact_sha256": hashlib.sha256(persisted_final_bytes).hexdigest(),
        "final_receipt_sha256": receipt["receipt_sha256"],
    }
    _write_artifact(DEFAULT_COMPLETION_SEAL, completion)
    return result


def _final_gate_metrics_valid(
    value: Mapping[str, Any], *, require_pass: bool = True
) -> bool:
    metrics = value.get("metrics")
    seeds = value.get("seeds")
    reference = value.get("reference", {}).get("one_shot_final", {})
    if (
        not isinstance(metrics, Mapping)
        or set(metrics)
        != {
            "orientation_free_max_seed_d",
            "bootstrap_max_seed_d_mean",
            "bootstrap_max_seed_d_ci_95",
            "threshold_max_d_upper_95",
            "gate_passed",
        }
        or not isinstance(seeds, Mapping)
        or set(seeds) != {str(seed) for seed in EVALUATION_SEEDS}
        or not isinstance(reference.get("score_customers"), int)
        or isinstance(reference.get("score_customers"), bool)
        or reference["score_customers"] < UCI502_MIN_SPLIT_CUSTOMERS
    ):
        return False
    if require_pass and metrics.get("gate_passed") is not True:
        return False
    points = []
    for seed in EVALUATION_SEEDS:
        prediction = seeds[str(seed)]
        if (
            not isinstance(prediction, Mapping)
            or set(prediction)
            != {"auc", "d", "real_customers", "synthetic_customers"}
            or prediction.get("real_customers")
            != reference["score_customers"]
            or prediction.get("synthetic_customers")
            != reference["score_customers"]
            or not _orientation_row_valid(prediction)
        ):
            return False
        points.append(float(prediction["d"]))
    return _gate_metrics_valid(
        {
            "gate_passed": metrics["gate_passed"],
            "metrics": {
                key: item for key, item in metrics.items() if key != "gate_passed"
            },
        },
        point_values=points,
    )


def _final_result_valid(
    value: Mapping[str, Any],
    *,
    expected_protocol_sha256: Any,
    expected_exploratory_sha256: Any,
    expected_receipt_sha256: Any,
    expected_behavior_profile_sha256: Any,
    require_pass: bool,
) -> bool:
    if not all(
        _sha256_value(item)
        for item in (
            expected_protocol_sha256,
            expected_exploratory_sha256,
            expected_receipt_sha256,
            expected_behavior_profile_sha256,
        )
    ):
        return False
    if set(value) != FINAL_RESULT_KEYS:
        return False
    containers = (
        "protocol",
        "bindings",
        "behavior_profile",
        "reference",
        "seeds",
        "metrics",
    )
    if not all(isinstance(value.get(key), Mapping) for key in containers):
        return False
    try:
        _assert_aggregate_only(value)
        profile = BehaviorProfileV7.from_dict(value["behavior_profile"])
    except (KeyError, TypeError, ValueError):
        return False

    protocol = value["protocol"]
    bindings = value["bindings"]
    reference = value["reference"]
    development = reference.get("development_training")
    one_shot = reference.get("one_shot_final")
    if not isinstance(development, Mapping) or not isinstance(one_shot, Mapping):
        return False

    split_counts_valid = (
        isinstance(one_shot.get("eligible_customers"), int)
        and not isinstance(one_shot.get("eligible_customers"), bool)
        and isinstance(one_shot.get("transform_customers"), int)
        and not isinstance(one_shot.get("transform_customers"), bool)
        and isinstance(one_shot.get("score_customers"), int)
        and not isinstance(one_shot.get("score_customers"), bool)
        and one_shot.get("transform_customers", 0) >= UCI502_MIN_SPLIT_CUSTOMERS
        and one_shot.get("score_customers", 0) >= UCI502_MIN_SPLIT_CUSTOMERS
        and one_shot.get("eligible_customers")
        == one_shot.get("transform_customers") + one_shot.get("score_customers")
        and all(
            _sha256_value(one_shot.get(key))
            for key in (
                "eligible_customer_ids_sha256",
                "transform_customer_ids_sha256",
                "score_customer_ids_sha256",
            )
        )
        and one_shot.get("transform_customer_ids_sha256")
        != one_shot.get("score_customer_ids_sha256")
    )
    return (
        value.get("schema_version") == 7
        and value.get("artifact")
        == "entity_fidelity_v7_uci502_one_shot_final"
        and value.get("protocol_sha256") == expected_protocol_sha256
        and set(protocol) == FINAL_PROTOCOL_KEYS
        and protocol.get("protocol_sha256") == expected_protocol_sha256
        and protocol.get("evaluation_partition") == "uci502_score_half"
        and protocol.get("one_shot_final_scored") is True
        and protocol.get("development_gate_recomputed_and_exactly_matched")
        is True
        and protocol.get("orientation_free_statistic") == "D=max(AUC,1-AUC)"
        and protocol.get("bootstrap_repeats") == BOOTSTRAP_REPEATS
        and protocol.get("shared_real_bootstrap_indices_across_seeds") is True
        and protocol.get("independent_synthetic_bootstrap_indices_across_seeds")
        is True
        and set(bindings) == FINAL_BINDING_KEYS
        and all(_sha256_value(bindings.get(key)) for key in FINAL_BINDING_KEYS)
        and bindings.get("exploratory_artifact_sha256")
        == expected_exploratory_sha256
        and bindings.get("final_receipt_sha256") == expected_receipt_sha256
        and bindings.get("behavior_profile_sha256")
        == expected_behavior_profile_sha256
        and profile.sha256 == expected_behavior_profile_sha256
        and profile.as_dict() == value["behavior_profile"]
        and set(reference) == FINAL_REFERENCE_KEYS
        and set(development) == FINAL_DEVELOPMENT_REFERENCE_KEYS
        and development.get("dataset_id") == UCI352_DATASET_ID
        and development.get("source_url") == UCI352_SOURCE_URL
        and development.get("workbook_sha256") == UCI352_WORKBOOK_SHA256
        and development.get("manifest_sha256") == UCI352_MANIFEST_SHA256
        and development.get("license") == UCI352_LICENSE
        and development.get("downloader_sha256")
        == sha256_file(ROOT / "scripts" / "download_retail_reference.py")
        and development.get("calibration_customer_ids_sha256")
        == EXPECTED_PARTITION_HASHES["calibration"]
        and development.get("calibration_customers")
        == UCI352_CALIBRATION_CUSTOMERS
        and development.get("full_workbook_customer_ids_scanned_for_routing")
        is True
        and development.get(
            "noncalibration_transaction_values_materialized_or_used"
        )
        is False
        and development.get(
            "uci352_diagnostic_transactions_materialized_or_used"
        )
        is False
        and development.get(
            "uci352_prior_holdout_transactions_materialized_or_used"
        )
        is False
        and set(one_shot) == FINAL_ONE_SHOT_REFERENCE_KEYS
        and one_shot.get("dataset_id") == UCI502_DATASET_ID
        and one_shot.get("source_url") == UCI502_SOURCE_URL
        and one_shot.get("archive_sha256") == UCI502_ARCHIVE_SHA256
        and one_shot.get("workbook_sha256") == UCI502_WORKBOOK_SHA256
        and one_shot.get("manifest_sha256") == UCI502_MANIFEST_SHA256
        and one_shot.get("license") == UCI502_LICENSE
        and one_shot.get("downloader_sha256")
        == sha256_file(ROOT / "scripts" / "download_retail_ii_reference.py")
        and one_shot.get("sheet") == UCI502_SHEET
        and one_shot.get("fixed_window")
        == f"[{UCI502_WINDOW_START}, {UCI502_WINDOW_END})"
        and one_shot.get("split_assignment") == UCI502_SPLIT_ASSIGNMENT
        and one_shot.get("excluded_uci352_calibration_customer_ids_sha256")
        == EXPECTED_PARTITION_HASHES["calibration"]
        and one_shot.get("transaction_time_disjoint") is True
        and one_shot.get("customer_disjoint_from_profile_and_classifier_training")
        is True
        and one_shot.get("same_retailer") is True
        and one_shot.get("source_independent") is False
        and one_shot.get("temporal_direction")
        == "historical_before_training_period"
        and one_shot.get("prospective_validation") is False
        and one_shot.get("transaction_sheet_parsed_or_scored_before_receipt")
        is False
        and one_shot.get("first_worksheet_opened_after_receipt") is True
        and one_shot.get("second_worksheet_opened") is False
        and one_shot.get("nonselected_transactions_materialized") is False
        and one_shot.get("transform_amount_ecdf_fit_on_transform_only") is True
        and one_shot.get(
            "transform_used_for_profile_generator_classifier_threshold_or_abort"
        )
        is False
        and one_shot.get("score_half_is_only_real_final_evaluation") is True
        and split_counts_valid
        and _final_gate_metrics_valid(value, require_pass=require_pass)
    )


def verify_v7_artifacts(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    exploratory_path: Path = DEFAULT_EXPLORATORY,
    attempt_path: Path = DEFAULT_DEVELOPMENT_ATTEMPT,
    receipt_path: Path = DEFAULT_SEAL_RECEIPT,
    final_path: Path = DEFAULT_OUTPUT,
    seal_path: Path = DEFAULT_COMPLETION_SEAL,
) -> dict[str, Any]:
    supplied = (
        protocol_path,
        exploratory_path,
        attempt_path,
        receipt_path,
        final_path,
        seal_path,
    )
    canonical = (
        DEFAULT_PROTOCOL,
        DEFAULT_EXPLORATORY,
        DEFAULT_DEVELOPMENT_ATTEMPT,
        DEFAULT_SEAL_RECEIPT,
        DEFAULT_OUTPUT,
        DEFAULT_COMPLETION_SEAL,
    )
    if any(
        not _same_absolute_path(candidate, expected)
        for candidate, expected in zip(supplied, canonical, strict=True)
    ):
        raise ValueError("v7 artifact verification paths are canonical")
    (
        protocol_path,
        exploratory_path,
        attempt_path,
        receipt_path,
        final_path,
        seal_path,
    ) = canonical
    protocol = load_frozen_v7_protocol(protocol_path)
    checks = {
        "attempt_exists": attempt_path.exists(),
        "attempt_valid": False,
        "exploratory_exists": exploratory_path.exists(),
        "exploratory_valid": False,
        "receipt_exists": receipt_path.exists(),
        "receipt_valid": False,
        "final_exists": final_path.exists(),
        "final_valid": False,
        "completion_seal_exists": seal_path.exists(),
        "completion_seal_valid": False,
        "one_shot_artifacts_complete": (
            receipt_path.exists() and final_path.exists() and seal_path.exists()
        ),
    }
    if attempt_path.exists():
        try:
            attempt = json.loads(_read_regular_file_bytes(attempt_path))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        else:
            checks["attempt_valid"] = isinstance(
                attempt, Mapping
            ) and _development_attempt_valid(attempt, protocol)
    exploratory: Mapping[str, Any] | None = None
    exploratory_sha256: str | None = None
    if exploratory_path.exists():
        try:
            exploratory_bytes = _read_regular_file_bytes(exploratory_path)
            exploratory = _parse_passed_exploratory(
                exploratory_bytes, protocol
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        else:
            checks["exploratory_valid"] = True
            exploratory_sha256 = hashlib.sha256(exploratory_bytes).hexdigest()
    receipt: Mapping[str, Any] | None = None
    if (
        receipt_path.exists()
        and exploratory is not None
        and exploratory_sha256 is not None
    ):
        try:
            loaded_receipt = json.loads(_read_regular_file_bytes(receipt_path))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        else:
            checks["receipt_valid"] = isinstance(
                loaded_receipt, Mapping
            ) and _final_receipt_valid(
                loaded_receipt,
                protocol,
                exploratory_sha256=exploratory_sha256,
                recomputed_exploratory_sha256=_canonical_hash(exploratory),
                behavior_profile_sha256=protocol["generator"][
                    "behavior_profile_sha256"
                ],
            )
            if checks["receipt_valid"]:
                receipt = loaded_receipt
    final_bytes: bytes | None = None
    if final_path.exists() and receipt is not None and exploratory is not None:
        try:
            final_bytes = _read_regular_file_bytes(final_path)
            final = json.loads(final_bytes)
            if not isinstance(final, Mapping):
                raise ValueError("one-shot final artifact must be a JSON object")
            if not all(
                isinstance(final.get(key), Mapping)
                for key in (
                    "protocol",
                    "bindings",
                    "behavior_profile",
                    "reference",
                    "seeds",
                    "metrics",
                )
            ):
                raise ValueError("one-shot final aggregate containers are malformed")
            _assert_aggregate_only(final)
            profile = BehaviorProfileV7.from_dict(final["behavior_profile"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        else:
            development = final.get("reference", {}).get(
                "development_training", {}
            )
            one_shot = final.get("reference", {}).get("one_shot_final", {})
            development_keys = {
                "dataset_id",
                "source_url",
                "workbook_sha256",
                "manifest_sha256",
                "license",
                "downloader_sha256",
                "calibration_customer_ids_sha256",
                "calibration_customers",
                "full_workbook_customer_ids_scanned_for_routing",
                "noncalibration_transaction_values_materialized_or_used",
                "uci352_diagnostic_transactions_materialized_or_used",
                "uci352_prior_holdout_transactions_materialized_or_used",
            }
            one_shot_keys = {
                "dataset_id",
                "source_url",
                "archive_sha256",
                "workbook_sha256",
                "manifest_sha256",
                "license",
                "downloader_sha256",
                "sheet",
                "fixed_window",
                "split_assignment",
                "excluded_uci352_calibration_customer_ids_sha256",
                "transaction_time_disjoint",
                "customer_disjoint_from_profile_and_classifier_training",
                "same_retailer",
                "source_independent",
                "temporal_direction",
                "prospective_validation",
                "transaction_sheet_parsed_or_scored_before_receipt",
                "first_worksheet_opened_after_receipt",
                "second_worksheet_opened",
                "nonselected_transactions_materialized",
                "eligible_customers",
                "eligible_customer_ids_sha256",
                "transform_customers",
                "score_customers",
                "transform_customer_ids_sha256",
                "score_customer_ids_sha256",
                "transform_amount_ecdf_fit_on_transform_only",
                "transform_used_for_profile_generator_classifier_threshold_or_abort",
                "score_half_is_only_real_final_evaluation",
            }
            split_counts_valid = (
                isinstance(one_shot.get("eligible_customers"), int)
                and not isinstance(one_shot.get("eligible_customers"), bool)
                and isinstance(one_shot.get("transform_customers"), int)
                and not isinstance(one_shot.get("transform_customers"), bool)
                and isinstance(one_shot.get("score_customers"), int)
                and not isinstance(one_shot.get("score_customers"), bool)
                and one_shot.get("transform_customers", 0)
                >= UCI502_MIN_SPLIT_CUSTOMERS
                and one_shot.get("score_customers", 0)
                >= UCI502_MIN_SPLIT_CUSTOMERS
                and one_shot.get("eligible_customers")
                == one_shot.get("transform_customers")
                + one_shot.get("score_customers")
                and all(
                    _sha256_value(one_shot.get(key))
                    for key in (
                        "eligible_customer_ids_sha256",
                        "transform_customer_ids_sha256",
                        "score_customer_ids_sha256",
                    )
                )
                and one_shot.get("transform_customer_ids_sha256")
                != one_shot.get("score_customer_ids_sha256")
            )
            checks["final_valid"] = (
                set(final) == FINAL_RESULT_KEYS
                and final.get("schema_version") == 7
                and final.get("artifact")
                == "entity_fidelity_v7_uci502_one_shot_final"
                and final.get("protocol_sha256") == protocol["protocol_sha256"]
                and final.get("protocol", {}).get("protocol_sha256")
                == protocol["protocol_sha256"]
                and final.get("protocol", {}).get("evaluation_partition")
                == "uci502_score_half"
                and final.get("protocol", {}).get("one_shot_final_scored") is True
                and final.get("protocol", {}).get(
                    "development_gate_recomputed_and_exactly_matched"
                )
                is True
                and final.get("protocol", {}).get("orientation_free_statistic")
                == "D=max(AUC,1-AUC)"
                and final.get("protocol", {}).get("bootstrap_repeats")
                == BOOTSTRAP_REPEATS
                and final.get("protocol", {}).get(
                    "shared_real_bootstrap_indices_across_seeds"
                )
                is True
                and final.get("protocol", {}).get(
                    "independent_synthetic_bootstrap_indices_across_seeds"
                )
                is True
                and set(final.get("protocol", {}))
                == {
                    "protocol_sha256",
                    "evaluation_partition",
                    "one_shot_final_scored",
                    "development_gate_recomputed_and_exactly_matched",
                    "orientation_free_statistic",
                    "bootstrap_repeats",
                    "shared_real_bootstrap_indices_across_seeds",
                    "independent_synthetic_bootstrap_indices_across_seeds",
                }
                and set(final.get("bindings", {}))
                == {
                    "exploratory_artifact_sha256",
                    "final_receipt_sha256",
                    "behavior_profile_sha256",
                }
                and final.get("bindings", {}).get(
                    "exploratory_artifact_sha256"
                )
                == exploratory_sha256
                and final.get("bindings", {}).get("final_receipt_sha256")
                == receipt.get("receipt_sha256")
                and final.get("bindings", {}).get("behavior_profile_sha256")
                == profile.sha256
                and profile.sha256
                == protocol["generator"]["behavior_profile_sha256"]
                and profile.as_dict() == final.get("behavior_profile")
                and set(final.get("reference", {}))
                == {"development_training", "one_shot_final"}
                and set(development) == development_keys
                and development.get("dataset_id") == UCI352_DATASET_ID
                and development.get("source_url") == UCI352_SOURCE_URL
                and development.get("workbook_sha256")
                == UCI352_WORKBOOK_SHA256
                and development.get("manifest_sha256")
                == UCI352_MANIFEST_SHA256
                and development.get("license") == UCI352_LICENSE
                and development.get("downloader_sha256")
                == sha256_file(ROOT / "scripts" / "download_retail_reference.py")
                and development.get("calibration_customer_ids_sha256")
                == EXPECTED_PARTITION_HASHES["calibration"]
                and development.get("calibration_customers")
                == UCI352_CALIBRATION_CUSTOMERS
                and development.get(
                    "full_workbook_customer_ids_scanned_for_routing"
                )
                is True
                and development.get(
                    "noncalibration_transaction_values_materialized_or_used"
                )
                is False
                and development.get(
                    "uci352_diagnostic_transactions_materialized_or_used"
                )
                is False
                and development.get(
                    "uci352_prior_holdout_transactions_materialized_or_used"
                )
                is False
                and set(one_shot) == one_shot_keys
                and one_shot.get("dataset_id") == UCI502_DATASET_ID
                and one_shot.get("source_url") == UCI502_SOURCE_URL
                and one_shot.get("archive_sha256") == UCI502_ARCHIVE_SHA256
                and one_shot.get("workbook_sha256") == UCI502_WORKBOOK_SHA256
                and one_shot.get("manifest_sha256") == UCI502_MANIFEST_SHA256
                and one_shot.get("license") == UCI502_LICENSE
                and one_shot.get("downloader_sha256")
                == sha256_file(ROOT / "scripts" / "download_retail_ii_reference.py")
                and one_shot.get("sheet") == UCI502_SHEET
                and one_shot.get("fixed_window")
                == f"[{UCI502_WINDOW_START}, {UCI502_WINDOW_END})"
                and one_shot.get("split_assignment") == UCI502_SPLIT_ASSIGNMENT
                and one_shot.get(
                    "excluded_uci352_calibration_customer_ids_sha256"
                )
                == EXPECTED_PARTITION_HASHES["calibration"]
                and one_shot.get("transaction_time_disjoint") is True
                and one_shot.get(
                    "customer_disjoint_from_profile_and_classifier_training"
                )
                is True
                and one_shot.get("same_retailer") is True
                and one_shot.get("source_independent") is False
                and one_shot.get("temporal_direction")
                == "historical_before_training_period"
                and one_shot.get("prospective_validation") is False
                and one_shot.get(
                    "transaction_sheet_parsed_or_scored_before_receipt"
                )
                is False
                and one_shot.get("first_worksheet_opened_after_receipt") is True
                and one_shot.get("second_worksheet_opened") is False
                and one_shot.get("nonselected_transactions_materialized") is False
                and one_shot.get(
                    "transform_amount_ecdf_fit_on_transform_only"
                )
                is True
                and one_shot.get(
                    "transform_used_for_profile_generator_classifier_threshold_or_abort"
                )
                is False
                and one_shot.get("score_half_is_only_real_final_evaluation")
                is True
                and split_counts_valid
                and _final_gate_metrics_valid(final)
            )
            if checks["final_valid"]:
                checks["final_valid"] = _final_result_valid(
                    final,
                    expected_protocol_sha256=protocol["protocol_sha256"],
                    expected_exploratory_sha256=exploratory_sha256,
                    expected_receipt_sha256=receipt.get("receipt_sha256"),
                    expected_behavior_profile_sha256=protocol["generator"][
                        "behavior_profile_sha256"
                    ],
                    require_pass=True,
                )
    if (
        seal_path.exists()
        and final_bytes is not None
        and receipt is not None
        and checks["final_valid"]
    ):
        try:
            completion = json.loads(_read_regular_file_bytes(seal_path))
            if not isinstance(completion, Mapping):
                raise ValueError("V7 completion seal must be a JSON object")
            _assert_aggregate_only(completion)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        else:
            checks["completion_seal_valid"] = (
                set(completion) == COMPLETION_SEAL_KEYS
                and completion.get("schema_version") == 7
                and completion.get("artifact")
                == "entity_fidelity_v7_completion_seal"
                and completion.get("protocol_sha256")
                == protocol["protocol_sha256"]
                and completion.get("status") == "complete"
                and completion.get("final_artifact_sha256")
                == hashlib.sha256(final_bytes).hexdigest()
                and completion.get("final_receipt_sha256")
                == receipt.get("receipt_sha256")
            )
    checks["internal_completion_chain_valid"] = (
        checks["attempt_valid"]
        and checks["exploratory_valid"]
        and checks["receipt_valid"]
        and checks["final_valid"]
        and checks["completion_seal_valid"]
    )
    return {
        "protocol_sha256": protocol["protocol_sha256"],
        "checks": checks,
        "verified": all(checks.values()),
        "verification_scope": "internal consistency only",
        "release_requires_external_signed_or_worm_custody": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "preregister", help="freeze the V7 correction before calibration access"
    )
    freeze = commands.add_parser("freeze", help="freeze the calibration-fitted v7 protocol")
    freeze.add_argument("--reference-xlsx", type=Path, default=DEFAULT_REFERENCE)
    freeze.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    exploratory = commands.add_parser(
        "exploratory", help="run frozen UCI352 calibration cross-fit futility"
    )
    exploratory.add_argument("--reference-xlsx", type=Path, default=DEFAULT_REFERENCE)
    exploratory.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    exploratory.add_argument("--output", type=Path, default=DEFAULT_EXPLORATORY)
    final = commands.add_parser(
        "final", help="consume the canonical receipt and score UCI502 once"
    )
    final.add_argument("--confirmation", required=True)
    verify = commands.add_parser("verify", help="verify all v7 artifact bindings")
    args = parser.parse_args()
    if args.command == "preregister":
        result = write_v7_preregistration()
    elif args.command == "freeze":
        result = freeze_v7_protocol(args.reference_xlsx, args.protocol)
    elif args.command == "exploratory":
        result = run_calibration_exploratory(
            args.reference_xlsx, args.output, protocol_path=args.protocol
        )
    elif args.command == "final":
        result = run_entity_fidelity_v7_final(
            one_shot_final_confirmation=args.confirmation,
        )
    else:
        result = verify_v7_artifacts()
    print(json.dumps(result, sort_keys=True))
    if args.command == "exploratory" and result["gate_passed"] is not True:
        raise SystemExit(1)
    if args.command == "final" and result["metrics"]["gate_passed"] is not True:
        raise SystemExit(1)
    if args.command == "verify" and result["verified"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
