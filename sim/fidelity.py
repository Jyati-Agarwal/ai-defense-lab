"""Measured real-vs-synthetic marginal fidelity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.population import PopulationSimulator


GATE_EVENTS = 200_000
GATE_THRESHOLD = 0.60
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "fidelity.json"


def _dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import lightgbm
        import numpy as np
        import pandas as pd
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
    except ImportError as exc:
        raise RuntimeError(
            "fidelity requires pandas, scikit-learn, and lightgbm; install the project dependencies"
        ) from exc
    return np, pd, lightgbm, roc_auc_score, StratifiedKFold


def _reference_rows(path: Path) -> tuple[Any, dict[str, Any]]:
    np, pd, _, _, _ = _dependencies()
    frame = pd.read_csv(path)
    input_rows = len(frame)
    if {"Time", "Amount"}.issubset(frame.columns):
        if "Class" in frame:
            frame = frame.loc[frame["Class"] == 0]
        normalized = pd.DataFrame(
            {
                "amount_minor": np.rint(frame["Amount"].astype(float) * 100).astype("int64"),
                "hour_proxy": ((frame["Time"].astype(float) // 3600) % 24).astype("int16"),
            }
        )
        time_semantics = "ULB Time is elapsed seconds; modulo-24h is only an elapsed-time proxy, not local clock hour"
    elif {"amount_minor", "hour_local"}.issubset(frame.columns):
        if "is_fraud" in frame:
            frame = frame.loc[~frame["is_fraud"].astype(bool)]
        normalized = frame[["amount_minor", "hour_local"]].rename(columns={"hour_local": "hour_proxy"}).copy()
        time_semantics = "source declares hour_local"
    else:
        raise ValueError("reference CSV needs Time/Amount or amount_minor/hour_local columns")
    normalized = normalized.replace([np.inf, -np.inf], np.nan).dropna()
    normalized = normalized.loc[
        (normalized["amount_minor"] >= 0)
        & normalized["hour_proxy"].between(0, 23, inclusive="both")
    ].astype({"amount_minor": "int64", "hour_proxy": "int16"})
    return normalized, {
        "input_rows": input_rows,
        "legitimate_usable_rows": len(normalized),
        "time_field_semantics": time_semantics,
    }


def _features(amount_minor: Any, hour_proxy: Any) -> Any:
    np, _, _, _, _ = _dependencies()
    hour_angle = 2 * math.pi * np.asarray(hour_proxy, dtype=float) / 24
    return np.column_stack(
        (
            np.log1p(np.asarray(amount_minor, dtype=float)),
            np.sin(hour_angle),
            np.cos(hour_angle),
        )
    )


def run_fidelity(
    reference_csv: Path,
    output: Path,
    *,
    events: int,
    seed: int = 20260827,
    jobs: int = 1,
    reference_source: str | None = None,
) -> dict[str, Any]:
    np, _, lightgbm, roc_auc_score, StratifiedKFold = _dependencies()
    if events < 100:
        raise ValueError("fidelity run needs at least 100 events")
    started = time.perf_counter()
    reference, row_counts = _reference_rows(reference_csv)
    minimum_calibration = max(1_000, events // 4)
    if len(reference) < events + minimum_calibration:
        raise ValueError(
            f"reference needs at least {events + minimum_calibration:,} legitimate rows for disjoint calibration and evaluation"
        )
    shuffled = reference.sample(frac=1, random_state=seed).reset_index(drop=True)
    evaluation = shuffled.iloc[:events]
    calibration = shuffled.iloc[events:]
    pairs = list(zip(calibration["amount_minor"].tolist(), calibration["hour_proxy"].tolist(), strict=True))
    simulator = PopulationSimulator(
        seed=seed,
        customers=800,
        merchants_per_category=1,
        agentic_rate=0,
        calibration_pairs=pairs,
    )
    synthetic_amount = np.empty(events, dtype="int64")
    synthetic_hour = np.empty(events, dtype="int16")
    for index, event in enumerate(simulator.iter_legitimate(events)):
        synthetic_amount[index] = event["txn"]["amount_minor"]
        synthetic_hour[index] = event["txn"]["hour_local"]

    real_x = _features(evaluation["amount_minor"].to_numpy(), evaluation["hour_proxy"].to_numpy())
    synthetic_x = _features(synthetic_amount, synthetic_hour)
    x = np.vstack((real_x, synthetic_x))
    y = np.concatenate((np.zeros(events, dtype="int8"), np.ones(events, dtype="int8")))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for fold, (train_index, test_index) in enumerate(folds.split(x, y)):
        classifier = lightgbm.LGBMClassifier(
            objective="binary",
            n_estimators=50,
            learning_rate=0.08,
            num_leaves=15,
            max_depth=4,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=1.0,
            random_state=seed + fold,
            n_jobs=jobs,
            verbosity=-1,
        )
        classifier.fit(x[train_index], y[train_index])
        aucs.append(float(roc_auc_score(y[test_index], classifier.predict_proba(x[test_index])[:, 1])))

    auc_mean = float(np.mean(aucs))
    reference_sha256 = hashlib.sha256(reference_csv.read_bytes()).hexdigest()
    reference_metadata = {
        "path": str(reference_csv),
        "sha256": reference_sha256,
        **row_counts,
        "calibration_rows": len(calibration),
        "evaluation_rows": len(evaluation),
        "calibration_evaluation_disjoint": True,
    }
    manifest_path = reference_csv.with_suffix(".manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("marginals_sha256") != reference_sha256:
            raise ValueError("reference manifest SHA-256 does not match the measured CSV")
        reference_metadata["manifest"] = {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "dataset_id": manifest.get("dataset_id"),
            "source": manifest.get("source"),
            "parquet_url": manifest.get("parquet_url"),
        }
    if reference_source:
        reference_metadata["source"] = reference_source
    result = {
        "schema_version": 1,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "events_generated": events,
        "reference": reference_metadata,
        "scope": "Narrow two-marginal smoke gate only: amount and elapsed-time-modulo-24 proxy; not behavioral or entity-state fidelity",
        "method": "Calibrate on one reference partition, then LightGBM real-vs-synthetic discrimination on a disjoint partition with stratified 5-fold ROC AUC",
        "features": ["log1p_amount_minor", "elapsed_hour_mod_24_sin", "elapsed_hour_mod_24_cos"],
        "fold_auc": aucs,
        "auc_mean": auc_mean,
        "auc_std": float(np.std(aucs)),
        "threshold_max_auc": GATE_THRESHOLD,
        "threshold_passed": auc_mean <= GATE_THRESHOLD,
        "gate_size_required": GATE_EVENTS,
        "gate_passed": events == GATE_EVENTS and auc_mean <= GATE_THRESHOLD,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "lightgbm_version": lightgbm.__version__,
        "seed": seed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=int, default=20_000, help="development run size")
    parser.add_argument("--gate-200k", action="store_true", help="run the required 200,000-event gate")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--reference-source", help="source URL recorded in the measurement artifact")
    args = parser.parse_args()
    events = GATE_EVENTS if args.gate_200k else args.events
    result = run_fidelity(
        args.reference_csv,
        args.output,
        events=events,
        seed=args.seed,
        jobs=args.jobs,
        reference_source=args.reference_source,
    )
    print(json.dumps({"auc_mean": result["auc_mean"], "events": events, "gate_passed": result["gate_passed"], "output": str(args.output)}))
    if args.gate_200k and not result["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
