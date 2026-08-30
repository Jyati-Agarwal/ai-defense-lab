"""Frozen model bundle: layer models, logistic fusion, and TreeSHAP reasons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from defend.features import FEATURE_COLUMNS, FEATURE_GROUPS, FeatureBatch

from .fusion import FusionModel
from .tabular import CalibratedLightGBM
from .text import TextModel


LAYER_ORDER = ("structural", "semantic", "provenance", "temporal", "economic")
REASON_MESSAGES = {
    "hash_matches_cart": "The payment no longer matches the signed cart hash.",
    "signature_chain_valid": "A mandate-chain signature does not verify.",
    "signer_did_in_registry": "A signer is absent from the trusted registry.",
    "delegation_depth": "The agent delegated farther than this intent normally permits.",
    "present_flag_vs_behavioral_evidence": "The claimed user presence conflicts with observed behavior.",
    "cart_mcc_in_intent": "The cart merchant category falls outside the signed intent.",
    "payee_in_allowlist": "The payee is outside the signed allowlist.",
    "payment_direction_consistency": "The payment direction conflicts with the checkout presentation.",
    "structural_floor": "A deterministic high-confidence check set a non-bypassable risk floor.",
    "intent_cart_semantic_drift": "The purchase meaning diverges from the user's playback intent.",
    "precedent_support": "Verified legitimate precedents contextualize this purchase.",
    "contextualized_semantic_drift": "Intent drift remains after legitimate precedent context.",
    "playback_constraint_consistency": "The purchase conflicts with a signed playback constraint.",
    "category_match": "The cart category does not match the requested category.",
    "intent_specificity": "The signed intent is unusually broad or underspecified.",
    "injection_fingerprint_score": "Retrieved content contains hidden-instruction fingerprints.",
    "tool_call_topic_shift": "The agent's tool path changed topic before payment.",
    "retrieved_domain_novelty": "The agent relied on a domain absent from prior trusted activity.",
    "negotiation_speed_zscore": "The agent reached payment unusually quickly.",
    "ttl_utilisation_pct": "The mandate was used unusually late in its lifetime.",
    "hour_deviation": "The payment time differs from this customer's usual hours.",
    "amount_zscore_entity": "The amount differs sharply from this customer's history.",
    "velocity_1h": "The customer's one-hour payment velocity is elevated.",
    "velocity_24h": "The customer's daily payment velocity is elevated.",
    "credit_utilisation_regime_change": "Credit utilisation jumped to near exhaustion during elevated payment velocity.",
    "price_vs_market_median": "The price is high relative to the market reference.",
    "budget_utilisation_pct": "The payment consumes an unusual share of the mandate budget.",
    "fanout_count_per_intent": "One intent is being fanned out across multiple payments.",
    "payee_novelty": "The beneficiary is new for this customer.",
    "beneficiary_in_degree": "The beneficiary's incoming-payment graph is unusual.",
    "shared_device_cluster_size": "The device is shared across an unusual customer cluster.",
    "connected_component_size": "The payment belongs to an unusually large graph component.",
}


@dataclass
class DefenseModel:
    seed: int = 20260827
    version: str = "intent-integrity-v1"
    tree: CalibratedLightGBM | None = field(default=None, init=False)
    layer_models: dict[str, CalibratedLightGBM] = field(default_factory=dict, init=False)
    text: TextModel | None = field(default=None, init=False)
    fusion: FusionModel | None = field(default=None, init=False)
    ablations: dict[str, FusionModel] = field(default_factory=dict, init=False)
    history: list[dict[str, object]] = field(default_factory=list)

    def fit(self, batch: FeatureBatch, targets: Iterable[int]) -> "DefenseModel":
        y = np.asarray(list(targets), dtype=int)
        self.tree = CalibratedLightGBM(tuple(FEATURE_COLUMNS), self.seed).fit(batch.frame, y)
        self.text = TextModel(self.seed).fit(batch.text_vectors, batch.has_text, y)
        for name in LAYER_ORDER[1:]:
            columns = tuple(FEATURE_GROUPS[name])
            self.layer_models[name] = CalibratedLightGBM(columns, self.seed).fit(batch.frame, y)
        training_layers = self.layer_scores(batch, use_oof=True)
        self.fusion = FusionModel(self.seed).fit(training_layers, y)
        return self

    def layer_scores(self, batch: FeatureBatch, use_oof: bool = False) -> np.ndarray:
        structural = batch.frame["structural_floor"].to_numpy(dtype=float)
        columns = [structural]
        for name in LAYER_ORDER[1:]:
            model = self.layer_models[name]
            if use_oof:
                if model.oof_calibrated_ is None:
                    raise RuntimeError("layer model has no out-of-fold predictions")
                values = model.oof_calibrated_.copy()
            else:
                values = model.predict_proba(batch.frame)
            if name == "semantic" and self.text is not None:
                text_values = self.text.oof_ if use_oof else self.text.predict_proba(batch.text_vectors, batch.has_text)
                values = np.where(batch.has_text, 0.5 * values + 0.5 * text_values, 0.0)
            seed_values = batch.frame[f"layer_{name}"].to_numpy(dtype=float)
            values = 0.40 * values + 0.60 * seed_values
            columns.append(values)
        return np.column_stack(columns).astype(float, copy=False)

    def predict(self, batch: FeatureBatch, ablation: str | None = None) -> dict[str, object]:
        if self.tree is None or self.fusion is None:
            raise RuntimeError("defense model is not fitted")
        layers = self.layer_scores(batch)
        fusion_input = layers.copy()
        fusion_model = self.fusion
        if ablation:
            if ablation not in {"without_semantic", "without_provenance"}:
                raise KeyError(ablation)
            excluded = ablation.removeprefix("without_")
            fusion_input[:, LAYER_ORDER.index(excluded)] = 0.0
        fused = fusion_model.predict_proba(fusion_input)
        fused = np.maximum(fused, layers[:, 0])
        reasons = self._reason_codes(batch, layers)
        return {"layers": layers, "fused": fused, "reason_codes": reasons}

    def _reason_codes(self, batch: FeatureBatch, layers: np.ndarray) -> list[list[str]]:
        if self.fusion is None or self.fusion.estimator is None:
            raise RuntimeError("fusion is not fitted")
        coefficients = self.fusion.estimator.coef_[0]
        selected = np.argmax(np.abs(layers[:, 1:] * coefficients[None, 1:]), axis=1) + 1
        explained: dict[int, np.ndarray] = {}
        for layer_index in set(int(value) for value in selected):
            positions = np.flatnonzero(selected == layer_index)
            model = self.layer_models[LAYER_ORDER[layer_index]]
            values = np.zeros((len(batch.frame), len(model.feature_names)), dtype=float)
            values[positions] = model.shap_values(batch.frame.iloc[positions])
            explained[layer_index] = values

        output: list[list[str]] = []
        for row_index, layer_index in enumerate(selected):
            model = self.layer_models[LAYER_ORDER[int(layer_index)]]
            values = explained[int(layer_index)][row_index] * float(coefficients[int(layer_index)])
            ranked = sorted(
                zip(model.feature_names, values, strict=True),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            if layers[row_index, 0] > 0:
                floor_feature = (
                    "credit_utilisation_regime_change"
                    if float(batch.frame.iloc[row_index]["credit_utilisation_regime_change"]) > 0
                    else "structural_floor"
                )
                ranked.insert(0, (floor_feature, 10.0 * float(layers[row_index, 0])))
            ranked = ranked[:3]
            if len(ranked) < 3:
                fallback = sorted(
                    ((name, float(batch.frame.iloc[row_index][name])) for name in FEATURE_COLUMNS if name not in {item[0] for item in ranked}),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )
                ranked.extend(fallback[: 3 - len(ranked)])
            reasons = []
            for feature, contribution in ranked:
                if contribution >= 0:
                    reasons.append(REASON_MESSAGES[feature])
                else:
                    reasons.append(f"Risk reduced: {feature.replace('_', ' ')} matched learned legitimate behavior.")
            output.append(reasons)
        return output
