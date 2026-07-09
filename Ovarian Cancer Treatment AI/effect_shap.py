"""
Effect-SHAP: Shapley-value attribution for the CaML-OP ITE function tau_hat(X).

Implements the explainability layer described in the thesis
(Section "Explainability layer" / "Effect-SHAP") that is documented in
main.tex but was never built in code.py. The attribution target is the
ensemble treatment-effect estimate itself, tau_hat(X) = 0.5*tau_R(X) +
0.5*tau_DR(X), over the five raw covariates (age, stage, race, ethnicity,
intent) -- not the XGBoost leaf-embedding features, matching the thesis's
description of phi_ji as the contribution of feature j to
tau_hat(X_i) - E_X[tau_hat(X)].

CaML-OP's tau_hat is a composite of an XGBoost leaf-encoder feeding two
forest-based causal learners (NonParamDML + ForestDRLearner), not a single
tree model that admits an exact TreeExplainer, so attribution uses SHAP's
model-agnostic Permutation explainer over the black-box predict() function.

Usage
-----
    from effect_shap import CaMLOPEffectModel, compute_effect_shap, top_features

    model = CaMLOPEffectModel(seed=42).fit(X_tr, T_tr, Y_tr)
    shap_values, feature_names = compute_effect_shap(model, X_tr, X_te)
    pos, neg = top_features(shap_values[0], feature_names, k=3)
"""
from __future__ import annotations

import numpy as np
import shap

from econml.dml import NonParamDML
from econml.dr import ForestDRLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Importing from code.py: the module is named "code" (shadowing the stdlib
# `code` module) because it lives alongside this file and Python resolves
# same-directory modules first when run as scripts from this folder. This
# mirrors how code.py itself is normally invoked (`python code.py`).
from code import LeafEncoder

FEATURE_NAMES = ["age", "stage", "race", "ethnicity", "intent"]


class CaMLOPEffectModel:
    """Wraps the fitted CaML-OP ensemble (leaf encoder + R-Learner +
    ForestDRLearner) as a single callable tau_hat(X_raw) -> effect estimate
    over the five raw covariates, so it can be explained model-agnostically."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.enc: LeafEncoder | None = None
        self.r_model = None
        self.dr_model = None

    def _rf_reg(self):
        return RandomForestRegressor(n_estimators=200, max_depth=6,
                                     min_samples_leaf=5, random_state=self.seed)

    def _rf_clf(self):
        return RandomForestClassifier(n_estimators=200, max_depth=6,
                                      min_samples_leaf=5, random_state=self.seed)

    def fit(self, X_tr, T_tr, Y_tr) -> "CaMLOPEffectModel":
        self.enc = LeafEncoder(seed=self.seed).fit(X_tr, Y_tr)
        X_tr_a = self.enc.transform(X_tr)

        self.r_model = NonParamDML(
            model_y=self._rf_reg(), model_t=self._rf_clf(),
            model_final=self._rf_reg(),
            discrete_treatment=True, cv=5, random_state=self.seed,
        )
        self.r_model.fit(Y_tr, T_tr, X=X_tr_a)

        self.dr_model = ForestDRLearner(
            model_regression=self._rf_reg(), model_propensity=self._rf_clf(),
            n_estimators=300, min_samples_leaf=5,
            cv=5, random_state=self.seed,
        )
        self.dr_model.fit(Y_tr, T_tr, X=X_tr_a)
        return self

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        tau_r, tau_dr = self.predict_components(X_raw)
        return 0.5 * tau_r + 0.5 * tau_dr

    def predict_components(self, X_raw: np.ndarray):
        """The R-Learner and DR-Learner point estimates separately, before
        ensembling. R is robust to outcome-model misspecification, DR to
        propensity-model misspecification (thesis: "R-Learner"/"DR-Learner"
        sections) -- their disagreement is a structural/model-form
        uncertainty signal distinct from either component's own sampling
        uncertainty (see NarrativeInputs.model_disagreement)."""
        X_raw = np.atleast_2d(X_raw)
        X_a = self.enc.transform(X_raw)
        tau_r = np.asarray(self.r_model.effect(X_a))
        tau_dr = np.asarray(self.dr_model.effect(X_a))
        return tau_r, tau_dr

    def predict_interval(self, X_raw: np.ndarray, alpha: float = 0.05):
        """Approximate uncertainty interval, mirroring code.py's
        fit_caml_op exactly: ForestDRLearner's native CI half-width widened
        10% to acknowledge ensemble noise (thesis: "Ensemble and
        uncertainty reporting"). Returns (lo, hi) arrays over the ensemble
        point estimate from predict()."""
        X_raw = np.atleast_2d(X_raw)
        X_a = self.enc.transform(X_raw)
        tau = self.predict(X_raw)
        lo_dr, hi_dr = self.dr_model.effect_interval(X_a, alpha=alpha)
        half_width = (np.asarray(hi_dr) - np.asarray(lo_dr)) / 2 * 1.10
        return tau - half_width, tau + half_width


def compute_effect_shap(model: CaMLOPEffectModel, X_background: np.ndarray,
                        X_explain: np.ndarray, n_background: int = 50,
                        seed: int = 42):
    """Model-agnostic Shapley attribution of tau_hat over the raw covariates.

    Returns (shap_values, feature_names) where shap_values has shape
    (len(X_explain), len(FEATURE_NAMES)).
    """
    rng = np.random.RandomState(seed)
    n_bg = min(n_background, len(X_background))
    bg_idx = rng.choice(len(X_background), size=n_bg, replace=False)
    background = X_background[bg_idx]

    explainer = shap.Explainer(model.predict, background,
                               algorithm="permutation",
                               feature_names=FEATURE_NAMES)
    sv = explainer(X_explain)
    return np.asarray(sv.values), FEATURE_NAMES


def top_features(shap_row: np.ndarray, feature_names: list[str], k: int = 3):
    """Top-k positive and negative Effect-SHAP drivers for one patient, as
    (feature_name, phi) pairs, matching the thesis's "top three Effect-SHAP
    drivers in each direction" prompt input for the LLM narrative module."""
    order = np.argsort(shap_row)
    neg = [(feature_names[i], float(shap_row[i]))
           for i in order[:k] if shap_row[i] < 0]
    pos = [(feature_names[i], float(shap_row[i]))
           for i in order[::-1][:k] if shap_row[i] > 0]
    return pos, neg
