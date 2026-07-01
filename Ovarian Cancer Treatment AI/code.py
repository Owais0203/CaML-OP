"""
CaML-OP: Full Benchmark v2 — Aligned with the Exposé
=====================================================
Adds five things missing from the previous version:

  (1) ATE error and MAE metrics (Section 8.1.1)
  (2) Loop over DGP scenarios 1–4 (Section 7.8.2)
  (3) ForestDRLearner with native 95% CI reporting (Section 7.5.3)
  (4) Subgroup analysis: age band, FIGO stage, race (Section 8.3)
  (5) Cox PH + RSF predictive baselines (Section 7.2, 8.1.3)

Outputs
-------
caml_op_outputs/
  summary_pehe.csv             per (scenario, method) PEHE/MAE/ATE-err
  summary_value.csv            per (scenario, method) AIPW policy value
  summary_predictive.csv       Cox PH / RSF / XGBoost AUC, AUPRC, Brier
  summary_ci_coverage.csv      coverage of CaML-OP's 95% CIs vs true tau
  summary_subgroup.csv         subgroup PEHE per method
  pehe_per_run.csv             per-run details
  *.png                        plots (see Section 7 below)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy import stats

from econml.dml import NonParamDML, CausalForestDML
from econml.dr import DRLearner, ForestDRLearner
from econml.metalearners import SLearner, TLearner, XLearner

# Survival baselines (used as binary 5-year classifiers — see Section 8.1.3)
from lifelines import CoxPHFitter

SEED = 42
np.random.seed(SEED)

OUT_DIR = "caml_op_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. DATA
# ============================================================
def load_tcga(path="tcga_ov_master_ml.csv"):
    df = pd.read_csv(path)
    df = df[['age_at_diagnosis', 'figo_stage', 'race', 'ethnicity',
             'treatments.treatment_intent_type',
             'five_year_survival', 'platinum_therapy']].copy()
    df.columns = ['age', 'stage', 'race', 'ethnicity', 'intent', 'y', 't']
    df = df.dropna()
    for col in ['stage', 'race', 'ethnicity', 'intent']:
        df[col] = df[col].astype('category').cat.codes
    df['age'] = pd.to_numeric(df['age'], errors='coerce') / 365.25
    df = df.dropna()
    X = df[['age', 'stage', 'race', 'ethnicity', 'intent']].values.astype(float)
    T = df['t'].values.astype(int)
    Y = df['y'].values.astype(int)
    return X, T, Y, df


# ============================================================
# 2. SEMI-SYNTHETIC DGP — all four scenarios (Section 7.8.2)
# ============================================================
def standardise(X):
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def generate_dgp(X_raw, seed=0, scenario=4):
    rng = np.random.RandomState(seed)
    Xs = standardise(X_raw)
    n, d = Xs.shape

    coef_rng = np.random.RandomState(123)
    beta0 = coef_rng.normal(0, 0.5, size=d)
    beta1 = coef_rng.normal(0, 0.5, size=d)

    f = Xs @ beta0
    age, stage = Xs[:, 0], Xs[:, 1]
    if scenario == 1:
        tau = 0.5 * np.ones(n)                          # constant
    elif scenario == 2:
        tau = 0.4 * age + 0.3 * stage                   # linear
    elif scenario == 3:
        tau = 1.0 * sigmoid(Xs @ beta1) - 0.5           # nonlinear
    elif scenario == 4:
        tau = 1.0 * sigmoid(Xs @ beta1) - 0.5 + 0.4 * age * stage
    else:
        raise ValueError(f"Unknown scenario {scenario}")

    e = sigmoid(-0.6 * age - 0.4 * stage)
    T = (rng.rand(n) < e).astype(int)
    p0 = sigmoid(f)
    p1 = sigmoid(f + tau)
    Y0 = (rng.rand(n) < p0).astype(int)
    Y1 = (rng.rand(n) < p1).astype(int)
    Y = np.where(T == 1, Y1, Y0)
    tau_true = p1 - p0
    return Y, T, tau_true, e


# ============================================================
# 3. LEAF ENCODER (Section 7.4)
# ============================================================
class LeafEncoder:
    def __init__(self, n_estimators=100, max_depth=4, lr=0.1, seed=SEED):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=lr, eval_metric="logloss",
            random_state=seed, verbosity=0, use_label_encoder=False,
        )
        self.leaf_max_ = None

    def fit(self, X, Y):
        self.model.fit(X, Y)
        leaves = self.model.apply(X).astype(int)
        self.leaf_max_ = (leaves.max(axis=0) + 1).astype(int)
        return self

    def transform(self, X):
        leaves = self.model.apply(X).astype(int)
        oh_blocks = []
        for t in range(leaves.shape[1]):
            block = np.zeros((leaves.shape[0], int(self.leaf_max_[t])))
            block[np.arange(leaves.shape[0]), leaves[:, t]] = 1.0
            oh_blocks.append(block)
        return np.hstack([X] + oh_blocks)


# ============================================================
# 4. CAUSAL ESTIMATORS
# ============================================================
def make_rf_reg(seed):
    return RandomForestRegressor(n_estimators=200, max_depth=6,
                                 min_samples_leaf=5, random_state=seed)

def make_rf_clf(seed):
    return RandomForestClassifier(n_estimators=200, max_depth=6,
                                  min_samples_leaf=5, random_state=seed)


def fit_s_learner(X_tr, T_tr, Y_tr, X_te, seed):
    m = SLearner(overall_model=make_rf_reg(seed))
    m.fit(Y_tr, T_tr, X=X_tr)
    return m.effect(X_te), None  # no CI

def fit_t_learner(X_tr, T_tr, Y_tr, X_te, seed):
    m = TLearner(models=make_rf_reg(seed))
    m.fit(Y_tr, T_tr, X=X_tr)
    return m.effect(X_te), None

def fit_x_learner(X_tr, T_tr, Y_tr, X_te, seed):
    m = XLearner(models=make_rf_reg(seed),
                 propensity_model=make_rf_clf(seed))
    m.fit(Y_tr, T_tr, X=X_tr)
    return m.effect(X_te), None

def fit_causal_forest(X_tr, T_tr, Y_tr, X_te, seed):
    m = CausalForestDML(
        model_y=make_rf_reg(seed), model_t=make_rf_clf(seed),
        discrete_treatment=True, n_estimators=300,
        min_samples_leaf=5, cv=5, random_state=seed,
    )
    m.fit(Y_tr, T_tr, X=X_tr)
    pt = m.effect(X_te)
    lo, hi = m.effect_interval(X_te, alpha=0.05)
    return pt, (lo, hi)

def fit_r_learner(X_tr, T_tr, Y_tr, X_te, seed):
    m = NonParamDML(
        model_y=make_rf_reg(seed), model_t=make_rf_clf(seed),
        model_final=make_rf_reg(seed),
        discrete_treatment=True, cv=5, random_state=seed,
    )
    m.fit(Y_tr, T_tr, X=X_tr)
    return m.effect(X_te), None

def fit_dr_learner(X_tr, T_tr, Y_tr, X_te, seed):
    m = DRLearner(
        model_propensity=make_rf_clf(seed),
        model_regression=make_rf_reg(seed),
        model_final=make_rf_reg(seed),
        cv=5, random_state=seed,
    )
    m.fit(Y_tr, T_tr, X=X_tr)
    return m.effect(X_te), None

def fit_caml_op(X_tr, T_tr, Y_tr, X_te, seed):
    """CaML-OP with leaf-augmented R+DR ensemble.
    Uses ForestDRLearner so we can report 95% CIs (Section 7.5.3)."""
    enc = LeafEncoder(seed=seed).fit(X_tr, Y_tr)
    X_tr_a = enc.transform(X_tr)
    X_te_a = enc.transform(X_te)

    # R-component (no native CIs from NonParamDML)
    r_model = NonParamDML(
        model_y=make_rf_reg(seed), model_t=make_rf_clf(seed),
        model_final=make_rf_reg(seed),
        discrete_treatment=True, cv=5, random_state=seed,
    )
    r_model.fit(Y_tr, T_tr, X=X_tr_a)
    tau_R = r_model.effect(X_te_a)

    # DR-component using ForestDRLearner (native CIs)
    fdr = ForestDRLearner(
        model_regression=make_rf_reg(seed),
        model_propensity=make_rf_clf(seed),
        n_estimators=300, min_samples_leaf=5,
        cv=5, random_state=seed,
    )
    fdr.fit(Y_tr, T_tr, X=X_tr_a)
    tau_DR = fdr.effect(X_te_a)
    lo_DR, hi_DR = fdr.effect_interval(X_te_a, alpha=0.05)

    # Ensemble for the point estimate
    tau_ens = 0.5 * tau_R + 0.5 * tau_DR

    # Use the ForestDRLearner CIs for the ensemble
    # (R has no CI; we widen the DR CI by 10% to acknowledge the ensemble noise)
    half_width = (hi_DR - lo_DR) / 2 * 1.10
    lo_ens = tau_ens - half_width
    hi_ens = tau_ens + half_width
    return tau_ens, (lo_ens, hi_ens)


METHODS = {
    "S-Learner":     fit_s_learner,
    "T-Learner":     fit_t_learner,
    "X-Learner":     fit_x_learner,
    "Causal Forest": fit_causal_forest,
    "R-Learner":     fit_r_learner,
    "DR-Learner":    fit_dr_learner,
    "CaML-OP":       fit_caml_op,
}


# ============================================================
# 5. PREDICTIVE BASELINES (Section 7.2, 8.1.3)
# ============================================================
# IMPORTANT NOTE FOR THE THESIS:
# The exposé lists Cox PH and RSF as predictive baselines. These are
# survival models designed for time-to-event data. Our outcome (5-year
# survival) is binary, so we adapt them as follows:
#
#   - Cox PH:  fit with duration=5 for all patients, event=Y. The hazard
#              ratio still fits the binary outcome under a fixed horizon.
#              Predict 1 - S(5) as P(Y=1).
#   - RSF:     does not work with constant durations. We replace it with
#              Random Forest classifier on the binary outcome — a non-linear
#              tree-based predictive baseline that serves the same role
#              ("non-linear ensemble for prognostic prediction"). Document
#              this substitution in the thesis methods section.
#   - XGBoost: standard binary classifier (the encoder backbone).
#
# All three predict marginal P(Y=1|X) by averaging predictions across
# T in {0,1}, so they serve as PREDICTIVE-only baselines that do not use
# treatment information at test time.

def fit_predictive_baselines(X_tr, T_tr, Y_tr, X_te, Y_te, seed):
    out = {}
    feat = ['age', 'stage', 'race', 'ethnicity', 'intent']
    X_tr_df = pd.DataFrame(X_tr, columns=feat)
    X_te_df = pd.DataFrame(X_te, columns=feat)
    X_tr_df['t'] = T_tr

    # ---- Cox PH (fixed horizon t=5) ----
    try:
        cph_df = X_tr_df.copy()
        cph_df['duration'] = 5.0
        cph_df['event'] = Y_tr
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cph_df, duration_col='duration', event_col='event')
        preds = []
        for t_val in [0, 1]:
            X_te_eval = X_te_df.copy()
            X_te_eval['t'] = t_val
            sf = cph.predict_survival_function(X_te_eval, times=[5.0])
            preds.append(1.0 - sf.values.ravel())
        p = np.mean(preds, axis=0)
        # AUC requires both classes present
        if len(np.unique(Y_te)) == 2:
            out["Cox PH"] = {
                "AUC":   roc_auc_score(Y_te, p),
                "AUPRC": average_precision_score(Y_te, p),
                "Brier": brier_score_loss(Y_te, p),
            }
    except Exception:
        pass

    # ---- Random Forest classifier (substitute for RSF; see note above) ----
    try:
        rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                    min_samples_leaf=5, random_state=seed)
        rf.fit(X_tr_df.values, Y_tr)
        preds = []
        for t_val in [0, 1]:
            X_te_eval = X_te_df.copy()
            X_te_eval['t'] = t_val
            preds.append(rf.predict_proba(X_te_eval.values)[:, 1])
        p = np.mean(preds, axis=0)
        if len(np.unique(Y_te)) == 2:
            out["RF (RSF surrogate)"] = {
                "AUC":   roc_auc_score(Y_te, p),
                "AUPRC": average_precision_score(Y_te, p),
                "Brier": brier_score_loss(Y_te, p),
            }
    except Exception:
        pass

    # ---- XGBoost classifier ----
    try:
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                eval_metric='logloss', random_state=seed,
                                verbosity=0, use_label_encoder=False)
        clf.fit(X_tr_df.values, Y_tr)
        preds = []
        for t_val in [0, 1]:
            X_te_eval = X_te_df.copy()
            X_te_eval['t'] = t_val
            preds.append(clf.predict_proba(X_te_eval.values)[:, 1])
        p = np.mean(preds, axis=0)
        if len(np.unique(Y_te)) == 2:
            out["XGBoost"] = {
                "AUC":   roc_auc_score(Y_te, p),
                "AUPRC": average_precision_score(Y_te, p),
                "Brier": brier_score_loss(Y_te, p),
            }
    except Exception:
        pass

    return out


# ============================================================
# 6. METRICS (Section 8.1.1)
# ============================================================
def pehe(true, pred):
    return float(np.sqrt(np.mean((true - pred) ** 2)))

def mae(true, pred):
    return float(np.mean(np.abs(true - pred)))

def ate_error(true, pred):
    return float(abs(np.mean(true) - np.mean(pred)))

def aipw_policy_value(tau_pred, Y, T, e_true):
    pi = (tau_pred > 0).astype(int)
    e_clip = np.clip(e_true, 0.05, 0.95)
    w = np.where(pi == 1, 1.0 / e_clip, 1.0 / (1.0 - e_clip))
    agree = (pi == T).astype(float)
    return float(np.mean(Y * agree * w))

def ci_coverage(tau_true, lo, hi):
    """Fraction of true tau values inside the predicted 95% CI."""
    inside = (tau_true >= lo) & (tau_true <= hi)
    return float(np.mean(inside))

def ci_width(lo, hi):
    return float(np.mean(hi - lo))


# ============================================================
# 7. SUBGROUP ANALYSIS (Section 8.3)
# ============================================================
def subgroup_indices(X_te_raw):
    """
    Returns dict of subgroup name -> boolean mask on the test set.
    age column is index 0 (years), stage is index 1, race is index 2.
    Race codes are arbitrary integers post-encoding; we use top-3 most
    common levels rather than re-mapping to White/Black/Other.
    """
    age   = X_te_raw[:, 0]
    stage = X_te_raw[:, 1]
    race  = X_te_raw[:, 2]
    groups = {}
    groups["age<50"]    = age < 50
    groups["age50-65"]  = (age >= 50) & (age < 65)
    groups["age>=65"]   = age >= 65
    # FIGO stage: low/mid/high using the encoded ordinals
    stage_med = np.median(stage)
    groups["stageLow"]  = stage <  stage_med
    groups["stageHigh"] = stage >= stage_med
    # Race: most common vs rest
    most_common = np.bincount(race.astype(int)).argmax()
    groups["raceMajority"] = race == most_common
    groups["raceMinority"] = race != most_common
    return groups


# ============================================================
# 8. EXPERIMENT LOOP — scenarios x runs
# ============================================================
def run_all(X_raw, scenarios=(1, 2, 3, 4), n_runs=10, verbose=True):
    rows_pehe, rows_value, rows_pred = [], [], []
    rows_ci, rows_subgroup = [], []
    calib_data = {}  # (scenario, method) -> {true, pred} for run 0

    for scen in scenarios:
        if verbose:
            print(f"\n{'='*70}\nSCENARIO {scen}\n{'='*70}")

        for run in range(n_runs):
            Y_full, T_full, tau_full, e_full = generate_dgp(
                X_raw, seed=run, scenario=scen)

            idx = np.arange(len(X_raw))
            tr_idx, te_idx = train_test_split(
                idx, test_size=0.25, random_state=run, stratify=T_full)
            X_tr, X_te = X_raw[tr_idx], X_raw[te_idx]
            T_tr, T_te = T_full[tr_idx], T_full[te_idx]
            Y_tr, Y_te = Y_full[tr_idx], Y_full[te_idx]
            tau_te, e_te = tau_full[te_idx], e_full[te_idx]

            keep = (e_full[tr_idx] > 0.05) & (e_full[tr_idx] < 0.95)
            X_tr, T_tr, Y_tr = X_tr[keep], T_tr[keep], Y_tr[keep]

            if verbose:
                print(f"\n  Run {run+1}/{n_runs}  test n={len(te_idx)}, "
                      f"true ATE={tau_te.mean():+.3f}")

            # Reference policy values
            row_v = {"scenario": scen, "run": run,
                     "TreatAll":  aipw_policy_value(np.ones_like(T_te), Y_te, T_te, e_te),
                     "TreatNone": aipw_policy_value(-np.ones_like(T_te), Y_te, T_te, e_te),
                     "Oracle":    aipw_policy_value(tau_te,             Y_te, T_te, e_te)}

            row_p = {"scenario": scen, "run": run}
            row_c = {"scenario": scen, "run": run}
            subgrps = subgroup_indices(X_te)

            for name, fn in METHODS.items():
                try:
                    tau_pred, ci = fn(X_tr, T_tr, Y_tr, X_te, run)
                    tau_pred = np.asarray(tau_pred).ravel()

                    row_p[f"PEHE_{name}"]   = pehe(tau_te, tau_pred)
                    row_p[f"MAE_{name}"]    = mae(tau_te, tau_pred)
                    row_p[f"ATEerr_{name}"] = ate_error(tau_te, tau_pred)
                    row_v[name] = aipw_policy_value(tau_pred, Y_te, T_te, e_te)

                    # CI metrics if available
                    if ci is not None:
                        lo, hi = np.asarray(ci[0]).ravel(), np.asarray(ci[1]).ravel()
                        row_c[f"Cov_{name}"]   = ci_coverage(tau_te, lo, hi)
                        row_c[f"Width_{name}"] = ci_width(lo, hi)

                    # Subgroup PEHE
                    for g_name, g_mask in subgrps.items():
                        if g_mask.sum() < 10:   # skip tiny subgroups
                            continue
                        rows_subgroup.append({
                            "scenario": scen, "run": run, "method": name,
                            "subgroup": g_name, "n": int(g_mask.sum()),
                            "PEHE": pehe(tau_te[g_mask], tau_pred[g_mask]),
                        })

                    if run == 0:
                        calib_data[(scen, name)] = {
                            "tau_true": tau_te, "tau_pred": tau_pred}

                    if verbose:
                        print(f"    {name:14s}  PEHE={row_p[f'PEHE_{name}']:.4f}  "
                              f"V={row_v[name]:.4f}")
                except Exception as e:
                    print(f"    {name:14s}  FAILED: {e}")

            # Predictive baselines (Cox PH, RSF, XGBoost) — only on real data
            # behaviour, but we apply them on the synthetic Y here for
            # consistency within each run.
            try:
                pred_results = fit_predictive_baselines(
                    X_tr, T_tr, Y_tr, X_te, Y_te, run)
                for m_name, met in pred_results.items():
                    rows_pred.append({"scenario": scen, "run": run,
                                      "method": m_name, **met})
            except Exception as e:
                print(f"    [predictive baselines]  FAILED: {e}")

            rows_pehe.append(row_p)
            rows_value.append(row_v)
            rows_ci.append(row_c)

    return (pd.DataFrame(rows_pehe),
            pd.DataFrame(rows_value),
            pd.DataFrame(rows_ci),
            pd.DataFrame(rows_subgroup),
            pd.DataFrame(rows_pred),
            calib_data)


# ============================================================
# 9. PLOTS (faceted by scenario)
# ============================================================
sns.set_style("whitegrid")
PALETTE = {
    "S-Learner": "#9ecae1", "T-Learner": "#6baed6", "X-Learner": "#4292c6",
    "Causal Forest": "#9e9ac8", "R-Learner": "#fdae6b", "DR-Learner": "#fd8d3c",
    "CaML-OP": "#d62728",
}


def plot_pehe_by_scenario(pehe_df, methods, path):
    """Grouped bar chart: scenario on x, methods as colored bars."""
    rows = []
    for m in methods:
        col = f"PEHE_{m}"
        for scen, sub in pehe_df.groupby("scenario"):
            rows.append({"method": m, "scenario": scen,
                         "mean": sub[col].mean(),
                         "ci": 1.96 * sub[col].std(ddof=1) / np.sqrt(len(sub))})
    long = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    scenarios = sorted(long["scenario"].unique())
    width = 0.11
    for i, m in enumerate(methods):
        d = long[long["method"] == m].sort_values("scenario")
        x = np.arange(len(scenarios)) + i * width - (len(methods) - 1) * width / 2
        ax.bar(x, d["mean"].values, width=width, yerr=d["ci"].values,
               color=PALETTE[m], label=m, capsize=2,
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels([f"Scenario {s}" for s in scenarios])
    ax.set_ylabel("PEHE  (lower is better)")
    ax.set_title("PEHE by DGP scenario (mean ± 95% CI across runs)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_metric_grid(pehe_df, methods, path):
    """3-panel: PEHE, MAE, ATE error per method (averaged over scenarios)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric, label in zip(
            axes, ["PEHE", "MAE", "ATEerr"],
            ["PEHE", "MAE", "ATE Error"]):
        means = pd.Series({m: pehe_df[f"{metric}_{m}"].mean() for m in methods})
        stds  = pd.Series({m: pehe_df[f"{metric}_{m}"].std(ddof=1) for m in methods})
        ci    = 1.96 * stds / np.sqrt(len(pehe_df))
        order = means.sort_values().index.tolist()
        ax.bar(range(len(order)), means.loc[order], yerr=ci.loc[order],
               color=[PALETTE[m] for m in order],
               capsize=4, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"{label} (avg over scenarios)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_policy_by_scenario(value_df, methods, path):
    rows = []
    for m in methods + ["Oracle", "TreatAll", "TreatNone"]:
        for scen, sub in value_df.groupby("scenario"):
            rows.append({"method": m, "scenario": scen,
                         "mean": sub[m].mean(),
                         "ci": 1.96 * sub[m].std(ddof=1) / np.sqrt(len(sub))})
    long = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(value_df["scenario"].unique()),
                             figsize=(4 * value_df["scenario"].nunique(), 4),
                             sharey=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    for ax, scen in zip(axes, sorted(value_df["scenario"].unique())):
        d = long[long["scenario"] == scen].set_index("method")
        order = methods + ["Oracle", "TreatAll", "TreatNone"]
        colors = [PALETTE[m] for m in methods] + ["#2ca02c", "#bdbdbd", "#525252"]
        ax.bar(range(len(order)), d.loc[order, "mean"].values,
               yerr=d.loc[order, "ci"].values, color=colors,
               capsize=3, edgecolor="black", linewidth=0.4)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Scenario {scen}")
        ax.axhline(d.loc["Oracle", "mean"], color="#2ca02c", ls="--",
                   lw=0.8, alpha=0.6)
    axes[0].set_ylabel("AIPW policy value")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_subgroup_pehe(sub_df, methods, path):
    """Heatmap: rows = method, cols = subgroup, value = mean PEHE."""
    pivot = (sub_df.groupby(["method", "subgroup"])["PEHE"].mean()
             .unstack("subgroup"))
    pivot = pivot.loc[methods]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                cbar_kws={"label": "Mean PEHE"}, ax=ax)
    ax.set_title("Subgroup PEHE per method (averaged over scenarios & runs)")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_predictive_baselines(pred_df, path):
    """Bar chart of AUC, AUPRC, Brier for Cox PH, RSF, XGBoost."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["AUC", "AUPRC", "Brier"]):
        means = pred_df.groupby("method")[metric].mean()
        stds  = pred_df.groupby("method")[metric].std(ddof=1)
        ci    = 1.96 * stds / np.sqrt(pred_df.groupby("method").size())
        order = (means.sort_values(ascending=(metric == "Brier"))
                 .index.tolist())
        ax.bar(range(len(order)), means.loc[order], yerr=ci.loc[order],
               capsize=4, edgecolor="black", linewidth=0.5,
               color=["#9e9ac8", "#fdae6b", "#9ecae1"][:len(order)])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=15)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} ({'higher' if metric != 'Brier' else 'lower'} is better)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_ci_coverage(ci_df, path):
    """Bar chart: 95% CI empirical coverage per method (target=0.95)."""
    cov_cols = [c for c in ci_df.columns if c.startswith("Cov_")]
    methods  = [c.replace("Cov_", "") for c in cov_cols]
    means    = ci_df[cov_cols].mean()
    stds     = ci_df[cov_cols].std(ddof=1)
    ci       = 1.96 * stds / np.sqrt(len(ci_df))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(methods)), means.values, yerr=ci.values,
           color=[PALETTE[m] for m in methods],
           capsize=4, edgecolor="black", linewidth=0.5)
    ax.axhline(0.95, color="black", ls="--", lw=1, label="nominal 95%")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Empirical coverage of 95% CIs across runs")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ============================================================
# 10. MAIN
# ============================================================
def main(n_runs=10, scenarios=(1, 2, 3, 4)):
    print("Loading TCGA-OV covariates...")
    X, _, _, _ = load_tcga()
    print(f"N = {len(X)}")

    pehe_df, value_df, ci_df, sub_df, pred_df, calib_data = run_all(
        X, scenarios=scenarios, n_runs=n_runs)

    methods = list(METHODS.keys())

    # ---- Summary tables ----
    print("\n" + "=" * 70)
    print("SUMMARY: PEHE / MAE / ATE error (mean over scenarios + runs)")
    print("=" * 70)
    rows = []
    for m in methods:
        col = f"PEHE_{m}"
        if col not in pehe_df.columns:
            print(f"  [skip] {m} produced no results")
            continue
        rows.append({
            "method": m,
            "PEHE":  pehe_df[f"PEHE_{m}"].mean(),
            "MAE":   pehe_df[f"MAE_{m}"].mean(),
            "ATEerr": pehe_df[f"ATEerr_{m}"].mean(),
        })
    sum_pehe = pd.DataFrame(rows).round(4).sort_values("PEHE")
    print(sum_pehe.to_string(index=False))
    sum_pehe.to_csv(os.path.join(OUT_DIR, "summary_pehe.csv"), index=False)
    methods_ok = sum_pehe["method"].tolist()

    print("\n" + "=" * 70)
    print("PER-SCENARIO PEHE (lower is better)")
    print("=" * 70)
    per_scen = []
    for scen in scenarios:
        sub = pehe_df[pehe_df.scenario == scen]
        for m in methods_ok:
            per_scen.append({"scenario": scen, "method": m,
                             "PEHE": sub[f"PEHE_{m}"].mean()})
    per_scen_df = pd.DataFrame(per_scen)
    print(per_scen_df.pivot(index="method", columns="scenario", values="PEHE")
          .round(4).to_string())

    print("\n" + "=" * 70)
    print("POLICY VALUE per method (mean across scenarios + runs)")
    print("=" * 70)
    val_methods = [m for m in methods_ok if m in value_df.columns]
    sum_val = pd.DataFrame({
        m: [value_df[m].mean(), value_df[m].std(ddof=1)]
        for m in val_methods + ["Oracle", "TreatAll", "TreatNone"]
    }, index=["mean", "sd"]).T.round(4).sort_values("mean", ascending=False)
    print(sum_val.to_string())
    sum_val.to_csv(os.path.join(OUT_DIR, "summary_value.csv"))

    print("\n" + "=" * 70)
    print("PREDICTIVE BASELINES (Cox PH / RSF / XGBoost)")
    print("=" * 70)
    if not pred_df.empty:
        pred_summary = pred_df.groupby("method")[["AUC", "AUPRC", "Brier"]].mean().round(4)
        print(pred_summary.to_string())
        pred_summary.to_csv(os.path.join(OUT_DIR, "summary_predictive.csv"))

    print("\n" + "=" * 70)
    print("95% CI COVERAGE (target = 0.95)  for methods that report CIs")
    print("=" * 70)
    cov_cols = [c for c in ci_df.columns if c.startswith("Cov_")]
    if cov_cols:
        cov_summary = pd.DataFrame({
            c.replace("Cov_", ""): [ci_df[c].mean(), ci_df[c].std(ddof=1),
                                     ci_df[c.replace("Cov_", "Width_")].mean()]
            for c in cov_cols
        }, index=["coverage", "coverage_sd", "mean_width"]).T.round(4)
        print(cov_summary.to_string())
        cov_summary.to_csv(os.path.join(OUT_DIR, "summary_ci_coverage.csv"))

    print("\n" + "=" * 70)
    print("SUBGROUP PEHE (averaged over scenarios + runs)")
    print("=" * 70)
    if not sub_df.empty:
        sub_summary = (sub_df.groupby(["method", "subgroup"])["PEHE"]
                       .mean().unstack("subgroup").round(4))
        sub_summary = sub_summary.loc[methods]
        print(sub_summary.to_string())
        sub_summary.to_csv(os.path.join(OUT_DIR, "summary_subgroup.csv"))

    pehe_df.to_csv(os.path.join(OUT_DIR, "pehe_per_run.csv"), index=False)

    # ---- Plots ----
    print(f"\nWriting plots to ./{OUT_DIR}/")
    plot_pehe_by_scenario(pehe_df, methods_ok,
                          os.path.join(OUT_DIR, "01_pehe_by_scenario.png"))
    plot_metric_grid(pehe_df, methods_ok,
                     os.path.join(OUT_DIR, "02_metric_grid.png"))
    plot_policy_by_scenario(value_df, val_methods,
                            os.path.join(OUT_DIR, "03_policy_by_scenario.png"))
    if not sub_df.empty:
        plot_subgroup_pehe(sub_df, methods_ok,
                           os.path.join(OUT_DIR, "04_subgroup_pehe.png"))
    if not pred_df.empty:
        plot_predictive_baselines(pred_df,
                                  os.path.join(OUT_DIR, "05_predictive.png"))
    if cov_cols:
        plot_ci_coverage(ci_df,
                         os.path.join(OUT_DIR, "06_ci_coverage.png"))
    print("Done.")


if __name__ == "__main__":
    # Default: all 4 scenarios, 10 runs each (~30-40 min on laptop CPU)
    main(n_runs=10, scenarios=(1, 2, 3, 4))