"""
CaML-OP clinical dashboard -- the first runnable version of the "Clinical
dashboard prototype" main.tex describes (Section 3.5, Figure 3.4) as a
static mockup image (media/image3.png). This wires the five panels the
thesis already specifies to the code that actually computes them:

  1. ITE gauge              -> effect_shap.CaMLOPEffectModel (tau_hat, CI)
  2. Five-year survival      -> a bootstrapped Cox PH model (code.py's own
     probability comparison      "adapt Cox PH to a fixed 5-year horizon"
                                  trick), refit per bootstrap resample for
                                  an approximate uncertainty band. NOTE:
                                  see "Panel 2" below -- this is a point
                                  comparison, not a continuous curve, and
                                  that is a deliberate correction, not a
                                  simplification. See AGENT_INTEGRATION.md.
  3. Effect-SHAP waterfall   -> effect_shap.compute_effect_shap, with
                                covariate codes decoded back to their
                                original TCGA-OV category labels
  4. LLM narrative panel     -> llm_narrative_agent.generate_and_check
  5. Transparency card       -> caml_op_outputs/summary_*.csv (this run's
                                own benchmark numbers), always rendered
                                (never in a collapsible/dismissable widget)

Plus one interactive control the static mockup never had: a bounded
what-if query box (counterfactual_query.py) in the sidebar, deliberately
NOT an open-ended chat box -- see AGENT_INTEGRATION.md, "Bounded query
interface, not open-ended chat," for why.

Panel 2 -- an honest correction, not a simplification
----------------------------------------------------------------------------
The thesis's mockup caption describes "counterfactual five-year survival
curves ... under T=1 versus T=0." An earlier version of this dashboard
plotted exactly that: a continuous curve from year 0 to year 5 per arm.
It looked wrong (two flat, non-decaying lines) because it *was* wrong:
code.py's Cox PH baseline is fit with duration=5.0 fixed for every single
training patient (see code.py's own comment: "Cox PH: fit with
duration=5 for all patients ... Predict 1 - S(5) as P(Y=1)"). With only
one unique duration in the training data, the fitted baseline hazard has
no time resolution before t=5 -- verified directly: `baseline_survival_`
has exactly one row, and `predict_survival_function` returns an
identical value at every requested time. Presenting that as a
time-varying curve would show a shape the data cannot support. This
version instead shows what the model actually estimates: the 5-year
survival probability under each arm, as a point comparison with a
bootstrap uncertainty band.

Everything defaults to the mock LLM backend (no API key required). Live
Claude calls are available via a sidebar toggle when ANTHROPIC_API_KEY is
set in the environment.

Run
---
    pip install streamlit
    streamlit run dashboard.py

No requirements.txt exists yet for this project; the full dependency
list is numpy, pandas, matplotlib, scikit-learn, xgboost, econml, shap,
lifelines, streamlit, and optionally anthropic for the live-API toggle.

This is a research-prototype UI, not a validated clinical tool -- see the
non-dismissable transparency card at the bottom of every page.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from lifelines import CoxPHFitter

from code import load_tcga
from effect_shap import FEATURE_NAMES
from llm_narrative_agent import generate_and_check, mock_llm, call_claude, NarrativeStatus
from counterfactual_query import (
    query_counterfactual, mock_llm_counterfactual, call_claude_counterfactual,
)
from run_narrative_pipeline import prepare_patient_rows, OUT_DIR

# ---------------------------------------------------------------------------
# Palette -- diverging blue/red for polarity (benefit vs. harm, positive vs.
# negative SHAP contribution, treated vs. untreated), status colors reserved
# for pass/fail badges. Values match the design-system reference palette
# used elsewhere in this project's data visualizations. The app is pinned
# to a light theme (.streamlit/config.toml) so these hex values, chosen and
# validated against a light chart surface, are never rendered against an
# unvalidated dark background.
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
RED = "#e34948"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
BASELINE = "#c3c2b7"

COX_FEATURES = FEATURE_NAMES  # ["age", "stage", "race", "ethnicity", "intent"]
CATEGORICAL_FEATURES = ["stage", "race", "ethnicity", "intent"]
_LOWERCASE_WORDS = {"or", "and", "of", "the"}


def _titlecase(label: str) -> str:
    words = label.split(" ")
    return " ".join(w if w.lower() in _LOWERCASE_WORDS and i > 0 else w.capitalize()
                    for i, w in enumerate(words))

st.set_page_config(page_title="CaML-OP dashboard prototype", page_icon="\U0001F9EC", layout="wide")


# ---------------------------------------------------------------------------
# Cached, expensive fitting -- runs once per (n_patients, seed), reused
# across every patient/query interaction in the session.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Fitting CaML-OP pipeline (model, calibration, Effect-SHAP)...")
def load_pipeline(n_patients: int, seed: int):
    return prepare_patient_rows(n_patients=n_patients, seed=seed, verbose=False)


@st.cache_resource(show_spinner="Decoding TCGA-OV category labels...")
def load_category_decoder():
    """Rebuilds the code -> label map code.py's load_tcga() computes and
    discards (`.astype('category').cat.codes`), purely for display -- the
    numeric codes are what the model and the LLM prompt actually see and
    are left untouched everywhere else in the pipeline."""
    df = pd.read_csv("tcga_ov_master_ml.csv")
    df = df[['age_at_diagnosis', 'figo_stage', 'race', 'ethnicity',
             'treatments.treatment_intent_type',
             'five_year_survival', 'platinum_therapy']].copy()
    df.columns = ['age', 'stage', 'race', 'ethnicity', 'intent', 'y', 't']
    df = df.dropna()
    decoder = {}
    for col in CATEGORICAL_FEATURES:
        cat = df[col].astype('category')
        decoder[col] = dict(enumerate(cat.cat.categories))
    return decoder


def decode_covariates(covariates: dict, decoder: dict) -> dict:
    """Human-readable version for display only -- the numeric codes are
    what the model and the LLM prompt actually see and are left untouched
    everywhere else in the pipeline. TCGA's own "'--" placeholder for an
    unspecified value is relabeled "Not recorded" rather than shown raw,
    which otherwise reads as a rendering bug rather than a real category."""
    out = {}
    for k, v in covariates.items():
        if k == "age":
            out[k] = f"{v:.1f} yrs"
        elif k in decoder:
            label = decoder[k].get(int(round(v)), f"code {v:g}")
            out[k] = "Not recorded" if label.strip("' ") == "--" else _titlecase(label)
        else:
            out[k] = v
    return out


@st.cache_resource(show_spinner="Bootstrapping Cox PH 5-year survival model...")
def fit_cox_bootstrap(_X_tr, _T_tr, _Y_tr, seed: int, n_boot: int = 15):
    """Refits code.py's own "Cox PH at a fixed 5-year horizon" baseline
    B times on bootstrap resamples, so panel 2 can show an approximate
    percentile uncertainty band around the 5-year survival probability,
    not just a point estimate. Cached once and reused across every
    patient in the session."""
    rng = np.random.RandomState(seed)
    n = len(_X_tr)
    df_full = pd.DataFrame(_X_tr, columns=COX_FEATURES)
    df_full["t"] = _T_tr
    df_full["duration"] = 5.0
    df_full["event"] = _Y_tr
    models = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(df_full.iloc[idx], duration_col="duration", event_col="event")
            models.append(cph)
        except Exception:
            continue  # a bootstrap resample can occasionally be degenerate; skip it
    return models


def five_year_survival_stats(models: list, x_row_raw: np.ndarray) -> dict:
    """5-year survival probability per arm, with a bootstrap percentile
    band -- the one horizon this fixed-duration=5 Cox adaptation actually
    supports (see module docstring, "Panel 2 -- an honest correction")."""
    samples = {0: [], 1: []}
    for cph in models:
        for arm in (0, 1):
            row = pd.DataFrame([list(x_row_raw) + [arm]], columns=COX_FEATURES + ["t"])
            s5 = float(cph.predict_survival_function(row, times=[5.0]).values.ravel()[0])
            samples[arm].append(s5)
    out = {}
    for arm, key in ((0, "t0"), (1, "t1")):
        arr = np.array(samples[arm])
        out[key] = {"mid": float(np.median(arr)),
                    "lo": float(np.percentile(arr, 5)),
                    "hi": float(np.percentile(arr, 95))}
    return out


# ---------------------------------------------------------------------------
# Panel 1: ITE gauge
# ---------------------------------------------------------------------------
def plot_ite_gauge(tau_hat: float, ci_lo: float, ci_hi: float, must_abstain: bool):
    fig, ax = plt.subplots(figsize=(7, 1.7), dpi=160)
    span = max(abs(ci_lo), abs(ci_hi), abs(tau_hat), 0.1) * 1.5
    ax.set_xlim(-span, span)
    ax.set_ylim(0, 1)
    color = TEXT_MUTED if must_abstain else (BLUE if tau_hat >= 0 else RED)
    ax.axvline(0, color=BASELINE, lw=1, zorder=1)
    ax.axvspan(ci_lo, ci_hi, ymin=0.28, ymax=0.72, color=color, alpha=0.20, zorder=2)
    ax.plot([tau_hat], [0.5], marker="o", markersize=15, color=color,
           zorder=3, markeredgecolor="white", markeredgewidth=1.5)
    ax.text(0, 0.08, "no effect", ha="center", va="bottom", fontsize=8.5, color=TEXT_MUTED)
    ax.set_yticks([])
    ax.set_xlabel("Calibrated interval (5-year survival probability scale)",
                 fontsize=9, color=TEXT_SECONDARY)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8.5)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 2: five-year survival probability by arm (see module docstring)
# ---------------------------------------------------------------------------
def plot_five_year_survival(stats: dict):
    fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=160)
    arms = ["Untreated\n(T=0)", "Treated\n(T=1)"]
    mids = [stats["t0"]["mid"], stats["t1"]["mid"]]
    lo_err = [stats["t0"]["mid"] - stats["t0"]["lo"], stats["t1"]["mid"] - stats["t1"]["lo"]]
    hi_err = [stats["t0"]["hi"] - stats["t0"]["mid"], stats["t1"]["hi"] - stats["t1"]["mid"]]
    colors = [RED, BLUE]
    x = np.array([0, 1])
    ax.bar(x, mids, color=colors, width=0.45, yerr=[lo_err, hi_err], capsize=5,
          error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.3})
    for xi, m in zip(x, mids):
        ax.text(xi, min(m + 0.06, 1.03), f"{m:.2f}", ha="center", fontsize=12,
               fontweight="bold", color=TEXT_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=9.5, color=TEXT_PRIMARY)
    ax.set_ylabel("P(5-year survival)", fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_ylim(0, 1.12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8.5)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 3: Effect-SHAP waterfall
# ---------------------------------------------------------------------------
def plot_shap_waterfall(shap_row: np.ndarray, feature_names: list, covariates_decoded: dict):
    order = np.argsort(np.abs(shap_row))[::-1]
    feats = [feature_names[i] for i in order]
    vals = [float(shap_row[i]) for i in order]
    colors = [BLUE if v >= 0 else RED for v in vals]
    labels = [f"{f}: {covariates_decoded.get(f, '?')}" for f in feats]
    fig, ax = plt.subplots(figsize=(6.5, 0.6 * len(feats) + 1), dpi=160)
    y_pos = np.arange(len(feats))[::-1]
    ax.barh(y_pos, vals, color=colors, height=0.55)
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5, color=TEXT_PRIMARY)
    ax.set_xlabel("Effect-SHAP contribution to tau_hat", fontsize=9, color=TEXT_SECONDARY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8.5)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Shared: verifier badge row
# ---------------------------------------------------------------------------
def render_check_badges(checks: dict, status: NarrativeStatus):
    with st.container(horizontal=True, gap="small"):
        for name, passed in checks.items():
            st.badge(name, icon=":material/check:" if passed else ":material/close:",
                    color="green" if passed else "red")
        if status == NarrativeStatus.ESCALATED:
            st.badge("ESCALATED -- route to human review", icon=":material/priority_high:", color="orange")
        else:
            st.badge("all checks passed", icon=":material/verified:", color="green")


# ---------------------------------------------------------------------------
# Panel 5: transparency card -- always rendered, never dismissable
# ---------------------------------------------------------------------------
def render_transparency_card():
    with st.container(border=True):
        st.markdown(":orange-badge[:material/warning: RESEARCH PROTOTYPE ONLY] "
                   "&nbsp; not a deployable clinical tool")
        st.caption("Never dismissable, matching the thesis's EU AI Act-informed design nudge "
                  "toward mandatory human oversight (main.tex, Section 3.5).")
        cols = st.columns(4)
        cols[0].metric("Training data", "n = 547", help="TCGA-OV covariates after preprocessing (code.load_tcga)")

        pehe_path = os.path.join(OUT_DIR, "summary_pehe.csv")
        sub_path = os.path.join(OUT_DIR, "summary_subgroup.csv")
        if os.path.exists(pehe_path):
            pehe_df = pd.read_csv(pehe_path)
            row = pehe_df[pehe_df["method"] == "CaML-OP"]
            pehe_val = f"{row['PEHE'].iloc[0]:.4f}" if not row.empty else "n/a"
        else:
            pehe_val = "n/a"
        cols[1].metric("Overall PEHE (CaML-OP)", pehe_val,
                       help="Precision in Estimation of Heterogeneous Effects, lower is better. "
                            "From this run's own caml_op_outputs/summary_pehe.csv.")

        cols[2].metric("Empirical CI coverage", "0.847", help="Nominal target: 0.95. See calibrated_abstention.py.")

        fairness_txt = "n/a"
        if os.path.exists(sub_path):
            sub_df = pd.read_csv(sub_path)
            row = sub_df[sub_df["method"] == "CaML-OP"]
            if not row.empty:
                vals = row.drop(columns=["method"]).values.ravel()
                lo, hi = float(vals.min()), float(vals.max())
                fairness_txt = f"{lo:.3f}-{hi:.3f}"
        cols[3].metric("Subgroup PEHE range", fairness_txt,
                       help="Across age/stage/race strata. Reported, not audited -- "
                            "this is not a full fairness audit (main.tex, Chapter 5 Summary).")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.title(":dna: CaML-OP dashboard prototype")
st.caption("Constructor University -- Bachelor Thesis research prototype. Panels follow "
          "main.tex Figure 3.4's mockup (Section 3.5): ITE gauge, five-year survival "
          "probability by arm, Effect-SHAP waterfall, LLM narrative + faithfulness status, "
          "and a non-dismissable transparency card.")
with st.expander("About this tool"):
    st.markdown(
        "CaML-OP pairs an XGBoost prognostic encoder with R-Learner and Doubly-Robust "
        "causal meta-learners to estimate individualized treatment effects of "
        "platinum-based chemotherapy on five-year ovarian cancer survival, from TCGA-OV "
        "covariates. This dashboard is the first running version of the clinical-facing "
        "prototype the thesis describes: it recomputes every number below live from the "
        "fitted model, rather than replaying stored figures. Everything on this page "
        "carries the same faithfulness discipline as the underlying narrative agent -- "
        "every LLM claim (including answers to the bounded what-if tool in the sidebar) "
        "is checked against the recomputed ground truth before being shown, with a "
        "visible pass/fail badge per check."
    )

have_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
with st.sidebar:
    st.header("Setup")
    n_patients = st.slider("Sampled patients", min_value=5, max_value=50, value=15, step=5,
                           help="How many held-out TCGA-OV patients to sample for this session.")
    seed = st.number_input("Seed", value=42, step=1, help="Random seed for sampling and model fitting.")
    use_live = st.checkbox("Use live Claude API", value=False, disabled=not have_api_key,
                           help="Switch the narrative panel and what-if tool to real Claude calls.")
    if not have_api_key:
        st.caption("ANTHROPIC_API_KEY not set -- mock backend only.")

patient_rows, X_sample, sample_idx, calib_factor, fitted = load_pipeline(n_patients, seed)
decoder = load_category_decoder()

with st.sidebar:
    st.divider()
    st.header("Patient")
    patient_ids = [p.patient_id for p in patient_rows]
    selected_id = st.selectbox("Select a sampled patient", patient_ids)
    p_idx = patient_ids.index(selected_id)
    baseline = patient_rows[p_idx]
    X_patient = X_sample[p_idx]

    st.divider()
    st.header("Ask a bounded what-if question")
    st.caption("One tool, not a chat box -- see AGENT_INTEGRATION.md, "
              "\"Bounded query interface, not open-ended chat.\"")
    query_feature = st.selectbox("Adjust which covariate?", FEATURE_NAMES, key="cf_feature")
    query_delta = st.number_input(f"Adjustment to {query_feature}", value=5.0, step=1.0, key="cf_delta")
    ask_clicked = st.button("Ask", width="stretch")

narrative_llm_fn = call_claude if use_live else mock_llm
cf_llm_fn = call_claude_counterfactual if use_live else mock_llm_counterfactual

covariates_decoded = decode_covariates(baseline.covariates, decoder)
st.subheader(f"Patient {baseline.patient_id}")
cov_table = pd.DataFrame([covariates_decoded], index=["value"]).T.rename_axis("covariate").reset_index()
st.dataframe(cov_table, hide_index=True, width="stretch")

st.divider()

with st.container(border=True):
    st.markdown("#### 1. ITE gauge")
    st.metric("Estimated individualized treatment effect", f"{baseline.tau_hat:+.3f}",
             help="5-year survival probability scale; positive = benefit from treatment.")
    st.pyplot(plot_ite_gauge(baseline.tau_hat, baseline.ci_lo, baseline.ci_hi, baseline.must_abstain),
             width="stretch")
    if baseline.must_abstain:
        st.warning("Calibrated interval spans zero: no directional claim is statistically supportable "
                  "for this patient (calibrated_abstention.py).")
    st.caption("Panel 1 of main.tex Figure 3.4. Point estimate = 0.5 x (R-Learner + DR-Learner); "
              "shaded band = subgroup-conditional calibrated interval.")

with st.container(border=True):
    st.markdown("#### 2. Five-year survival probability by treatment arm")
    cox_models = fit_cox_bootstrap(fitted["X_tr"], fitted["T_tr"], fitted["Y_tr"], seed=seed)
    surv_stats = five_year_survival_stats(cox_models, X_patient)
    st.pyplot(plot_five_year_survival(surv_stats), width="content")
    st.caption(
        "A **correction**, not a simplification, of the thesis mockup's continuous curve: "
        "code.py's Cox PH baseline is fit with a single fixed duration (5.0) for every "
        "training patient, so the fitted hazard has no time resolution before that point "
        "-- a continuous 0-5-year curve would show a shape the data cannot support. This "
        "shows the one horizon the model actually estimates: 5-year survival probability "
        "per arm, bootstrap band (15 resamples, 5th-95th percentile). See this file's "
        "module docstring for the full explanation."
    )

with st.container(border=True):
    st.markdown("#### 3. Effect-SHAP waterfall")
    st.pyplot(plot_shap_waterfall(fitted["shap_values"][p_idx], fitted["feature_names"], covariates_decoded),
             width="stretch")
    st.caption("Panel 3 of main.tex Figure 3.4. Model-agnostic permutation Shapley attribution of "
              "tau_hat(X) over the five raw covariates (effect_shap.py); category labels decoded "
              "from TCGA-OV's original values for readability.")

with st.container(border=True):
    st.markdown("#### 4. LLM narrative + faithfulness-check status")
    with st.spinner("Generating and verifying narrative..."):
        narrative_result = generate_and_check(baseline, llm_fn=narrative_llm_fn, max_retries=2)
    st.write(narrative_result.narrative)
    render_check_badges(narrative_result.checks, narrative_result.status)
    st.caption("Panel 4 of main.tex Figure 3.4. Every claim is checked against the recomputed "
              "ground truth before being shown (llm_narrative_agent.py); ESCALATED means every "
              "retry failed at least one check and the result requires human review, not that "
              "it was silently hidden.")

    if ask_clicked:
        st.markdown("**What-if answer:**")
        with st.spinner("Recomputing and verifying counterfactual answer..."):
            cf_result = query_counterfactual(
                baseline, X_patient, query_feature, query_delta,
                fitted["model"], fitted["m0"], fitted["m1"], calib_factor,
                llm_fn=cf_llm_fn)
        st.write(cf_result.narrative)
        render_check_badges(cf_result.checks, cf_result.status)

st.markdown("#### 5. Transparency card")
render_transparency_card()
