"""
CaML-OP clinical dashboard -- the first runnable version of the "Clinical
dashboard prototype" main.tex describes (Section 3.5, Figure 3.4) as a
static mockup image (media/image3.png). This wires the five panels the
thesis already specifies to the code that actually computes them:

  1. ITE gauge            -> effect_shap.CaMLOPEffectModel (tau_hat, CI)
  2. Counterfactual        -> a bootstrapped Cox PH model (code.py's own
     survival curves          "adapt Cox PH to a fixed 5-year horizon" trick),
                              refit per bootstrap resample for an approximate
                              percentile uncertainty band
  3. Effect-SHAP waterfall -> effect_shap.compute_effect_shap
  4. LLM narrative panel   -> llm_narrative_agent.generate_and_check
  5. Transparency card     -> caml_op_outputs/summary_*.csv (this run's own
                              benchmark numbers), always rendered (never in
                              a collapsible/dismissable widget)

Plus one interactive control the static mockup never had: a bounded
what-if query box (counterfactual_query.py) in the sidebar, deliberately
NOT an open-ended chat box -- see AGENT_INTEGRATION.md, "Bounded query
interface, not open-ended chat," for why.

Everything defaults to the mock LLM backend (no API key required). Live
Claude calls are available via a sidebar toggle when ANTHROPIC_API_KEY is
set in the environment.

Run
---
    pip install streamlit
    streamlit run dashboard.py

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

from effect_shap import FEATURE_NAMES
from llm_narrative_agent import generate_and_check, mock_llm, call_claude, NarrativeStatus
from counterfactual_query import (
    query_counterfactual, mock_llm_counterfactual, call_claude_counterfactual,
)
from run_narrative_pipeline import prepare_patient_rows, OUT_DIR

# ---------------------------------------------------------------------------
# Palette -- diverging blue/red for polarity (benefit vs. harm, positive vs.
# negative SHAP contribution), status colors reserved for pass/fail badges.
# Values match the design-system reference palette used elsewhere in this
# project's data visualizations.
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
RED = "#e34948"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WARNING = "#fab219"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
BASELINE = "#c3c2b7"

COX_FEATURES = FEATURE_NAMES  # ["age", "stage", "race", "ethnicity", "intent"]

st.set_page_config(page_title="CaML-OP dashboard prototype", layout="wide")


# ---------------------------------------------------------------------------
# Cached, expensive fitting -- runs once per (n_patients, seed), reused
# across every patient/query interaction in the session.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Fitting CaML-OP pipeline (model, calibration, Effect-SHAP)...")
def load_pipeline(n_patients: int, seed: int):
    return prepare_patient_rows(n_patients=n_patients, seed=seed, verbose=False)


@st.cache_resource(show_spinner="Bootstrapping Cox PH survival curves...")
def fit_cox_bootstrap(_X_tr, _T_tr, _Y_tr, seed: int, n_boot: int = 15):
    """Refits code.py's own "Cox PH at a fixed 5-year horizon" baseline
    B times on bootstrap resamples, so panel 2 can show an approximate
    percentile uncertainty band, not just a point estimate. Cached once and
    reused across every patient in the session -- only the cheap
    predict_survival_function step runs per patient."""
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


def survival_curves_for_patient(models: list, x_row_raw: np.ndarray, times: np.ndarray) -> dict:
    curves = {0: [], 1: []}
    for cph in models:
        for arm in (0, 1):
            row = pd.DataFrame([list(x_row_raw) + [arm]], columns=COX_FEATURES + ["t"])
            sf = cph.predict_survival_function(row, times=times)
            curves[arm].append(sf.values.ravel())
    out = {}
    for arm, key in ((0, "t0"), (1, "t1")):
        arr = np.array(curves[arm])
        out[key] = {"mid": np.median(arr, axis=0),
                    "lo": np.percentile(arr, 5, axis=0),
                    "hi": np.percentile(arr, 95, axis=0)}
    return out


# ---------------------------------------------------------------------------
# Panel 1: ITE gauge
# ---------------------------------------------------------------------------
def plot_ite_gauge(tau_hat: float, ci_lo: float, ci_hi: float, must_abstain: bool):
    fig, ax = plt.subplots(figsize=(6, 1.6))
    span = max(abs(ci_lo), abs(ci_hi), abs(tau_hat), 0.1) * 1.6
    ax.set_xlim(-span, span)
    ax.set_ylim(0, 1)
    color = TEXT_MUTED if must_abstain else (BLUE if tau_hat >= 0 else RED)
    ax.axvline(0, color=BASELINE, lw=1, zorder=1)
    ax.axvspan(ci_lo, ci_hi, ymin=0.25, ymax=0.75, color=color, alpha=0.22, zorder=2)
    ax.plot([tau_hat], [0.5], marker="o", markersize=16, color=color, zorder=3)
    ax.text(tau_hat, 0.9, f"{tau_hat:+.3f}", ha="center", va="bottom",
           fontsize=12, color=TEXT_PRIMARY, fontweight="bold")
    ax.text(0, 0.05, "no effect", ha="center", va="bottom", fontsize=8, color=TEXT_MUTED)
    ax.set_yticks([])
    ax.set_xlabel("Estimated ITE (5-year survival probability scale) with calibrated interval",
                 fontsize=8, color=TEXT_SECONDARY)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 2: counterfactual survival curves
# ---------------------------------------------------------------------------
def plot_survival_curves(times: np.ndarray, curves: dict):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(times, curves["t1"]["mid"], color=BLUE, lw=2, label="Treated (T=1)")
    ax.fill_between(times, curves["t1"]["lo"], curves["t1"]["hi"], color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(times, curves["t0"]["mid"], color=RED, lw=2, label="Untreated (T=0)")
    ax.fill_between(times, curves["t0"]["lo"], curves["t0"]["hi"], color=RED, alpha=0.18, linewidth=0)
    ax.set_xlabel("Years since diagnosis", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("Estimated survival probability", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 3: Effect-SHAP waterfall
# ---------------------------------------------------------------------------
def plot_shap_waterfall(shap_row: np.ndarray, feature_names: list, covariates: dict):
    order = np.argsort(np.abs(shap_row))[::-1]
    feats = [feature_names[i] for i in order]
    vals = [float(shap_row[i]) for i in order]
    colors = [BLUE if v >= 0 else RED for v in vals]
    labels = [f"{f} = {covariates.get(f, '?')}" for f in feats]
    fig, ax = plt.subplots(figsize=(6, 0.55 * len(feats) + 1))
    y_pos = np.arange(len(feats))[::-1]
    ax.barh(y_pos, vals, color=colors, height=0.6)
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlabel("Effect-SHAP contribution to tau_hat", fontsize=8, color=TEXT_SECONDARY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 5: transparency card -- always rendered, never dismissable
# ---------------------------------------------------------------------------
def render_transparency_card():
    with st.container(border=True):
        st.markdown("### :warning: RESEARCH PROTOTYPE ONLY -- not a deployable clinical tool")
        st.caption("This card is never dismissable, matching the thesis's EU AI Act-informed "
                  "design nudge toward mandatory human oversight (main.tex, Section 3.5).")
        cols = st.columns(4)
        cols[0].markdown("**Training data**")
        cols[0].write("TCGA-OV covariates, n=547 after preprocessing (real.load_tcga)")
        cols[1].markdown("**Empirical CI coverage**")
        cols[1].write("0.847 (nominal 0.95 target) -- see calibrated_abstention.py")

        pehe_path = os.path.join(OUT_DIR, "summary_pehe.csv")
        sub_path = os.path.join(OUT_DIR, "summary_subgroup.csv")
        if os.path.exists(pehe_path):
            pehe_df = pd.read_csv(pehe_path)
            row = pehe_df[pehe_df["method"] == "CaML-OP"]
            pehe_val = f"{row['PEHE'].iloc[0]:.4f}" if not row.empty else "n/a"
        else:
            pehe_val = "n/a (caml_op_outputs/summary_pehe.csv not found)"
        cols[2].markdown("**Overall PEHE (CaML-OP)**")
        cols[2].write(pehe_val)

        fairness_txt = "n/a"
        if os.path.exists(sub_path):
            sub_df = pd.read_csv(sub_path)
            row = sub_df[sub_df["method"] == "CaML-OP"]
            if not row.empty:
                vals = row.drop(columns=["method"]).values.ravel()
                lo, hi = float(vals.min()), float(vals.max())
                fairness_txt = f"{lo:.4f}-{hi:.4f} across age/stage/race subgroups"
        cols[3].markdown("**Subgroup PEHE range**")
        cols[3].write(fairness_txt)
        st.caption("Subgroup range is reported, not audited -- this is not a full fairness "
                  "audit (main.tex, Chapter 5 Summary).")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.title("CaML-OP dashboard prototype")
st.caption("Panels match main.tex Figure 3.4's mockup: ITE gauge, counterfactual survival "
          "curves, Effect-SHAP waterfall, LLM narrative + faithfulness status, and a "
          "non-dismissable transparency card.")

have_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
with st.sidebar:
    st.header("Setup")
    n_patients = st.slider("Sampled patients", min_value=5, max_value=50, value=15, step=5)
    seed = st.number_input("Seed", value=42, step=1)
    use_live = st.checkbox("Use live Claude API", value=False, disabled=not have_api_key)
    if not have_api_key:
        st.caption("ANTHROPIC_API_KEY not set -- mock backend only.")

patient_rows, X_sample, sample_idx, calib_factor, fitted = load_pipeline(n_patients, seed)

with st.sidebar:
    st.header("Patient")
    patient_ids = [p.patient_id for p in patient_rows]
    selected_id = st.selectbox("Select a sampled patient", patient_ids)
    p_idx = patient_ids.index(selected_id)
    baseline = patient_rows[p_idx]
    X_patient = X_sample[p_idx]

    st.header("Ask a bounded what-if question")
    st.caption("One tool, not a chat box -- see AGENT_INTEGRATION.md, "
              "\"Bounded query interface, not open-ended chat.\"")
    query_feature = st.selectbox("Adjust which covariate?", FEATURE_NAMES, key="cf_feature")
    query_delta = st.number_input(f"Adjustment to {query_feature}", value=5.0, step=1.0, key="cf_delta")
    ask_clicked = st.button("Ask")

narrative_llm_fn = call_claude if use_live else mock_llm
cf_llm_fn = call_claude_counterfactual if use_live else mock_llm_counterfactual

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. ITE gauge")
    st.pyplot(plot_ite_gauge(baseline.tau_hat, baseline.ci_lo, baseline.ci_hi, baseline.must_abstain),
              width='stretch')
    if baseline.must_abstain:
        st.warning("Calibrated interval spans zero: no directional claim is statistically supportable.")

with col2:
    st.subheader("3. Effect-SHAP waterfall")
    st.pyplot(plot_shap_waterfall(fitted["shap_values"][p_idx], fitted["feature_names"], baseline.covariates),
              width='stretch')

st.subheader("2. Counterfactual survival curves")
st.caption("Cox PH refit on 15 bootstrap resamples of the training fold (code.py's own "
          "fixed-5-year-horizon adaptation); band = 5th-95th percentile across resamples, "
          "an approximate uncertainty band, not a formally derived CI.")
cox_models = fit_cox_bootstrap(fitted["X_tr"], fitted["T_tr"], fitted["Y_tr"], seed=seed)
times = np.linspace(0, 5, 26)
curves = survival_curves_for_patient(cox_models, X_patient, times)
st.pyplot(plot_survival_curves(times, curves), width='stretch')

st.subheader("4. LLM narrative + faithfulness-check status")
with st.spinner("Generating and verifying narrative..."):
    narrative_result = generate_and_check(baseline, llm_fn=narrative_llm_fn, max_retries=2)
st.write(narrative_result.narrative)
check_cols = st.columns(len(narrative_result.checks) + 1)
for i, (name, passed) in enumerate(narrative_result.checks.items()):
    with check_cols[i]:
        (st.success if passed else st.error)(name)
with check_cols[-1]:
    if narrative_result.status == NarrativeStatus.ESCALATED:
        st.warning("ESCALATED: route to human review")
    else:
        st.success("all checks passed")

if ask_clicked:
    with st.spinner("Recomputing and verifying counterfactual answer..."):
        cf_result = query_counterfactual(
            baseline, X_patient, query_feature, query_delta,
            fitted["model"], fitted["m0"], fitted["m1"], calib_factor,
            llm_fn=cf_llm_fn)
    st.markdown("**What-if answer:**")
    st.write(cf_result.narrative)
    cf_cols = st.columns(len(cf_result.checks) + 1)
    for i, (name, passed) in enumerate(cf_result.checks.items()):
        with cf_cols[i]:
            (st.success if passed else st.error)(name)
    with cf_cols[-1]:
        if cf_result.status == NarrativeStatus.ESCALATED:
            st.warning("ESCALATED: route to human review")
        else:
            st.success("all checks passed")

st.subheader("5. Transparency card")
render_transparency_card()
