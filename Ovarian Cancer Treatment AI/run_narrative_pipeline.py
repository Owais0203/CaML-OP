"""
End-to-end demo: TCGA-OV covariates -> CaML-OP tau_hat -> Effect-SHAP ->
calibrated abstention -> LLM narrative agent -> faithfulness checks ->
empirical pass-rate.

This ties together code.py (data loading, CaML-OP estimator),
effect_shap.py (explainability layer + interval), calibrated_abstention.py
(the radical piece: decide whether a directional claim is even
statistically supportable) and llm_narrative_agent.py (the narrative agent
+ faithfulness checks) to actually run the architecture main.tex describes
but code.py alone never executes.

By default this uses the semi-synthetic DGP (scenario 4, the hardest) so
tau_hat can be compared against a known ground truth, and the mock LLM
backend so it runs without any API key. Pass --live to use a real Claude
call (requires ANTHROPIC_API_KEY) and produce a genuine empirical
pass-rate suitable for citing in the thesis instead of the mock's.

Usage
-----
    python run_narrative_pipeline.py [--n-patients 30] [--live]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split

from code import load_tcga, generate_dgp
from effect_shap import CaMLOPEffectModel, compute_effect_shap, top_features, FEATURE_NAMES
from calibrated_abstention import fit_calibration_factor, should_abstain
from llm_narrative_agent import NarrativeInputs, run_eval, mock_llm, call_claude

OUT_DIR = "caml_op_outputs"


def prepare_patient_rows(n_patients: int = 30, seed: int = 42, verbose: bool = True):
    """Shared data/model/calibration prep, factored out so narrative_research_eval.py
    can reuse the exact same patients (and their raw covariates, for the
    subgroup breakdown) instead of duplicating this fitting logic.

    Returns (patient_rows, X_sample, sample_idx, calib_factor).
    """
    def log(msg):
        if verbose:
            print(msg)

    log("Loading TCGA-OV covariates...")
    X_raw, _, _, _ = load_tcga()

    log("Simulating outcomes under DGP scenario 4 (interaction, hardest case)...")
    Y, T, tau_true, e = generate_dgp(X_raw, seed=seed, scenario=4)

    idx = np.arange(len(X_raw))
    tr_idx, te_idx = train_test_split(idx, test_size=0.25, random_state=seed, stratify=T)
    keep = (e[tr_idx] > 0.05) & (e[tr_idx] < 0.95)
    X_tr, T_tr, Y_tr = X_raw[tr_idx][keep], T[tr_idx][keep], Y[tr_idx][keep]
    X_te, tau_true_te = X_raw[te_idx], tau_true[te_idx]

    n_patients = min(n_patients, len(X_te))
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(len(X_te), size=n_patients, replace=False)
    cal_idx = np.setdiff1d(np.arange(len(X_te)), sample_idx)  # disjoint calibration split
    X_sample = X_te[sample_idx]

    log(f"Fitting CaML-OP effect model on {len(X_tr)} training patients...")
    model = CaMLOPEffectModel(seed=seed).fit(X_tr, T_tr, Y_tr)
    tau_hat = model.predict(X_sample)
    ci_lo, ci_hi = model.predict_interval(X_sample)
    half_width = (ci_hi - ci_lo) / 2

    log(f"Fitting split-conformal calibration factor on {len(cal_idx)} held-out "
        f"test patients disjoint from the {n_patients} narrated above...")
    tau_hat_cal = model.predict(X_te[cal_idx])
    cal_lo, cal_hi = model.predict_interval(X_te[cal_idx])
    calib_factor = fit_calibration_factor(
        tau_true_te[cal_idx], tau_hat_cal, (cal_hi - cal_lo) / 2, target_coverage=0.95)
    must_abstain = np.array([
        should_abstain(tau_hat[i], half_width[i], calib_factor)
        for i in range(n_patients)
    ])
    log(f"Calibration factor: {calib_factor:.3f}x reported half-width. "
        f"{must_abstain.sum()}/{n_patients} sampled patients must abstain "
        f"from a directional sign claim once calibrated.")

    log(f"Computing Effect-SHAP for {n_patients} sampled patients "
        f"(model-agnostic permutation explainer)...")
    shap_values, feature_names = compute_effect_shap(model, X_tr, X_sample, seed=seed)

    patient_rows = []
    for i in range(n_patients):
        pos, neg = top_features(shap_values[i], feature_names, k=3)
        covariates = dict(zip(FEATURE_NAMES, X_sample[i].round(2).tolist()))
        patient_rows.append(NarrativeInputs(
            patient_id=f"P{sample_idx[i]:04d}",
            covariates=covariates,
            tau_hat=float(tau_hat[i]),
            ci_lo=float(ci_lo[i]),
            ci_hi=float(ci_hi[i]),
            top_positive=pos,
            top_negative=neg,
            must_abstain=bool(must_abstain[i]),
        ))
    return patient_rows, X_sample, sample_idx, calib_factor


def main(n_patients: int = 30, seed: int = 42, live: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)
    patient_rows, X_sample, sample_idx, calib_factor = prepare_patient_rows(n_patients, seed)
    must_abstain = np.array([inp.must_abstain for inp in patient_rows])

    llm_fn = call_claude if live else mock_llm
    model_name = "claude-sonnet-5" if live else "mock"
    backend = "Claude (live API)" if live else "mock template (no API key required)"
    print(f"Generating narratives + running faithfulness checks via {backend}...")
    results = run_eval(
        patient_rows, llm_fn=llm_fn, model_name=model_name,
        max_retries=2, max_workers=(8 if live else 1),
        out_csv=os.path.join(OUT_DIR, "narrative_eval.csv"),
        cache_path=os.path.join(OUT_DIR, ".narrative_cache.jsonl"),
    )

    print("\n" + "=" * 70)
    print(f"NARRATIVE FAITHFULNESS -- empirical pass-rate (n={len(results)}, "
          f"backend={backend}, prompt_version={results['prompt_version'].iloc[0]})")
    print("=" * 70)
    for col in ["pass_sign", "pass_features", "pass_numeric", "pass_overall"]:
        print(f"  {col:16s} {results[col].mean():.3f}")
    print(f"  {'mean_attempts':16s} {results['attempts'].mean():.2f}  "
          f"(1.00 = every narrative passed on the first try, no retries needed)")
    print(f"  {'abstain_rate':16s} {must_abstain.mean():.3f}  "
          f"(fraction of patients where calibration found the sign not "
          f"distinguishable from zero)")
    print(f"\nWrote per-patient detail to {OUT_DIR}/narrative_eval.csv")
    if not live:
        print("\nNOTE: this run used the mock LLM backend. These pass-rates "
              "describe the checking logic, not an LLM's behaviour, and "
              "must not be cited as the thesis's empirical result for RQ2. "
              "Re-run with --live and ANTHROPIC_API_KEY set for a real number.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patients", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--live", action="store_true",
                        help="Use a real Claude call instead of the mock LLM "
                             "(requires the anthropic package and ANTHROPIC_API_KEY).")
    args = parser.parse_args()
    main(n_patients=args.n_patients, seed=args.seed, live=args.live)
