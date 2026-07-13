"""
Research-grade evaluation harness for the narrative agent.

Three additions beyond the bare pass-rate `run_narrative_pipeline.py`
already reports, aimed at the gap between "working prototype" and evidence
a thesis chapter could actually cite:

  1. Subgroup breakdown -- ties RQ2 (narrative fidelity) to RQ3 (subgroup
     fairness), which the thesis currently evaluates separately. Is the
     narrative agent equally faithful across the same age/stage/race
     subgroups the causal estimator is already audited on in Section
     "Subgroup analysis"? Reuses code.py's own subgroup_indices() so the
     subgroup definitions are identical to the ones already used for PEHE.

  2. Verifier stress test -- the previous version of this module had one
     hand-written adversarial example proving the checks *can* catch a
     lie. That's a proof of existence, not a validation. This treats
     check_sign/check_features/check_numeric as a diagnostic classifier
     against a synthetically labeled dataset (known ground truth by
     construction) and reports sensitivity/specificity, the standard way
     to validate a detector rather than spot-check it.

  3. Ablation table -- isolates what the retry loop actually buys. The
     deterministic mock_llm never errs, so it can't demonstrate retry's
     benefit (there's nothing to recover from). This uses a stochastic
     mock LLM with a configurable, injected error rate to show pass-rate
     as a function of max_retries under controlled conditions -- an
     explicit simulation, not a claim about any real LLM's error rate.
     Replace the stochastic mock with call_claude to get the real number.

Usage
-----
    python narrative_research_eval.py [--n-patients 40]
"""
from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from code import subgroup_indices
from llm_narrative_agent import (
    NarrativeInputs, NarrativeOutput, CitedValue,
    check_sign, check_features, check_numeric,
    default_verifiers, generate_and_check, mock_llm, run_eval,
    _dominant_uncertainty_source,
)
from run_narrative_pipeline import prepare_patient_rows

OUT_DIR = "caml_op_outputs"


# ---------------------------------------------------------------------------
# Wilson score interval -- appropriate near p=0 or p=1, unlike the normal
# approximation, which matters here since several pass-rates below are 1.0.
# ---------------------------------------------------------------------------
def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ---------------------------------------------------------------------------
# 1. Subgroup breakdown (ties RQ2 to RQ3)
# ---------------------------------------------------------------------------
def subgroup_pass_rates(results: pd.DataFrame, X_sample: np.ndarray,
                        patient_ids: list, min_n: int = 5) -> pd.DataFrame:
    groups = subgroup_indices(X_sample)
    id_to_pass = dict(zip(results.patient_id, results.pass_overall))
    rows = []
    for g_name, mask in groups.items():
        ids = [pid for pid, m in zip(patient_ids, mask) if m]
        passes = [id_to_pass[pid] for pid in ids if pid in id_to_pass]
        if len(passes) < min_n:
            continue
        lo, hi = wilson_ci(sum(passes), len(passes))
        rows.append({"subgroup": g_name, "n": len(passes),
                    "pass_rate": np.mean(passes), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def compare_subgroups(results: pd.DataFrame, X_sample: np.ndarray, patient_ids: list,
                      group_a: str, group_b: str) -> dict:
    """Fisher's exact test on pass/fail counts between two subgroups --
    appropriate given the small per-subgroup counts here, matching the test
    the thesis's own subgroup PEHE analysis would need at this sample size
    but doesn't currently run."""
    groups = subgroup_indices(X_sample)
    id_to_pass = dict(zip(results.patient_id, results.pass_overall))

    def counts(mask):
        ids = [pid for pid, m in zip(patient_ids, mask) if m]
        passes = [id_to_pass[pid] for pid in ids if pid in id_to_pass]
        return sum(passes), len(passes) - sum(passes)

    a_pass, a_fail = counts(groups[group_a])
    b_pass, b_fail = counts(groups[group_b])
    odds_ratio, p_value = fisher_exact([[a_pass, a_fail], [b_pass, b_fail]])
    return {"group_a": group_a, "group_b": group_b,
           "a_pass": a_pass, "a_fail": a_fail, "b_pass": b_pass, "b_fail": b_fail,
           "odds_ratio": odds_ratio, "p_value": p_value}


# ---------------------------------------------------------------------------
# 2. Verifier stress test -- checks as a diagnostic classifier
# ---------------------------------------------------------------------------
def _base_inputs(must_abstain: bool = False, tau_hat: float = 0.08,
                 mu0: float = 0.30, mu1: float = 0.38,
                 tau_r: float = 0.07, tau_dr: float = 0.09) -> NarrativeInputs:
    return NarrativeInputs(
        "STRESS", {"age": 55.0, "stage": 2, "race": 0, "ethnicity": 0, "intent": 1},
        tau_hat=tau_hat, ci_lo=tau_hat - 0.10, ci_hi=tau_hat + 0.10,
        mu0=mu0, mu1=mu1, tau_r=tau_r, tau_dr=tau_dr,
        top_positive=[("age", 0.03), ("stage", 0.01)],
        top_negative=[("ethnicity", -0.02)],
        must_abstain=must_abstain,
    )


def _stress_cases():
    """(name, ground-truth is_faithful, inputs, output) tuples. Ground
    truth is known by construction -- each case is deliberately built to
    be faithful or to contain exactly one class of violation. Every
    "faithful" case's output must satisfy ALL five default verifiers, not
    just the one the case name targets -- adding mu0/mu1 citation and
    dominant_uncertainty_source to every case (including ones testing an
    unrelated check) is what keeps the ground-truth labels accurate
    against the full gate, not just the original three checks."""
    inp = _base_inputs()
    abstain_inp = _base_inputs(must_abstain=True, tau_hat=0.005, mu0=0.40, mu1=0.401,
                               tau_r=0.001, tau_dr=0.003)
    true_drivers = inp.true_drivers()
    dom = _dominant_uncertainty_source(inp)          # "identification" for these numbers
    dom_abstain = _dominant_uncertainty_source(abstain_inp)
    mu_cites = [CitedValue("mu0", inp.mu0), CitedValue("mu1", inp.mu1)]
    mu_cites_abstain = [CitedValue("mu0", abstain_inp.mu0), CitedValue("mu1", abstain_inp.mu1)]

    cases = [
        ("faithful_baseline", True, inp,
         NarrativeOutput("x", "positive", ["age"], [CitedValue("tau_hat", inp.tau_hat)] + mu_cites, dom)),
        ("faithful_cites_all_drivers", True, inp,
         NarrativeOutput("x", "positive", list(true_drivers), [CitedValue("tau_hat", inp.tau_hat)] + mu_cites, dom)),
        ("faithful_numeric_within_atol_floor", True, inp,
         NarrativeOutput("x", "positive", ["age"], [CitedValue("tau_hat", inp.tau_hat + 0.001)] + mu_cites, dom)),
        ("wrong_sign", False, inp,
         NarrativeOutput("x", "negative", ["age"], [CitedValue("tau_hat", inp.tau_hat)] + mu_cites, dom)),
        ("hallucinated_feature", False, inp,
         NarrativeOutput("x", "positive", ["race"], [CitedValue("tau_hat", inp.tau_hat)] + mu_cites, dom)),
        ("hallucinated_quantity_tag", False, inp,
         NarrativeOutput("x", "positive", ["age"], [CitedValue("shap:race", 0.05)] + mu_cites, dom)),
        ("numeric_off_by_50pct", False, inp,
         NarrativeOutput("x", "positive", ["age"], [CitedValue("tau_hat", inp.tau_hat * 1.5)] + mu_cites, dom)),
        ("empty_citation_when_drivers_exist", False, inp,
         NarrativeOutput("x", "positive", [], [CitedValue("tau_hat", inp.tau_hat)] + mu_cites, dom)),
        ("mixed_true_and_hallucinated_feature", False, inp,
         NarrativeOutput("x", "positive", ["age", "race"], [CitedValue("tau_hat", inp.tau_hat)] + mu_cites, dom)),
        ("abstains_correctly", True, abstain_inp,
         NarrativeOutput("x", "indeterminate", ["age"], [CitedValue("tau_hat", abstain_inp.tau_hat)] + mu_cites_abstain, dom_abstain)),
        ("overconfident_should_abstain", False, abstain_inp,
         NarrativeOutput("x", "positive", ["age"], [CitedValue("tau_hat", abstain_inp.tau_hat)] + mu_cites_abstain, dom_abstain)),
        ("abstains_when_it_shouldnt", False, inp,
         NarrativeOutput("x", "indeterminate", ["age"], [CitedValue("tau_hat", inp.tau_hat)] + mu_cites, dom)),
        # -- causal-structure checks --
        ("counterfactual_contradiction", False, inp,
         NarrativeOutput("x", "positive", ["age"], [CitedValue("tau_hat", inp.tau_hat),
                         CitedValue("mu0", inp.mu1), CitedValue("mu1", inp.mu0)], dom)),  # arms swapped
        ("counterfactual_missing_citation", False, inp,
         NarrativeOutput("x", "positive", ["age"], [CitedValue("tau_hat", inp.tau_hat)], dom)),  # no mu0/mu1 at all
        ("wrong_dominant_uncertainty_source", False, inp,
         NarrativeOutput("x", "positive", ["age"], [CitedValue("tau_hat", inp.tau_hat)] + mu_cites,
                         "sampling" if dom != "sampling" else "model_disagreement")),
    ]
    return cases


def run_verifier_stress_test() -> pd.DataFrame:
    """Runs every default verifier (not a hardcoded subset), so a new
    Verifier appended to default_verifiers() is automatically covered by
    this stress test without editing this function too."""
    verifiers = default_verifiers()
    rows = []
    for name, is_faithful, case_inp, out in _stress_cases():
        results = {v.name: v.check(out, case_inp) for v in verifiers}
        predicted_faithful = all(results.values())
        rows.append({
            "case": name, "true_faithful": is_faithful,
            "predicted_faithful": predicted_faithful,
            **{f"pass_{k}": v for k, v in results.items()},
            "correct": predicted_faithful == is_faithful,
        })
    return pd.DataFrame(rows)


def verifier_stress_test_summary(df: pd.DataFrame) -> dict:
    tp = int(((~df.true_faithful) & (~df.predicted_faithful)).sum())
    fn = int(((~df.true_faithful) & (df.predicted_faithful)).sum())
    tn = int(((df.true_faithful) & (df.predicted_faithful)).sum())
    fp = int(((df.true_faithful) & (~df.predicted_faithful)).sum())
    return {
        "n": len(df),
        "sensitivity_catches_unfaithful": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity_passes_faithful": tn / (tn + fp) if (tn + fp) else float("nan"),
        "accuracy": float(df.correct.mean()),
        "false_negatives": fn,  # unfaithful narratives that slipped through -- the dangerous error
        "false_positives": fp,  # faithful narratives wrongly rejected -- an annoyance, not a safety issue
    }


# ---------------------------------------------------------------------------
# 3. Ablation: isolate retry's contribution via a controlled stochastic mock
# ---------------------------------------------------------------------------
def make_stochastic_mock(error_rate: float = 0.3, seed: int = 0):
    """Simulates an imperfect LLM: each call independently has `error_rate`
    probability of introducing exactly one violation (wrong sign,
    hallucinated feature, or numeric error), else returns mock_llm's
    correct output. A controlled simulation for isolating the retry loop's
    benefit -- NOT a model of any real LLM's error rate. Swap in
    call_claude for a live ablation number."""
    rng = random.Random(seed)

    def stochastic(prompt: str, inputs: NarrativeInputs) -> NarrativeOutput:
        good = mock_llm(prompt, inputs)
        if rng.random() >= error_rate:
            return good
        violation = rng.choice(["sign", "feature", "numeric"])
        if violation == "sign" and good.sign in ("positive", "negative"):
            bad_sign = "negative" if good.sign == "positive" else "positive"
            return NarrativeOutput(good.narrative, bad_sign, good.cited_features,
                                   good.cited_values, good.dominant_uncertainty_source)
        if violation == "feature":
            all_feats = {"age", "stage", "race", "ethnicity", "intent"}
            hallucinated = list(all_feats - inputs.true_drivers())
            feat = rng.choice(hallucinated) if hallucinated else "age"
            return NarrativeOutput(good.narrative, good.sign, [feat],
                                   good.cited_values, good.dominant_uncertainty_source)
        bad_values = [CitedValue(cv.quantity, cv.value * 3 + 1) for cv in good.cited_values]
        return NarrativeOutput(good.narrative, good.sign, good.cited_features,
                               bad_values, good.dominant_uncertainty_source)

    return stochastic


def run_ablation(patient_rows: list, error_rate: float = 0.3,
                 n_trials: int = 200, seed: int = 0) -> pd.DataFrame:
    """Pass-rate vs. max_retries under a fixed injected error rate, each
    configuration evaluated over n_trials independent stochastic draws per
    patient so the pass-rate is a stable estimate, not one lucky/unlucky
    sequence of calls."""
    configs = [("no_retry", 0), ("retry_x1", 1), ("retry_x2", 2)]
    rows = []
    for label, max_retries in configs:
        passes, attempts = [], []
        for trial in range(n_trials):
            llm_fn = make_stochastic_mock(error_rate=error_rate, seed=seed * 10_000 + trial)
            for inp in patient_rows:
                r = generate_and_check(inp, llm_fn=llm_fn, max_retries=max_retries)
                passes.append(r.passed)
                attempts.append(r.attempts)
        lo, hi = wilson_ci(sum(passes), len(passes))
        rows.append({"config": label, "max_retries": max_retries,
                    "n": len(passes), "pass_rate": float(np.mean(passes)),
                    "ci_lo": lo, "ci_hi": hi,
                    "mean_attempts": float(np.mean(attempts))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_patients: int = 40, seed: int = 42):
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("1. SUBGROUP BREAKDOWN (ties RQ2 narrative fidelity to RQ3 fairness)")
    print("=" * 70)
    patient_rows, X_sample, sample_idx, calib_factor, _fitted = prepare_patient_rows(
        n_patients=n_patients, seed=seed, verbose=False)
    results = run_eval(patient_rows, llm_fn=mock_llm, model_name="mock", max_retries=2)
    patient_ids = [inp.patient_id for inp in patient_rows]

    sub_df = subgroup_pass_rates(results, X_sample, patient_ids)
    sub_df.to_csv(os.path.join(OUT_DIR, "narrative_subgroup_pass_rate.csv"), index=False)
    print(sub_df.to_string(index=False))
    if {"raceMajority", "raceMinority"}.issubset(set(sub_df.subgroup)):
        cmp = compare_subgroups(results, X_sample, patient_ids, "raceMajority", "raceMinority")
        print(f"\nFisher's exact test, raceMajority vs raceMinority: "
              f"odds_ratio={cmp['odds_ratio']:.3f}, p={cmp['p_value']:.3f}")
    print("NOTE: with the mock backend every narrative passes (self-consistent by "
          "construction), so this table currently shows 1.0 everywhere with the "
          "same caveat as elsewhere -- rerun with --live wiring for a real subgroup "
          "signal. What's being validated here is that the breakdown mechanism "
          "itself (using code.py's own subgroup_indices) is correct and ready.")

    print("\n" + "=" * 70)
    print("2. VERIFIER STRESS TEST (checks as a diagnostic classifier)")
    print("=" * 70)
    stress_df = run_verifier_stress_test()
    stress_df.to_csv(os.path.join(OUT_DIR, "narrative_verifier_stress_test.csv"), index=False)
    print(stress_df.to_string(index=False))
    summary = verifier_stress_test_summary(stress_df)
    print(f"\nsensitivity (catches unfaithful): {summary['sensitivity_catches_unfaithful']:.3f}")
    print(f"specificity (passes faithful):    {summary['specificity_passes_faithful']:.3f}")
    print(f"accuracy:                         {summary['accuracy']:.3f}")
    print(f"false negatives (unsafe -- unfaithful slipped through): {summary['false_negatives']}")
    print(f"false positives (annoying -- faithful wrongly rejected): {summary['false_positives']}")

    print("\n" + "=" * 70)
    print("3. ABLATION: pass-rate vs. max_retries under a controlled 30% "
          "injected error rate (stochastic mock -- see docstring caveat)")
    print("=" * 70)
    ablation_df = run_ablation(patient_rows[:5], error_rate=0.3, n_trials=200, seed=seed)
    ablation_df.to_csv(os.path.join(OUT_DIR, "narrative_ablation.csv"), index=False)
    print(ablation_df.to_string(index=False))
    print("\nNOTE: error_rate=0.3 is an assumed/injected rate for isolating the "
          "retry mechanism's effect, not a measurement of any real LLM. Rerun "
          "with call_claude in place of the stochastic mock for a live ablation.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patients", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(n_patients=args.n_patients, seed=args.seed)
