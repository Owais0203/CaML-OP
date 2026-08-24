# CaML-OP · ICBINB-BIO 2026 — change log

Every manuscript number that changed after the three P0 runs, and the file that produced it.
Old values are from the superseded five-field / leaky-encoder configuration.

## 1. Numbers replaced

### Primary CATE benchmark — `results/cate_benchmark_primary.csv` (task1_cate_benchmark.py)

| Manuscript location | Old | New |
|---|---|---|
| Abstract, §4.1 Table 1 — CaML-OP PEHE | 0.1610 (rank 3 of 7) | 0.0972, variant C (rank 2 of 9) |
| Abstract, §4.1 Table 1 — S-Learner PEHE | 0.1005 | 0.0911 |
| §4.1 Table 1 — Causal Forest PEHE | 0.1185 | 0.1326 |
| §4.1 Table 1 — X / T / R / DR PEHE | 0.1721 / 0.2232 / 0.2412 / 0.2661 | 0.1734 / 0.2202 / 0.2612 / 0.2752 |
| §4.1 Table 1 — MAE, all methods | see old table | replaced, all nine rows |
| §4.1 Table 1 — ATE error, all methods | see old table | replaced, all nine rows |
| §4.1 Table 1 — sign accuracy column | not present | added, all nine rows |
| §4.1 Table 1 — CaML-OP variants A and B | not present | added (0.1848 raw, 0.1711 leaky) |
| Abstract, §4.2 — CaML-OP sign accuracy | 0.577 | 0.5988 (highest of nine) |
| §4.2 — best-method sign accuracy | 0.659 (S-Learner) | 0.5988 (CaML-OP C); S-Learner 0.5760 |

### Paired bootstrap CIs — `results/paired_bootstrap_cis.csv` (task1_cate_benchmark.py)

New content in §4.1; n = 40 paired (DGP, seed) differences. C vs raw: PEHE −0.0875
(−0.0962, −0.0787), MAE −0.0678 (−0.0753, −0.0604), ATE error −0.0030 (−0.0132, +0.0073).
C vs leaky: PEHE −0.0739 (−0.0843, −0.0635), MAE −0.0526 (−0.0613, −0.0440).
C vs S-Learner: PEHE +0.0061 (−0.0022, +0.0145), MAE +0.0081 (+0.0006, +0.0157),
ATE error +0.0198 (+0.0091, +0.0302).

### Encoder ablation — `results/encoder_crossfit_ablation.csv` (task1_cate_benchmark.py)

| Manuscript location | Old | New |
|---|---|---|
| Abstract, §4.3, failure table, conclusion — representation gain | ~5% (0.1939 → 0.1836, scenario 4) | 47.4% overall (0.1848 → 0.0972) |
| §4.3 — per-DGP gain | not reported | 52.0 / 46.4 / 49.6 / 41.8% (DGP 1–4) |
| §4.3 — leaky-encoder gain | not distinguished | 7.4% overall |
| §4.3 — encoder hyperparameter range | 0.1814–0.1968 | removed; no longer the relevant comparison |

### Split-conformal calibration — `results/conformal_repeated_splits.csv`, `results/conformal_summary.csv` (task2_conformal.py)

| Manuscript location | Old | New |
|---|---|---|
| Abstract, §4.4, Fig. 3 — coverage at 0.80 / 0.90 / 0.95 / 0.99 | 0.762 / 0.867 / 0.919 / 0.974 | 0.799 / 0.902 / 0.955 / 0.990 |
| Abstract, §4.4, Fig. 3 — abstention at same targets | 0.771 / 0.864 / 0.910 / 0.977 | 0.850 / 0.958 / 0.994 / 1.000 |
| §4.4 — mean interval width | not reported | 0.323 / 0.450 / 0.613 / 0.984 |
| §4.4 — repetitions, calibration n, evaluation n | 5 seeds, halves | 30 repetitions, n = 106 / 107 |
| §4.4 — 95% intervals across repetitions | not reported | added, e.g. 0.683–0.918 for coverage at 0.80 |

### Live-LLM evaluation — `results/llm_verifier_ablation.csv`, `results/verifier_stress_tests.csv`, `results/freetext_validation.csv` (task3_llm_verifier.py, task3_validate_freetext.py)

New Table 2 (`tab:llm`) in §4.5, plus supporting prose. Two backends, 160 held-out states,
conditions A–D. Unsupported direction 0.176 (A, Sonnet, lenient rubric) → 0.000 in B/C/D
for both backends. Verified pass 0.450 → 0.781 with 0.219 escalation (Sonnet), 0.194 →
1.000 with 0.000 escalation (Haiku). Numeric error 0.469 → 0.000 (Haiku). Calls/case 2.14
and 2.59 in condition D. Stress tests: sensitivity and specificity 1.000 on the 15-case
regression suite and on 128 generated cases. Cost $5.92 and $1.76; latency 12.4 s and 9.9 s
in condition D.

### Retained unchanged

The 30%-corruption retry mechanism test (0.736 / 0.930 / 0.978 at 0, 1, 2 retries, 1.345
calls) is unaffected by the covariate change and is kept in §4.5 as a control-loop test,
explicitly subordinate to the live-model rates.

## 2. Wording changed, not just renumbered

**The representation claim reversed direction.** The old text called the leaf-encoder gain
"small," "about 5%," and "comparable to hyperparameter variation," and made that one of
three headline failures. Under cross-fitting the gain is 47.4% with a paired CI excluding
zero, in all four DGPs. The 5% figure was an artifact of comparing raw covariates against a
leaky encoder, not a property of the representation.

Rewritten passages: abstract; third sentence of Introduction ¶3; contribution bullet 1;
§4.3 ¶2–3; failure-table row "Leakage hid a real representation gain"; conclusion ¶1.

The FINAL_ACTIONS stop condition triggers on the cross-fitted model *becoming* the best PEHE
method. It did not — the S-Learner still leads at 0.0911, though the paired PEHE interval
spans zero — so this was handled as an in-scope rewrite rather than escalated. It still needs
a scientific read before submission.

Consequential smaller rewrites: §2.1 (five fields → four pre-treatment fields, timing caveat
removed); §3.1 (leakage paragraph → out-of-fold encoder description); §3.2 (heuristic
interval-inflation → split-conformal procedure); §3.3 ("five raw fields" → "four raw
fields"); Limitations (treatment-intent and no-live-LLM limitations removed; single-model-
family backend, rubric-dependent free-text scoring, and un-rescored Haiku condition A added).

## 3. Numbers with no replacement value

These appeared in the old manuscript but have no counterpart in the P0 output. Nothing was
invented to fill the gaps.

| Old number | Where it was | What was done |
|---|---|---|
| Heuristic interval coverage 0.847 vs nominal 0.95 | abstract, §4.4, failure table | Removed. The split-conformal study replaces this claim. Restore if a new heuristic-coverage value exists. |
| Effect-scale sweep (multipliers 0.5 / 1.0 / 2.0) | §4.2, old Figure 2 left panel | Not re-run. §4.2 now reports sign accuracy from the primary rerun; one sentence notes the sweep used the superseded configuration. |
| Assignment-strength sweep, CaML-OP PEHE 0.1541 → 0.2162 | abstract, §4.3, failure table | Not re-run. Removed from the abstract and failure table; §4.3 keeps the qualitative caveat and flags the configuration explicitly. |

## 4. Figures

| File | Old content | New content |
|---|---|---|
| `sign_accuracy.png` | scenario-4 sign accuracy vs effect-scale multiplier, 6 methods | sign accuracy by method, primary four-field rerun, 9 bars, CaML-OP variants shaded, 0.5 reference line retained |
| `calibration_sweep.png` | coverage and abstention vs target, 5 seeds, no intervals | coverage and abstention vs target, 30 repetitions, 95% intervals as error bars |
| `architecture.png` | unchanged | unchanged; caption updated to name the split-conformal scores as evaluation-only |

Both figures are regenerated by `make_figures.py` from the values in the P0 results
document. Captions for `fig:diagnostics` were rewritten to match.

## 5. Structure

Section and subsection order is unchanged. One table was added (`tab:llm`, §4.5), per the
FINAL_ACTIONS instruction to add the live-LLM A/B/C/D table; the failure table is therefore
Table 3. Paired CIs are reported in text rather than as a table. The reference list is
unchanged: 19 entries, all still cited, no additions.
