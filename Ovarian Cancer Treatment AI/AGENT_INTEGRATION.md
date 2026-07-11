# Code review + agent-integration assessment

Scope: reviewed `code.py` against the thesis text (`CaML_OP_Thesis_LaTeX/CaML_OP_Thesis_LaTeX/main.tex`),
fixed two real bugs, and implemented the piece of the architecture the
thesis describes but the code never built — the explainability layer and
LLM narrative agent — to answer "how would a research agent fit into this
thesis" concretely rather than abstractly.

## 1. What the review found

`code.py`'s docstring says it implements "everything from the exposé," but
the thesis (`main.tex`) describes a five-stage pipeline — input, encoding,
causal estimation, **explainability**, output — plus a clinical dashboard
prototype. `code.py` only implements the first three stages. Sections
"Explainability layer" (Effect-SHAP), "LLM narrative and faithfulness
checks", "Clinical dashboard prototype", and RQ2 ("LLM narrative fidelity")
are documented in the thesis and even discussed in the Results/Discussion
chapters, but nothing in the repo computes SHAP values or generates a
narrative. The thesis is honest about this — the Limitations section
explicitly lists "LLM narrative empirical evaluation: No pass-rate,
failure-rate or clinician evaluation is reported... The architecture is
documented, but the empirical performance is not" — but that also means
the gap was a known, named target, not something to guess at.

Two further issues were bugs rather than gaps:

- **`aipw_policy_value` wasn't AIPW.** It's labeled "AIPW" throughout the
  code, the thesis prose ("off-policy AIPW estimated value"), and the plot
  axis labels, but the implementation was a plain inverse-propensity-weighted
  (Horvitz–Thompson) estimator with no outcome-regression augmentation term.
  That matters beyond naming: AIPW's whole point is variance reduction via
  double robustness, and the thesis's Discussion section leans on the policy
  value comparison (CaML-OP "not statistically distinguishable from the
  oracle") — a claim resting on IPW's higher-variance estimator is weaker
  than the same claim under true AIPW.
- **`use_label_encoder=False`** is a removed XGBoost parameter (confirmed
  against xgboost 3.2.0 in this session). It happened to be silently
  dropped rather than erroring, but that's not guaranteed across versions.

## 2. Fixes applied to `code.py`

- Added `fit_outcome_models()` (arm-specific `RandomForestClassifier`s for
  `E[Y|X,T=0]` and `E[Y|X,T=1]`) and rewrote `aipw_policy_value()` to the
  actual doubly-robust estimator (Dudik, Langford & Li, 2011):
  `V(pi) = mean[ mu_hat_pi(x) + 1{T=pi(x)}/e(x) * (Y - mu_hat_T(x)) ]`.
  The outcome model is fit once per (scenario, run) and shared across every
  method scored, so it's a fair common correction, not something that
  favours any one estimator. Verified: on a smoke test, `TreatNone`'s
  across-run standard deviation dropped from the old IPW estimator's noisy
  value to a tight ~0.003 — the expected signature of adding the
  augmentation term.
- Removed `use_label_encoder=False` from both `XGBClassifier` call sites.

Both changes are covered by an end-to-end run (`main(n_runs=2,
scenarios=(1,4))`) that completed cleanly against the actual TCGA-OV data.

**Not changed:** the encoder-leakage risk (XGBoost leaf encoder sees
training outcomes before causal cross-fitting) is real but already
documented in both `code.py`'s comments and the thesis's own "Encoder
leakage" limitation and "Internal validity" section — fixing it would mean
restructuring the cross-fitting protocol, which is a bigger scope change
than a code review fix and is already correctly flagged as a known,
acknowledged limitation rather than a silent bug.

## 3. New: the explainability + narrative agent, implemented

Three new files close the code/thesis gap:

- **`effect_shap.py`** — `CaMLOPEffectModel` wraps the fitted leaf-encoder +
  R-Learner + ForestDRLearner ensemble as a single `tau_hat(X_raw)`
  function over the five raw covariates (age, stage, race, ethnicity,
  intent), matching the thesis's definition of Effect-SHAP attribution
  (Section "Effect-SHAP"). Because that ensemble isn't a single tree model,
  attribution uses SHAP's model-agnostic permutation explainer rather than
  an exact `TreeExplainer`.
- **`llm_narrative_agent.py`** — the narrative generator + three
  faithfulness checks the thesis specifies (sign match, named-feature
  match, numeric tolerance of 15%), plus a batch `run_eval()` harness. The
  LLM call is dependency-injected: a deterministic `mock_llm` runs the
  checking logic without any API key (for testing), and `call_claude` (real
  Anthropic API call, requires `ANTHROPIC_API_KEY`) is a one-line swap for
  producing an actual empirical result.
- **`run_narrative_pipeline.py`** — wires the three pieces together:
  loads TCGA-OV, fits CaML-OP on the semi-synthetic scenario 4 (hardest
  case), computes Effect-SHAP per patient, generates narratives, runs the
  faithfulness checks, and writes `caml_op_outputs/narrative_eval.csv`.

### Bugs the checks themselves had, found by actually running them

Writing this wasn't a paper exercise — running the pipeline against real
model outputs surfaced three bugs in the checking logic itself, which is
the same kind of gap the thesis flags for the (unbuilt) original: an
architecture that looks right on paper can still fail the moment it's
run against real text.

1. `check_numeric` flagged the literal "95" in "95% interval" as an
   uncited number, because the fixed confidence-level constant isn't a
   per-patient model output. Fixed by excluding percentage-tagged tokens.
2. The exclusion above initially used a lookahead regex that let "95%"
   partially match as a bare "9" via backtracking. Fixed by matching
   `number%?` as one token and dropping any token ending in `%` outright.
3. `check_sign` treated the noun "benefit" as a positive-direction signal,
   so "a **reduced** survival benefit" (a correct negative-effect
   narrative) failed because "benefit" alone tripped the positive-word
   list. Fixed by dropping ambiguous nouns and keying only on directional
   modifiers (increase/reduce/higher/lower/etc).
4. `check_features` originally required *every* top-3-per-direction SHAP
   feature (up to 6) to be named, which no narrative could satisfy inside
   the thesis's own two-to-three-sentence cap. Relaxed to: no hallucinated
   feature may be named, and at least one true driver must be — matching
   the thesis's actual wording ("names the dominant... features") rather
   than an unstated completeness requirement.

All four were confirmed via unit tests and one adversarial test (a
deliberately lying `llm_fn` that gets the sign wrong, cites a hallucinated
feature, and invents a number) — the checks correctly reject it on all
three axes. See the test snippets referenced in the commit; the pipeline
was run end-to-end against real TCGA-OV data with the mock backend.

### Important: the mock backend's pass-rate is not a thesis result

Running `run_narrative_pipeline.py` without `--live` uses `mock_llm`, a
template generator that always produces well-formed text. Its 100%
pass-rate demonstrates the checking logic works, not that an LLM would
pass. Producing the actual number the thesis's Limitations section is
missing requires `--live` with `ANTHROPIC_API_KEY` set, calling a real
model, and — per the thesis's own text — ideally a clinician panel to
validate that "faithful" (passes automated checks) also means clinically
sound.

## 4. Assessment: how a research agent fits into this thesis

**As a component of CaML-OP itself.** The narrative module described in
Section "LLM narrative and faithfulness checks" already *is* a minimal
agent: a generator (LLM) constrained by a strict prompt, gated by a
verifier (three deterministic checks) before anything reaches the
dashboard. That generator/verifier loop is the right shape for this
use case — it doesn't need planning, tool use, or multi-turn reasoning,
just a tight act→check→gate cycle with the verifier kept outside the LLM's
own judgment. `llm_narrative_agent.py` is that loop, now runnable.

**As a tool for building and maintaining this research pipeline** — this is
what today's session actually was. The most valuable thing an agent did
here wasn't writing code from scratch; it was cross-referencing 950 lines
of thesis prose against 800 lines of implementation and surfacing a
mismatch a single-file code review would miss (the whole explainability
stage silently absent) and a naming bug that a numerical-methods reviewer,
not a code reviewer, would need to catch (IPW mislabeled as AIPW). That
combination — reading the spec and the implementation together, then
verifying fixes by actually running them against the real 547-patient
dataset rather than reasoning from the diff — is where an agent adds the
most leverage on a thesis codebase like this one, more than as a
narrative-generation backend.

**Recommended next uses, in priority order:**

1. Run `run_narrative_pipeline.py --live` with a real API key to produce
   the actual empirical pass-rate the thesis's Limitations section is
   missing, then feed that into the RQ2 answer and Discussion chapter (the
   thesis currently says only "the architecture demonstrates a workflow
   design").
2. The raw-vs-leaf-augmented covariate ablation the Discussion chapter
   flags as "the most direct test of the encoder hypothesis" and "a
   priority for follow-up work" (line ~824) is a well-defined, mechanical
   experiment — rerun `code.py`'s benchmark with the `LeafEncoder` step
   disabled and diff the PEHE tables. This is exactly the kind of
   long-running, well-specified sweep (the full benchmark already takes
   30–40 minutes) that suits an agent running in the background while the
   researcher does something else.
3. A cross-fitted variant of `LeafEncoder` (fit leaf assignment out-of-fold
   rather than on the full training set) would let the "encoder leakage"
   limitation be *measured* rather than just acknowledged — worth a
   companion ablation to (2).
4. Ongoing: use an agent to keep `main.tex` and the codebase honest with
   each other going forward — today's biggest single finding was that the
   two had drifted apart, and that class of bug doesn't show up in either
   document read alone.

**Caveats.** The faithfulness checks in `llm_narrative_agent.py` are
deliberately narrow and rule-based, not themselves LLM-graded, precisely
because the thesis wants a hard gate a clinician can audit rather than
another opaque model judging the first one. Keep it that way — an
LLM-as-judge faithfulness checker would undermine the exact transparency
argument the thesis makes about the EU AI Act (Section "Relation to prior
work"). And nothing here should be read as validating the narratives
clinically: passing the three automated checks means the text didn't
contradict the model's own numbers, not that it's sound clinical
communication — that still needs the clinician evaluation the thesis
already says is missing.

## 5. Four technical improvements applied to the narrative agent

The first version of `llm_narrative_agent.py` checked free-text narratives
with regex, and running it against real output (not just reading the code)
surfaced three prose-parsing failures documented in the module's
CHANGELOG (a fixed "95%" constant flagged as an uncited number, "benefit"
misread as a positive-direction word, and an unrealistic
all-features-must-be-named requirement). Rather than keep patching regexes,
the module was rewritten around four changes:

1. **Structured output instead of prose parsing.** The model now submits a
   narrative via a forced tool call (`submit_clinical_narrative`) with typed
   fields: `sign` (enum), `cited_features` (list), `cited_values` (each
   number tagged with *which* quantity it claims to represent, e.g.
   `{"quantity": "tau_hat", "value": 0.007}`). Checks validate these fields
   directly — set membership and tagged-value comparison — with no text
   search, so there is no class of "ordinary English word collides with a
   covariate name" bug left to have. This also fixed a correctness gap the
   old checker had structurally: it verified a cited number was close to
   *some* reference value, never that it was close to the *specific*
   quantity it was cited as. Tagged values fix that by construction.
2. **Closed-loop retry.** `generate_and_check` now loops: on failure, the
   specific failing check(s) are fed back to the model as a targeted
   correction request (e.g. *"cited_features must be drawn only from
   ['age','stage']"*), and it gets up to `max_retries` attempts. Verified
   with a test double (`make_flaky_mock`) that deliberately fails once then
   succeeds — confirmed the loop converges in exactly 2 attempts, not just
   unit-tested in isolation.
3. **Scale-consistent numeric tolerance.** `check_numeric` now uses
   `numpy.isclose`-style combined absolute+relative tolerance
   (`max(tol * |true|, atol)`, `atol=0.005`) instead of pure relative
   tolerance, which was unforgivingly tight near `tau_hat ~ 0` (15% of
   0.002 allows only 0.0003 absolute deviation — tighter than the `.3f`
   display precision used everywhere else in the pipeline).
4. **Production hardening.** `temperature=0.0` pinned in `call_claude` (a
   pass-rate should be reproducible run-to-run); a network-level
   `with_backoff` wrapper distinct from the content-level retry in (2);
   incremental CSV writes (`run_eval` appends per-patient, so a crash
   partway through a live run over ~137 patients doesn't lose completed
   results); a disk-backed cache keyed on `(prompt, model)` so re-running
   after a checking-logic tweak doesn't re-spend API calls on identical
   prompts; optional `max_workers` concurrency via `ThreadPoolExecutor`
   (stress-tested at 8 workers / 30 patients with randomized latency — no
   lost or duplicated rows); and a `PROMPT_VERSION` hash stamped into every
   output row so a stored pass-rate can't silently go stale if the prompt
   changes later.

## 6. Radical improvement: calibration-aware abstention

**The idea.** The thesis measures something specific: CaML-OP's nominal
95% interval has *empirical* coverage of only 0.847 (Section "Uncertainty
interval coverage") — the reported intervals are already known, by the
thesis's own analysis, to be too narrow. The narrative module as specified
was going to ignore that finding completely: it asserts a confident
"positive"/"negative" sign claim for every patient with `tau_hat != 0`,
using the very interval the thesis itself flagged as unreliable. Nothing
in the original architecture connects the interval's *measured*
reliability to whether the agent should make a directional claim at all.

**What I built.** `calibrated_abstention.py` fits a split-conformal
calibration factor — the multiplicative inflation of the reported
half-width needed to actually achieve 95% coverage — using a calibration
split of the semi-synthetic test set (which uniquely exposes `tau_true`,
unlike real data). If a patient's *calibrated* interval still spans zero,
the narrative agent is required to declare `sign: "indeterminate"` instead
of asserting benefit or harm — wired directly into
`llm_narrative_agent.py`'s tool schema and `check_sign`, with a dedicated
prompt branch and a mock-narrative variant, all unit-tested including the
adversarial case of a mock that wrongly asserts confidence when it should
abstain (correctly rejected).

**This is not a hypothetical.** I ran it against the actual pipeline
(`code.py`'s own `fit_caml_op`, 4 runs of scenario 4, n=548):

| | reported (uncalibrated) | calibrated (95% target) |
|---|---|---|
| coverage | 0.854 (thesis reports 0.847 — consistent) | 0.949 |
| calibration factor | 1x (baseline) | **1.586x** half-width |
| fraction where sign is indistinguishable from zero | 0.838 | **0.967** |

Read plainly: even the narrower, *reported* interval already can't
distinguish sign from zero for 84% of patients. Once widened enough to
actually deliver its stated coverage, **97% of patients' estimated effects
are not statistically distinguishable from no effect at all.** The
end-to-end demo (`run_narrative_pipeline.py`) reproduces this on a live
15-patient sample: all 15 correctly come back `indeterminate`, including
one with `tau_hat = 0.174` — a large point estimate that still isn't
reliably signed once the interval is honest about its own uncertainty.

**Why this is the right place for an agent to add value, not just a
statistics fix.** This number could have been computed as a static table
in Chapter 5. Making it a live *decision* the narrative agent consults per
patient — rather than a caveat printed once in the Limitations section — is
what turns "we know our intervals under-cover" from an acknowledged
weakness into an operational constraint the deployed system actually
respects. It's also a natural point for the thesis to take an explicit
position it currently doesn't: if honest calibration means the system
can rarely assert a direction, is the right response (a) an abstention-
heavy dashboard, (b) tightening the causal estimator until it can support
confident claims more often, or (c) some middle ground (e.g. reporting
magnitude-only, without sign, when calibrated CI spans zero)? The
mechanism is built either way — which policy to adopt is a thesis
decision, not a code one.

**Designed but not built (honest scope note):** a natural extension is a
*counterfactual-contrastive* consistency check — since CaML-OP's
`fit_outcome_models` (added in the AIPW fix) already estimates
`E[Y|X,T=0]` and `E[Y|X,T=1]` separately, the narrative agent could
generate a short claim about each arm plus the ITE claim, and a fourth
check could verify the three are mutually consistent (e.g. the arm-1 claim
implies higher survival than the arm-0 claim iff tau_hat > 0). This would
catch a class of internal contradiction a single-output narrative
structurally cannot expose. I did not build this — it needs a second
generation call per patient (cost/latency roughly doubles) and a new
consistency-check design, and calibrated abstention already existed as a
tighter, cheaper win using infrastructure already in the codebase. Flagging
it here as the next candidate if the pass-rate work in item 3 (Section 4
above) turns up systematic sign-consistency issues that abstention alone
doesn't explain.

## 7. Raising the evaluation from a pass-rate to citable evidence

A single aggregate pass-rate number is a demo result, not a thesis result.
`narrative_research_eval.py` adds three things a reviewer would actually
ask for, all built on the existing harness rather than new infrastructure:

1. **Subgroup breakdown, tying RQ2 to RQ3.** `subgroup_pass_rates()` reuses
   `code.py`'s own `subgroup_indices()` — the identical age/stage/race
   definitions already used for the PEHE subgroup table — so narrative
   faithfulness can be audited on the same axes as ITE estimation accuracy.
   `compare_subgroups()` runs a Fisher's exact test (appropriate given the
   small per-subgroup counts here, same constraint the thesis's own
   subgroup analysis already names) between `raceMajority` and
   `raceMinority`. Nobody had asked whether the *narrative layer* introduces
   its own fairness gap on top of the causal estimator's — this makes that
   question answerable, not answered: on the mock backend every narrative
   passes by construction, so the table currently validates the
   *mechanism*, not a real subgroup signal. A `--live` run is what would
   turn this into an actual RQ3-adjacent finding.
2. **Verifier stress test as a diagnostic classifier, not a demo.** The
   single hand-written adversarial example from Section 5 proved the
   checks *can* catch a lie; it didn't validate them systematically.
   `run_verifier_stress_test()` builds twelve cases with known ground-truth
   faithfulness by construction (wrong sign, hallucinated feature,
   hallucinated quantity tag, numeric error, empty citation, mixed
   true/hallucinated citation, correct and incorrect abstention) and
   reports sensitivity/specificity the way a diagnostic test is validated —
   distinguishing the dangerous error (an unfaithful narrative that slips
   through, a false negative) from the merely annoying one (a faithful
   narrative wrongly rejected, a false positive). Result on the current
   checks: 12/12 correct, sensitivity = specificity = 1.0 on this case set.
   That's a statement about coverage of *this* case set, not a proof of
   completeness — it's a regression suite, and it should grow every time a
   new failure mode is found (the way items 1-4 in Section 5 were each
   found by running the thing, not by inspection).
3. **Ablation quantifying what retry actually buys.** The deterministic
   `mock_llm` never errs, so it structurally cannot demonstrate the retry
   loop's value — there's nothing to recover from. `make_stochastic_mock()`
   injects a configurable, explicit error rate (wrong sign / hallucinated
   feature / numeric error, chosen uniformly) so `run_ablation()` can
   measure pass-rate as a function of `max_retries` under controlled
   conditions. At an assumed 30% per-call error rate (n=1000 trials per
   configuration, Wilson CIs):

   | config | max_retries | pass_rate | 95% CI |
   |---|---|---|---|
   | no_retry | 0 | 0.765 | [0.738, 0.790] |
   | retry_x1 | 1 | 0.939 | [0.922, 0.952] |
   | retry_x2 | 2 | 0.980 | [0.969, 0.987] |

   The 30% error rate is an assumed, injected constant for isolating the
   mechanism's effect — not a measurement of any real LLM's behavior.
   Swapping `make_stochastic_mock` for `call_claude` in `run_ablation`
   turns this from "how much does retry help against an assumed error
   rate" into "how much does retry help against Claude's actual error
   rate," which is the number worth citing.

All three write CSVs to `caml_op_outputs/` (`narrative_subgroup_pass_rate`,
`narrative_verifier_stress_test`, `narrative_ablation`) and are structured
so re-running with `call_claude` in place of the mock is the only change
needed to turn each from "mechanism validated" into "empirical result."

## 8. Causal-structure verifiers, not generic LLM engineering

Everything up to this point (structured output, retry, composable
verifiers, escalation) is generic LLM-agent engineering that happens to
sit on top of a causal model — none of it uses anything specific to
*causal inference*. Four mechanisms were added to close that gap, ranked
by how directly they reuse causal machinery already in the codebase
versus needing new plumbing.

**Counterfactual-contrastive consistency (`check_counterfactual_consistency`).**
The agent previously verified a single scalar (`tau_hat`) against a single
CI. `fit_outcome_models` (added for the AIPW fix in `code.py`) already
estimates the two potential outcomes separately — `E[Y|X,T=0]` (`mu0`) and
`E[Y|X,T=1]` (`mu1`). The agent is now required to cite both, and a new
check verifies the cited pair is *ordered* consistently with the declared
sign (a "benefit" claim requires `mu1 > mu0`, not just that their
difference happens to be reported as positive). This is the
potential-outcomes framework itself as a verification structure — it
catches a class of internal contradiction a single-number check
structurally cannot see, and it's also the one place an LLM does
something more than translate a pre-computed fact: it has to state and
keep two related but distinct causal claims consistent, not just echo one
number correctly.

**Uncertainty decomposition and synthesis (`check_dominant_uncertainty`).**
CaML-OP's R-Learner and DR-Learner are robust to *different*
misspecifications (R survives a wrong outcome model, DR survives a wrong
propensity model — the explicit rationale for ensembling them in the
thesis's "Ensemble and uncertainty reporting" section). Their disagreement
(`|tau_r - tau_dr|`, now exposed via `CaMLOPEffectModel.predict_components`)
is a second, causally distinct uncertainty signal from CI width: it
measures sensitivity to *which* nuisance model you trust, not sampling
noise. Combined with a third signal (identification fragility, next), the
agent must now name which of three sources dominates for *this* patient
and say so in the narrative, checked against a deterministic ranking
(`_dominant_uncertainty_source`). This is the mechanism's most genuinely
"agentic" piece: synthesizing across several continuous signals into one
coherent, correctly-prioritized statement is a task a fixed template can't
scale to gracefully (it would need a hand-authored rule for every
combination of which signal is largest), but an LLM can — and it's still
checked, not left to the model's judgment about which one to report.

**Identification-confidence via E-value (`sensitivity_analysis.py`).**
Every uncertainty mechanism up to this point (calibrated abstention, R/DR
disagreement) answers "how much does sampling noise / model choice affect
this estimate" — none asks "how robust is this estimate to the
confounding the thesis itself says is likely present" (Section
"Identification assumptions" is explicit that performance status,
comorbidities and physician preference are unmeasured and plausibly
confound treatment assignment). Added a per-patient E-value (VanderWeele &
Ding, 2017) computed from `mu0`/`mu1`: the minimum strength an unmeasured
confounder would need to fully explain away the finding. A low E-value
feeds directly into the dominant-uncertainty synthesis above as the
"identification" signal. Stated plainly, as elsewhere in this codebase
when a threshold is a judgment call rather than a derived constant: the
fragility cutoff (E-value < 2.0) is a documented, provisional convention,
not empirically calibrated to how strong an unmeasured confounder like
performance status plausibly is in this specific clinical context — a
domain reviewer should revisit it, not treat it as settled.

**Subgroup-conditional calibration (`fit_calibration_factors_by_subgroup`).**
The one purely statistical addition, with no LLM role at all — calling it
an "agent improvement" would be inaccurate. `calibrated_abstention.py`
previously fit one global calibration factor for the whole cohort, even
though the subgroup PEHE table already shows reliability isn't uniform
across age/stage/race. Now fits a factor per subgroup (falling back to the
global factor below `min_n=20`, the same guard used elsewhere in this
pipeline) and a patient gets the most conservative factor across every
subgroup they belong to. Validated against the real pipeline: factors
ranged 1.41x–1.83x across subgroups (vs. a single global 1.59x), a
real, not noise-level, spread — `stageHigh` patients need meaningfully
less inflation than `stageLow` patients to hit honest 95% coverage. This
changes only the precomputed `must_abstain` flag the agent already
consumes; nothing about the agent's own behavior changes.

All four are verified: the two new checks pass a dedicated adversarial
stress test (contradicted arm ordering, missing arm citation, and a wrong
declared dominant source are all correctly caught — see the three new
cases in `narrative_research_eval.py`'s stress suite, 15/15 correct
overall), and `run_narrative_pipeline.py` runs end-to-end against the real
TCGA-OV data with all four wired in, producing narratives that visibly
differ in their stated dominant uncertainty source from patient to
patient rather than a fixed template phrase.

## Bounded query interface, not open-ended chat (`counterfactual_query.py`)

A chat-style extension to the narrative module was proposed mid-session
("what if a clinician could ask follow-up questions about a patient's
estimate?"). Before building it, a literature check was run specifically
to test that idea rather than assume it was a good one. The results
sharpened the design rather than confirming the premise:

- **Open-ended clinical LLM chat has a measured, non-trivial hallucination
  rate.** One cited study found a 12.5% hallucination rate for LLMs
  generating cancer treatment information; broader safety surveys put
  "problematic response" rates at 21.6%–43.2% depending on model, with
  unsafe response rates of 5%–13%. This is a direct argument against a
  freeform text box for clinical questions, not just a generic caution.
- **Conversational XAI produces faster understanding and higher trust than
  static dashboards, but the same literature documents it promoting
  overreliance** — users accepting recommendations too readily without
  critical evaluation, an effect that **builds over repeated turns**, not
  just present on a single response. This lands directly on the thesis's
  own "Automation bias and clinician oversight" limitation, now with
  measured RCT evidence behind it rather than a theoretical concern.
- **IMPACT** ("an interactive multi-disease prevention and counterfactual
  treatment system using explainable AI and a multimodal LLM",
  PMC12192960) is published precedent for a different shape: bounded
  interactive counterfactual querying — "what adjustment to this feature
  reduces risk" — over a model, rather than open conversation. This is
  real precedent that a bounded tool-query pattern is a recognized,
  published approach to this problem, not something invented from
  scratch for this thesis.
- **VeriFact** (verifying LLM-generated clinical text against structured
  EHR data) already justified the base narrative agent's per-claim
  verification design (see above); the same discipline needed to extend
  to any follow-up interaction, not relax for the sake of a conversational
  feel.

**What was built.** One bounded tool, `query_counterfactual` — not a chat
box. A clinician (or dashboard) asks a single structured question ("what
if `age` were adjusted by −5?"), `counterfactual_query.py` recomputes
`tau_hat`/`mu0`/`mu1`/the calibrated interval under the perturbed
covariate using the already-fitted CaML-OP model (no refitting — a bounded
query has to be cheap enough to answer interactively), and an LLM narrates
the delta via the same structured-tool-call pattern as the base narrative
agent. The answer is checked by three verifiers before being shown,
mirroring `llm_narrative_agent.py`'s generate → check → targeted-feedback
→ retry loop exactly (including `NarrativeStatus.ESCALATED` as the
terminal state):

- `check_cf_numeric` — every cited value (baseline/counterfactual
  `tau_hat`, `delta_tau`, `mu0_cf`, `mu1_cf`, interval bounds) matches the
  recomputed ground truth within tolerance, not left to the model's own
  arithmetic.
- `check_cf_direction` — the declared "improves"/"worsens"/
  "no_meaningful_change" must match the actual sign of the recomputed
  `delta_tau` past a fixed epsilon, so a claimed direction can't drift from
  the numbers backing it.
- `check_cf_calibration_restated` — **the concrete countermeasure to the
  overreliance-builds-over-repeated-turns finding above.** The literature
  said a one-time disclaimer isn't enough; this makes that a hard,
  per-query verifier instead of a prompt instruction the model could drift
  away from three turns into a session. Every single answer must correctly
  restate whether the counterfactual estimate is calibration-reliable,
  freshly recomputed for that specific counterfactual point — an LLM that
  "forgets" to restate it on query 3 fails exactly the same check that
  catches it on query 1.

**Verified, not just asserted.** `counterfactual_query.py` is runnable
standalone (`python counterfactual_query.py`) against the real TCGA-OV
pipeline with zero API key required (mock backend): it fits the model
once via `run_narrative_pipeline.prepare_patient_rows`'s now-exposed
`fitted` dict, asks two real what-if questions about a sampled patient's
top SHAP driver, and prints both. It also includes a retry-loop smoke test
using a flaky mock that deliberately omits the calibration restatement
once (the literature's specific failure mode) — the loop catches it and
self-corrects on retry, asserted programmatically (`attempts == 2`), not
just eyeballed.

**Deliberately not done.** No dashboard UI wiring — the thesis's
"Clinical dashboard prototype" (Section 3.5) is a static mockup image, not
running code, and this tool is not yet wired into anything a clinician
would click. No live pass-rate evaluation of this tool's verifiers against
a real LLM backend (same caveat as the base narrative agent's RQ2 number:
the mock backend's pass-rate describes the checking logic, not an LLM's
behavior). Both are natural next steps, not silently dropped.
