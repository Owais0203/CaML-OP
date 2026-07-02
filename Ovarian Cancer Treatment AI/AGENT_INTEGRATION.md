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
