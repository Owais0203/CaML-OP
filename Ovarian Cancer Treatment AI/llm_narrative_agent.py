"""
LLM narrative agent + faithfulness checks.

Implements the piece of the CaML-OP architecture that main.tex describes in
detail (Section "LLM narrative and faithfulness checks", RQ2) but that
code.py never built. The thesis's own Limitations section names this
explicitly: "No pass-rate, failure-rate or clinician evaluation is reported
for the narrative module. The architecture is documented, but the empirical
performance is not." This module is a runnable version of that architecture
plus the missing evaluation harness.

Design: a minimal generator/verifier agent loop.
  1. generate(...)  -- an LLM produces a 2-3 sentence narrative from a
     constrained prompt: patient covariates, tau_hat, its uncertainty
     interval, and the top Effect-SHAP drivers in each direction.
  2. Three deterministic, non-LLM checks gate the narrative before display:
       - check_sign:      recommendation sign matches tau_hat
       - check_features:  named features match the top Effect-SHAP features
       - check_numeric:   every cited number is within `tol` (thesis: 15%)
                           of a number the model actually produced
  3. A narrative is marked PASS only if all three checks succeed.

The LLM call is dependency-injected (`llm_fn`) so the checking logic and
batch evaluation harness are runnable and testable without any API key.
Swap in `call_claude` (requires ANTHROPIC_API_KEY) to generate real
narratives and get an actual empirical pass-rate for the thesis -- the
`mock_llm` default is a template generator for testing only, not a stand-in
for a clinician evaluation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

ALL_FEATURES = ["age", "stage", "race", "ethnicity", "intent"]


@dataclass
class NarrativeInputs:
    patient_id: str
    covariates: dict
    tau_hat: float
    ci_lo: float
    ci_hi: float
    top_positive: list = field(default_factory=list)  # [(feature, phi), ...], phi > 0
    top_negative: list = field(default_factory=list)  # [(feature, phi), ...], phi < 0


PROMPT_TEMPLATE = """You are a clinical decision-support narrative generator.
Given the structured outputs of a causal machine-learning model estimating
the individualized effect of platinum-based chemotherapy on five-year
ovarian cancer survival, write a two- to three-sentence clinical summary.

Rules (must follow exactly):
- State whether the estimated effect is positive (benefit) or negative (harm),
  matching the sign of tau_hat below.
- State the approximate magnitude of tau_hat and its uncertainty interval.
- Name the model's stated top contributing feature(s) below, in each
  direction that is present. Do not invent or omit named features.
- Any numeric value you cite must be drawn from the values given below.
- Do not introduce outside medical knowledge, guidelines, or claims not
  derivable from the numbers given.
- Two to three sentences only.

Patient covariates: {covariates}
Estimated ITE (tau_hat): {tau_hat:.3f}
95%% uncertainty interval: [{ci_lo:.3f}, {ci_hi:.3f}]
Top features increasing benefit: {top_positive}
Top features decreasing benefit: {top_negative}
"""


def build_prompt(inputs: NarrativeInputs) -> str:
    return PROMPT_TEMPLATE.format(
        covariates=inputs.covariates,
        tau_hat=inputs.tau_hat,
        ci_lo=inputs.ci_lo,
        ci_hi=inputs.ci_hi,
        top_positive=", ".join(f"{f} ({v:+.3f})" for f, v in inputs.top_positive) or "none",
        top_negative=", ".join(f"{f} ({v:+.3f})" for f, v in inputs.top_negative) or "none",
    )


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
def mock_llm(prompt: str, inputs: NarrativeInputs) -> str:
    """Deterministic template narrative used when no LLM is configured.
    Exists so the faithfulness-check logic and eval harness below are
    runnable without an API key -- NOT a substitute for a real model in the
    thesis's empirical evaluation."""
    direction = "an increased" if inputs.tau_hat > 0 else "a reduced"
    feats = (inputs.top_positive[:1] + inputs.top_negative[:1])
    feat_txt = " and ".join(f[0] for f in feats) if feats else "the modelled covariates"
    return (
        f"This patient's estimated treatment effect is {inputs.tau_hat:+.3f} "
        f"(95% interval {inputs.ci_lo:.3f} to {inputs.ci_hi:.3f}), indicating "
        f"{direction} five-year survival benefit from platinum therapy. "
        f"The estimate is driven mainly by {feat_txt}, and the interval width "
        f"reflects meaningful uncertainty in this individualized estimate."
    )


def call_claude(prompt: str, inputs: NarrativeInputs,
                model: str = "claude-sonnet-5") -> str:
    """Real narrative generation via the Anthropic API. Requires the
    `anthropic` package and ANTHROPIC_API_KEY. Not used by default -- pass
    llm_fn=call_claude to run_eval(...) to produce an actual empirical
    pass-rate for the thesis instead of the mock generator's output."""
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


# ---------------------------------------------------------------------------
# Faithfulness checks (thesis: "LLM narrative and faithfulness checks")
# ---------------------------------------------------------------------------
# Captures a number with an optional trailing "%" as one token, so e.g.
# "95%" is recognised as a percentage and dropped whole below rather than
# risk a partial match like "9" surviving a lookahead-based exclusion.
# Percentages are excluded from the fidelity check because "95%" is the
# fixed confidence-level constant repeated in every narrative, not a
# per-patient number the model produced.
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+%?")


def _word_in(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def check_sign(narrative: str, tau_hat: float) -> bool:
    """The recommendation sign must match tau_hat: benefit language for
    tau_hat > 0, harm/reduction language for tau_hat < 0, not both."""
    text = narrative.lower()
    # "benefit"/"survival benefit" is excluded deliberately: it is used as a
    # neutral noun in both directions ("a reduced survival benefit" is a
    # harm claim), so treating it as a positive-direction signal caused
    # every correctly-worded negative-tau narrative to be flagged as
    # contradictory. Direction is carried by the modifier, not the noun.
    positive_terms = ["increase", "improv", "higher"]
    negative_terms = ["decrease", "reduc", "harm", "lower", "worse"]
    has_pos = any(t in text for t in positive_terms)
    has_neg = any(t in text for t in negative_terms)
    if tau_hat > 0:
        return has_pos and not has_neg
    if tau_hat < 0:
        return has_neg and not has_pos
    return True


def check_features(narrative: str, inputs: NarrativeInputs) -> bool:
    """No hallucinated driver: every feature named in the narrative must be
    one of the true top Effect-SHAP drivers, and at least one true driver
    must be named (word-boundary matched to avoid substring false
    positives, e.g. "race" inside "embrace").

    This intentionally does not require *every* top-3-per-direction feature
    to be named. The thesis's own prompt caps narratives at two to three
    sentences while allowing up to six candidate drivers (top three in each
    direction); requiring all of them to be enumerated in that space is not
    a realistic reading of "names the dominant Effect-SHAP features" and
    would make the check fail well-written narratives, not just unfaithful
    ones."""
    text = narrative.lower()
    named = {f for f, _ in inputs.top_positive + inputs.top_negative}
    mentioned = {f for f in ALL_FEATURES if _word_in(text, f)}
    if not named:
        return not mentioned
    if mentioned - named:
        return False  # a feature was named that isn't a true driver
    return len(mentioned) > 0  # at least one true driver was actually named


def check_numeric(narrative: str, inputs: NarrativeInputs, tol: float = 0.15) -> bool:
    """Every numeric value cited in the narrative must be within `tol`
    (relative, thesis default 15%) of one of the model's own reported
    numbers (tau_hat, ci_lo, ci_hi, or a SHAP phi value)."""
    reference = [abs(v) for v in (
        [inputs.tau_hat, inputs.ci_lo, inputs.ci_hi]
        + [v for _, v in inputs.top_positive + inputs.top_negative]
    ) if abs(v) > 1e-9]
    if not reference:
        return True
    cited = [float(x) for x in NUMBER_RE.findall(narrative) if not x.endswith("%")]
    for c in cited:
        c_abs = abs(c)
        if c_abs < 1e-9:
            continue
        if not any(abs(c_abs - r) / r <= tol for r in reference):
            return False
    return True


@dataclass
class NarrativeResult:
    inputs: NarrativeInputs
    narrative: str
    pass_sign: bool
    pass_features: bool
    pass_numeric: bool

    @property
    def passed(self) -> bool:
        return self.pass_sign and self.pass_features and self.pass_numeric


def generate_and_check(inputs: NarrativeInputs,
                       llm_fn: Callable[[str, NarrativeInputs], str] = mock_llm,
                       tol: float = 0.15) -> NarrativeResult:
    prompt = build_prompt(inputs)
    narrative = llm_fn(prompt, inputs)
    return NarrativeResult(
        inputs=inputs,
        narrative=narrative,
        pass_sign=check_sign(narrative, inputs.tau_hat),
        pass_features=check_features(narrative, inputs),
        pass_numeric=check_numeric(narrative, inputs, tol=tol),
    )


def run_eval(patient_rows: list[NarrativeInputs],
            llm_fn: Callable[[str, NarrativeInputs], str] = mock_llm,
            tol: float = 0.15, out_csv: Optional[str] = None) -> pd.DataFrame:
    """Batch faithfulness evaluation across patients, producing the
    empirical pass-rate the thesis (RQ2 / Limitations) flags as unreported.
    Pass llm_fn=call_claude to get a real result instead of the mock's."""
    rows = []
    for inp in patient_rows:
        r = generate_and_check(inp, llm_fn=llm_fn, tol=tol)
        rows.append({
            "patient_id": inp.patient_id,
            "tau_hat": inp.tau_hat,
            "narrative": r.narrative,
            "pass_sign": r.pass_sign,
            "pass_features": r.pass_features,
            "pass_numeric": r.pass_numeric,
            "pass_overall": r.passed,
        })
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df
