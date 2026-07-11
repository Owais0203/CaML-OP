"""
Bounded counterfactual query tool -- the interactive extension the earlier
literature check (see AGENT_INTEGRATION.md, "Bounded query interface, not
open-ended chat") recommended building instead of a freeform chat box.

Why bounded, not chat
----------------------------------------------------------------------------
Two findings from that literature check drove this design, not just the
recommendation in the abstract:

  1. Open-ended clinical LLM chat has a measured, non-trivial hallucination
     rate: one cited study found 12.5% for cancer-treatment questions;
     broader safety surveys put "problematic response" rates at 21.6%-43.2%
     depending on model. That is a direct argument against giving the model
     a freeform text box for clinical questions.
  2. Conversational XAI is documented to induce automation bias / overreliance
     that *builds over repeated turns*, not just on a single response --
     which means a one-time disclaimer at the top of a chat is not a
     sufficient countermeasure.

IMPACT (an interactive multi-disease prevention and counterfactual
treatment system, PMC12192960) is published precedent for the alternative
this module implements: ONE bounded tool -- "what if this covariate were
adjusted?" -- with a structured, checkable answer, not an open conversation.

What's actually enforced here (not just documented)
----------------------------------------------------------------------------
The countermeasure to finding (2) above is made a hard verifier, not a
prompt instruction the model can drift away from over a session:
`check_cf_calibration_restated` fails the answer if `calibration_status`
does not match the freshly recomputed calibrated-abstention result for the
counterfactual point -- on EVERY query, not just the first. An LLM that
"forgets" to restate uncertainty on turn 3 of a session fails the same
check that catches it on turn 1.

The rest of the design deliberately reuses llm_narrative_agent.py's
pattern rather than inventing a new one: structured tool-call output
(no prose parsing), a composable Verifier list, a generate -> check ->
targeted-feedback -> retry loop, and NarrativeStatus.ESCALATED as the
terminal state when a bounded number of retries doesn't produce a
compliant answer -- exactly the VeriFact-style "check every claim against
ground truth" discipline the base narrative agent already applies,
extended to a second, narrower kind of claim (a claim about a delta,
not just a single estimate).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np

from effect_shap import FEATURE_NAMES
from calibrated_abstention import should_abstain
from llm_narrative_agent import NarrativeInputs, CitedValue, NarrativeStatus


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class CounterfactualInputs:
    patient_id: str
    feature: str
    delta: float
    orig_value: float
    new_value: float
    baseline: NarrativeInputs           # the patient's already-shown estimate
    tau_hat_cf: float
    ci_lo_cf: float
    ci_hi_cf: float
    mu0_cf: float
    mu1_cf: float
    must_abstain_cf: bool

    def delta_tau(self) -> float:
        return self.tau_hat_cf - self.baseline.tau_hat

    def quantities(self) -> dict:
        return {
            "tau_hat_baseline": self.baseline.tau_hat,
            "tau_hat_cf": self.tau_hat_cf,
            "delta_tau": self.delta_tau(),
            "mu0_cf": self.mu0_cf,
            "mu1_cf": self.mu1_cf,
            "ci_lo_cf": self.ci_lo_cf,
            "ci_hi_cf": self.ci_hi_cf,
        }

    def true_direction(self, eps: float) -> str:
        d = self.delta_tau()
        if d > eps:
            return "improves"
        if d < -eps:
            return "worsens"
        return "no_meaningful_change"

    def true_calibration_status(self) -> str:
        return "not_statistically_distinguishable" if self.must_abstain_cf else "reliable"


@dataclass
class CounterfactualOutput:
    narrative: str
    direction: str                      # "improves" | "worsens" | "no_meaningful_change"
    cited_values: list                  # list[CitedValue]
    calibration_status: str             # "reliable" | "not_statistically_distinguishable"


def recompute_counterfactual(model, m0, m1, X_patient_raw: np.ndarray,
                             baseline: NarrativeInputs, feature: str, delta: float,
                             calib_factor: float) -> CounterfactualInputs:
    """Recompute the CaML-OP estimate under one perturbed raw covariate,
    reusing the already-fitted model/outcome-models (see
    run_narrative_pipeline.prepare_patient_rows's `fitted` return) rather
    than refitting anything -- a bounded query must be cheap enough to
    answer interactively."""
    if feature not in FEATURE_NAMES:
        raise ValueError(f"feature must be one of {FEATURE_NAMES}, got {feature!r}")
    idx = FEATURE_NAMES.index(feature)
    X_cf = np.atleast_2d(np.array(X_patient_raw, dtype=float).copy())
    orig_value = float(X_cf[0, idx])
    X_cf[0, idx] = orig_value + delta
    new_value = float(X_cf[0, idx])

    tau_hat_cf = float(model.predict(X_cf)[0])
    ci_lo_cf, ci_hi_cf = model.predict_interval(X_cf)
    mu0_cf = float(m0.predict_proba(X_cf)[:, 1][0])
    mu1_cf = float(m1.predict_proba(X_cf)[:, 1][0])
    half_width_cf = (float(ci_hi_cf[0]) - float(ci_lo_cf[0])) / 2
    must_abstain_cf = should_abstain(tau_hat_cf, half_width_cf, calib_factor)

    return CounterfactualInputs(
        patient_id=baseline.patient_id, feature=feature, delta=delta,
        orig_value=orig_value, new_value=new_value, baseline=baseline,
        tau_hat_cf=tau_hat_cf, ci_lo_cf=float(ci_lo_cf[0]), ci_hi_cf=float(ci_hi_cf[0]),
        mu0_cf=mu0_cf, mu1_cf=mu1_cf, must_abstain_cf=bool(must_abstain_cf),
    )


# ---------------------------------------------------------------------------
# Prompt + tool schema
# ---------------------------------------------------------------------------
DIRECTION_EPS = 0.01  # |delta_tau| below this counts as "no_meaningful_change"

COUNTERFACTUAL_PROMPT_TEMPLATE = """You are answering ONE bounded what-if question about a causal machine-learning estimate you have already shown this clinician for this patient. Submit your answer via the submit_counterfactual_answer tool. Do not introduce any claim, medical or otherwise, beyond the numbers given below.

Rules (must follow exactly):
- "direction" must be "improves" if delta_tau > {eps}, "worsens" if delta_tau < -{eps}, else "no_meaningful_change", where delta_tau = tau_hat_cf - tau_hat_baseline.
- "cited_values" must tag every number you state using exactly these tags: tau_hat_baseline, tau_hat_cf, delta_tau, mu0_cf, mu1_cf, ci_lo_cf, ci_hi_cf. Do not cite a number under the wrong tag or invent a tag not in this list.
- "calibration_status" MUST be restated on this answer even though it may already have been stated in an earlier answer this session: "not_statistically_distinguishable" if the counterfactual estimate is not reliably distinguishable from zero at a calibrated confidence level (see status below), otherwise "reliable".
- The "narrative" must be 1-2 sentences, must state the baseline estimate, the counterfactual estimate, and the calibration_status in plain language.

Patient: {patient_id}
Question: what if '{feature}' were adjusted by {delta:+.2f} (from {orig_value:.2f} to {new_value:.2f})?

Baseline estimated effect (tau_hat_baseline): {tau_hat_baseline:.3f}
Counterfactual estimated effect (tau_hat_cf): {tau_hat_cf:.3f} [{ci_lo_cf:.3f}, {ci_hi_cf:.3f}]
Counterfactual expected outcome if untreated (mu0_cf): {mu0_cf:.3f}
Counterfactual expected outcome if treated (mu1_cf): {mu1_cf:.3f}
Calibrated reliability at this adjusted value: {calib_note}
"""

RETRY_SUFFIX = """

Your previous answer was rejected by an automated check for this reason: {feedback}
Previous answer: {previous}
Submit a corrected version via submit_counterfactual_answer that fixes this while still following all rules above.
"""

PROMPT_VERSION = hashlib.sha256(
    (COUNTERFACTUAL_PROMPT_TEMPLATE + RETRY_SUFFIX).encode()).hexdigest()[:12]


def build_cf_prompt(q: CounterfactualInputs, eps: float = DIRECTION_EPS,
                    feedback: Optional[str] = None,
                    previous: Optional[CounterfactualOutput] = None) -> str:
    calib_note = ("NOT statistically distinguishable from zero" if q.must_abstain_cf
                  else "statistically supportable")
    prompt = COUNTERFACTUAL_PROMPT_TEMPLATE.format(
        eps=eps, patient_id=q.patient_id, feature=q.feature, delta=q.delta,
        orig_value=q.orig_value, new_value=q.new_value,
        tau_hat_baseline=q.baseline.tau_hat, tau_hat_cf=q.tau_hat_cf,
        ci_lo_cf=q.ci_lo_cf, ci_hi_cf=q.ci_hi_cf,
        mu0_cf=q.mu0_cf, mu1_cf=q.mu1_cf, calib_note=calib_note,
    )
    if feedback and previous:
        prompt += RETRY_SUFFIX.format(feedback=feedback, previous=asdict(previous))
    return prompt


COUNTERFACTUAL_TOOL = {
    "name": "submit_counterfactual_answer",
    "description": "Submit the answer to a bounded what-if query about the patient's estimate.",
    "input_schema": {
        "type": "object",
        "properties": {
            "narrative": {"type": "string"},
            "direction": {"type": "string",
                         "enum": ["improves", "worsens", "no_meaningful_change"]},
            "cited_values": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quantity": {"type": "string"},
                        "value": {"type": "number"},
                    },
                    "required": ["quantity", "value"],
                },
            },
            "calibration_status": {"type": "string",
                                   "enum": ["reliable", "not_statistically_distinguishable"]},
        },
        "required": ["narrative", "direction", "cited_values", "calibration_status"],
    },
}


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
def mock_llm_counterfactual(prompt: str, q: CounterfactualInputs,
                            eps: float = DIRECTION_EPS) -> CounterfactualOutput:
    """Deterministic, self-consistent-by-construction backend so the
    check/retry loop is runnable and testable without an API key -- same
    role mock_llm plays in llm_narrative_agent.py."""
    direction = q.true_direction(eps)
    calib_status = q.true_calibration_status()
    delta_tau = q.delta_tau()
    verb = {"improves": "improves", "worsens": "worsens",
           "no_meaningful_change": "does not meaningfully change"}[direction]
    calib_txt = ("this counterfactual estimate is not statistically distinguishable "
                "from zero at a calibrated confidence level" if q.must_abstain_cf
                else "this counterfactual estimate is statistically supportable")
    narrative = (
        f"Adjusting {q.feature} by {q.delta:+.2f} {verb} the estimated effect "
        f"from {q.baseline.tau_hat:+.3f} to {q.tau_hat_cf:+.3f} (delta {delta_tau:+.3f}); "
        f"{calib_txt}."
    )
    cited_values = [
        CitedValue("tau_hat_baseline", q.baseline.tau_hat),
        CitedValue("tau_hat_cf", q.tau_hat_cf),
        CitedValue("delta_tau", delta_tau),
        CitedValue("mu0_cf", q.mu0_cf),
        CitedValue("mu1_cf", q.mu1_cf),
        CitedValue("ci_lo_cf", q.ci_lo_cf),
        CitedValue("ci_hi_cf", q.ci_hi_cf),
    ]
    return CounterfactualOutput(narrative=narrative, direction=direction,
                                cited_values=cited_values, calibration_status=calib_status)


def make_flaky_cf_mock(fail_first_n: int = 1):
    """Test double: fails `fail_first_n` times by omitting the calibration
    restatement (the exact overreliance-mitigation failure mode this
    module's verifier exists to catch) before returning the correct
    answer, so the retry loop is exercised end-to-end. Not used by the
    real pipeline."""
    calls = {"n": 0}

    def flaky(prompt: str, q: CounterfactualInputs, eps: float = DIRECTION_EPS) -> CounterfactualOutput:
        calls["n"] += 1
        good = mock_llm_counterfactual(prompt, q, eps)
        if calls["n"] <= fail_first_n:
            bad_status = "reliable" if good.calibration_status != "reliable" else "not_statistically_distinguishable"
            return CounterfactualOutput(narrative=good.narrative, direction=good.direction,
                                        cited_values=good.cited_values,
                                        calibration_status=bad_status)
        return good

    return flaky


def call_claude_counterfactual(prompt: str, q: CounterfactualInputs,
                               eps: float = DIRECTION_EPS,
                               model: str = "claude-sonnet-5", temperature: float = 0.0,
                               max_tokens: int = 300) -> CounterfactualOutput:
    """Real answer generation via the Anthropic API with a forced tool
    call, mirroring llm_narrative_agent.call_claude. Requires the
    `anthropic` package and ANTHROPIC_API_KEY."""
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        tools=[COUNTERFACTUAL_TOOL],
        tool_choice={"type": "tool", "name": "submit_counterfactual_answer"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            data = block.input
            return CounterfactualOutput(
                narrative=data["narrative"], direction=data["direction"],
                cited_values=[CitedValue(**cv) for cv in data.get("cited_values", [])],
                calibration_status=data["calibration_status"],
            )
    raise RuntimeError("Model did not call submit_counterfactual_answer")


# ---------------------------------------------------------------------------
# Verifiers -- same discipline as llm_narrative_agent.py: check the
# structured claim against recomputed ground truth, not the prose.
# ---------------------------------------------------------------------------
def check_cf_numeric(output: CounterfactualOutput, q: CounterfactualInputs,
                     tol: float = 0.15, atol: float = 0.005) -> bool:
    quantities = q.quantities()
    for cv in output.cited_values:
        true_val = quantities.get(cv.quantity)
        if true_val is None:
            return False
        if abs(cv.value - true_val) > max(tol * abs(true_val), atol):
            return False
    return True


def check_cf_direction(output: CounterfactualOutput, q: CounterfactualInputs,
                       eps: float = DIRECTION_EPS) -> bool:
    return output.direction == q.true_direction(eps)


def check_cf_calibration_restated(output: CounterfactualOutput, q: CounterfactualInputs) -> bool:
    """The hard-enforced version of the literature's overreliance
    countermeasure: this must be correct on every single query, not just
    the first one in a session."""
    return output.calibration_status == q.true_calibration_status()


@dataclass
class CFVerifier:
    name: str
    check: Callable[[CounterfactualOutput, CounterfactualInputs], bool]
    feedback: Callable[[CounterfactualInputs], str]


def _cf_numeric_feedback(q: CounterfactualInputs) -> str:
    return (f"one or more cited_values did not match its declared quantity within "
           f"tolerance; valid quantities are {sorted(q.quantities())}.")


def _cf_direction_feedback(q: CounterfactualInputs) -> str:
    return f"direction must be '{q.true_direction(DIRECTION_EPS)}' given delta_tau={q.delta_tau():+.3f}."


def _cf_calibration_feedback(q: CounterfactualInputs) -> str:
    return (f"calibration_status must be '{q.true_calibration_status()}' for this "
           f"counterfactual point -- restate it even if a prior answer already stated it.")


def default_cf_verifiers(tol: float = 0.15, atol: float = 0.005,
                         eps: float = DIRECTION_EPS) -> list:
    return [
        CFVerifier("cf_numeric", lambda o, q: check_cf_numeric(o, q, tol=tol, atol=atol),
                  _cf_numeric_feedback),
        CFVerifier("cf_direction", lambda o, q: check_cf_direction(o, q, eps=eps),
                  _cf_direction_feedback),
        CFVerifier("cf_calibration_restated", check_cf_calibration_restated,
                  _cf_calibration_feedback),
    ]


@dataclass
class CounterfactualResult:
    query: CounterfactualInputs
    output: CounterfactualOutput
    checks: dict
    status: NarrativeStatus = NarrativeStatus.PASSED
    attempts: int = 1

    @property
    def passed(self) -> bool:
        return self.status == NarrativeStatus.PASSED

    @property
    def narrative(self) -> str:
        return self.output.narrative


def generate_and_check_counterfactual(
        q: CounterfactualInputs,
        llm_fn: Callable = mock_llm_counterfactual,
        tol: float = 0.15, atol: float = 0.005, eps: float = DIRECTION_EPS,
        max_retries: int = 2, verifiers: Optional[list] = None) -> CounterfactualResult:
    """Same generate -> check -> targeted-feedback -> retry loop as
    llm_narrative_agent.generate_and_check, applied to a bounded what-if
    answer instead of the initial narrative."""
    verifiers = verifiers if verifiers is not None else default_cf_verifiers(tol, atol, eps)
    feedback, previous = None, None
    checks, output = {}, None
    for attempt in range(1, max_retries + 2):
        prompt = build_cf_prompt(q, eps=eps, feedback=feedback, previous=previous)
        output = llm_fn(prompt, q)
        checks = {v.name: v.check(output, q) for v in verifiers}
        if all(checks.values()):
            return CounterfactualResult(q, output, checks, NarrativeStatus.PASSED, attempts=attempt)
        feedback = " ".join(v.feedback(q) for v in verifiers if not checks[v.name])
        previous = output
    return CounterfactualResult(q, output, checks, NarrativeStatus.ESCALATED, attempts=attempt)


def query_counterfactual(baseline: NarrativeInputs, X_patient_raw: np.ndarray,
                         feature: str, delta: float, model, m0, m1, calib_factor: float,
                         llm_fn: Callable = mock_llm_counterfactual,
                         max_retries: int = 2) -> CounterfactualResult:
    """Top-level entry point a dashboard or CLI calls: one bounded
    what-if question in, one verified answer out."""
    q = recompute_counterfactual(model, m0, m1, X_patient_raw, baseline, feature, delta, calib_factor)
    return generate_and_check_counterfactual(q, llm_fn=llm_fn, max_retries=max_retries)


if __name__ == "__main__":
    from run_narrative_pipeline import prepare_patient_rows

    patient_rows, X_sample, sample_idx, calib_factor, fitted = prepare_patient_rows(
        n_patients=5, seed=42, verbose=True)
    baseline = patient_rows[0]
    X_patient = X_sample[0]

    print("\n" + "=" * 70)
    print(f"Baseline narrative for {baseline.patient_id}: tau_hat={baseline.tau_hat:+.3f}")
    print("=" * 70)

    feature = baseline.true_drivers().pop() if baseline.true_drivers() else FEATURE_NAMES[0]
    for delta in (-5.0, 5.0):
        result = query_counterfactual(
            baseline, X_patient, feature, delta,
            fitted["model"], fitted["m0"], fitted["m1"], calib_factor,
            llm_fn=mock_llm_counterfactual)
        print(f"\nQuery: what if '{feature}' changed by {delta:+.1f}?")
        print(f"  status={result.status.value} attempts={result.attempts} checks={result.checks}")
        print(f"  {result.narrative}")

    print("\n" + "=" * 70)
    print("Retry-loop smoke test (flaky mock omits calibration restatement once)")
    print("=" * 70)
    flaky_result = query_counterfactual(
        baseline, X_patient, feature, 5.0,
        fitted["model"], fitted["m0"], fitted["m1"], calib_factor,
        llm_fn=make_flaky_cf_mock(fail_first_n=1))
    print(f"  status={flaky_result.status.value} attempts={flaky_result.attempts} "
         f"(expect attempts=2: fails once, then a corrected retry passes)")
    assert flaky_result.attempts == 2, "flaky mock should be caught and corrected on retry"
    assert flaky_result.passed
    print("  OK")
