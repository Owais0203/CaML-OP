"""
LLM narrative agent + faithfulness checks.

Implements the piece of the CaML-OP architecture that main.tex describes in
detail (Section "LLM narrative and faithfulness checks", RQ2) but that
code.py never built. The thesis's own Limitations section names this
explicitly: "No pass-rate, failure-rate or clinician evaluation is reported
for the narrative module. The architecture is documented, but the empirical
performance is not." This module is a runnable version of that architecture
plus the missing evaluation harness.

Design: generator/verifier agent loop with structured output.
  1. generate(...)  -- an LLM produces a 2-3 sentence narrative *plus*
     machine-checkable structured claims (sign, cited features, cited
     numeric values each tagged with which quantity they claim to be) via
     a forced tool call. This replaces parsing the free-text narrative with
     regex/keyword heuristics, which is fragile by construction (see
     CHANGELOG below for three concrete failures that approach had).
  2. Three deterministic checks validate the structured claims exactly:
       - check_sign:      declared sign matches sign(tau_hat), or is
                          "indeterminate" when NarrativeInputs.must_abstain
                          is set by calibrated_abstention.py (see that
                          module: the reported CI is known to under-cover,
                          so a raw-tau_hat-sign claim can overstate
                          confidence the interval doesn't actually support)
       - check_features:  every cited feature is a true top Effect-SHAP
                           driver (set membership, not text search)
       - check_numeric:   every cited value matches the *specific*
                           quantity it claims to represent, within a
                           combined absolute+relative tolerance
  3. On failure, the specific failing check(s) are fed back to the model
     and it gets up to `max_retries` attempts to produce a compliant
     narrative -- a real generate -> check -> regenerate loop, not a
     single-shot generate-then-filter.

The LLM call is dependency-injected (`llm_fn`) so the checking logic and
batch evaluation harness are runnable and testable without any API key.
Swap in `call_claude` (requires `anthropic` + ANTHROPIC_API_KEY) to
generate real narratives and get an actual empirical pass-rate -- the
`mock_llm` default is a template generator for testing only.

CHANGELOG (prose-parsing version -> structured version)
---------------------------------------------------------------------------
The first version of this module checked free-text narratives with regex.
Running it against real model output (not just reading the code) surfaced
three failures baked into that approach:
  1. check_numeric flagged the fixed "95%" confidence-level constant as an
     uncited number.
  2. check_sign treated the noun "benefit" as a positive-direction signal,
     so "a reduced survival benefit" (a correct negative-effect claim)
     failed.
  3. check_features required every one of up to six candidate SHAP drivers
     to be named inside a two-to-three sentence cap, and separately, the
     covariate short names ("age", "stage", "intent") are ordinary English
     words that can appear in prose for reasons unrelated to citing them
     as a driver ("at this stage of treatment").
None of these were faithfulness failures -- they were prose-parsing
failures. Structured output (the model declares sign/features/values as
typed fields via a tool call, instead of us inferring them from text)
removes the entire class: there is no text to misparse.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class NarrativeInputs:
    patient_id: str
    covariates: dict
    tau_hat: float
    ci_lo: float
    ci_hi: float
    top_positive: list = field(default_factory=list)  # [(feature, phi), ...], phi > 0
    top_negative: list = field(default_factory=list)  # [(feature, phi), ...], phi < 0
    # Set by calibrated_abstention.should_abstain() against a *calibrated*
    # interval (not the raw ci_lo/ci_hi above, which are the reported,
    # known-under-covering interval). When True, the model must not assert
    # a directional sign -- see check_sign and the "indeterminate" tool
    # schema value.
    must_abstain: bool = False

    def true_drivers(self) -> set:
        return {f for f, _ in self.top_positive + self.top_negative}

    def quantities(self) -> dict:
        """Every quantity the model is allowed to cite a number against,
        keyed by the tag it must use in cited_values."""
        q = {"tau_hat": self.tau_hat, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi}
        q.update({f"shap:{f}": v for f, v in self.top_positive + self.top_negative})
        return q


@dataclass
class CitedValue:
    quantity: str
    value: float


@dataclass
class NarrativeOutput:
    """The model's structured submission for one patient."""
    narrative: str
    sign: str                       # "positive" | "negative"
    cited_features: list            # list[str]
    cited_values: list              # list[CitedValue]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a clinical decision-support narrative generator.
Given the structured outputs of a causal machine-learning model estimating
the individualized effect of platinum-based chemotherapy on five-year
ovarian cancer survival, submit a narrative via the submit_clinical_narrative
tool.

Rules (must follow exactly):
- "sign" must be "positive" if tau_hat > 0 or "negative" if tau_hat < 0
  {abstain_rule}
- "cited_features" must contain only feature names drawn from the top
  drivers listed below. Do not invent or include features not listed.
- "cited_values" must tag every number you state in the narrative with
  which quantity it represents, using exactly one of these tags: tau_hat,
  ci_lo, ci_hi, or shap:<feature> for a listed driver. Do not cite a number
  under the wrong tag.
- The "narrative" field itself must be two to three sentences, must state
  the sign/magnitude of the effect and its uncertainty interval in plain
  language, must name the cited features, and must not introduce outside
  medical knowledge or claims not derivable from the numbers given.
{abstain_narrative_rule}

Patient covariates: {covariates}
Estimated ITE (tau_hat): {tau_hat:.3f}
95% uncertainty interval: [{ci_lo:.3f}, {ci_hi:.3f}]
Top features increasing benefit: {top_positive}
Top features decreasing benefit: {top_negative}
"""

ABSTAIN_RULE = ('-- EXCEPT: a calibrated reliability check on this estimate has '
                'determined the sign is not statistically distinguishable from '
                'zero, so "sign" must be "indeterminate" instead, regardless of '
                'the raw tau_hat value above.')
NO_ABSTAIN_RULE = "."
ABSTAIN_NARRATIVE_RULE = (
    '- Because sign is "indeterminate", the narrative must say the estimated '
    "effect is not reliably distinguishable from no effect at this patient's "
    'confidence level, and must not claim benefit or harm.')

RETRY_SUFFIX = """

Your previous submission was rejected by an automated faithfulness check
for this reason: {feedback}
Previous submission: {previous}
Submit a corrected version via submit_clinical_narrative that fixes this
while still following all rules above.
"""


def build_prompt(inputs: NarrativeInputs, feedback: Optional[str] = None,
                 previous: Optional[NarrativeOutput] = None) -> str:
    prompt = PROMPT_TEMPLATE.format(
        covariates=inputs.covariates,
        tau_hat=inputs.tau_hat,
        ci_lo=inputs.ci_lo,
        ci_hi=inputs.ci_hi,
        top_positive=", ".join(f"{f} ({v:+.3f})" for f, v in inputs.top_positive) or "none",
        top_negative=", ".join(f"{f} ({v:+.3f})" for f, v in inputs.top_negative) or "none",
        abstain_rule=ABSTAIN_RULE if inputs.must_abstain else NO_ABSTAIN_RULE,
        abstain_narrative_rule=ABSTAIN_NARRATIVE_RULE if inputs.must_abstain else "",
    )
    if feedback and previous:
        prompt += RETRY_SUFFIX.format(feedback=feedback, previous=asdict(previous))
    return prompt


# `main.tex` prose, `PROMPT_TEMPLATE` and `RETRY_SUFFIX` jointly define what a
# given pass-rate actually measured. Stamped into every result row so a
# stored pass-rate can't silently go stale if the prompt changes later.
PROMPT_VERSION = hashlib.sha256(
    (PROMPT_TEMPLATE + RETRY_SUFFIX).encode()).hexdigest()[:12]


NARRATIVE_TOOL = {
    "name": "submit_clinical_narrative",
    "description": "Submit the clinical narrative and its structured, machine-checkable claims.",
    "input_schema": {
        "type": "object",
        "properties": {
            "narrative": {"type": "string"},
            "sign": {"type": "string", "enum": ["positive", "negative", "indeterminate"]},
            "cited_features": {"type": "array", "items": {"type": "string"}},
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
        },
        "required": ["narrative", "sign", "cited_features", "cited_values"],
    },
}


# ---------------------------------------------------------------------------
# LLM backends -- llm_fn: (prompt: str, inputs: NarrativeInputs) -> NarrativeOutput
# ---------------------------------------------------------------------------
def mock_llm(prompt: str, inputs: NarrativeInputs) -> NarrativeOutput:
    """Deterministic, self-consistent-by-construction narrative used when no
    LLM is configured. Exists so the check/retry/harness logic below is
    runnable and testable without an API key -- NOT a substitute for a real
    model in the thesis's empirical evaluation."""
    feats = (inputs.top_positive[:1] + inputs.top_negative[:1])
    feat_names = [f for f, _ in feats]
    feat_txt = " and ".join(feat_names) if feat_names else "the modelled covariates"
    if inputs.must_abstain:
        sign = "indeterminate"
        narrative = (
            f"This patient's estimated treatment effect is {inputs.tau_hat:+.3f} "
            f"(95% interval {inputs.ci_lo:.3f} to {inputs.ci_hi:.3f}), but at a "
            f"calibrated confidence level this is not reliably distinguishable "
            f"from no effect. The estimate is most influenced by {feat_txt}, and "
            f"no directional benefit or harm claim can be made for this patient."
        )
    else:
        sign = "positive" if inputs.tau_hat > 0 else "negative"
        direction = "an increased" if sign == "positive" else "a reduced"
        narrative = (
            f"This patient's estimated treatment effect is {inputs.tau_hat:+.3f} "
            f"(95% interval {inputs.ci_lo:.3f} to {inputs.ci_hi:.3f}), indicating "
            f"{direction} five-year survival benefit from platinum therapy. "
            f"The estimate is driven mainly by {feat_txt}, and the interval width "
            f"reflects meaningful uncertainty in this individualized estimate."
        )
    cited_values = [
        CitedValue("tau_hat", inputs.tau_hat),
        CitedValue("ci_lo", inputs.ci_lo),
        CitedValue("ci_hi", inputs.ci_hi),
    ]
    return NarrativeOutput(narrative=narrative, sign=sign,
                           cited_features=feat_names, cited_values=cited_values)


def make_flaky_mock(fail_first_n: int = 1):
    """Test double: fails `fail_first_n` times (wrong sign) before returning
    mock_llm's correct output, so the retry loop (#2 below) can be exercised
    end-to-end rather than only unit-tested in isolation. Not used by the
    real pipeline."""
    calls = {"n": 0}

    def flaky(prompt: str, inputs: NarrativeInputs) -> NarrativeOutput:
        calls["n"] += 1
        if calls["n"] <= fail_first_n:
            good = mock_llm(prompt, inputs)
            bad_sign = "negative" if good.sign == "positive" else "positive"
            return NarrativeOutput(narrative=good.narrative, sign=bad_sign,
                                   cited_features=good.cited_features,
                                   cited_values=good.cited_values)
        return mock_llm(prompt, inputs)

    return flaky


def call_claude(prompt: str, inputs: NarrativeInputs,
                model: str = "claude-sonnet-5", temperature: float = 0.0,
                max_tokens: int = 400) -> NarrativeOutput:
    """Real narrative generation via the Anthropic API, using a forced tool
    call so the response is structured rather than free text. Requires the
    `anthropic` package and ANTHROPIC_API_KEY.

    temperature=0.0 by default so a reported pass-rate is reproducible
    run-to-run for the same patient set and prompt version.
    """
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=[NARRATIVE_TOOL],
        tool_choice={"type": "tool", "name": "submit_clinical_narrative"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            data = block.input
            return NarrativeOutput(
                narrative=data["narrative"],
                sign=data["sign"],
                cited_features=list(data.get("cited_features", [])),
                cited_values=[CitedValue(**cv) for cv in data.get("cited_values", [])],
            )
    raise RuntimeError("Model did not call submit_clinical_narrative")


def with_backoff(llm_fn: Callable, retries: int = 3, base_delay: float = 1.0):
    """Wraps an llm_fn with exponential backoff for transient API errors
    (network blips, rate limits) -- distinct from the faithfulness retry
    loop in generate_and_check, which retries on a *content* failure, not a
    *transport* failure."""

    def wrapped(prompt: str, inputs: NarrativeInputs) -> NarrativeOutput:
        last_exc = None
        for attempt in range(retries):
            try:
                return llm_fn(prompt, inputs)
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: any transient API failure
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
        raise last_exc

    return wrapped


# ---------------------------------------------------------------------------
# Faithfulness checks -- validate structured fields, not prose
# ---------------------------------------------------------------------------
def check_sign(output: NarrativeOutput, inputs: NarrativeInputs) -> bool:
    if inputs.must_abstain:
        return output.sign == "indeterminate"
    expected = "positive" if inputs.tau_hat > 0 else "negative"
    return output.sign == expected


def check_features(output: NarrativeOutput, inputs: NarrativeInputs) -> bool:
    """Every cited feature must be a true top Effect-SHAP driver (set
    membership on the model's own declared list -- no text search, so
    there is no risk of a covariate name like "stage" or "intent" being
    mistaken for a citation because it happens to appear in prose)."""
    true_drivers = inputs.true_drivers()
    cited = set(output.cited_features)
    if not true_drivers:
        return not cited
    if cited - true_drivers:
        return False  # hallucinated a feature that isn't a true driver
    return len(cited) > 0


def check_numeric(output: NarrativeOutput, inputs: NarrativeInputs,
                  tol: float = 0.15, atol: float = 0.005) -> bool:
    """Every cited value must match the *specific* quantity it claims to
    represent (via its tag), within a combined absolute+relative tolerance
    (numpy.isclose-style: allowed deviation = max(tol * |true|, atol)).

    The combined tolerance matters at small scale: a pure relative
    tolerance is unforgivingly tight near tau_hat ~ 0 (e.g. 15% of 0.002 is
    an allowed deviation of only 0.0003), tighter than the .3f display
    precision used everywhere else in the pipeline. atol=0.005 sets a floor
    matching that display precision.
    """
    quantities = inputs.quantities()
    for cv in output.cited_values:
        true_val = quantities.get(cv.quantity)
        if true_val is None:
            return False  # cited a quantity tag that doesn't exist -> hallucinated reference
        if abs(cv.value - true_val) > max(tol * abs(true_val), atol):
            return False
    return True


@dataclass
class NarrativeResult:
    inputs: NarrativeInputs
    output: NarrativeOutput
    pass_sign: bool
    pass_features: bool
    pass_numeric: bool
    attempts: int = 1

    @property
    def passed(self) -> bool:
        return self.pass_sign and self.pass_features and self.pass_numeric

    @property
    def narrative(self) -> str:
        return self.output.narrative


def _check(output: NarrativeOutput, inputs: NarrativeInputs,
          tol: float, atol: float) -> tuple:
    return (check_sign(output, inputs),
            check_features(output, inputs),
            check_numeric(output, inputs, tol=tol, atol=atol))


def _feedback_for(pass_sign: bool, pass_features: bool, pass_numeric: bool,
                  inputs: NarrativeInputs) -> str:
    msgs = []
    if not pass_sign:
        if inputs.must_abstain:
            msgs.append("sign must be 'indeterminate': the calibrated interval "
                       "for this patient spans zero, so no directional claim "
                       "is statistically supportable regardless of tau_hat's raw sign.")
        else:
            expected = "positive" if inputs.tau_hat > 0 else "negative"
            msgs.append(f"sign must be '{expected}' (tau_hat={inputs.tau_hat:+.3f}).")
    if not pass_features:
        msgs.append(f"cited_features must be drawn only from {sorted(inputs.true_drivers())}.")
    if not pass_numeric:
        msgs.append(f"one or more cited_values did not match its declared quantity "
                    f"within tolerance; valid quantities are {sorted(inputs.quantities())}.")
    return " ".join(msgs)


def generate_and_check(inputs: NarrativeInputs,
                       llm_fn: Callable[[str, NarrativeInputs], NarrativeOutput] = mock_llm,
                       tol: float = 0.15, atol: float = 0.005,
                       max_retries: int = 2) -> NarrativeResult:
    """Generate -> check -> (on failure) feed back the specific failing
    check(s) and regenerate, up to max_retries additional attempts."""
    feedback, previous = None, None
    for attempt in range(1, max_retries + 2):  # first attempt + max_retries retries
        prompt = build_prompt(inputs, feedback=feedback, previous=previous)
        output = llm_fn(prompt, inputs)
        pass_sign, pass_features, pass_numeric = _check(output, inputs, tol, atol)
        if pass_sign and pass_features and pass_numeric:
            return NarrativeResult(inputs, output, pass_sign, pass_features,
                                   pass_numeric, attempts=attempt)
        feedback = _feedback_for(pass_sign, pass_features, pass_numeric, inputs)
        previous = output
    return NarrativeResult(inputs, output, pass_sign, pass_features,
                           pass_numeric, attempts=attempt)


# ---------------------------------------------------------------------------
# Batch evaluation harness
# ---------------------------------------------------------------------------
class _DiskCache:
    """Thread-safe (prompt, model) -> raw NarrativeOutput cache, so
    re-running run_eval after a *checking-logic* change doesn't re-spend API
    calls for prompts that were already generated. Keyed on the prompt text
    itself (which already embeds PROMPT_VERSION indirectly, since it's built
    from PROMPT_TEMPLATE) plus the model name."""

    def __init__(self, path: Optional[str]):
        self.path = path
        self.lock = threading.Lock()
        self._data = {}
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    self._data[row["key"]] = row["output"]

    @staticmethod
    def _key(prompt: str, model: str) -> str:
        return hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[dict]:
        return self._data.get(self._key(prompt, model))

    def put(self, prompt: str, model: str, output: NarrativeOutput):
        key = self._key(prompt, model)
        row = asdict(output)
        self._data[key] = row
        if self.path:
            with self.lock:
                with open(self.path, "a") as f:
                    f.write(json.dumps({"key": key, "output": row}) + "\n")


def run_eval(patient_rows: list, llm_fn: Callable = mock_llm,
            tol: float = 0.15, atol: float = 0.005, max_retries: int = 2,
            out_csv: Optional[str] = None, cache_path: Optional[str] = None,
            model_name: str = "unknown", max_workers: int = 1) -> pd.DataFrame:
    """Batch faithfulness evaluation across patients, producing the
    empirical pass-rate the thesis (RQ2 / Limitations) flags as unreported.

    - out_csv is written incrementally (one row appended per completed
      patient) so a crash partway through a large/live run doesn't lose
      already-generated results.
    - cache_path, if given, memoizes (prompt, model_name) -> output so
      re-running after a checking-logic tweak doesn't re-call a live API
      for prompts already generated.
    - max_workers > 1 runs patients concurrently (each patient is
      independent), useful for a live run over the ~137-patient test set.
    """
    cache = _DiskCache(cache_path)
    if out_csv and os.path.exists(out_csv):
        os.remove(out_csv)
    lock = threading.Lock()
    rows = []

    def cached_llm_fn(prompt: str, inputs: NarrativeInputs) -> NarrativeOutput:
        hit = cache.get(prompt, model_name)
        if hit is not None:
            return NarrativeOutput(
                narrative=hit["narrative"], sign=hit["sign"],
                cited_features=hit["cited_features"],
                cited_values=[CitedValue(**cv) for cv in hit["cited_values"]],
            )
        output = llm_fn(prompt, inputs)
        cache.put(prompt, model_name, output)
        return output

    def process(inp: NarrativeInputs) -> dict:
        r = generate_and_check(inp, llm_fn=cached_llm_fn, tol=tol, atol=atol,
                               max_retries=max_retries)
        return {
            "patient_id": inp.patient_id,
            "tau_hat": inp.tau_hat,
            "narrative": r.narrative,
            "sign": r.output.sign,
            "cited_features": ";".join(r.output.cited_features),
            "cited_values": json.dumps(asdict_cited(r.output.cited_values)),
            "pass_sign": r.pass_sign,
            "pass_features": r.pass_features,
            "pass_numeric": r.pass_numeric,
            "pass_overall": r.passed,
            "attempts": r.attempts,
            "prompt_version": PROMPT_VERSION,
        }

    def append_row(row: dict):
        rows.append(row)
        if out_csv:
            with lock:
                pd.DataFrame([row]).to_csv(
                    out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    if max_workers <= 1:
        for inp in patient_rows:
            append_row(process(inp))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(process, inp): inp for inp in patient_rows}
            for fut in as_completed(futures):
                append_row(fut.result())

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("patient_id").reset_index(drop=True)
    return df


def asdict_cited(cited_values: list) -> list:
    return [asdict(cv) for cv in cited_values]
