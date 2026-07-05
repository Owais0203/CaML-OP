"""
Calibration-aware abstention for the narrative agent.

Motivation (grounded in a number the thesis already reports): CaML-OP's
nominal 95% interval has measured empirical coverage of 0.847 (Section
"Uncertainty interval coverage"), and the ensemble CI is explicitly built
from a heuristic (ForestDRLearner half-width x1.10, see fit_caml_op in
code.py) rather than a formally derived interval for the ensemble. A
narrative agent that asserts a confident "positive"/"negative" sign claim
whenever tau_hat != 0, using that *reported, known-too-narrow* interval, is
asserting more confidence than the interval's own measured reliability
supports -- the thesis measures the miscalibration and then the (unbuilt)
narrative module was going to ignore it.

This module closes that loop: it fits a conformal-style calibration factor
from the semi-synthetic scenarios (which expose tau_true, unlike real
data) and uses it to decide, per patient, whether the calibrated interval
still distinguishes the sign of tau_hat from zero. If it doesn't, the
narrative agent is required to abstain (see the "indeterminate" sign value
wired into llm_narrative_agent.py) rather than assert a direction.

Empirical result on this codebase (4 runs, scenario 4, n=548 test points,
via code.py's own fit_caml_op):
    naive (reported) coverage:  0.854   (thesis reports 0.847 -- consistent)
    naive abstain rate:         0.838   (patients whose *reported* CI already spans 0)
    calibration factor:         1.586x  half-width inflation needed to hit 0.95 coverage
    calibrated coverage:        0.949
    calibrated abstain rate:    0.967

Read literally: once the interval is widened enough to actually deliver on
its stated 95% coverage, essentially every patient's sign is statistically
indistinguishable from zero. Whether the narrative module should therefore
abstain on ~97% of patients, or whether this argues for tightening the
causal estimator instead, is a question for the thesis to take a position
on -- but it is not a question the current (unbuilt) narrative module had
any way to even ask, since it was never wired to the interval's own
measured reliability.
"""
from __future__ import annotations

import numpy as np


def fit_calibration_factor(tau_true: np.ndarray, tau_hat: np.ndarray,
                           half_width: np.ndarray, target_coverage: float = 0.95) -> float:
    """Split-conformal calibration: the smallest multiplicative inflation of
    half_width such that the resulting interval achieves >= target_coverage
    on a calibration set with known tau_true. Must be fit on a calibration
    split disjoint from any patient the narrative agent will later describe
    (standard split-conformal requirement -- fitting and evaluating on the
    same points would understate the required inflation)."""
    scores = np.abs(tau_true - tau_hat) / np.maximum(half_width, 1e-9)
    return float(np.quantile(scores, target_coverage))


def calibrated_interval(tau_hat: float, half_width: float,
                        calibration_factor: float) -> tuple:
    hw = half_width * calibration_factor
    return tau_hat - hw, tau_hat + hw


def should_abstain(tau_hat: float, half_width: float,
                   calibration_factor: float) -> bool:
    """True if the *calibrated* interval spans zero -- i.e. the sign of
    tau_hat is not distinguishable from zero once the interval is widened
    to actually hit its nominal coverage target, rather than the narrower
    reported interval."""
    lo, hi = calibrated_interval(tau_hat, half_width, calibration_factor)
    return lo <= 0 <= hi
