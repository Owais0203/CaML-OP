"""
E-value sensitivity analysis (VanderWeele & Ding, 2017) -- an
identification-confidence signal distinct from both of the statistical
uncertainty signals already in the pipeline (CI width from
calibrated_abstention.py; R/DR model-form disagreement from
effect_shap.py's predict_components).

Motivation: the thesis is explicit that conditional unconfoundedness is
doubtful for TCGA-OV -- performance status, comorbidities and physician
preference are not recorded, and each plausibly affects both treatment
and survival (Section "Identification assumptions"). Every uncertainty
mechanism built so far (calibrated abstention, R/DR disagreement) answers
"how much does sampling noise / model choice affect this estimate" --
none of them ask "how robust is this estimate to the confounding the
thesis itself says is likely present." The E-value does: it's the minimum
strength an unmeasured confounder would need (on both the
treatment-association and outcome-association legs) to fully explain away
the observed association, on the risk-ratio scale. A low E-value means a
weak, plausible confounder could nullify the finding; a high E-value means
only an implausibly strong one could.

This is deliberately a *per-patient* sensitivity signal (computed from
that patient's own mu0_hat/mu1_hat), not a single global E-value for the
whole cohort -- individualized treatment-effect claims warrant an
individualized robustness check.

Caveat, stated plainly: the E-value threshold below (2.0) is a documented,
provisional convention, not an empirically validated cutoff for this
clinical context -- there is no domain-calibrated prior here on how strong
an unmeasured confounder like performance status plausibly is for
platinum-therapy assignment. Treat FRAGILE_E_VALUE_THRESHOLD as a labelled
assumption a domain reviewer should revisit, not a derived constant.
"""
from __future__ import annotations

import numpy as np

FRAGILE_E_VALUE_THRESHOLD = 2.0


def risk_ratio(mu0: float, mu1: float, eps: float = 1e-6) -> float:
    """Approximate risk ratio from the two arm-level outcome estimates,
    clipped away from 0/1 so the ratio and its E-value stay finite for
    near-certain or near-impossible predicted outcomes."""
    mu0 = float(np.clip(mu0, eps, 1 - eps))
    mu1 = float(np.clip(mu1, eps, 1 - eps))
    return mu1 / mu0


def e_value(mu0: float, mu1: float) -> float:
    """E-value (VanderWeele & Ding, 2017) on the risk-ratio scale. Always
    expressed as >= 1 by taking RR on the side further from the null (1),
    per the original definition, so a higher e_value always means a more
    robust (harder to explain away) finding regardless of effect direction."""
    rr = risk_ratio(mu0, mu1)
    if rr < 1:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


def is_fragile(mu0: float, mu1: float, threshold: float = FRAGILE_E_VALUE_THRESHOLD) -> bool:
    """True if the estimate's E-value falls below the (provisional)
    fragility threshold -- i.e. a plausibly weak unmeasured confounder
    could nullify this specific patient's estimated effect."""
    return e_value(mu0, mu1) < threshold
