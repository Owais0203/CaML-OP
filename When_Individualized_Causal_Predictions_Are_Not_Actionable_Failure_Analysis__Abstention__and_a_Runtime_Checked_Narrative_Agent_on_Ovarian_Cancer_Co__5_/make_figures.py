"""Regenerate CaML-OP result figures from the P0 experimental results.

Figure 1 (sign_accuracy.png)  <- P0 Results section 1.1 (primary CATE benchmark)
Figure 2 (calibration_sweep.png) <- P0 Results section 2.1 (repeated split-conformal)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGSIZE = (5.29, 2.88)
DPI = 220
OUT = "."

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
})

# ---------------------------------------------------------------- Figure 1
# Sign accuracy by method, primary four-field + cross-fitted rerun.
methods = [
    ("CaML-OP [C, cross-fitted]", 0.5988),
    ("X-Learner",                 0.5879),
    ("CaML-OP [A, raw]",          0.5779),
    ("S-Learner",                 0.5760),
    ("Causal Forest",             0.5685),
    ("NonParamDML",               0.5670),
    ("T-Learner",                 0.5636),
    ("DRLearner",                 0.5617),
    ("CaML-OP [B, leaky]",        0.5517),
]
names = [m[0] for m in methods][::-1]
vals = [m[1] for m in methods][::-1]
colors = ["tab:red" if n.startswith("CaML-OP") else "tab:blue" for n in names]

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
y = np.arange(len(names))
ax.barh(y, vals, color=colors, height=0.68, edgecolor="none")
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=7.5)
ax.set_xlim(0.475, 0.625)
ax.axvline(0.5, color="tab:blue", linestyle="--", linewidth=1.2)
ax.set_xlabel("Effect-sign accuracy", fontsize=9)
ax.tick_params(axis="x", labelsize=8)
ax.grid(axis="y", visible=False)
for yi, v in zip(y, vals):
    ax.text(v + 0.0015, yi, f"{v:.4f}", va="center", fontsize=7)
fig.tight_layout()
fig.savefig(f"{OUT}/sign_accuracy.png", dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
# Repeated split-conformal: coverage and abstention vs nominal target.
targets = np.array([0.80, 0.90, 0.95, 0.99])
cov = np.array([0.799, 0.902, 0.955, 0.990])
cov_lo = np.array([0.683, 0.807, 0.901, 0.957])
cov_hi = np.array([0.918, 0.975, 1.000, 1.000])
abst = np.array([0.850, 0.958, 0.994, 1.000])
abst_lo = np.array([0.668, 0.820, 0.960, 1.000])
abst_hi = np.array([0.984, 1.000, 1.000, 1.000])

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
ax.errorbar(targets - 0.003, cov, yerr=[cov - cov_lo, cov_hi - cov],
            marker="o", color="tab:blue", capsize=3, linewidth=1.6,
            markersize=5, label="Achieved coverage")
ax.errorbar(targets + 0.003, abst, yerr=[abst - abst_lo, abst_hi - abst],
            marker="s", color="tab:orange", capsize=3, linewidth=1.6,
            markersize=5, label="Abstention rate")
ax.plot(targets, targets, linestyle="--", color="tab:green",
        linewidth=1.3, label="Nominal target")
ax.set_xlabel("Nominal calibration target", fontsize=9)
ax.set_ylabel("Rate", fontsize=9)
ax.set_ylim(0.63, 1.03)
ax.set_xticks(targets)
ax.set_xticklabels(["0.80", "0.90", "0.95", "0.99"])
ax.tick_params(labelsize=8)
ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{OUT}/calibration_sweep.png", dpi=DPI)
plt.close(fig)

print("figures written")