"""Stage 07 — recompute per-RQ sample sizes from the OBSERVED effect sizes
(alpha = 0.05, power = 0.80), reported side by side with the Synopsis's
assumed-effect numbers so the design is visibly updated with evidence.

Two independent proportions (the factor of 2 is required — an earlier version
of this project omitted it and was wrong by half):
    n_per_group = 2 * (z_{1-alpha/2} + z_{1-beta})**2 / h**2

Paired McNemar (discordant-pair formula):
    n = (z_{alpha/2}*sqrt(p_disc) + z_beta*sqrt(p_disc - d**2))**2 / d**2
where p_disc = (b+c)/n observed and d = (b-c)/n.

Outputs: outputs/sample_size_observed.csv, outputs/fig_power_curve.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUTS, append_log, update_manifest  # noqa: E402

ALPHA, POWER = 0.05, 0.80
Z_A = norm.ppf(1 - ALPHA / 2)
Z_B = norm.ppf(POWER)

# Assumed-effect sample sizes as submitted in the graded Synopsis
SYNOPSIS_ASSUMED = {"RQ1": 170, "RQ2": 34, "RQ3": 356, "RQ4": 564}


def n_two_prop(h: float) -> float:
    return 2 * (Z_A + Z_B) ** 2 / h**2 if h else np.inf


def n_mcnemar(b: int, c: int, n: int) -> float:
    if not n or (b + c) == 0:
        return np.inf
    p_disc = (b + c) / n
    d = abs(b - c) / n
    if d == 0 or p_disc <= d**2:
        return np.inf
    return (Z_A * np.sqrt(p_disc) + Z_B * np.sqrt(p_disc - d**2)) ** 2 / d**2


def main():
    tests = pd.read_csv(OUTPUTS / "stats_tests.csv")
    rows = []
    for _, t in tests.iterrows():
        if t["test"] == "two_proportion_z" and pd.notna(t["effect"]) and t["effect"] != 0:
            n_obs = n_two_prop(abs(float(t["effect"])))
            formula = "n = 2(z_a/2+z_b)^2/h^2"
        elif t["test"] == "mcnemar_exact" and pd.notna(t.get("n_discordant_b")):
            n_obs = n_mcnemar(int(t["n_discordant_b"]), int(t["n_discordant_c"]), int(t["n"]))
            formula = "discordant-pair formula"
        elif t["test"] == "welch_t" and pd.notna(t["effect"]) and t["effect"] != 0:
            # RQ2 primary: TWO independent groups (multi vs single questions),
            # effect = pooled-SD Cohen's d -> the factor of 2 applies here too
            n_obs = 2 * (Z_A + Z_B) ** 2 / float(t["effect"]) ** 2
            formula = "n per group = 2(z_a/2+z_b)^2/d^2 (two-sample t approximation)"
        elif t["test"] == "paired_t" and pd.notna(t.get("effect_dz")) and t["effect_dz"] != 0:
            # paired design: use the STANDARDIZED paired effect d_z =
            # mean_diff/sd(diff) emitted by 06 — the raw mean_diff is NOT a
            # standardized effect and would misscale n by 1/sd^2
            n_obs = (Z_A + Z_B) ** 2 / float(t["effect_dz"]) ** 2
            formula = "n = (z_a/2+z_b)^2/d_z^2 (paired t, d_z = mean_diff/sd_diff)"
        else:
            continue
        rows.append(dict(
            rq=t["rq"], comparison=t["comparison"], budget=t["budget"], test=t["test"],
            observed_effect=round(float(t["effect"]), 4) if np.isfinite(float(t["effect"])) else None,
            n_assumed_synopsis=SYNOPSIS_ASSUMED.get(t["rq"]),
            # blank (not -1) when the observed effect implies no finite n
            n_observed_effect=int(np.ceil(n_obs)) if np.isfinite(n_obs) else None,
            formula=formula, alpha=ALPHA, power=POWER,
            primary_comparison=bool(t["primary_comparison"]),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "sample_size_observed.csv", index=False)
    print(df[df.primary_comparison].to_string(index=False))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = np.linspace(0.05, 0.6, 200)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(hs, [n_two_prop(h) for h in hs], label="n per group (two-proportion, factor 2)")
    prim = df[(df.primary_comparison) & (df.test == "two_proportion_z")]
    for _, r in prim.iterrows():
        h = abs(r["observed_effect"])
        if h > 0.01:
            ax.scatter([h], [n_two_prop(h)], zorder=5)
            ax.annotate(f"{r['rq']} observed h={h:.2f}\n-> n={r['n_observed_effect']}",
                        (h, n_two_prop(h)), textcoords="offset points", xytext=(8, 8), fontsize=8)
    ax.axhline(600, color="green", ls="--", lw=1, label="n available (600 pooled)")
    ax.set_yscale("log")
    ax.set_xlabel("Cohen's h"); ax.set_ylabel("required n per group (log)")
    ax.set_title(f"Power analysis refresh — alpha={ALPHA}, power={POWER}\n"
                 "assumed (Synopsis) vs observed effect sizes")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS / "fig_power_curve.png", dpi=300)

    update_manifest(stage07=dict(n_rows=len(df)))
    append_log("Stage 07 power refresh",
               f"```\n{df[df.primary_comparison].to_string(index=False)}\n```")


if __name__ == "__main__":
    main()
