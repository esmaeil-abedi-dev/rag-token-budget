"""Stage 06 — statistics and figures. No API calls; consumes eval_records.parquet.

Reports BOTH test families, per the brief:
  pre-registered: two-proportion z with Cohen's h (RQ1/RQ3/RQ4), paired t (RQ2)
  paired-appropriate: McNemar exact with discordant counts (RQ1/RQ3)
Benjamini–Hochberg FDR within each RQ family; paired bootstrap 95% CIs
(10k resamples over questions); effect sizes reported with the CIs.

Outputs: fig_pareto, fig_accuracy_by_budget, fig_hop_breakdown, fig_by_dataset,
fig_structured_vs_prose, fig_latency_cost (all 300 dpi),
preliminary_results.csv, stats_tests.csv, graph_confound_checks.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).parent))
from common import BUDGETS, DATA, OUTPUTS, SEED, append_log, update_manifest  # noqa: E402

ARM_ORDER = ["naive_topk", "naive_topk_dedup", "rerank_topk", "compress_llmlingua",
             "summarize_recomp", "graph_select"]
SYNOPSIS_ARMS = ["naive_topk", "rerank_topk", "compress_llmlingua",
                 "summarize_recomp", "graph_select"]
N_BOOT = 10_000
RNG = np.random.default_rng(SEED)

PRIMARY_BUDGET = 1000  # pre-declared primary comparisons live at this budget


def cohens_h(p1: float, p2: float) -> float:
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def boot_ci(values: np.ndarray, idx_matrix: np.ndarray) -> tuple[float, float]:
    means = values[idx_matrix].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_frame(df: pd.DataFrame, arm_a: str, arm_b: str, budget) -> pd.DataFrame:
    a = df[(df.arm == arm_a) & (df.budget == budget)][["question_id", "em", "f1"]]
    b = df[(df.arm == arm_b) & (df.budget == budget)][["question_id", "em", "f1"]]
    return a.merge(b, on="question_id", suffixes=("_a", "_b"))


def mcnemar_exact(m: pd.DataFrame) -> dict:
    b = int(((m.em_a == 1) & (m.em_b == 0)).sum())
    c = int(((m.em_a == 0) & (m.em_b == 1)).sum())
    from statsmodels.stats.contingency_tables import mcnemar as _mc

    table = [[int(((m.em_a == 1) & (m.em_b == 1)).sum()), b],
             [c, int(((m.em_a == 0) & (m.em_b == 0)).sum())]]
    res = _mc(table, exact=True)
    return dict(b=b, c=c, statistic=float(res.statistic), p=float(res.pvalue))


def two_prop(m: pd.DataFrame) -> dict:
    from statsmodels.stats.proportion import proportions_ztest

    n = len(m)
    x1, x2 = int(m.em_a.sum()), int(m.em_b.sum())
    with np.errstate(invalid="ignore"):  # degenerate cells -> NaN p, handled by bh_adjust
        stat, p = proportions_ztest([x1, x2], [n, n])
    p1, p2 = x1 / n, x2 / n
    diffs = []
    idx = RNG.integers(0, n, size=(2000, n))
    ea, eb = m.em_a.to_numpy(), m.em_b.to_numpy()
    for row in idx:
        diffs.append(cohens_h(ea[row].mean(), eb[row].mean()))
    return dict(statistic=float(stat), p=float(p), h=cohens_h(p1, p2),
                h_ci_lo=float(np.percentile(diffs, 2.5)),
                h_ci_hi=float(np.percentile(diffs, 97.5)),
                p1=p1, p2=p2, n=n)


def bh_adjust(rows: list[dict]) -> None:
    """BH-FDR within each (rq, test) family, in place. NaN p-values (degenerate
    cells) are excluded from the family — one NaN must not erase every p_adj —
    and flagged instead."""
    from statsmodels.stats.multitest import multipletests

    df = pd.DataFrame(rows)
    for _, g in df.groupby(["rq", "test"]):
        valid = g[g["p_raw"].notna()]
        for ridx in g.index[g["p_raw"].isna()]:
            rows[ridx]["p_adj"] = None
            rows[ridx]["note"] = (rows[ridx].get("note", "") +
                                  " | p undefined (degenerate cell), excluded from FDR family")
        if not len(valid):
            continue
        adj = multipletests(valid["p_raw"], method="fdr_bh")[1]
        for i, (ridx, _) in enumerate(valid.iterrows()):
            rows[ridx]["p_adj"] = float(adj[i])
            rows[ridx]["family_size"] = len(valid)


def results_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (sweep, arm, budget), g in df.groupby(["sweep", "arm", "budget"]):
        g = g.sort_values("question_id")
        for hop, gg in [("all", g)] + list(g.groupby("hop_type")):
            idx = RNG.integers(0, len(gg), size=(N_BOOT, len(gg)))
            em_lo, em_hi = boot_ci(gg["em"].to_numpy().astype(float), idx)
            f1_lo, f1_hi = boot_ci(gg["f1"].to_numpy().astype(float), idx)
            row = dict(
                sweep=sweep, arm=arm, budget=budget, hop_type=hop, n=len(gg),
                em=round(gg["em"].mean(), 4), f1=round(gg["f1"].mean(), 4),
                faithfulness=round(gg["faithfulness"].mean(), 4),
                answer_relevance=round(gg["answer_relevance"].mean(), 4),
                mean_gen_input_tokens=round(gg["gen_input_tokens"].mean(), 1),
                mean_total_tokens=round(float((gg["gen_input_tokens"]
                                               + gg["assembly_input_tokens"]
                                               + gg["assembly_output_tokens"]).mean()), 1),
                mean_cost_usd=round(float((gg["cost_gen_usd"] + gg["cost_assembly_usd"]
                                           + gg.get("judge_cost_usd", 0)).mean()), 6),
                mean_latency_s=round(float((gg["latency_gen_s"]
                                            + gg["latency_assembly_s"]).mean()), 3),
                # bootstrap CIs on EVERY reported accuracy (brief), incl. hops
                em_ci_lo=round(em_lo, 4), em_ci_hi=round(em_hi, 4),
                f1_ci_lo=round(f1_lo, 4), f1_ci_hi=round(f1_hi, 4),
            )
            if hop == "all":
                # APT unit is EM per 1,000 input tokens — stated in the column
                # name so nobody misreads the prereg definition's raw ratio
                row.update(apt_generator_per_1k=round(
                               row["em"] / max(row["mean_gen_input_tokens"], 1) * 1000, 4),
                           apt_total_per_1k=round(
                               row["em"] / max(row["mean_total_tokens"], 1) * 1000, 4))
            out.append(row)
    return pd.DataFrame(out)


def results_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset slice with bootstrap CIs — EXPLORATORY (~75-150 questions per
    dataset is underpowered; significance claims attach only to pooled results,
    per the prereg)."""
    out = []
    for (sweep, dataset, arm, budget), g in df.groupby(["sweep", "dataset", "arm", "budget"]):
        g = g.sort_values("question_id")
        idx = RNG.integers(0, len(g), size=(N_BOOT, len(g)))
        em_lo, em_hi = boot_ci(g["em"].to_numpy().astype(float), idx)
        out.append(dict(sweep=sweep, dataset=dataset, arm=arm, budget=budget, n=len(g),
                        em=round(g["em"].mean(), 4),
                        em_ci_lo=round(em_lo, 4), em_ci_hi=round(em_hi, 4),
                        f1=round(g["f1"].mean(), 4),
                        faithfulness=round(g["faithfulness"].mean(), 4),
                        label="exploratory (underpowered slice)"))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- figures


def fig_pareto(res: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = res[(res.sweep == "primary") & (res.hop_type == "all")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, xcol, title in [
        (axes[0], "mean_gen_input_tokens", "APT_generator accounting (Synopsis metric)"),
        (axes[1], "mean_total_tokens", "APT_total accounting (incl. assembly tokens)"),
    ]:
        pts = []
        for arm in ARM_ORDER:
            g = sub[sub.arm == arm].sort_values("budget")
            if not len(g):
                continue
            ax.plot(g[xcol], g["em"], marker="o", label=arm)
            pts += list(zip(g[xcol], g["em"]))
        # Pareto frontier: not dominated (someone with <= tokens and >= EM);
        # index comparison so coincident points don't exempt each other
        frontier = [p for i, p in enumerate(pts)
                    if not any((q[0] <= p[0] and q[1] > p[1]) or (q[0] < p[0] and q[1] >= p[1])
                               for j, q in enumerate(pts) if j != i)]
        frontier.sort()
        if frontier:
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, "k--", lw=1.5, alpha=0.7, label="Pareto frontier")
        ax.set_xscale("log")
        ax.set_xlabel("mean input tokens per question (log)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Exact match")
    axes[0].legend(fontsize=8)
    fig.suptitle("Answer quality per token — the headline comparison (primary sweep, n=600)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "fig_pareto.png", dpi=300)


def fig_accuracy_by_budget(res: pd.DataFrame):
    import matplotlib.pyplot as plt

    sub = res[(res.sweep == "primary") & (res.hop_type == "all")]
    fig, ax = plt.subplots(figsize=(9, 6))
    for arm in ARM_ORDER:
        g = sub[sub.arm == arm].sort_values("budget")
        if not len(g):
            continue
        yerr = np.clip([g["em"] - g["em_ci_lo"], g["em_ci_hi"] - g["em"]], 0, None)
        ax.errorbar(g["budget"], g["em"], yerr=yerr, marker="o", capsize=4, label=arm)
    ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS)
    ax.set_xlabel("token budget"); ax.set_ylabel("Exact match (95% bootstrap CI)")
    ax.set_title("Accuracy vs token budget — primary sweep (n=600, paired bootstrap)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "fig_accuracy_by_budget.png", dpi=300)


def fig_hop_breakdown(res: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, hop in zip(axes, ["single", "multi"]):
        sub = res[(res.sweep == "primary") & (res.hop_type == hop)]
        for arm in ARM_ORDER:
            g = sub[sub.arm == arm].sort_values("budget")
            if len(g):
                yerr = np.clip([g["em"] - g["em_ci_lo"], g["em_ci_hi"] - g["em"]], 0, None)
                ax.errorbar(g["budget"], g["em"], yerr=yerr, marker="o", capsize=3, label=arm)
        ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS)
        ax.set_title(f"{hop}-hop questions (95% bootstrap CIs)")
        ax.set_xlabel("token budget"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Exact match"); axes[0].legend(fontsize=8)
    fig.suptitle("RQ2 — single-hop vs multi-hop (primary sweep)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "fig_hop_breakdown.png", dpi=300)


def fig_by_dataset(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    prim = df[df.sweep == "primary"]
    datasets = sorted(prim.dataset.unique())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    for ax, ds in zip(axes.flat, datasets):
        sub = prim[prim.dataset == ds]
        for arm in ARM_ORDER:
            g = (sub[sub.arm == arm].groupby("budget")["em"].mean().reset_index()
                 .sort_values("budget"))
            if len(g):
                ax.plot(g["budget"], g["em"], marker="o", label=arm)
        ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS)
        ax.set_title(f"{ds} (n={sub.question_id.nunique()}, descriptive)")
        ax.set_xlabel("token budget"); ax.set_ylabel("Exact match")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(datasets):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Per-dataset breakdown — exploratory (per-dataset slices are underpowered; "
                 "significance claims attach only to pooled results)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "fig_by_dataset.png", dpi=300)


def fig_structured_vs_prose(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    prose = df[(df.sweep == "primary") & (df.content_type == "prose")]
    struct = df[df.sweep == "structured"]
    arms = [a for a in ARM_ORDER if a in set(df.arm)]
    x = np.arange(len(arms)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))

    def mean_ci(series):
        v = series.to_numpy().astype(float)
        if not len(v):
            return np.nan, 0.0
        idx = RNG.integers(0, len(v), size=(2000, len(v)))
        m = v.mean()
        lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
        return m, max(m - lo, hi - m)

    pm_ci = [mean_ci(prose[prose.arm == a]["em"]) for a in arms]
    sm_ci = [mean_ci(struct[struct.arm == a]["em"]) for a in arms]
    pm = [m for m, _ in pm_ci]; sm = [m for m, _ in sm_ci]
    ax.bar(x - w / 2, pm, w, yerr=[e for _, e in pm_ci], capsize=4,
           label="prose (primary sweep)")
    ax.bar(x + w / 2, sm, w, yerr=[e for _, e in sm_ci], capsize=4,
           label="structured (RQ4 sweep)")
    for i, (a, b) in enumerate(zip(pm, sm)):
        if pd.notna(a) and pd.notna(b):  # EM of exactly 0.0 still gets its delta
            ax.annotate(f"{(b - a):+.2f}", (x[i], max(a, b) + 0.01), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(arms, rotation=20, ha="right")
    ax.set_ylabel("Exact match (pooled over budgets)")
    ax.set_title("RQ4 — accuracy on structured/table content vs prose, per arm")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "fig_structured_vs_prose.png", dpi=300)


def fig_latency_cost(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    prim = df[df.sweep == "primary"]
    arms = [a for a in ARM_ORDER if a in set(prim.arm)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    la = [prim[prim.arm == a]["latency_assembly_s"].mean() for a in arms]
    lg = [prim[prim.arm == a]["latency_gen_s"].mean() for a in arms]
    axes[0].bar(arms, la, label="assembly")
    axes[0].bar(arms, lg, bottom=la, label="generation")
    axes[0].set_ylabel("mean latency (s)"); axes[0].set_title("Latency per question")
    axes[0].tick_params(axis="x", rotation=20); axes[0].legend()
    ca = [prim[prim.arm == a]["cost_assembly_usd"].mean() * 1000 for a in arms]
    cg = [prim[prim.arm == a]["cost_gen_usd"].mean() * 1000 for a in arms]
    axes[1].bar(arms, ca, label="assembly")
    axes[1].bar(arms, cg, bottom=ca, label="generation")
    axes[1].set_ylabel("mean cost per question (m$)"); axes[1].set_title("Cost per question")
    axes[1].tick_params(axis="x", rotation=20); axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Latency and cost per arm (primary sweep; assembly vs generation)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "fig_latency_cost.png", dpi=300)


# ---------------------------------------------------------------- tests


def run_tests(df: pd.DataFrame) -> list[dict]:
    prim = df[df.sweep == "primary"]
    rows: list[dict] = []

    def add(rq, comparison, test, budget, d: dict, effect_name, effect, ci=(None, None),
            n=None, primary=False, note="", effect_dz=None):
        if test == "two_proportion_z":
            note = (note + " | " if note else "") + (
                "h CI is paired-bootstrap; p is the pre-registered UNPAIRED z — "
                "they can disagree under within-question correlation; McNemar "
                "carries the paired inference")
        rows.append(dict(rq=rq, comparison=comparison, test=test, budget=budget,
                         statistic=d.get("statistic"), p_raw=d["p"], p_adj=None,
                         effect_name=effect_name, effect=effect,
                         effect_ci_lo=ci[0], effect_ci_hi=ci[1],
                         effect_dz=effect_dz,  # standardized paired effect for 07
                         n=n or d.get("n"), n_discordant_b=d.get("b"),
                         n_discordant_c=d.get("c"), primary_comparison=primary,
                         note=note))

    def discordant_odds(mc: dict) -> float:
        """Haldane-corrected b/c odds: finite for c=0, and 0/0 is not 'infinite'."""
        if mc["b"] == 0 and mc["c"] == 0:
            return float("nan")
        return (mc["b"] + 0.5) / (mc["c"] + 0.5)

    WINNERS_CURSE = ("best arm selected on the same data it is tested on "
                     "(max-selection); nominal p is anti-conservative — "
                     "flagged per review, reported as prereg'd")

    # RQ1: best Synopsis arm vs naive, per budget
    for budget in BUDGETS:
        by_em = (prim[(prim.budget == budget) & (prim.arm.isin(SYNOPSIS_ARMS))]
                 .groupby("arm")["em"].mean())
        if "naive_topk" not in by_em.index or len(by_em) < 2:
            print(f"  [06] RQ1 @ {budget}: arms missing (partial sweep) — skipped")
            continue
        best = by_em.drop("naive_topk").idxmax()
        m = paired_frame(prim, best, "naive_topk", budget)
        tp = two_prop(m)
        add("RQ1", f"{best} vs naive_topk", "two_proportion_z", budget, tp,
            "cohens_h", tp["h"], (tp["h_ci_lo"], tp["h_ci_hi"]),
            primary=(budget == PRIMARY_BUDGET), note=WINNERS_CURSE)
        mc = mcnemar_exact(m)
        add("RQ1", f"{best} vs naive_topk", "mcnemar_exact", budget, mc,
            "discordant_odds_haldane", discordant_odds(mc),
            n=len(m), primary=(budget == PRIMARY_BUDGET), note=WINNERS_CURSE)

    # RQ3: graph vs naive AND graph vs naive_dedup (the confound), per budget
    for other in ["naive_topk", "naive_topk_dedup", "rerank_topk"]:
        for budget in BUDGETS:
            m = paired_frame(prim, "graph_select", other, budget)
            if not len(m):
                continue
            tp = two_prop(m)
            add("RQ3", f"graph_select vs {other}", "two_proportion_z", budget, tp,
                "cohens_h", tp["h"], (tp["h_ci_lo"], tp["h_ci_hi"]),
                primary=(budget == PRIMARY_BUDGET and other == "naive_topk"))
            mc = mcnemar_exact(m)
            add("RQ3", f"graph_select vs {other}", "mcnemar_exact", budget, mc,
                "discordant_odds_haldane", discordant_odds(mc),
                n=len(m), primary=(budget == PRIMARY_BUDGET and other == "naive_topk"))

    # Sensitivity disclosure: graph hops=1 vs hops=2 at the 1000 budget
    sens = df[(df.sweep == "sensitivity_h1") & (df.budget == PRIMARY_BUDGET)
              & (df.arm == "graph_select_h1")]
    if len(sens):
        both = prim[(prim.arm == "graph_select") & (prim.budget == PRIMARY_BUDGET)][
            ["question_id", "em", "f1"]].merge(
            sens[["question_id", "em", "f1"]], on="question_id", suffixes=("_a", "_b"))
        if len(both):
            mc = mcnemar_exact(both)
            add("SENSITIVITY", "graph_select hops=2 vs hops=1", "mcnemar_exact",
                PRIMARY_BUDGET, mc, "discordant_odds_haldane", discordant_odds(mc),
                n=len(both),
                note="pre-registered hyperparameter sensitivity run, disclosed not tested")

    # RQ2: budget slope, single vs multi. UNIT OF ANALYSIS = THE QUESTION
    # (review round 1: per-(question x arm) slopes are 5x pseudo-replicated).
    # Per question: mean F1 across the 5 Synopsis arms at each budget, then the
    # least-squares slope over log2(budget).
    slopes = {}
    for hop, g in prim[prim.arm.isin(SYNOPSIS_ARMS)].groupby("hop_type"):
        per_q = g.pivot_table(index="question_id", columns="budget", values="f1",
                              aggfunc="mean")  # mean over arms -> one row/question
        counts = g.pivot_table(index="question_id", columns="budget", values="f1",
                               aggfunc="count")
        # Amendment 1 says "F1 averaged over the 5 arms": a question missing any
        # arm at any budget would mix arm compositions into its slope — drop it
        # and disclose rather than average unequal sets
        if set(BUDGETS) - set(per_q.columns):
            print(f"  [06] RQ2 {hop}: missing budget columns "
                  f"{sorted(set(BUDGETS) - set(per_q.columns))} — skipped (partial sweep)")
            continue
        complete = counts.eq(len(SYNOPSIS_ARMS)).all(axis=1)
        n_incomplete = int((~complete).sum())
        if n_incomplete:
            print(f"  [06] RQ2 {hop}: {n_incomplete} questions dropped "
                  f"(incomplete arm coverage; disclosed)")
        per_q = per_q[complete].dropna()
        if not len(per_q):
            continue
        x = np.log2(np.array(BUDGETS, dtype=float))
        y = per_q[BUDGETS].to_numpy()
        s = ((y - y.mean(axis=1, keepdims=True)) @ (x - x.mean())) / ((x - x.mean()) ** 2).sum()
        slopes[hop] = s
        t, p = sps.ttest_rel(per_q[4000], per_q[500])
        diffs = (per_q[4000] - per_q[500]).to_numpy()
        idx = RNG.integers(0, len(diffs), size=(2000, len(diffs)))
        ci = (float(np.percentile(diffs[idx].mean(axis=1), 2.5)),
              float(np.percentile(diffs[idx].mean(axis=1), 97.5)))
        sd_diff = float(diffs.std(ddof=1))
        add("RQ2", f"{hop}: mean-arm F1@4000 vs F1@500 (paired by question)",
            "paired_t", "500vs4000",
            dict(statistic=float(t), p=float(p)), "mean_diff",
            float(diffs.mean()), ci=ci, n=len(per_q),
            effect_dz=(float(diffs.mean()) / sd_diff) if sd_diff > 0 else None)
    if "multi" in slopes and "single" in slopes:
        sm, ss = slopes["multi"], slopes["single"]
        t, p = sps.ttest_ind(sm, ss, equal_var=False)
        sp = np.sqrt(((len(sm) - 1) * sm.var(ddof=1) + (len(ss) - 1) * ss.var(ddof=1))
                     / (len(sm) + len(ss) - 2))
        d_eff = (sm.mean() - ss.mean()) / sp if sp > 0 else float("nan")
        boot_d = []
        for _ in range(2000):
            bm = sm[RNG.integers(0, len(sm), len(sm))]
            bs = ss[RNG.integers(0, len(ss), len(ss))]
            bsp = np.sqrt(((len(bm) - 1) * bm.var(ddof=1) + (len(bs) - 1) * bs.var(ddof=1))
                          / (len(bm) + len(bs) - 2))
            boot_d.append((bm.mean() - bs.mean()) / bsp if bsp > 0 else np.nan)
        ci = (float(np.nanpercentile(boot_d, 2.5)), float(np.nanpercentile(boot_d, 97.5)))
        add("RQ2", "per-question F1-per-log2(budget) slope: multi vs single",
            "welch_t", "slope",
            dict(statistic=float(t), p=float(p)), "cohens_d", float(d_eff), ci=ci,
            n=len(sm) + len(ss), primary=True)
    else:
        print(f"  [06] RQ2 slope contrast skipped: hop types present = {sorted(slopes)}")

    # RQ4: structured sweep vs prose primary, per arm, AT THE PRIMARY BUDGET
    # (each question contributes exactly one record: pooling budgets would stack
    # 4 correlated records per question into a test that assumes independence).
    # Unpaired by design (different question sets) -> two-proportion only; the
    # dataset-vs-content-type confound is disclosed in the note.
    struct = df[(df.sweep == "structured") & (df.budget == PRIMARY_BUDGET)]
    prose = prim[(prim.content_type == "prose") & (prim.budget == PRIMARY_BUDGET)]
    for arm in SYNOPSIS_ARMS:
        a = struct[struct.arm == arm]["em"]
        b = prose[prose.arm == arm]["em"]
        if not len(a) or not len(b):
            continue
        from statsmodels.stats.proportion import proportions_ztest

        with np.errstate(invalid="ignore"):
            stat, p = proportions_ztest([int(a.sum()), int(b.sum())], [len(a), len(b)])
        h = cohens_h(a.mean(), b.mean())
        av, bv = a.to_numpy().astype(float), b.to_numpy().astype(float)
        boot_h = [cohens_h(av[RNG.integers(0, len(av), len(av))].mean(),
                           bv[RNG.integers(0, len(bv), len(bv))].mean())
                  for _ in range(2000)]
        ci = (float(np.percentile(boot_h, 2.5)), float(np.percentile(boot_h, 97.5)))
        add("RQ4", f"{arm}: structured vs prose @ {PRIMARY_BUDGET}", "two_proportion_z",
            PRIMARY_BUDGET,
            dict(statistic=float(stat) if np.isfinite(stat) else None,
                 p=float(p) if np.isfinite(p) else float("nan")),
            "cohens_h", float(h), ci=ci,
            n=len(a) + len(b), primary=(arm == "graph_select"),
            note="unpaired (different question sets); content type confounded with "
                 "dataset (WTQ vs prose benchmarks) — disclosed; McNemar not applicable")

    bh_adjust(rows)
    return rows


def confound_checks(df: pd.DataFrame) -> pd.DataFrame:
    prim = df[df.sweep == "primary"]
    out = []
    # budget parity: realized context tokens per arm at each budget
    for (arm, budget), g in prim.groupby(["arm", "budget"]):
        out.append(dict(check="budget_parity", arm=arm, budget=budget,
                        value=round(g["gen_context_tokens"].mean(), 1),
                        detail="mean realized context tokens"))
    # retrieval-success stratification
    for (arm, budget), g in prim.groupby(["arm", "budget"]):
        for flag, gg in g.groupby("retrieval_gold_in_pool"):
            out.append(dict(check="retrieval_stratification", arm=arm, budget=budget,
                            value=round(gg["em"].mean(), 4),
                            detail=f"EM | gold_in_pool={flag} (n={len(gg)})"))
    return pd.DataFrame(out)


def main():
    ev_path = DATA / "eval_records.parquet"
    if not ev_path.exists():
        raise SystemExit("[06] data/eval_records.parquet missing — run src/05_run.py first")
    df = pd.read_parquet(ev_path)
    # sensitivity_h1 stays in: its rows appear in preliminary_results and the
    # SENSITIVITY test row — the pre-registered disclosure must surface
    df = df[df.sweep.isin(["primary", "structured", "sensitivity_h1"])]
    if "failed" in df.columns:
        n_failed = int(df["failed"].sum())
        if n_failed:
            print(f"[06] excluding {n_failed} failed records "
                  f"(prereg exclusion rule 2, pairwise; disclosed)")
        df = df[~df["failed"].astype(bool)]
    print(f"[06] {len(df)} records")

    # scale sanity: final-looking outputs from smoke-scale data must shout
    n_full = len(pd.read_parquet(DATA / "questions_primary_clean.parquet"))
    n_prim = df[df.sweep == "primary"].question_id.nunique()
    if 0 < n_prim < n_full:
        print(f"[06] *** WARNING: primary sweep covers {n_prim}/{n_full} questions — "
              f"these are SMOKE-SCALE outputs, not the committed design ***")
        update_manifest(stage06_scale_warning=f"primary sweep n={n_prim} of {n_full}")

    res = results_table(df)
    base = DATA / "baseline_records.parquet"
    if base.exists():
        bdf = pd.read_parquet(base)
        if "failed" in bdf.columns:
            bdf = bdf[~bdf["failed"].astype(bool)]
        res = pd.concat([res, results_table(bdf)], ignore_index=True)
    res.to_csv(OUTPUTS / "preliminary_results.csv", index=False)
    results_by_dataset(df).to_csv(OUTPUTS / "preliminary_results_by_dataset.csv", index=False)

    fig_pareto(res)
    fig_accuracy_by_budget(res)
    fig_hop_breakdown(res)
    fig_by_dataset(df)
    fig_structured_vs_prose(df)
    fig_latency_cost(df)

    tests = run_tests(df)
    tdf = pd.DataFrame(tests)
    tdf.to_csv(OUTPUTS / "stats_tests.csv", index=False)

    cc = confound_checks(df)
    cc.to_csv(OUTPUTS / "graph_confound_checks.csv", index=False)

    n_sig = int((tdf["p_adj"] < 0.05).sum())
    summary = (f"results rows: {len(res)}; tests: {len(tdf)} "
               f"({n_sig} significant after BH-FDR); figures written (300 dpi)")
    print(summary)
    prim_pool = res[(res.sweep == "primary") & (res.hop_type == "all")]
    print(prim_pool.pivot_table(index="arm", columns="budget", values="em").round(3).to_string())
    update_manifest(stage06=dict(n_tests=len(tdf), n_significant_fdr=n_sig))
    append_log("Stage 06 analysis", f"```\n{summary}\n```")


if __name__ == "__main__":
    main()
