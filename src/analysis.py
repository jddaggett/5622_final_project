"""Follow-up analyses, read from the outputs of `python -m src.run` (no retraining).

- ranking vs. threshold by age group; fairness ablation (socioeconomic features dropped).
- k-means clusters of training patients, missed-stroke concentration, per-cluster cutoffs.
- recall vs. referral capacity; reweighting to younger age mixes.

All cutoffs come from training out-of-fold scores; the test set is only evaluated.
Usage: python -m src.analysis [--results DIR] [--n-boot N]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score

from .config import BETA, FEATURES, HEADLINE_MODEL, N_BOOT, RANDOM_STATE, RESULTS_DIR, SLICES, TARGET
from .models import FeatureEncoder
from .train import f2_optimal_threshold

FLEXIBLE = "grad_boost"
WORST_SLICE = "18-44"
# pre-filter the slice list to the two groups used repeatedly
AGE_GROUPS = [(name, fn) for group, name, fn in SLICES if group == "age"]
INCOME_TIERS = [(name, fn) for group, name, fn in SLICES if group == "income"]
K_RANGE = range(2, 9)
SILHOUETTE_SAMPLE = 20_000  # silhouette is O(n^2), so scored on a training subsample
BUDGETS = np.round(np.arange(0.05, 0.501, 0.05), 2)
YOUNG_SHARES = np.round(np.arange(0.0, 1.001, 0.05), 2)
BATCH = 250  # bootstrap replicates per batch, to bound memory


# ---------------------------------------------------------------- helpers

def _ci(x) -> tuple[float, float]:
    """2.5th and 97.5th percentiles of an array of bootstrap values"""
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def _f2(tp, fp, fn) -> np.ndarray:
    """F2 from raw counts; works on scalars or arrays of bootstrap replicates"""
    tp, fp, fn = (np.asarray(a, dtype=float) for a in (tp, fp, fn))
    b2 = BETA**2
    denom = (1 + b2) * tp + b2 * fn + fp
    return np.divide((1 + b2) * tp, denom, out=np.zeros_like(denom), where=denom > 0)


def _recall(tp, fn) -> np.ndarray:
    tp, fn = np.asarray(tp, dtype=float), np.asarray(fn, dtype=float)
    return np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)


def _counts(y: np.ndarray, flag: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """tp, fp, fn inside `mask`; works on 1-D arrays or 2-D bootstrap batches.

    When called with 2-D arrays (shape batch × n_patients), axis=-1 sums
    across patients so each row of the result is one replicate's counts.
    """
    tp = (y & flag & mask).sum(axis=-1)
    fp = (~y & flag & mask).sum(axis=-1)
    fn = (y & ~flag & mask).sum(axis=-1)
    # stack into a single array so callers can unpack with *counts or index by column
    return np.stack([tp, fp, fn], axis=-1)


def boot_counts(
    y: np.ndarray, flags: dict[str, np.ndarray], masks: dict[str, np.ndarray], n_boot: int
) -> dict[tuple[str, str], np.ndarray]:
    """Bootstrap (tp, fp, fn) for every (flag, mask) pair on shared resamples.

    shared resamples make differences paired: correlated errors cancel and CIs are narrower.
    strokes and non-strokes are resampled separately so every replicate keeps the full set's
    stroke count (same reason as slice_ci.py).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    out = {(f, m): [] for f in flags for m in masks}
    for start in range(0, n_boot, BATCH):
        b = min(BATCH, n_boot - start)
        # idx shape: (b, n_patients) — each row is one replicate's row indices
        idx = np.hstack([rng.choice(pos, (b, len(pos))), rng.choice(neg, (b, len(neg)))])
        yb = y[idx]
        for m, mask in masks.items():
            mb = mask[idx]  # (b, n_patients) boolean mask for this slice
            for f, flag in flags.items():
                out[(f, m)].append(_counts(yb, flag[idx], mb))
    # vstack the per-batch lists into a single (n_boot, 3) array per (flag, mask) pair
    return {k: np.vstack(v) for k, v in out.items()}


def _flags(df: pd.DataFrame, model: str, thr: dict[str, float]) -> np.ndarray:
    """hard referral decisions for one model at its threshold"""
    return df[f"{model}_proba"].to_numpy() >= thr[model]


# ---------------------------------------------------------------- ranking vs. threshold, fairness

def ranking_vs_threshold(test: pd.DataFrame, thr: dict[str, float], n_boot: int) -> pd.DataFrame:
    """Per age group: recall at the global cutoff vs. when referring the same share of every group.

    If ranking quality is similar across groups, recall at a matched share is similar too,
    and the young-adult failure comes from the single cutoff, not from the ranking.
    """
    y = test[TARGET].to_numpy().astype(bool)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for model in [HEADLINE_MODEL, FLEXIBLE]:
        s = test[f"{model}_proba"].to_numpy()
        # the model's overall referral rate — used as the matched budget for every age group
        share = float((s >= thr[model]).mean())
        for name, select in AGE_GROUPS:
            mask = select(test).to_numpy()
            ys, ss = y[mask], s[mask]
            pos, neg = np.flatnonzero(ys), np.flatnonzero(~ys)
            aucs, recalls = [], []
            for _ in range(n_boot):
                i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
                yb, sb = ys[i], ss[i]
                aucs.append(roc_auc_score(yb, sb))
                # np.quantile(sb, 1 - share) is the score cutoff that refers exactly `share`
                # of this group — refer the top `share` fraction by score
                recalls.append((yb & (sb >= np.quantile(sb, 1 - share))).sum() / yb.sum())
            row = {
                "model": model, "slice": name, "n": int(mask.sum()), "n_pos": int(ys.sum()),
                "matched_share": share,
                "referred_at_global_cutoff": float((ss >= thr[model]).mean()),
                "recall_at_global_cutoff": float((ys & (ss >= thr[model])).sum() / ys.sum()),
                "recall_at_matched_share": float((ys & (ss >= np.quantile(ss, 1 - share))).sum()
                                                 / ys.sum()),
                "roc_auc": float(roc_auc_score(ys, ss)),
            }
            row["recall_at_matched_share_lo"], row["recall_at_matched_share_hi"] = _ci(recalls)
            row["roc_auc_lo"], row["roc_auc_hi"] = _ci(aucs)
            rows.append(row)
    return pd.DataFrame(rows)


def fairness(test: pd.DataFrame, thr: dict[str, float], n_boot: int) -> pd.DataFrame:
    """Main model vs. the same model without Income, Education, NoDocbcCost (paired CIs)."""
    y = test[TARGET].to_numpy().astype(bool)
    # pairs: (main model name, its -ses counterpart)
    pairs = [(m, f"{m}-ses") for m in [HEADLINE_MODEL, FLEXIBLE]]
    # pre-compute flags for all four models at once so boot_counts can share resamples
    flags = {m: _flags(test, m, thr) for pair in pairs for m in pair}
    masks = {"all": np.ones(len(test), bool), **{n: f(test).to_numpy() for n, f in INCOME_TIERS}}
    boot = boot_counts(y, flags, masks, n_boot)
    rows = []
    for main, ses in pairs:
        for m, mask in masks.items():
            # point estimates from the actual test set
            pm, ps = _counts(y, flags[main], mask), _counts(y, flags[ses], mask)
            # bootstrap arrays: shape (n_boot, 3) — columns are tp, fp, fn
            bm, bs = boot[(main, m)], boot[(ses, m)]
            row = {
                "model": main, "slice": m, "n": int(mask.sum()), "n_pos": int((y & mask).sum()),
                "referred_main": float(flags[main][mask].mean()),
                "referred_ses": float(flags[ses][mask].mean()),
                "recall_main": float(_recall(pm[0], pm[2])),
                "recall_ses": float(_recall(ps[0], ps[2])),
                "f2_main": float(_f2(*pm)), "f2_ses": float(_f2(*ps)),
            }
            # paired differences: ses minus main on the same resamples
            d_rec = _recall(bs[:, 0], bs[:, 2]) - _recall(bm[:, 0], bm[:, 2])
            d_f2 = _f2(bs[:, 0], bs[:, 1], bs[:, 2]) - _f2(bm[:, 0], bm[:, 1], bm[:, 2])
            row["diff_recall_lo"], row["diff_recall_hi"] = _ci(d_rec)
            row["diff_f2_lo"], row["diff_f2_hi"] = _ci(d_f2)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- unsupervised clustering

def clusters(
    oof: pd.DataFrame, test: pd.DataFrame, thr: dict[str, float], n_boot: int
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    """k-means on training patients; misses per cluster; per-cluster vs. global cutoffs."""
    # use the same encoding as the models: numeric columns scaled to SD 1, binaries left 0/1
    enc = FeatureEncoder(FEATURES).fit(oof[FEATURES])
    Xtr, Xte = enc.transform(oof[FEATURES]), enc.transform(test[FEATURES])

    # try k = 2..8, pick by silhouette (scored on a subsample because silhouette is O(n^2))
    silhouettes, fits = {}, {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xtr)
        silhouettes[k] = float(silhouette_score(
            Xtr, km.labels_, sample_size=SILHOUETTE_SAMPLE, random_state=RANDOM_STATE))
        fits[k] = km
    k = max(silhouettes, key=silhouettes.get)
    km = fits[k]
    c_tr, c_te = km.labels_, km.predict(Xte)

    # describe each cluster by which features deviate most from the training average
    mu, sd = oof[FEATURES].mean(), oof[FEATURES].std()
    y_tr = oof[TARGET].to_numpy().astype(bool)
    y_te = test[TARGET].to_numpy().astype(bool)
    model = HEADLINE_MODEL
    s_tr, s_te = oof[f"{model}_proba"].to_numpy(), test[f"{model}_proba"].to_numpy()
    global_flags = s_te >= thr[model]

    # per-cluster F2-optimal cutoffs from training out-of-fold scores only
    cutoffs = {}
    for c in range(k):
        m = c_tr == c
        cutoffs[c] = (f2_optimal_threshold(y_tr[m], s_tr[m]) if y_tr[m].any() else np.inf)
    # apply each test patient's cluster's cutoff
    cluster_flags = s_te >= np.array([cutoffs[c] for c in c_te])

    # masks for the keep-rule comparison: per cluster, plus "all" and the worst slice
    masks = {f"cluster {c}": c_te == c for c in range(k)}
    young = next(f for n, f in AGE_GROUPS if n == WORST_SLICE)(test).to_numpy()
    masks |= {"all": np.ones(len(test), bool), WORST_SLICE: young}
    boot = boot_counts(y_te, {"global": global_flags, "per_cluster": cluster_flags}, masks, n_boot)

    misses = (~global_flags & y_te).sum()
    profile_rows = []
    for c in range(k):
        m_tr, m_te = c_tr == c, c_te == c
        # z-scores for this cluster's mean vs. the overall training mean, sorted by magnitude
        z = ((oof.loc[m_tr, FEATURES].mean() - mu) / sd).sort_values(key=abs, ascending=False)
        g = _counts(y_te, global_flags, m_te)
        pc = _counts(y_te, cluster_flags, m_te)
        bg = boot[("global", f"cluster {c}")]
        row = {
            "cluster": c,
            "defining_traits": ", ".join(f"{f} {v:+.1f} SD" for f, v in z.head(4).items()),
            "train_n": int(m_tr.sum()), "train_stroke_rate": float(y_tr[m_tr].mean()),
            "mean_age_band": float(oof.loc[m_tr, "Age"].mean()),
            "test_n": int(m_te.sum()), "test_strokes": int(y_te[m_te].sum()),
            "share_of_strokes": float(y_te[m_te].sum() / y_te.sum()),
            "share_of_missed": float((~global_flags & y_te & m_te).sum() / misses),
            "recall_global": float(_recall(g[0], g[2])),
            "cutoff_global": float(thr[model]), "cutoff_cluster": float(cutoffs[c]),
            "referred_global": float(global_flags[m_te].mean()),
            "referred_cluster": float(cluster_flags[m_te].mean()),
            "recall_cluster": float(_recall(pc[0], pc[2])),
        }
        row["recall_global_lo"], row["recall_global_hi"] = _ci(_recall(bg[:, 0], bg[:, 2]))
        profile_rows.append(row)
    profiles = pd.DataFrame(profile_rows)

    # keep-rule comparison: pre-registered in project notes 2.8 before running
    summary_rows = []
    for m in ["all", WORST_SLICE]:
        g, pc = _counts(y_te, global_flags, masks[m]), _counts(y_te, cluster_flags, masks[m])
        bg, bp = boot[("global", m)], boot[("per_cluster", m)]
        # per-replicate differences: cluster minus global for each metric
        d_f2 = _f2(bp[:, 0], bp[:, 1], bp[:, 2]) - _f2(bg[:, 0], bg[:, 1], bg[:, 2])
        d_rec = _recall(bp[:, 0], bp[:, 2]) - _recall(bg[:, 0], bg[:, 2])
        summary_rows.append({
            "slice": m, "n": int(masks[m].sum()), "n_pos": int(y_te[masks[m]].sum()),
            "referred_global": float(global_flags[masks[m]].mean()),
            "referred_cluster": float(cluster_flags[masks[m]].mean()),
            "f2_global": float(_f2(*g)), "f2_cluster": float(_f2(*pc)),
            "recall_global": float(_recall(g[0], g[2])), "recall_cluster": float(_recall(pc[0], pc[2])),
            "diff_f2_lo": _ci(d_f2)[0], "diff_f2_hi": _ci(d_f2)[1],
            "diff_recall_lo": _ci(d_rec)[0], "diff_recall_hi": _ci(d_rec)[1],
        })
    comparison = pd.DataFrame(summary_rows).set_index("slice")
    a, w = comparison.loc["all"], comparison.loc[WORST_SLICE]
    # rule a: overall F2 strictly improves (CI entirely above 0)
    rule_a = a["diff_f2_lo"] > 0
    # rule b: worst-slice recall improves AND overall F2 is not significantly worse (CI includes 0)
    rule_b = w["recall_cluster"] > w["recall_global"] and a["diff_f2_lo"] <= 0 <= a["diff_f2_hi"]
    verdict = {
        "k": k, "silhouettes": silhouettes, "model": model,
        "rule_a_overall_f2_up": bool(rule_a), "rule_b_worst_recall_up_f2_not_worse": bool(rule_b),
        "keep_per_cluster_cutoffs": bool(rule_a or rule_b),
    }
    return verdict, profiles, comparison.reset_index(), {"train": c_tr, "test": c_te}


# ---------------------------------------------------------------- capacity, age mix

def _recall_at_budgets(y: np.ndarray, s: np.ndarray, budgets: np.ndarray, tiebreak: np.ndarray):
    """recall when referring the top `budget` share by score.

    ties broken by `tiebreak` (a fixed random array) so patients with equal scores
    are ordered consistently across calls.
    """
    # sort patients by descending score, then by tiebreak for ties
    order = np.lexsort((tiebreak, -s))
    # cumulative true positives as we walk down the ranked list
    cum_tp = np.cumsum(y[order])
    # how many to refer at each budget
    k = np.maximum(np.ceil(budgets * len(y)).astype(int), 1)
    return cum_tp[k - 1] / y.sum()


def capacity(test: pd.DataFrame, thr: dict[str, float], n_boot: int):
    """Recall vs. share of patients referred, with paired CIs for ensemble minus logistic regression."""
    y = test[TARGET].to_numpy().astype(bool)
    models = [HEADLINE_MODEL, FLEXIBLE, "random_forest"]
    rng = np.random.default_rng(RANDOM_STATE)
    # one fixed random order for tie-breaking; reused across models so comparisons are fair
    tiebreak = rng.random(len(y))
    scores = {m: test[f"{m}_proba"].to_numpy() for m in models}
    # point estimates at each budget
    point = {m: _recall_at_budgets(y, scores[m], BUDGETS, tiebreak) for m in models}

    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    boot = {m: [] for m in models}
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        tb = rng.random(len(i))  # fresh tiebreak for this replicate
        for m in models:
            boot[m].append(_recall_at_budgets(y[i], scores[m][i], BUDGETS, tb))
    boot = {m: np.array(v) for m, v in boot.items()}  # (n_boot, len(BUDGETS))

    rows = []
    for j, b in enumerate(BUDGETS):
        row = {"budget": float(b), **{f"recall_{m}": float(point[m][j]) for m in models}}
        # paired diff at each budget: index j across all replicates
        for m in models[1:]:
            row[f"diff_{m}_lo"], row[f"diff_{m}_hi"] = _ci(boot[m][:, j] - boot[HEADLINE_MODEL][:, j])
        rows.append(row)
    table = pd.DataFrame(rows)

    # smooth curves for the figure (200 points vs. 10 for the CI table)
    grid = np.linspace(0.005, 1.0, 200)
    curves = {m: _recall_at_budgets(y, scores[m], grid, tiebreak) for m in models}
    # each model's actual operating point: (share referred, recall) at its F2 cutoff
    operating = {m: (float((scores[m] >= thr[m]).mean()),
                     float((y & (scores[m] >= thr[m])).sum() / y.sum())) for m in models}
    # age-rule operating points for every possible cutoff band (for the figure's dashed line)
    age_points = [(float((test["Age"] >= k).mean()), float((y & (test["Age"] >= k)).sum() / y.sum()), k)
                  for k in range(1, 14)]
    age_op = round(thr["age_rule"] * 13)  # the band chosen for the age rule
    return table, grid, curves, operating, age_points, age_op


def age_mix(test: pd.DataFrame, thr: dict[str, float], n_boot: int):
    """F2 of GB minus logistic regression after reweighting to a given share of 18-44s.

    thresholds are held fixed; other age groups keep their relative sizes.
    reweighting scales the counts from each group, not the actual patients,
    so this is purely arithmetic — no refitting.
    """
    y = test[TARGET].to_numpy().astype(bool)
    young = next(f for n, f in AGE_GROUPS if n == WORST_SLICE)(test).to_numpy()
    p0 = float(young.mean())  # current young share in the test set (~21.5%)
    models = [HEADLINE_MODEL, FLEXIBLE]
    flags = {m: _flags(test, m, thr) for m in models}
    masks = {"young": young, "older": ~young}
    # get bootstrap counts for young and older separately; share the resamples across models
    boot = boot_counts(y, flags, masks, n_boot)
    # point-estimate counts for young and older for each model
    point = {(m, g): _counts(y, flags[m], masks[g]) for m in models for g in masks}

    rows = []
    for s in YOUNG_SHARES:
        # scale factors so young patients make up share s and older make up (1-s)
        # dividing by p0 and (1-p0) normalises before applying the new shares
        w_y, w_o = s / p0, (1 - s) / (1 - p0)
        # reweighted point-estimate counts: blend young and older with the new weights
        f2 = {m: _f2(*(w_y * point[(m, "young")] + w_o * point[(m, "older")])) for m in models}
        # same reweighting on bootstrap arrays for CIs
        fb = {m: (w_y * boot[(m, "young")] + w_o * boot[(m, "older")]) for m in models}
        # .T transposes (n_boot, 3) to (3, n_boot) so *fb unpacks to tp, fp, fn arrays
        d = _f2(*fb[FLEXIBLE].T) - _f2(*fb[HEADLINE_MODEL].T)
        lo, hi = _ci(d)
        rows.append({"young_share": float(s), f"f2_{HEADLINE_MODEL}": float(f2[HEADLINE_MODEL]),
                     f"f2_{FLEXIBLE}": float(f2[FLEXIBLE]), "diff_lo": lo, "diff_hi": hi})
    table = pd.DataFrame(rows)
    # first young_share where GB is significantly better (CI lower bound > 0)
    sig = table[table["diff_lo"] > 0]
    flip = float(sig["young_share"].min()) if len(sig) else None
    return table, p0, flip


# ---------------------------------------------------------------- figures

def plot_capacity(path: Path, grid, curves, operating, age_points, age_op) -> None:
    labels = {HEADLINE_MODEL: "Logistic regression (tuned)", FLEXIBLE: "Gradient boosting",
              "random_forest": "Random forest"}
    # drawn at IEEE column width (3.5 in) so fonts print at their real size
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for m, r in curves.items():
        line, = ax.plot(grid, r, label=labels[m], lw=1.2)
        x, yv = operating[m]
        ax.plot(x, yv, "o", color=line.get_color(), ms=4)  # filled circle = F2 operating point
    ax.plot([p[0] for p in age_points], [p[1] for p in age_points], "s--", color="gray",
            ms=3, lw=0.8, label="Age rule (refer band ≥ k)")
    op = next(p for p in age_points if p[2] == age_op)
    ax.plot(op[0], op[1], "s", color="black", ms=4)  # black square = tuned age rule's cutoff
    ax.plot([0, 1], [0, 1], ":", color="lightgray", label="Random referral")
    ax.set_xlabel("Share of patients referred", fontsize=7.5)
    ax.set_ylabel("Recall (share of strokes caught)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_slices(path: Path, results: Path) -> None:
    ci = pd.read_csv(results / "slice_ci.csv")
    sl = pd.read_csv(results / "slices.csv")
    models = {HEADLINE_MODEL: "Logistic regression (tuned)", FLEXIBLE: "Gradient boosting"}
    names = ["all"] + [n for _, n, _ in SLICES]
    # drawn at IEEE column width (3.5 in)
    fig, ax = plt.subplots(figsize=(3.5, 4.4))
    metrics = pd.read_csv(results / "metrics.csv")
    for j, (m, label) in enumerate(models.items()):
        c = ci[ci["model"] == m].set_index("slice").loc[names]
        recall = sl[sl["model"] == m].set_index("slice")["recall"].to_dict()
        # the "all" slice isn't in slices.csv; read it from the overall metrics instead
        recall["all"] = metrics.query("model == @m and split == 'test'")["recall"].iloc[0]
        point = np.array([recall[n] for n in names], dtype=float)
        ypos = np.arange(len(names)) + (j - 0.5) * 0.3  # vertical offset separates the two models
        err = [point - c["recall_lo"].to_numpy(), c["recall_hi"].to_numpy() - point]
        ax.errorbar(point, ypos, xerr=err, fmt="o", ms=3, capsize=1.5, elinewidth=1, label=label)
    c0 = ci[ci["model"] == HEADLINE_MODEL].set_index("slice").loc[names]
    ax.set_yticks(np.arange(len(names)))
    labels = [f"{'All' if n == 'all' else n}  ({c0.loc[n, 'n']:,} / {c0.loc[n, 'n_pos']:,})" for n in names]
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Recall (95% CI). Labels: patients / strokes", fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False,
              handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    out = args.results / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(args.results / "test_predictions.csv")
    oof = pd.read_csv(args.results / "oof_predictions.csv")
    info = json.loads((args.results / "best_params.json").read_text())
    thr = {m: v["threshold"] for m, v in info.items()}
    n = args.n_boot
    pd.set_option("display.width", 200)
    summary: dict = {}

    print("R5: ranking vs. threshold by age group")
    rvt = ranking_vs_threshold(test, thr, n)
    rvt.to_csv(out / "ranking_vs_threshold.csv", index=False)
    print(rvt.round(3).to_string(index=False))

    print("\nR5: fairness ablation (drop Income, Education, NoDocbcCost)")
    fair = fairness(test, thr, n)
    fair.to_csv(out / "fairness_ablation.csv", index=False)
    print(fair.round(3).to_string(index=False))

    print("\nR4: k-means clusters")
    verdict, profiles, comparison, _ = clusters(oof, test, thr, n)
    profiles.to_csv(out / "cluster_profiles.csv", index=False)
    comparison.to_csv(out / "cluster_cutoffs.csv", index=False)
    print("silhouette by k:", {k: round(v, 3) for k, v in verdict["silhouettes"].items()},
          "-> k =", verdict["k"])
    print(profiles.round(3).to_string(index=False))
    print(comparison.round(3).to_string(index=False))
    print("keep rule:", {k: v for k, v in verdict.items() if k.startswith(("rule", "keep"))})
    summary["clusters"] = verdict

    print("\nR6: capacity")
    cap, grid, curves, operating, age_points, age_op = capacity(test, thr, n)
    cap.to_csv(out / "capacity.csv", index=False)
    print(cap.round(3).to_string(index=False))
    plot_capacity(out / "fig_capacity.png", grid, curves, operating, age_points, age_op)
    summary["operating_points"] = operating

    print("\nR6: age mix")
    mix, p0, flip = age_mix(test, thr, n)
    mix.to_csv(out / "age_mix.csv", index=False)
    print(mix.round(3).to_string(index=False))
    print(f"current 18-44 share {p0:.3f}; GB significantly better from share: {flip}")
    summary["age_mix"] = {"current_young_share": p0, "flip_share": flip}

    plot_slices(out / "fig_slices.png", args.results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nAnalysis written to {out}")
    return summary


if __name__ == "__main__":
    main()