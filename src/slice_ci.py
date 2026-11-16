"""Bootstrap 95% CIs for overall and per-slice F2 and recall, plus paired differences vs. the baseline.

Reads the outputs of `python -m src.run`, so no retraining is needed.
Usage: python -m src.slice_ci [--results DIR] [--n-boot N]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BASELINE_MODEL, BETA, CI_MODELS, N_BOOT, RANDOM_STATE, RESULTS_DIR, SLICES, TARGET

# batch process bootstrap replicas as to not have 100M+ element arrays
BATCH = 250


def _scores(y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """F2 and recall for every row in a 2-D batch of bootstrap replicates.

    y and pred are 2-D: shape (batch_size, n_patients). Each row is one replicate.
    Summing axis=1 gives per-replicate counts, then formulas give per-replicate metrics.
    """
    tp = (y & pred).sum(axis=1)
    fp = (~y & pred).sum(axis=1)
    fn = (y & ~pred).sum(axis=1)
    b2 = BETA**2
    denom = (1 + b2) * tp + b2 * fn + fp
    # np.divide with where= writes 0 where the denominator is 0 (no positives in a replicate)
    f2 = np.divide((1 + b2) * tp, denom, out=np.zeros(len(tp)), where=denom > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(len(tp)), where=(tp + fn) > 0)
    return f2, recall


def _ci(x: np.ndarray) -> tuple[float, float]:
    """2.5th and 97.5th percentiles of a 1-D array of bootstrap values."""
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def slice_cis(
    preds: pd.DataFrame, thresholds: dict[str, float], models: list[str], n_boot: int
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    # prepend a synthetic "all patients" slice so the overall CI uses the same code path
    everyone = ("all", "all", lambda d: pd.Series(True, index=d.index))
    for group, name, select in [everyone, *SLICES]:

        # filter the test set to this slice
        sub = preds[select(preds).to_numpy()]
        y = sub[TARGET].to_numpy().astype(bool)

        # indices of strokes and non-strokes within the slice
        pos, neg = np.flatnonzero(y), np.flatnonzero(~y)

        # hard referral decisions for every model at their fixed thresholds.
        # computed once here so the loop doesn't repeat the >= comparison per batch
        flags = {m: sub[f"{m}_proba"].to_numpy() >= thresholds[m] for m in models}

        # accumulate per-replicate F2 and recall across batches for each model
        f2 = {m: [] for m in models}
        rec = {m: [] for m in models}

        for start in range(0, n_boot, BATCH):
            b = min(BATCH, n_boot - start)

            # build the resampled index matrix
            # each row is one bootstrap replicate
            # non-stroke indices drawn with replacement from their respective pools
            idx = np.hstack([
                rng.choice(pos, (b, len(pos))),
                rng.choice(neg, (b, len(neg))),
            ])
            # idx[i] is the row indices for the i-th replicate
            yb = y[idx]

            for m in models:
                # _scores returns two 1-D arrays of length b: one F2 and one recall per replicate
                f, r = _scores(yb, flags[m][idx])
                f2[m].append(f)
                rec[m].append(r)

        # concatenate the per-batch lists into a single 1-D array of n_boot values
        f2 = {m: np.concatenate(v) for m, v in f2.items()}
        rec = {m: np.concatenate(v) for m, v in rec.items()}

        # write one row per model
        for m in models:
            row = {"group": group, "slice": name, "model": m, "n": len(y), "n_pos": len(pos)}
            row["f2_lo"], row["f2_hi"] = _ci(f2[m])
            row["recall_lo"], row["recall_hi"] = _ci(rec[m])
            if m != BASELINE_MODEL:
                row["diff_f2_lo"], row["diff_f2_hi"] = _ci(f2[m] - f2[BASELINE_MODEL])
            rows.append(row)

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)

    preds = pd.read_csv(args.results / "test_predictions.csv")
    info = json.loads((args.results / "best_params.json").read_text())
    # only compute CIs for models that were actually run
    models = [m for m in CI_MODELS if m in info]
    thresholds = {m: info[m]["threshold"] for m in models}

    out = slice_cis(preds, thresholds, models, args.n_boot)
    out.to_csv(args.results / "slice_ci.csv", index=False)
    print(f"Bootstrap CIs ({args.n_boot} resamples) written to {args.results / 'slice_ci.csv'}")
    return out


if __name__ == "__main__":
    main()