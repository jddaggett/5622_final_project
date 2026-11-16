"""End-to-end entry point: load, split, train, evaluate, bootstrap CIs, write results.

Usage: python -m src.run [--data PATH] [--out DIR] [--sample FRAC]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import slice_ci
from .config import AGE_BANDS, BASELINE_MODEL, DATA_PATH, HEADLINE_MODEL, RESULTS_DIR, TARGET
from .data import load_data, split, subsample
from .evaluate import classification_metrics, odds_ratios, slice_metrics, worst_slice
from .models import build_models
from .train import fit_model


def _ci(ci: pd.DataFrame, model: str, slice_name: str, metric: str) -> str:
    row = ci[(ci["model"] == model) & (ci["slice"] == slice_name)].iloc[0]
    return f"[{row[f'{metric}_lo']:.3f}, {row[f'{metric}_hi']:.3f}]"


def print_headline(metrics: pd.DataFrame, slices: pd.DataFrame, ci: pd.DataFrame) -> None:
    """The headline number with the baseline, references, and worst slice beside it."""
    test = metrics[metrics["split"] == "test"].set_index("model")
    print("\n" + "=" * 72)
    print("HEADLINE (test set)")
    for model, label in [
        (HEADLINE_MODEL, "headline"),
        (BASELINE_MODEL, "declared baseline"),
    ]:
        r = test.loc[model]
        print(f"  {label:<18} {model:<14} F2 = {r['f2']:.3f} {_ci(ci, model, 'all', 'f2')}"
              f"  recall = {r['recall']:.3f}  flagged = {(r['tp'] + r['fp']) / r['n']:.1%}")
    for model in ["flag_all", "age_rule", "dummy"]:
        r = test.loc[model]
        print(f"  {'reference':<18} {model:<14} F2 = {r['f2']:.3f}"
              f"  recall = {r['recall']:.3f}  flagged = {(r['tp'] + r['fp']) / r['n']:.1%}")
    w = worst_slice(slices[slices["model"] == HEADLINE_MODEL])
    print(f"  worst slice        {w['slice']:<14} F2 = {w['f2']:.3f} "
          f"{_ci(ci, HEADLINE_MODEL, w['slice'], 'f2')}  recall = {w['recall']:.3f} "
          f"{_ci(ci, HEADLINE_MODEL, w['slice'], 'recall')}  n = {w['n']}, strokes = {w['n_pos']}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate stroke referral models.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="path to the BRFSS CSV")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="output directory")
    parser.add_argument("--sample", type=float, default=None,
                        help="use a stratified fraction of rows, for quick test runs")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)

    # for testing, small frac of rows
    if args.sample:
        df = subsample(df, args.sample)

    X_train, X_test, y_train, y_test = split(df)
    print(
        f"train: {len(X_train)} rows, {int(y_train.sum())} strokes | "
        f"test: {len(X_test)} rows, {int(y_test.sum())} strokes"
    )

    metric_rows: list[dict] = []
    slice_frames: list[pd.DataFrame] = []
    info: dict[str, dict] = {}
    predictions = X_test.assign(**{TARGET: y_test})
    # Out-of-fold training scores, so analysis (e.g. per-cluster cutoffs) can pick
    # thresholds without touching the test set.
    oof_predictions = X_train.assign(**{TARGET: y_train})

    for spec in build_models():
        print(f"fitting {spec.name} ...", flush=True)
        start = time.perf_counter()

        # if tuned_from is specified, use those params
        fixed = None
        if spec.tuned_from:
            fixed = info[spec.tuned_from]["params"]

        # fit the model, pull out classifier from sklearn pipeline
        result = fit_model(spec, X_train, y_train, params=fixed)
        clf = result.estimator.named_steps["clf"]

        info[spec.name] = {"threshold": result.threshold, "params": result.best_params}
        if hasattr(clf, "n_iter_"):
            # Shows whether boosting stopped early or hit max_iter, and whether saga converged.
            info[spec.name]["n_iter"] = int(np.max(clf.n_iter_))
        if spec.name == "age_rule":
            info[spec.name]["refer_age_band_at_least"] = round(result.threshold * AGE_BANDS)
        print(f"  done in {time.perf_counter() - start:.0f}s", flush=True)

        if result.cv_results is not None:
            keep = [c for c in result.cv_results if c.startswith("param_")]
            keep += ["mean_test_score", "std_test_score", "rank_test_score"]
            result.cv_results[keep].to_csv(args.out / f"cv_results_{spec.name}.csv", index=False)
        if isinstance(clf, LogisticRegression):
            odds_ratios(result.estimator).to_csv(args.out / f"odds_ratios_{spec.name}.csv",
                                                 index=False)

        # run the best fit model on the test split and save probabilities
        # also save oof probabilities for this model
        test_proba = result.estimator.predict_proba(X_test[spec.features])[:, 1]
        predictions[f"{spec.name}_proba"] = test_proba
        oof_predictions[f"{spec.name}_proba"] = result.oof_proba

        # CV metrics use the same oof predictions the threshold was picked on,
        # so they run optimistic. The test split is the held-out estimate.
        for split_name, y, proba in (
            ("cv", y_train, result.oof_proba),
            ("test", y_test, test_proba),
        ):
            metric_rows.append({
                "model": spec.name,
                "split": split_name,
                "threshold": result.threshold,
                **classification_metrics(y, proba, result.threshold),
            })

        if spec.in_slices:
            slices = slice_metrics(X_test, y_test, test_proba, result.threshold)
            slices.insert(0, "model", spec.name)
            slice_frames.append(slices)

    metrics = pd.DataFrame(metric_rows)
    slices = pd.concat(slice_frames, ignore_index=True)

    metrics.to_csv(args.out / "metrics.csv", index=False)
    slices.to_csv(args.out / "slices.csv", index=False)
    predictions.to_csv(args.out / "test_predictions.csv", index=False)
    oof_predictions.to_csv(args.out / "oof_predictions.csv", index=False)
    (args.out / "best_params.json").write_text(json.dumps(info, indent=2, default=str))

    pd.set_option("display.width", 160)
    cols = ["model", "threshold", "f2", "precision", "recall", "pr_auc", "roc_auc", "tn", "fp", "fn", "tp"]
    print("\nTest set metrics")
    print(metrics.loc[metrics["split"] == "test", cols].to_string(index=False, float_format="%.3f"))

    print("\nWorst slice per model (lowest test F2)")
    for model, group in slices.groupby("model", sort=False):
        w = worst_slice(group)
        flag = "  [low support]" if w["low_support"] else ""
        print(
            f"  {model:<26} {w['slice']:<12} F2={w['f2']:.3f} recall={w['recall']:.3f} "
            f"n={w['n']} positives={w['n_pos']}{flag}"
        )

    ci = slice_ci.main(["--results", str(args.out)])
    print_headline(metrics, slices, ci)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()