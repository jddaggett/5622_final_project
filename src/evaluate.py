"""Overall and per-slice evaluation metrics."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import BETA, MIN_SLICE_POSITIVES, SLICES


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, Any]:
    """F2, precision, recall, PR-AUC, ROC-AUC, and confusion counts at a threshold."""
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n, n_pos = len(y_true), int(y_true.sum())
    # AUCs are undefined when a subset contains only one class. (ValueError if only 1 class present)
    both_classes = 0 < n_pos < n
    return {
        "n": n,
        "n_pos": n_pos,
        "f2": fbeta_score(y_true, pred, beta=BETA, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, proba) if both_classes else np.nan,
        "roc_auc": roc_auc_score(y_true, proba) if both_classes else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def slice_metrics(
    X: pd.DataFrame, y: pd.Series, proba: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Metrics for every slice in config.SLICES, using the model's global threshold."""
    y_arr = y.to_numpy()
    rows = []
    for group, name, select in SLICES:
        # apply slice lambda to data to get boolean mask then apply to y and proba
        mask = select(X).to_numpy()
        m = classification_metrics(y_arr[mask], proba[mask], threshold)

        # **m <- (n, n_pos, f2, precision, recall, pr_auc, roc_auc, tn, fp, fn, tp) 
        rows.append({
            "group": group,
            "slice": name,
            **m,
            "low_support": m["n_pos"] < MIN_SLICE_POSITIVES,
        })
    return pd.DataFrame(rows)


def worst_slice(slices: pd.DataFrame) -> pd.Series:
    """Lowest-F2 slice among slices with at least one positive."""
    eligible = slices[slices["n_pos"] > 0]
    return eligible.loc[eligible["f2"].idxmin()]


_UNITS = {
    "BMI": "per 1 BMI point",
    "Age": "per age band (5 years)",
    "Education": "per education level",
    "Income": "per income bracket",
}


def odds_ratios(pipeline: Any) -> pd.DataFrame:
    """Odds ratio per natural unit for each logistic regression input.

    Scaled coefficients are divided by the scaler's SD to undo standardization.
    class_weight='balanced' mostly shifts the intercept; slopes are only exactly
    unaffected if the logistic model is correctly specified, so treat them as approximate.
    """
    ct = pipeline.named_steps["encode"].ct_
    coef = pipeline.named_steps["clf"].coef_.ravel()
    names = ct.get_feature_names_out()
    scaler = ct.named_transformers_["scale"]
    sd = dict(zip(scaler.feature_names_in_, scaler.scale_))
    rows = []
    for name, c in zip(names, coef):
        kind, feature = name.split("__", 1)
        if kind == "scale":
            c_unit, unit = c / sd[feature], _UNITS.get(feature, "per unit")
        elif kind == "age":
            band = feature.split("_", 1)[1]
            c_unit, unit, feature = c, f"band {band} vs. band 1 (18-24)", "Age"
        else:
            c_unit, unit = c, "male vs. female" if feature == "Sex" else "yes vs. no"
        rows.append({"feature": feature, "unit": unit, "coef": c_unit, "odds_ratio": np.exp(c_unit)})
    return pd.DataFrame(rows)