"""Cross-validated tuning, F2 threshold selection, and final fitting."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from .config import BETA, CV_FOLDS, N_JOBS, RANDOM_STATE
from .models import ModelSpec


@dataclass
class FitResult:
    name: str
    estimator: Pipeline
    threshold: float
    best_params: dict[str, Any]
    oof_proba: np.ndarray
    cv_results: pd.DataFrame | None


def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def f2_optimal_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    """Probability cutoff that maximizes F-beta on the given predictions."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    # The last precision/recall pair has no matching threshold.
    p, r = precision[:-1], recall[:-1]
    denom = BETA**2 * p + r
    f = np.divide((1 + BETA**2) * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[np.argmax(f)])


def fit_model(
    spec: ModelSpec, X: pd.DataFrame, y: pd.Series, params: dict[str, Any] | None = None
) -> FitResult:
    """Tune on CV (PR-AUC), pick an F2 threshold from out-of-fold probabilities, refit on all of X.

    params: fixed hyperparameters to apply instead of tuning (used by ablations).
    """
    X = X[spec.features]
    cv = _cv()
    # clone pipeline so spec isn't fitted
    pipeline = clone(spec.pipeline).set_params(**(params or {}))
    cv_results = None
    if spec.param_grid and not params:
        # PR-AUC is threshold-free, so the grid search isn't tied to a cutoff
        # that gets chosen afterwards.
        search = GridSearchCV(
            pipeline, spec.param_grid, scoring="average_precision", cv=cv, n_jobs=N_JOBS
        )
        # fit training data on pipeline with best params from CV (highest mean PR-AUC)
        search.fit(X, y)
        estimator, best_params = search.best_estimator_, search.best_params_
        cv_results = pd.DataFrame(search.cv_results_)
    else:
        # no parameter tuning, just fit with specified params
        estimator, best_params = pipeline.fit(X, y), dict(params or {})

    # out of fold probabilities that set the decision threshold for this model
    oof = cross_val_predict(
        clone(estimator), X, y, cv=cv, method="predict_proba", n_jobs=N_JOBS
    )[:, 1]
    if spec.fixed_threshold is not None:
        threshold = spec.fixed_threshold
    else:
        threshold = f2_optimal_threshold(y.to_numpy(), oof)
    return FitResult(spec.name, estimator, threshold, best_params, oof, cv_results)