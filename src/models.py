"""Model pipelines: reference rules, logistic regression, RF, GB, and ablations."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    AGE_BANDS, BINARY, CONTINUOUS, FEATURES, POST_STROKE, POST_STROKE_CONTINUOUS, RANDOM_STATE,
    SES,
)

MAIN_FEATURES = FEATURES
ALL_FEATURES = FEATURES + POST_STROKE
NO_SES_FEATURES = [c for c in FEATURES if c not in SES]


@dataclass
class ModelSpec:
    """A named pipeline, the features it uses, and its hyperparameter grid.

    fixed_threshold: use this cutoff instead of tuning one for F2 (reference rules).
    in_slices: include in slice analysis.
    tuned_from: for ablations, reuse that model's tuned hyperparameters instead of
    re-tuning, so the ablation changes only one thing.
    """

    name: str
    pipeline: Pipeline
    features: list[str]
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    fixed_threshold: float | None = None
    in_slices: bool = True
    tuned_from: str | None = None


class FeatureEncoder(TransformerMixin, BaseEstimator):
    """Scale numeric and banded columns, pass binary columns through, optionally one-hot Age.

    A wrapper (rather than a bare ColumnTransformer) so `onehot_age` is a tunable
    hyperparameter that ablations can inherit via `tuned_from`.
    """

    def __init__(self, features: list[str], onehot_age: bool = False):
        self.features = features
        self.onehot_age = onehot_age

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEncoder":
        binary = [c for c in self.features if c in BINARY or c == "DiffWalk"]
        onehot = ["Age"] if self.onehot_age and "Age" in self.features else []
        scaled = [c for c in self.features if c not in binary + onehot]
        parts = [("scale", StandardScaler(), scaled), ("bin", "passthrough", binary)]
        if onehot:
            # Band 1 (18-24) is the reference, so each age coefficient is relative to it.
            parts.append(("age", OneHotEncoder(drop="first", sparse_output=False), onehot))
        self.ct_ = ColumnTransformer(parts).fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.ct_.transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return self.ct_.get_feature_names_out()


class AgeRule(ClassifierMixin, BaseEstimator):
    """Reference rule: a DummyClassifier-like model whose score is the patient's age band."""

    def fit(self, X: pd.DataFrame, y: Any) -> "AgeRule":
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # normalize scores to probabilities in [0, 1].
        s = X["Age"].to_numpy(dtype=float) / AGE_BANDS
        return np.column_stack([1 - s, s])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Never used for results, but cross_val_predict rejects estimators without it.
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _smote(features: list[str]) -> SMOTENC:
    # SMOTENC keeps binary and banded columns on valid codes instead of interpolating
    # them to values like Age=7.4. It runs only during fit inside an imblearn Pipeline.
    continuous = CONTINUOUS + POST_STROKE_CONTINUOUS
    categorical = [c for c in features if c not in continuous]
    return SMOTENC(categorical_features=categorical, random_state=RANDOM_STATE)


def _pipeline(clf: ClassifierMixin, features: list[str], smote: bool = False) -> Pipeline:
    steps = [("smote", _smote(features))] if smote else []
    steps += [("encode", FeatureEncoder(features)), ("clf", clf)]
    return Pipeline(steps)


def _logreg() -> LogisticRegression:
    """The declared baseline, exactly as locked in the proposal."""
    return LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, solver="lbfgs")


def _logreg_tunable() -> LogisticRegression:
    # saga because lbfgs can't do L1, and liblinear also penalizes the intercept, which
    # distorts scores at small C. l1_ratio selects L1 vs. L2 (penalty= deprecated as of sklearn 1.8)
    return LogisticRegression(
        class_weight="balanced", solver="saga", max_iter=2000, random_state=RANDOM_STATE
    )


def _random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
    )


def _grad_boost() -> HistGradientBoostingClassifier:
    # HistGradientBoosting because GradientBoostingClassifier has no class_weight.
    return HistGradientBoostingClassifier(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )


LR_GRID = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "clf__l1_ratio": [0.0, 1.0],
    "encode__onehot_age": [False, True],
}
# Run 1's optimum (max_depth=20, min_samples_leaf=50) sat on the grid edge; this extends past it.
RF_GRID = {"clf__max_depth": [20, None], "clf__min_samples_leaf": [20, 50, 200]}
# Run 1's optimum (learning_rate=0.05, max_leaf_nodes=15) sat on the grid edge; this extends past it.
# min_samples_leaf=20 never ranked in the top 6 of run 2, and 1000 won, so 2000 replaced it.
GB_GRID = {
    "clf__learning_rate": [0.01, 0.025, 0.05, 0.1],
    "clf__max_leaf_nodes": [4, 8, 15, 31],
    "clf__min_samples_leaf": [200, 1000, 2000],
}


def build_models() -> list[ModelSpec]:
    """Reference rules and main models first, then ablations (which reuse main-model tuning)."""
    f = MAIN_FEATURES
    return [
        # References: a useless floor, "refer everyone", and the best age-only cutoff.
        ModelSpec("dummy", Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]), f,
                  fixed_threshold=0.5, in_slices=False),
        ModelSpec("flag_all", Pipeline([("clf", DummyClassifier(strategy="constant", constant=1))]),
                  f, fixed_threshold=0.5, in_slices=False),
        ModelSpec("age_rule", Pipeline([("clf", AgeRule())]), ["Age"],
                  in_slices=False),
        ModelSpec("logreg", _pipeline(_logreg(), f), f),
        ModelSpec("logreg_tuned", _pipeline(_logreg_tunable(), f), f, param_grid=LR_GRID),
        ModelSpec("random_forest", _pipeline(_random_forest(), f), f, param_grid=RF_GRID),
        ModelSpec("grad_boost", _pipeline(_grad_boost(), f), f, param_grid=GB_GRID),
        # Ablation 1: SMOTENC oversampling on the ensembles.
        ModelSpec("random_forest+smote", _pipeline(_random_forest(), f, smote=True), f,
                  tuned_from="random_forest"),
        ModelSpec("grad_boost+smote", _pipeline(_grad_boost(), f, smote=True), f,
                  tuned_from="grad_boost"),
        # Ablation 2: add the likely post-stroke features to measure how much they inflate results.
        ModelSpec("logreg_tuned+post_stroke", _pipeline(_logreg_tunable(), ALL_FEATURES),
                  ALL_FEATURES, tuned_from="logreg_tuned"),
        ModelSpec("random_forest+post_stroke", _pipeline(_random_forest(), ALL_FEATURES),
                  ALL_FEATURES, tuned_from="random_forest"),
        ModelSpec("grad_boost+post_stroke", _pipeline(_grad_boost(), ALL_FEATURES),
                  ALL_FEATURES, tuned_from="grad_boost"),
        # Ablation 3 (fairness): drop the socioeconomic features to see whether income acts as a proxy.
        ModelSpec("logreg_tuned-ses", _pipeline(_logreg_tunable(), NO_SES_FEATURES),
                  NO_SES_FEATURES, tuned_from="logreg_tuned"),
        ModelSpec("grad_boost-ses", _pipeline(_grad_boost(), NO_SES_FEATURES),
                  NO_SES_FEATURES, tuned_from="grad_boost"),
    ]