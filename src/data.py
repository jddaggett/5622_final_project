"""Loading and splitting the CDC BRFSS health indicators dataset."""

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, DATA_URL, FEATURES, POST_STROKE, RANDOM_STATE, TARGET, TEST_SIZE


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the UCI CSV, downloading it first if missing.

    The file has no missing values; duplicate rows are kept.
    """
    if not path.exists():
        print(f"{path} not found, downloading from {DATA_URL}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path).drop(columns="ID")


def subsample(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Stratified random fraction of rows, for quick test runs."""
    return df.groupby(TARGET, group_keys=False).sample(frac=frac, random_state=RANDOM_STATE)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split. Returns X_train, X_test, y_train, y_test.

    X includes the post-stroke columns so the ablation can use them; each model
    selects its own feature subset.
    """
    X, y = df[FEATURES + POST_STROKE], df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )