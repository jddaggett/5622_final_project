"""Project-wide constants: paths, split settings, feature groups, and slices."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "cdc_diabetes_health_indicators.csv"
DATA_URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
RESULTS_DIR = ROOT / "results"

RANDOM_STATE = 42 # for reproducing results
TEST_SIZE = 0.2
CV_FOLDS = 5
BETA = 2.0 # F-beta with beta=2: recall is twice as important as precision
N_JOBS = 6 # how many models are fitted at once
N_BOOT = 2000 # how many bootstrap resamples go into each CI

BASELINE_MODEL = "logreg"
HEADLINE_MODEL = "logreg_tuned"
CI_MODELS = [
    "logreg", "logreg_tuned", "random_forest", "grad_boost", "logreg_tuned-ses", "grad_boost-ses",
]

TARGET = "Stroke"
CONTINUOUS = ["BMI"]
# Encoded as integer bands (e.g., Age=1 means 18-24, the first 5 year band)
ORDINAL = ["Age", "Education", "Income"]

# Many features are yes/no survey questions which make the decision difficult
# Diabetes was the original label
BINARY = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "HeartDiseaseorAttack", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "Sex",
    "Diabetes_binary",
]
FEATURES = CONTINUOUS + ORDINAL + BINARY
AGE_BANDS = 13  # BRFSS Age codes 1-13

# plausibly consequences of a stroke: excluded from main models
# added as features in post-stroke ablation models
POST_STROKE = ["DiffWalk", "GenHlth", "MentHlth", "PhysHlth"]

# "how many days in the last month have you experienced poor mental/physical health (0-30)"
POST_STROKE_CONTINUOUS = ["MentHlth", "PhysHlth"]

# Socioeconomic features dropped in the fairness ablation (income acts as a risk signal).
SES = ["Income", "Education", "NoDocbcCost"]

# Slices with fewer test positives than this are flagged as low-support.
MIN_SLICE_POSITIVES = 10

SliceFn = Callable[[pd.DataFrame], pd.Series]

# BRFSS codes: Age 1-5 = 18-44, 6-8 = 45-59, 9-13 = 60+; Sex 0 = female, 1 = male;
# Income 1-4 = under $25k, 5-7 = $25k-75k, 8 = $75k+.
# These slices were chosen and locked in based on risk factors and a 
# fairness check before any training or evaluation
SLICES: list[tuple[str, str, SliceFn]] = [
    ("age", "18-44", lambda d: d["Age"] <= 5),
    ("age", "45-59", lambda d: d["Age"].between(6, 8)),
    ("age", "60+", lambda d: d["Age"] >= 9),
    ("sex", "Female", lambda d: d["Sex"] == 0),
    ("sex", "Male", lambda d: d["Sex"] == 1),
    ("hypertension", "HighBP=1", lambda d: d["HighBP"] == 1),
    ("hypertension", "HighBP=0", lambda d: d["HighBP"] == 0),
    ("smoking", "Smoker=1", lambda d: d["Smoker"] == 1),
    ("smoking", "Smoker=0", lambda d: d["Smoker"] == 0),
    ("diabetes", "Diabetes=1", lambda d: d["Diabetes_binary"] == 1),
    ("diabetes", "Diabetes=0", lambda d: d["Diabetes_binary"] == 0),
    ("income", "<$25k", lambda d: d["Income"] <= 4),
    ("income", "$25-75k", lambda d: d["Income"].between(5, 7)),
    ("income", ">=$75k", lambda d: d["Income"] == 8),
]