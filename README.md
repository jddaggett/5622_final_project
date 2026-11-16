# Preventative Stroke Intervention Model

CSCI 5622 (Machine Learning) final project — Jackson Daggett

## How to run

Requires Python 3.11+. Run everything from the repository root.

1. Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # macOS / Linux
   .venv\Scripts\Activate.ps1       # Windows PowerShell
   pip install -r requirements.txt
   ```

2. Reproduce the headline number:

   ```bash
   python -m src.run
   ```

   This trains every model, computes slice metrics and bootstrap CIs, and ends with a
   `HEADLINE` block. It took me about 22 minutes on a 6-core CPU with 16 GB of RAM. The seed is
   fixed (`RANDOM_STATE = 42` in `src/config.py`), and repeated runs give identical numbers.

   Expected `HEADLINE` block:

   ```
   HEADLINE (test set)
     headline           logreg_tuned   F2 = 0.339 [0.329, 0.350]  recall = 0.632  flagged = 21.5%
     declared baseline  logreg         F2 = 0.339 [0.329, 0.350]  recall = 0.632  flagged = 21.6%
     reference          flag_all       F2 = 0.175  recall = 1.000  flagged = 100.0%
     reference          age_rule       F2 = 0.239  recall = 0.604  flagged = 35.1%
     reference          dummy          F2 = 0.000  recall = 0.000  flagged = 0.0%
     worst slice        18-44          F2 = 0.095 [0.042, 0.156]  recall = 0.098 [0.043, 0.163]  n = 10910, strokes = 92
   ```

3. Run the follow-up analyses and figures. This reads the outputs of step 2, so run it after:

   ```bash
   python -m src.analysis
   ```

   It takes about 5 minutes and doesn't retrain anything. It covers k-means clusters of
   training patients and per-cluster cutoffs, recall at a matched referral share by age group,
   the fairness ablation, recall vs. referral capacity, and reweighting to younger age mixes.
   Every cutoff comes from training out-of-fold scores.

Optional flags for `src.run`:
- `--sample FRAC` runs on a stratified fraction of the data (for example `--sample 0.05`
  finishes in a few minutes). Useful for testing code changes; the numbers will differ.
- `--data PATH` uses a different CSV location.
- `--out DIR` writes to a different output directory (default `results/`). Pass the same
  directory to the other scripts with `--results DIR`.

`python -m src.slice_ci` recomputes only the bootstrap CIs from saved predictions.

## Overview

A model that helps a primary care physician decide which patients to refer for a fast-tracked
cardiovascular workup, using demographic and clinical survey data. A readable logistic
regression model and two flexible ensembles (random forest, gradient boosting) are compared
against a declared logistic regression baseline and simple reference rules, using a metric
chosen for the asymmetric cost of errors.

## Results

Test set: 50,736 patients, 2,058 strokes. Every cutoff was chosen on training out-of-fold scores.

| Model | F2 [95% CI] | Recall | Referred |
|---|---|---|---|
| Never refer (`dummy`) | 0.000 | 0.000 | 0% |
| Refer everyone (`flag_all`) | 0.175 | 1.000 | 100% |
| Age rule, refer 65+ (`age_rule`) | 0.239 | 0.604 | 35.1% |
| Logistic regression, declared baseline (`logreg`) | 0.339 [0.329, 0.350] | 0.632 | 21.6% |
| **Logistic regression, tuned (`logreg_tuned`, deployed)** | **0.339 [0.329, 0.350]** | **0.632** | **21.5%** |
| Random forest (`random_forest`) | 0.341 [0.329, 0.354] | 0.550 | 16.5% |
| Gradient boosting (`grad_boost`) | 0.340 [0.329, 0.351] | 0.615 | 20.5% |

- **Recommendation:** deploy the tuned logistic regression, which is effectively the declared
  baseline, with one cutoff for every patient. Neither ensemble beats it (paired ΔF2 CIs
  include 0), and its odds ratios show what drives each referral.
- **Worst slice:** ages 18–44, n = 10,910, 92 strokes, recall 0.098 [0.043, 0.163]. The model
  ranks young patients well (recall 0.543 when each age group's top 21.5% is referred), but at
  a 0.84% stroke rate few of them clear the single cutoff.
- **Clustering (k-means, k = 2):** missed strokes concentrate in the higher-income, younger
  cluster (31% of strokes, 57% of misses). Cluster-specific cutoffs lowered F2 (ΔF2 −0.016
  [−0.026, −0.007]), so one global cutoff is kept.
- **Fairness ablation:** dropping income, education and "couldn't afford a doctor" costs 0.013
  F2 and moves referrals toward higher-income patients. The features are kept.
- **What would reverse the recommendation:** a clinic serving only adults under 45, where
  gradient boosting is significantly better (ΔF2 +0.048 [0.007, 0.097]; 16 vs. 9 of 92 strokes).

## Problem framing

- **Decision:** whether to refer a patient for a fast-tracked cardiovascular workup (blood
  pressure monitoring, lipid panel, ECG). The workup, not the model, decides on any medication.
- **Decision maker:** the patient's primary care physician (PCP).
- **False positive:** a patient without elevated risk is referred. Cost: an unneeded workup,
  meaning time, money, and anxiety.
- **False negative:** a high-risk patient is not referred. Their risk factors stay undetected
  and untreated, and they may suffer a preventable, potentially fatal or debilitating stroke.

A missed stroke costs far more than an unneeded workup, which drives the metric choice below.
The flag rate is also a capacity question: every flagged patient needs a workup slot.

## Data

[CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
(UCI ID 891): a cleaned version of the CDC's 2015 BRFSS telephone survey with 253,680 responses
and 21 health and demographic features (high blood pressure, cholesterol, BMI, smoking,
diabetes, heart disease, physical activity, age, sex, income, and others).

- **Target:** `Stroke` (1 = ever told they had a stroke). 10,292 positives (4.1%).
- `Diabetes_binary`, the dataset's original target, is used as a feature.
- The full file is used, not the 70,692-row version that is balanced 50/50 on diabetes.
- The CSV is included at `data/raw/cdc_diabetes_health_indicators.csv`. If it is missing, the
  pipeline downloads it from UCI.

### Limitations of this dataset

- **Label meaning:** `Stroke` is stroke *history*, not a future stroke. The model finds
  profiles resembling stroke survivors, which is a proxy for risk rather than a true
  prospective prediction.
- **Reverse causation:** `DiffWalk`, `GenHlth`, `PhysHlth`, and `MentHlth` are plausibly
  consequences of a stroke. They are excluded from the main models, with one ablation that
  includes them to measure how much they inflate results.
- **Coarser features:** self-reported and mostly binary. Age is in 13 five-year bands, income in
  8 brackets, and there is no glucose. There are no missing values. The 24,206 duplicate rows are
  expected at this coarseness (different people, identical answers) and are kept.

### Why not the original dataset

The memo proposed the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
(Soriano, 2021; 5,110 records, 249 strokes). It was dropped before the proposal for two reasons:

1. **Too few positives.** A pilot confirmed the feedback that it is too small for RF and GB:
   both ensembles lost to logistic regression (test F2 0.354 and 0.331 vs. 0.403, on 50 test
   strokes), and only 4 strokes under age 45 landed in training.
2. **Unreliable provenance.** [Gibson et al. (2026)](https://doi.org/10.1186/s12916-026-04981-y)
   found irregular patient IDs, implausible glucose and age distributions, and unrealistically
   little missing data. The dataset failed all nine TRIPOD+AI provenance checks, and the
   authors concluded it is likely synthetic or fabricated.

The pilot code was replaced by the BRFSS pipeline, so the pilot is not reproducible from this
repository. The original dataset is not included.

## Design

| Decision | Choice |
|---|---|
| Split | 80/20 train/test, stratified on `Stroke`, seed 42 |
| Model selection | Stratified 5-fold CV on the training set, grid search scored on PR-AUC |
| Imbalance handling | `class_weight="balanced"` on all learned models. SMOTENC inside training folds only, as an ablation. The test set is never resampled. |
| Primary metric | F2-score (recall treated as 2x as important as precision) |
| Secondary metrics | PR-AUC, ROC-AUC, confusion matrix at the operating threshold |
| Operating threshold | One per model, chosen to maximize F2 on out-of-fold training predictions |
| Uncertainty | 95% bootstrap CIs (2,000 resamples of the test set, stratified by class); paired differences vs. the baseline |

Accuracy is not reported: always predicting "no stroke" scores over 95% while catching no strokes.

### Models

| Name | Role | Details |
|---|---|---|
| `dummy` | Sanity floor | Always predicts "no stroke" |
| `flag_all` | Reference rule | Refers everyone |
| `age_rule` | Reference rule | Refers everyone at or above an age band; the band is tuned for F2 on training data |
| `logreg` | **Declared baseline** (locked) | `LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, solver="lbfgs")` |
| `logreg_tuned` | Readable model | Logistic regression with C, L1 vs. L2, and ordinal vs. one-hot age tuned by CV. Odds ratios are written to `results/`. |
| `random_forest` | Flexible model | 200 trees; depth and leaf size tuned |
| `grad_boost` | Flexible model | `HistGradientBoostingClassifier` with early stopping; learning rate, leaves, and leaf size tuned |

**Ablations.** `*+smote` adds SMOTENC to an ensemble. `*+post_stroke` adds the four likely
post-stroke features. `*-ses` drops the socioeconomic features (`Income`, `Education`,
`NoDocbcCost`) as a fairness check. Ablations reuse the hyperparameters tuned for the matching
main model, so each changes only one thing.

### Changes since the proposal

- **No locked item changed.** The split, the metric (F2), and the baseline are as declared.
- **Decision reworded** from "prescribe anticoagulants and order a workup" to "refer for a
  workup." Anticoagulants are prescribed for specific indications such as atrial fibrillation,
  which this data can't see, and the model flags about a fifth of patients. A referral is a
  decision the model can actually support.
- **Added** the `flag_all` and `age_rule` references, because the dummy's F2 of 0 is
  uninformative; a tuned logistic model, so the readable model is tuned as honestly as the
  ensembles; the socioeconomic (`*-ses`) ablation; and the analyses in `src/analysis.py`.
- **Corrected two claims in the proposal.** With balanced class weights, logistic regression
  does not give calibrated probabilities (see Implementation notes). And F2 treats recall as
  twice as important as precision; 4 is the coefficient β² in the formula, not the weight.

## Slice analysis

Test-set F2, precision, recall, and confusion counts are reported for each slice, with bootstrap
CIs for F2 and recall. All slices below were named in the proposal.

| Slice | Definition | Rationale |
|---|---|---|
| Age (young) | age bands 1–5 (18–44) | Strokes are rare here, so recall may suffer from few training positives. |
| Age (middle) | age bands 6–8 (45–59) | Transition-risk cohort where confidence and stakes are both elevated. |
| Age (senior) | age bands 9–13 (60+) | Highest incidence, holds most positives, and is expected to drive overall performance. |
| Sex | male / female | Sex differences in stroke incidence and risk-factor weighting. |
| Hypertension | `HighBP` = 1 vs 0 | Dominant risk factor. Checks performance on non-hypertensive patients. |
| Smoking | `Smoker` = 1 vs 0 | Behavioral risk factor. BRFSS only records 100+ lifetime cigarettes. |
| Diabetes | `Diabetes_binary` = 1 vs 0 | Major risk factor (stroke rate 9.3% with diabetes or prediabetes vs. 3.2% without). |
| Income | < $25k / $25–75k / ≥ $75k | Stroke rate falls from 8.1% to 3.9% to 1.7%. Checks whether lower-income patients are under-served. |

## Project structure

```
.
├── data/raw/                  # BRFSS CSV
├── results/                   # generated outputs (git-ignored)
├── src/
│   ├── config.py              # paths, constants, feature groups, slice definitions
│   ├── data.py                # loading (with download), stratified split
│   ├── models.py              # pipelines, reference rules, hyperparameter grids
│   ├── train.py               # CV tuning, F2 threshold selection, refit
│   ├── evaluate.py            # overall and per-slice metrics, odds ratios
│   ├── slice_ci.py            # bootstrap CIs for overall and per-slice F2 and recall
│   ├── analysis.py            # clustering, ranking vs. threshold, fairness, capacity, age mix, figures
│   └── run.py                 # end-to-end entry point
└── requirements.txt
```

## Outputs and implementation notes

### Outputs

| File | Contents |
|---|---|
| `results/metrics.csv` | Per-model CV (out-of-fold) and test metrics at the chosen threshold |
| `results/slices.csv` | Per-model, per-slice test metrics with sample sizes and low-support flags |
| `results/slice_ci.csv` | 95% bootstrap CIs for F2 and recall (overall and per slice) and paired F2 differences vs. `logreg` |
| `results/test_predictions.csv` | Test rows with true labels and each model's score |
| `results/best_params.json` | Per model: threshold, selected hyperparameters, iteration count |
| `results/cv_results_*.csv` | Mean and SD of CV PR-AUC for every grid configuration |
| `results/odds_ratios_*.csv` | Logistic regression odds ratios per natural unit (per BMI point, per age band, yes vs. no) |
| `results/oof_predictions.csv` | Training rows with each model's out-of-fold score, used to choose cutoffs in the analysis |
| `results/analysis/*.csv` | From `src.analysis`: cluster profiles and cutoffs, ranking vs. threshold, fairness ablation, capacity, age mix |
| `results/analysis/fig_*.png` | Recall vs. share referred, and recall by slice with 95% CIs |

CV metrics are computed on the same out-of-fold predictions used to pick the threshold, so
they are optimistic. Use the test numbers as the held-out estimate.

### Implementation notes
- **Encoding:** age, education, and income are integer-coded bands, scaled as ordinal values.
  The tuned logistic model's grid also tried one-hot age, which didn't help. Binary columns
  pass through unchanged.
- **Parallelism:** `N_JOBS` in `config.py` is set to 6. Each parallel random forest fit holds
  its own forest in memory, so raise it only if you have RAM to spare.

## References

- CDC Diabetes Health Indicators [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53919
- Gibson, A.D. et al. (2026). Evidence of unreliable data and poor data provenance in clinical
  prediction model research and clinical practice. *BMC Medicine*, 24, 386.
  https://doi.org/10.1186/s12916-026-04981-y
- Soriano, F. (2021). Stroke Prediction Dataset. Kaggle.
  https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
