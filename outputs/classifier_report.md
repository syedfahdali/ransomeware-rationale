# Classifier report — ransomware vs rest (CIC-MalMem-2022)

Generated: 2026-10-10T16:57:02+00:00

## Setup

- Model: xgboost; best params: `{"subsample": 0.7, "reg_lambda": 1.0, "reg_alpha": 0.0, "min_child_weight": 1, "max_depth": 8, "learning_rate": 0.2, "colsample_bytree": 0.8}`
- Tuning: RandomizedSearchCV n_iter=30 (requested 30), 3-fold CV, metric=average_precision, budget=1200.0s
- Early stopping on 10% train slice -> 191 trees; refit on full train
- Calibration: sigmoid on val (5813 rows)
- Train/val/test rows: 40540/5813/11670; seed=42
- Duplicate rows removed before splitting (see outputs/data_summary.md)

**Acceptance (Phase 2): test ROC-AUC >= 0.95.**
Observed raw test ROC-AUC = 0.92726.
## Test metrics

### Raw booster (uncalibrated)

- n=11670, accuracy=0.8847, precision=0.6566, recall=0.6092, F1=0.6320
- ROC-AUC=0.92726, PR-AUC=0.69773, Brier=0.08221
- confusion: TN=9170 FP=604 FN=741 TP=1155

### Calibrated

- n=11670, accuracy=0.8867, precision=0.6803, recall=0.5712, F1=0.6210
- ROC-AUC=0.92726, PR-AUC=0.69773, Brier=0.08495
- confusion: TN=9265 FP=509 FN=813 TP=1083

## Validation metrics (calibration split)

### Raw booster (uncalibrated)

- n=5813, accuracy=0.9085, precision=0.7358, recall=0.6928, F1=0.7137
- ROC-AUC=0.95193, PR-AUC=0.80893, Brier=0.06366
- confusion: TN=4618 FP=238 FN=294 TP=663

### Calibrated

- n=5813, accuracy=0.9105, precision=0.7721, recall=0.6479, F1=0.7045
- ROC-AUC=0.95193, PR-AUC=0.80893, Brier=0.06548
- confusion: TN=4673 FP=183 FN=337 TP=620

## Figures

- `outputs/figs/confusion_test.png` — confusion matrix
- `outputs/figs/reliability_test.png` — reliability curve
- `outputs/figs/roc_test.png`, `outputs/figs/pr_test.png`
