# FinGuard AI — Corporate Distress Prediction

A forward-looking corporate financial-distress classification project on **2,108 NSE mainboard non-financial firms** over FY2012–2023. The project focuses on leakage control, temporal validation, imbalanced classification, and explainability.

## Research question

Can firm-level financial information distinguish healthy, vulnerable, and distressed firms, and how much harder is the problem when the model must identify distress **one year ahead**?

## Data & target

- Final panel: **25,296 firm-year observations** and 34 columns
- Three-tier ICR-based distress label
- ICR is excluded from the final predictors because it contributes directly to target construction
- 27 final predictive features after feature screening

## Methodology

- Chronological train/validation/OOT design rather than a random split for the main evaluation
- Models include logistic regression, Random Forest, and XGBoost
- Assessment emphasizes accuracy, balanced accuracy, Macro F1, ROC-AUC, and class-level recall
- SHAP is used to interpret tree-model predictions
- A separate one-year-ahead target evaluates the early-warning setting

## Canonical out-of-time results

The final XGBoost pipeline evaluates FY2022–2023 as a genuinely future-period OOT sample. The saved final specification reports:

- **Accuracy: 94.66%**
- **Balanced Accuracy: 89.53%**
- **Macro F1: 0.890**
- **ROC-AUC: 0.987**

A separate robustness specification reports **93.45% accuracy, 91.42% balanced accuracy, 0.879 Macro F1, and 0.986 ROC-AUC**. The difference is retained rather than silently collapsed: the two figures correspond to different model specifications and should not be treated as interchangeable.

The headline is not accuracy alone: class-level performance matters because missing a distressed firm is materially different from misclassifying a healthy firm.

## One-year-ahead early warning

The temporal early-warning experiment uses a later held-out period to predict distress one year ahead. Its lower performance relative to contemporaneous classification is an important finding: **future distress is materially harder to predict than the current financial state**.

## Key takeaway

FinGuard demonstrates a complete risk-modeling workflow: **target construction → leakage audit → feature screening → temporal validation → model comparison → OOT evaluation → explainability → early-warning extension**.

## Limitations

This is a predictive modeling exercise, not a causal model of why firms become distressed. Class imbalance, measurement choices in the distress label, and temporal regime changes can affect performance.

## Notebooks

1. `01_data_audit_and_prep.ipynb` — data audit, target construction, feature screening, leakage control, and temporal dataset preparation
2. `02_modeling_and_shap.ipynb` — canonical final XGBoost evaluation, OOT performance, feature importance, and SHAP explainability

## Data availability

Raw/private source data are not included. Place the required source files under `./data/` before executing the notebooks.
