# Pipeline

Setup (once):

```
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run in order from the project root (`project-7/`):

```
python scripts/01_build_datasets.py
python scripts/02_baseline_random_forest.py
python scripts/03_boosting_optuna.py
python scripts/04_evaluate_and_select_model.py
python scripts/05_generate_predictions.py
```

- `01_build_datasets.py` merges every raw CSV in `project-7-files/` into
  `outputs/learn_features.parquet` and `outputs/test_features.parquet`.
- `02_baseline_random_forest.py` / `03_boosting_optuna.py` tune each model
  by 5-fold cross-validated ROC AUC and save the best params + fitted
  model under `outputs/`.
- `04_evaluate_and_select_model.py` recomputes honest out-of-fold
  predictions for both models, saves the confusion matrices, ROC curves
  and feature importances to `outputs/figures/`, and picks the model with
  the higher out-of-fold ROC AUC (`outputs/model_selection.json`).
- `05_generate_predictions.py` retrains the selected model on the full
  learning set and writes `predictions.csv` at the project root.

`scripts/common/` holds the shared loading, feature-engineering and
preprocessing code reused by every step above, so that the learn and test
sets are always transformed identically.
