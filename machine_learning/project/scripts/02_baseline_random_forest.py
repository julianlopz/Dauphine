"""Step 2: Random Forest baseline.

Random forests are used as the reference model requested in the subject
(section 2). Hyperparameters are selected with a grid search under 5-fold
stratified cross-validation, scored by ROC AUC (the target is binary:
"I" vs "T"). Results (CV table + best params + best estimator) are saved
under outputs/ for the evaluation and prediction scripts to reuse.
"""

import json
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from common import config, modeling

PARAM_GRID = {
    "model__n_estimators": [300],
    "model__max_depth": [None, 20, 30],
    "model__min_samples_leaf": [1, 5, 20],
    "model__max_features": ["sqrt", 0.3],
}


def main():
    learn = pd.read_parquet(config.OUTPUT_DIR / "learn_features.parquet")
    y = learn[modeling.TARGET_COL]
    X = learn.drop(columns=[modeling.TARGET_COL])

    preprocessor, categorical, boolean, numeric = modeling.random_forest_preprocessor(X)
    print(f"{len(categorical)} categorical, {len(boolean)} boolean, {len(numeric)} numeric features")

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)),
        ]
    )

    cv = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    search = GridSearchCV(
        pipe,
        PARAM_GRID,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=2,
    )

    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start
    print(f"Grid search done in {elapsed:.1f}s")
    print("Best params:", search.best_params_)
    print("Best CV ROC AUC:", search.best_score_)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(config.OUTPUT_DIR / "rf_cv_results.csv", index=False)

    joblib.dump(search.best_estimator_, config.MODELS_DIR / "random_forest.joblib")

    with open(config.OUTPUT_DIR / "rf_best_params.json", "w") as f:
        json.dump(
            {"best_params": search.best_params_, "best_cv_roc_auc": search.best_score_},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
