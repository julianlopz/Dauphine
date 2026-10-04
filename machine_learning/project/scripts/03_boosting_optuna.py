"""Step 3: gradient boosting, tuned with Optuna.

The subject recommends boosting as the model most likely to give the best
performance (section 2). We use scikit-learn's
HistGradientBoostingClassifier: it natively supports categorical features
and missing values, which fits this dataset's mostly *structural*
missingness (section 1.2) much better than a one-hot + impute pipeline.

Hyperparameters are optimised with Optuna; each trial is scored by the mean
ROC AUC over the same 5-fold stratified cross-validation used for the
random forest, so that both models are compared on an equal footing.
"""

import json

import joblib
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from common import config, modeling

N_TRIALS = 40


def make_model(categorical_features, **params):
    return HistGradientBoostingClassifier(
        categorical_features=categorical_features,
        random_state=config.RANDOM_STATE,
        early_stopping=False,
        **params,
    )


def objective(trial, X, y, categorical_features, cv):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 500),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
    }
    model = make_model(categorical_features, **params)
    scores = cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
    return scores.mean()


def main():
    learn = pd.read_parquet(config.OUTPUT_DIR / "learn_features.parquet")
    y = learn[modeling.TARGET_COL]
    X_raw = learn.drop(columns=[modeling.TARGET_COL])
    X, categorical_cols = modeling.prepare_for_hgb(X_raw)
    categorical_mask = [c in categorical_cols for c in X.columns]

    cv = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, X, y, categorical_mask, cv),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("Best params:", study.best_params)
    print("Best CV ROC AUC:", study.best_value)

    study.trials_dataframe().to_csv(config.OUTPUT_DIR / "hgb_optuna_trials.csv", index=False)

    best_model = make_model(categorical_mask, **study.best_params)
    best_model.fit(X, y)
    joblib.dump(best_model, config.MODELS_DIR / "hist_gradient_boosting.joblib")
    joblib.dump(categorical_mask, config.MODELS_DIR / "hgb_categorical_mask.joblib")

    with open(config.OUTPUT_DIR / "hgb_best_params.json", "w") as f:
        json.dump({"best_params": study.best_params, "best_cv_roc_auc": study.best_value}, f, indent=2)


if __name__ == "__main__":
    main()
