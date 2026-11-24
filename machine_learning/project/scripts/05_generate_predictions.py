"""Step 5: retrain the selected model on the full learn set and predict on test.

Writes predictions.csv at the project root with exactly the two columns
required by the submission format, in order: Pkey, target.
"""

import json

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from common import config, modeling


def load_best_params(name):
    with open(config.OUTPUT_DIR / f"{name}_best_params.json") as f:
        return json.load(f)["best_params"]


def main():
    learn = pd.read_parquet(config.OUTPUT_DIR / "learn_features.parquet")
    test = pd.read_parquet(config.OUTPUT_DIR / "test_features.parquet")

    y = learn[modeling.TARGET_COL]
    X_learn = learn.drop(columns=[modeling.TARGET_COL])

    with open(config.OUTPUT_DIR / "model_selection.json") as f:
        winner = json.load(f)["winner"]
    print(f"Retraining selected model on the full learn set: {winner}")

    if winner == "random_forest":
        params = load_best_params("rf")
        preprocessor, *_ = modeling.random_forest_preprocessor(X_learn)
        model = Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=config.RANDOM_STATE,
                        n_jobs=-1,
                        n_estimators=params["model__n_estimators"],
                        max_depth=params["model__max_depth"],
                        min_samples_leaf=params["model__min_samples_leaf"],
                        max_features=params["model__max_features"],
                    ),
                ),
            ]
        )
        model.fit(X_learn, y)
        predictions = model.predict(test)
    else:
        params = load_best_params("hgb")
        X_learn_hgb, categorical_cols = modeling.prepare_for_hgb(X_learn)
        categorical_mask = [c in categorical_cols for c in X_learn_hgb.columns]
        model = HistGradientBoostingClassifier(
            categorical_features=categorical_mask,
            random_state=config.RANDOM_STATE,
            early_stopping=False,
            **params,
        )
        model.fit(X_learn_hgb, y)
        X_test_hgb, _ = modeling.prepare_for_hgb(test)
        X_test_hgb = X_test_hgb[X_learn_hgb.columns]  # keep identical column order/dtypes
        predictions = model.predict(X_test_hgb)

    joblib.dump(model, config.MODELS_DIR / "final_model.joblib")

    output = pd.DataFrame({"Pkey": test["Pkey"], "target": predictions})
    out_path = config.PROJECT_ROOT / "predictions.csv"
    output.to_csv(out_path, index=False)
    print(f"Wrote {len(output)} predictions to {out_path}")
    print(output["target"].value_counts())


if __name__ == "__main__":
    main()
