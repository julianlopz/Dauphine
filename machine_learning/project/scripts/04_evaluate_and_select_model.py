"""Step 4: unbiased evaluation of both models and final model selection.

For each candidate (random forest, gradient boosting) we recompute
out-of-fold predictions with `cross_val_predict` on the same 5-fold split
used during tuning: every prediction comes from a model that never saw
that individual during training, which is what lets us report an honest
estimate of future performance (confusion matrix, ROC curve, accuracy,
ROC AUC) instead of reusing in-sample predictions.

The model with the higher out-of-fold ROC AUC is written to
outputs/model_selection.json and is the one script 05 will retrain on the
full learn set to produce predictions.csv.
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from common import config, modeling

POS_LABEL = "T"


def load_best_params(name):
    with open(config.OUTPUT_DIR / f"{name}_best_params.json") as f:
        return json.load(f)["best_params"]


def evaluate_random_forest(X, y, cv):
    params = load_best_params("rf")
    preprocessor, *_ = modeling.random_forest_preprocessor(X)
    pipe = Pipeline(
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
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)
    pipe.fit(X, y)  # full-data fit, only used for feature importance below
    return proba, pipe


def evaluate_hgb(X_raw, y, cv):
    params = load_best_params("hgb")
    X, categorical_cols = modeling.prepare_for_hgb(X_raw)
    categorical_mask = [c in categorical_cols for c in X.columns]
    model = HistGradientBoostingClassifier(
        categorical_features=categorical_mask,
        random_state=config.RANDOM_STATE,
        early_stopping=False,
        **params,
    )
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)
    model.fit(X, y)
    return proba, model, X


def plot_confusion_matrix(y_true, y_pred, labels, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, proba_pos, path, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true == POS_LABEL, proba_pos, ax=ax, name=title)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(names, importances, path, title, top_n=20):
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(range(len(order)), importances[order][::-1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(np.array(names)[order][::-1])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize(name, y, proba, classes):
    pos_idx = list(classes).index(POS_LABEL)
    y_pred = classes[np.argmax(proba, axis=1)]
    auc = roc_auc_score(y == POS_LABEL, proba[:, pos_idx])
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    print(f"\n=== {name} (out-of-fold) ===")
    print(f"ROC AUC: {auc:.4f}  Accuracy: {acc:.4f}")
    print(classification_report(y, y_pred))
    plot_confusion_matrix(
        y, y_pred, classes, config.FIGURES_DIR / f"{name}_confusion_matrix.png", f"{name} - confusion matrix"
    )
    plot_roc(y, proba[:, pos_idx], config.FIGURES_DIR / f"{name}_roc_curve.png", f"{name} - ROC curve")
    return {"roc_auc": auc, "accuracy": acc, "classification_report": report}, y_pred


def main():
    learn = pd.read_parquet(config.OUTPUT_DIR / "learn_features.parquet")
    y = learn[modeling.TARGET_COL]
    X = learn.drop(columns=[modeling.TARGET_COL])
    cv = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    rf_proba, rf_pipe = evaluate_random_forest(X, y, cv)
    rf_metrics, rf_pred = summarize("random_forest", y, rf_proba, rf_pipe.classes_)

    hgb_proba, hgb_model, X_hgb = evaluate_hgb(X, y, cv)
    hgb_metrics, hgb_pred = summarize("hist_gradient_boosting", y, hgb_proba, hgb_model.classes_)

    # --- feature importance -------------------------------------------------
    rf_names = rf_pipe.named_steps["prep"].get_feature_names_out()
    rf_importance = rf_pipe.named_steps["model"].feature_importances_
    plot_feature_importance(
        rf_names, rf_importance, config.FIGURES_DIR / "random_forest_feature_importance.png",
        "Random forest - feature importance (impurity)",
    )

    perm = permutation_importance(
        hgb_model, X_hgb, y, scoring="roc_auc", n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    plot_feature_importance(
        X_hgb.columns, perm.importances_mean, config.FIGURES_DIR / "hgb_feature_importance.png",
        "Gradient boosting - permutation importance (ROC AUC)",
    )

    # --- error analysis: biggest misclassifications of the winning model ---
    winner = "random_forest" if rf_metrics["roc_auc"] >= hgb_metrics["roc_auc"] else "hist_gradient_boosting"
    proba, pred = (rf_proba, rf_pred) if winner == "random_forest" else (hgb_proba, hgb_pred)
    classes = rf_pipe.classes_ if winner == "random_forest" else hgb_model.classes_
    pos_idx = list(classes).index(POS_LABEL)
    confidence = np.abs(proba[:, pos_idx] - 0.5)
    errors = X.assign(
        Pkey=learn["Pkey"], true_target=y, predicted=pred, predicted_proba_T=proba[:, pos_idx]
    )[pred != y].assign(confidence=confidence[pred != y])
    errors.sort_values("confidence", ascending=False).head(50).to_csv(
        config.OUTPUT_DIR / "top_misclassifications.csv", index=False
    )

    with open(config.OUTPUT_DIR / "model_evaluation.json", "w") as f:
        json.dump({"random_forest": rf_metrics, "hist_gradient_boosting": hgb_metrics}, f, indent=2)

    with open(config.OUTPUT_DIR / "model_selection.json", "w") as f:
        json.dump({"winner": winner}, f, indent=2)

    print(f"\nSelected model: {winner}")


if __name__ == "__main__":
    main()
