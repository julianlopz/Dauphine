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


from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder


##################
#Config
###################
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_DIR.parent

DATA_DIR = PROJECT_ROOT / "project-7-files"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

for d in (OUTPUT_DIR, FIGURES_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
ID_COL = "Pkey"
TARGET_COL = "target"
N_CV_FOLDS = 5

# Approximate Paris coordinates in the Lambert-93 projection (in metres).
# Used to derive a "distance to Paris" feature as a cheap proxy for
# centrality / access to the main economic hub.
PARIS_LAMBERT93_X = 652_709.0
PARIS_LAMBERT93_Y = 6_862_008.0


####################3
#Modeling
######################

ID_COL = "Pkey"
TARGET_COL = "target"

# Columns dropped because they are either pure identifiers, free-text
# duplicates of a code already kept (Nom de la commune / du département /
# de la région), or a very-high-cardinality alternative to a coarser
# column we keep instead (INSEE_CODE -> dep/Reg/lat/long, club_federation
# -> club_category, X/Y -> distance_to_paris_km already derived from them).
DROP_COLS = [
    "INSEE_CODE",
    "Nom de la commune",
    "Nom du département",
    "Nom de la région",
    "club_federation",
    "X",
    "Y",
]

CATEGORICAL_COLS = [
    "highest_credential",
    "Job_42",
    "HOUSEHOLD",
    "Sex",
    "Act",
    "EMP_CONTRACT",
    "Eco_sect",
    "Employee_count",
    "Job_dep",
    "COMPANY_CATEGORY",
    "type_of_contract",
    "labor_force_status",
    "Job_condition",
    "Job_description_N2",
    "Job_description_N1",
    "Former_job_42",
    "FORMER_EMP_CONTRACT",
    "Eco_sect_former_job",
    "former_dep",
    "Job_dep_former_job",
    "type_of_contract_former_job",
    "labor_force_status_former_job",
    "Job_condition_former_job",
    "COMPANY_CATEGORY_former_job",
    "Employee_count_former_job",
    "Former_job_description_N2",
    "Former_job_description_N1",
    "club_category",
    "city_type",
    "dep",
    "Reg",
]

BOOLEAN_COLS = [
    "IS_STUDENT",
    "has_emp_contract",
    "has_job_detail",
    "has_retired_former",
    "has_retired_job_detail",
    "has_pension",
    "has_club",
]

NUMERIC_COLS = [
    "Age_2020",
    "PAY",
    "working_hours",
    "retirement_age",
    "working_hours_former_job",
    "retirement_pay",
    "LAT",
    "long",
    "INHABITANTS",
    "distance_to_paris_km",
    "log_inhabitants",
]


def get_feature_columns(df):
    """Restrict the column groups above to what is actually in `df`."""
    cols = set(df.columns) - {ID_COL, TARGET_COL} - set(DROP_COLS)
    categorical = [c for c in CATEGORICAL_COLS if c in cols]
    boolean = [c for c in BOOLEAN_COLS if c in cols]
    numeric = [c for c in NUMERIC_COLS if c in cols]
    known = set(categorical + boolean + numeric)
    leftover = cols - known
    if leftover:
        raise ValueError(f"Unclassified columns, update common/py: {leftover}")
    return categorical, boolean, numeric


def _stringify(X):
    """Cast every column to plain strings while preserving NaN markers.

    Some categorical columns (e.g. club_category) come out of the merges
    as floats because of upstream NaNs; mixing floats and strings in the
    same column later crashes OneHotEncoder, so we normalise everything
    to strings here, before imputation/encoding.
    """
    X = X.copy()
    for col in X.columns:
        na_mask = X[col].isna()
        X[col] = X[col].astype(str)
        X.loc[na_mask, col] = np.nan
    return X


def random_forest_preprocessor(df):
    categorical, boolean, numeric = get_feature_columns(df)

    categorical_pipe = Pipeline(
        steps=[
            ("stringify", FunctionTransformer(_stringify, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, categorical),
            ("num", numeric_pipe, numeric + boolean),
        ]
    )
    return preprocessor, categorical, boolean, numeric


def prepare_for_hgb(df):
    """Cast categorical columns to pandas 'category' dtype in place.

    HistGradientBoostingClassifier(categorical_features="from_dtype") then
    picks them up automatically, NaNs included, with no further encoding.
    """
    categorical, boolean, numeric = get_feature_columns(df)
    df = df.copy()
    for c in categorical:
        df[c] = df[c].astype("category")
    for c in boolean:
        df[c] = df[c].astype(float)
    feature_cols = categorical + boolean + numeric
    return df[feature_cols], categorical





######################
#Actual script
######################


POS_LABEL = "T"

#Load best parameters from random forest (script 2) and from gradient boosting (script 3)
def load_best_params(name):
    with open(OUTPUT_DIR / f"{name}_best_params.json") as f:
        return json.load(f)["best_params"]

#-----------------------
#Random forest
#-----------------------
def evaluate_random_forest(X, y, cv):
    #Load the optimized Random Forest hyperparameters.
    params = load_best_params("rf")
    preprocessor, *_ = random_forest_preprocessor(X)

    #Build the pipeline
    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    n_estimators=params["model__n_estimators"],
                    max_depth=params["model__max_depth"],
                    min_samples_leaf=params["model__min_samples_leaf"],
                    max_features=params["model__max_features"],
                ),
            ),
        ]
    )

    #Generate out of fold predictions
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)
    #trains temporary copies of the pipeline using the cross-validation splits and generates predictions for the held-out samples.

    #Fit the pipeline on the full dataset
    pipe.fit(X, y)  # full-data fit, only used for feature importance below
    return proba, pipe


#----------------------
#Gradient boosting
#----------------------
def evaluate_hgb(X_raw, y, cv):
    #Load the optimized gradiend boosting parameters
    params = load_best_params("hgb")
    X, categorical_cols = prepare_for_hgb(X_raw)
    categorical_mask = [c in categorical_cols for c in X.columns]

    #Create the classifier
    model = HistGradientBoostingClassifier(
        categorical_features=categorical_mask,
        random_state=RANDOM_STATE,
        early_stopping=False,
        **params,
    )

    #Generate out-of-fold probabilities
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)

    #Fit the model on the full dataset
    model.fit(X, y)
    return proba, model, X

#---------------------
#Plots of the results
#---------------------
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

#------------------
#Evaluation metrics
#------------------
#This function takes the actual targets and the out-of-fold probabilities and produces the model's evaluation results.
def summarize(name, y, proba, classes):
    pos_idx = list(classes).index(POS_LABEL)
    y_pred = classes[np.argmax(proba, axis=1)] #convert the probability predictions into class predictions.

    #Calculate AUC and accuracy
    auc = roc_auc_score(y == POS_LABEL, proba[:, pos_idx])
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    
    #Show results
    print(f"\n=== {name} (out-of-fold) ===")
    print(f"ROC AUC: {auc:.4f}  Accuracy: {acc:.4f}")
    print(classification_report(y, y_pred))

    #Plot results
    plot_confusion_matrix(
        y, y_pred, classes, FIGURES_DIR / f"{name}_confusion_matrix.png", f"{name} - confusion matrix"
    )
    plot_roc(y, proba[:, pos_idx], FIGURES_DIR / f"{name}_roc_curve.png", f"{name} - ROC curve")
    return {"roc_auc": auc, "accuracy": acc, "classification_report": report}, y_pred

#define main script
def main():
    #Load dataset (X and Y)
    learn = pd.read_parquet(OUTPUT_DIR / "learn_features.parquet")
    y = learn[TARGET_COL]
    X = learn.drop(columns=[TARGET_COL])

    #Cross validation
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    #Train and evaluate the Random forest
    rf_proba, rf_pipe = evaluate_random_forest(X, y, cv)
    rf_metrics, rf_pred = summarize("random_forest", y, rf_proba, rf_pipe.classes_)

    #Train and evaluate the gradient boosting
    hgb_proba, hgb_model, X_hgb = evaluate_hgb(X, y, cv)
    hgb_metrics, hgb_pred = summarize("hist_gradient_boosting", y, hgb_proba, hgb_model.classes_)

    # --- feature importance -------------------------------------------------
    #Random forest feature importance
    rf_names = rf_pipe.named_steps["prep"].get_feature_names_out()
    rf_importance = rf_pipe.named_steps["model"].feature_importances_
    plot_feature_importance(
        rf_names, rf_importance, FIGURES_DIR / "random_forest_feature_importance.png",
        "Random forest - feature importance (impurity)",
    )

    #Gradient boosting permutation importance
    perm = permutation_importance(
        hgb_model, X_hgb, y, scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    plot_feature_importance(
        X_hgb.columns, perm.importances_mean, FIGURES_DIR / "hgb_feature_importance.png",
        "Gradient boosting - permutation importance (ROC AUC)",
    )

    # --- error analysis: biggest misclassifications of the winning model ---
    winner = "random_forest" if rf_metrics["roc_auc"] >= hgb_metrics["roc_auc"] else "hist_gradient_boosting"

    #Get winning model predictions
    proba, pred = (rf_proba, rf_pred) if winner == "random_forest" else (hgb_proba, hgb_pred)
    classes = rf_pipe.classes_ if winner == "random_forest" else hgb_model.classes_
    pos_idx = list(classes).index(POS_LABEL)

    #Calculate predictions confidence
    confidence = np.abs(proba[:, pos_idx] - 0.5)

    #Identify missclased observations
    errors = X.assign(
        Pkey=learn["Pkey"], true_target=y, predicted=pred, predicted_proba_T=proba[:, pos_idx]
    )[pred != y].assign(confidence=confidence[pred != y])

    #Save the 50 biggest errors (more confident about being errors)
    errors.sort_values("confidence", ascending=False).head(50).to_csv(
        OUTPUT_DIR / "top_misclassifications.csv", index=False
    )

    #Save model and metrics
    with open(OUTPUT_DIR / "model_evaluation.json", "w") as f:
        json.dump({"random_forest": rf_metrics, "hist_gradient_boosting": hgb_metrics}, f, indent=2)

    with open(OUTPUT_DIR / "model_selection.json", "w") as f:
        json.dump({"winner": winner}, f, indent=2)

    print(f"\nSelected model: {winner}")


if __name__ == "__main__":
    main()
