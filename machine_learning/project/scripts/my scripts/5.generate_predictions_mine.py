"""Step 5: retrain the selected model on the full learn set and predict on test.

Writes predictions.csv at the project root with exactly the two columns
required by the submission format, in order: Pkey, target.
"""

import json

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

#from common import config, modeling
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

#Load the best hyperparameters for both models
def load_best_params(name):
    with open(OUTPUT_DIR / f"{name}_best_params.json") as f:
        return json.load(f)["best_params"]


def main():
    #Load data train and test
    learn = pd.read_parquet(OUTPUT_DIR / "learn_features.parquet")
    test = pd.read_parquet(OUTPUT_DIR / "test_features.parquet")

    y = learn[TARGET_COL]
    X_learn = learn.drop(columns=[TARGET_COL])

    #Keep only the winning model
    with open(OUTPUT_DIR / "model_selection.json") as f:
        winner = json.load(f)["winner"]
    print(f"Retraining selected model on the full learn set: {winner}")

    #-----------------
    #Models to predict
    #-----------------

    #Random forest wins
    if winner == "random_forest":
        params = load_best_params("rf") #Load best parameters
        preprocessor, *_ = random_forest_preprocessor(X_learn)

        #Create the pipeline
        model = Pipeline(
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

        #Train the model
        model.fit(X_learn, y)

        #Predict with the trained model
        predictions = model.predict(test)

    #Gradient boosting wins
    else:
        params = load_best_params("hgb") #Load best parameters
        X_learn_hgb, categorical_cols = prepare_for_hgb(X_learn)
        categorical_mask = [c in categorical_cols for c in X_learn_hgb.columns]

        #Create the model
        model = HistGradientBoostingClassifier(
            categorical_features=categorical_mask,
            random_state=RANDOM_STATE,
            early_stopping=False,
            **params,
        )

        #Train the model
        model.fit(X_learn_hgb, y)
        X_test_hgb, _ = prepare_for_hgb(test)
        X_test_hgb = X_test_hgb[X_learn_hgb.columns]  # keep identical column order/dtypes

        #Predict with the trained model
        predictions = model.predict(X_test_hgb)

    #Save final trained model
    joblib.dump(model, MODELS_DIR / "final_model.joblib")

    #Save predictions as a csv file
    output = pd.DataFrame({"Pkey": test["Pkey"], "target": predictions})
    out_path = PROJECT_ROOT / "predictions.csv"
    output.to_csv(out_path, index=False)

    #Final summary
    print(f"Wrote {len(output)} predictions to {out_path}")
    print(output["target"].value_counts())


if __name__ == "__main__":
    main()
