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

N_TRIALS = 40 #40 different combinations of hyperparameters

# A hyperparameter is a setting you choose before training, such as how many iterations the model performs or how complex its decision trees can become.

# The script doesn't manually define all 40 combinations. Instead, Optuna proposes combinations and uses previous results to guide its subsequent choices.

# One important distinction: 40 trials does not mean 40 cross-validation folds. Each trial is evaluated using five folds, 
# so the optimization performs up to 200 individual model fits, followed by one final fit on all the training data.


#This function creates a new instance of your classifier.
def make_model(categorical_features, **params):
    return HistGradientBoostingClassifier(
        categorical_features=categorical_features, #Tells the model which input columns are categorical.
        random_state=RANDOM_STATE, #Makes the model's randomized behavior reproducible where applicable.
        early_stopping=False, #Prevents the model from automatically stopping its training early based on an internal validation procedure.
        **params, #Passes additional hyperparameters to the classifier.
    )



# This function defines how Optuna evaluates each candidate model
def objective(trial, X, y, categorical_features, cv):
#1st optuna proposes hyperparameters (6 of them)
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 500),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
    }
#Optuna's proposed values are passed into make_model() to create a classifier for the current trial. (previous function)
    model = make_model(categorical_features, **params)

#This line evaluates the candidate model using five-fold cross-validation.
#With five folds, the dataset is divided into five subsets. Each evaluation trains on four subsets and validates on the remaining subset.
#This produces 5 differenc AUC scores
    scores = cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)

#Get mean of the 5 AUC scores. This is what we try to maximize
    return scores.mean()




def main():
    learn = pd.read_parquet(OUTPUT_DIR / "learn_features.parquet") #Load the training dataset
    
    #Separate target from feature columns
    y = learn[TARGET_COL] #Target
    X_raw = learn.drop(columns=[TARGET_COL]) #Feature

    #Prepare the features for HistGradientBoosting
    X, categorical_cols = prepare_for_hgb(X_raw)
    categorical_mask = [c in categorical_cols for c in X.columns]

    #Define the cross-validation strategy
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    #Create the Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    #Run Optuna's optimization
    #For each of the 40 trials, the hyperparameters are proposed, a model is created with them, evaluates with the Cross validation, Gets AUC and records to compare with the rest of trials
    study.optimize(
        lambda trial: objective(trial, X, y, categorical_mask, cv),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("Best params:", study.best_params)
    print("Best CV ROC AUC:", study.best_value)

    study.trials_dataframe().to_csv(OUTPUT_DIR / "hgb_optuna_trials.csv", index=False)

    #select and train only the best model
    #During optimization, the model is repeatedly trained on only four of the five folds. Once the best hyperparameters are found, the script creates a fresh model 
    #and fits it on the entire training dataset.
    
    #This final model is the one you can use later for predictions.
    #It is not an ensemble of the five cross-validation models. Those models were temporary evaluation models; this is a newly trained classifier.
    best_model = make_model(categorical_mask, **study.best_params)
    best_model.fit(X, y)
    joblib.dump(best_model, MODELS_DIR / "hist_gradient_boosting.joblib")
    joblib.dump(categorical_mask, MODELS_DIR / "hgb_categorical_mask.joblib")

    with open(OUTPUT_DIR / "hgb_best_params.json", "w") as f:
        json.dump({"best_params": study.best_params, "best_cv_roc_auc": study.best_value}, f, indent=2)


if __name__ == "__main__":
    main()
