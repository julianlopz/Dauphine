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
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder


#from common import config, modeling

#####################
#CONFIG
######################

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

####################
#Modeling
####################

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

#inspect all columns in the DF and determine which feature belongs on each group
def get_feature_columns(df):
    """Restrict the column groups above to what is actually in `df`."""
    cols = set(df.columns) - {ID_COL, TARGET_COL} - set(DROP_COLS)
    categorical = [c for c in CATEGORICAL_COLS if c in cols]
    boolean = [c for c in BOOLEAN_COLS if c in cols]
    numeric = [c for c in NUMERIC_COLS if c in cols]
    known = set(categorical + boolean + numeric)
    leftover = cols - known
    if leftover:
        raise ValueError(f"Unclassified columns, update common/modeling.py: {leftover}")
    return categorical, boolean, numeric


#converts categorical columns into strings while preserving missing values.
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


#This function builds the preprocessing pipeline for your Random Forest model.
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
    #missing values are replaced with the median of the corresponding feature.

    #Apply categrorical pipeline to categorical columns and numerical pipeline to numerical columns (and boolean too)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, categorical),
            ("num", numeric_pipe, numeric + boolean),
        ]
    )
    return preprocessor, categorical, boolean, numeric


#def prepare_for_hgb(df):
    """Cast categorical columns to pandas 'category' dtype in place.

    HistGradientBoostingClassifier(categorical_features="from_dtype") then
    picks them up automatically, NaNs included, with no further encoding.
    """
#    categorical, boolean, numeric = get_feature_columns(df)
#    df = df.copy()
#    for c in categorical:
#        df[c] = df[c].astype("category")
#    for c in boolean:
#        df[c] = df[c].astype(float)
#    feature_cols = categorical + boolean + numeric
#    return df[feature_cols], categorical


####################
#Actual script
####################
#Once the data is in the correct shape, here we train the random forest

#-----------------
#Define the hyperparameter grid
#----------------
#A hyperparameter is a setting you choose before training a model. Grid search tries different combinations
# of these settings to find the best-performing configuration

PARAM_GRID = {
    "model__n_estimators": [300], #Number of decision trees in the forest
    "model__max_depth": [None, 20, 30], #Maximum depth of each tree; None allows growth until other stopping conditions apply
    "model__min_samples_leaf": [1, 5, 20], #Minimum number of training samples required in a leaf
    "model__max_features": ["sqrt", 0.3], #Number or proportion of features considered when searching for the best split
}

#----------------------
#How many configurations will be tested?
#---------------------
# You have:
# 1 choice for n_estimators.
# 3 choices for max_depth.
# 3 choices for min_samples_leaf.
# 2 choices for max_features.

# Therefore:

# 1x3x3x2=18
# There are 18 distinct hyperparameter combinations.

# With five-fold cross-validation, each combination is trained five times:

# 18x5=90
# Grid search therefore performs 90 cross-validation model fits, followed by one additional fit of the winning pipeline on the entire training dataset.
# That's 91 fits in total, assuming all fits complete successfully.

def main():
    #Load dataset
    learn = pd.read_parquet(OUTPUT_DIR / "learn_features.parquet")
    y = learn[TARGET_COL] #contains the correct outcomes the model must learn to predict.
    X = learn.drop(columns=[TARGET_COL]) #contains the input features used to make predictions.

    preprocessor, categorical, boolean, numeric = random_forest_preprocessor(X) #load function created previously
    print(f"{len(categorical)} categorical, {len(boolean)} boolean, {len(numeric)} numeric features")

    #Build the actual pipeline
    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )

# The pipeline has two stages:

# prep: preprocess the input data.

# model: train the Random Forest on the transformed data.

# When you call pipe.fit(X, y), scikit-learn fits the preprocessing steps, transforms the input features, and then fits the classifier.
# When you use the fitted pipeline for predictions later, it automatically applies the same fitted preprocessing.
# That is why you save the entire pipeline rather than just the Random Forest classifier.


    #Cross validation
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Five-fold stratified cross-validation splits the dataset into five parts.
# Each part serves as the validation set once, while the other four parts are used for training.
# The stratification attempts to preserve the proportion of each target class in each fold. 
# This is particularly helpful if your target is imbalanced.

    #Running gridsearchCV
    search = GridSearchCV(
        pipe,
        PARAM_GRID,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=2,
    )

#GridSearchCV tests each of the 18 parameter configurations across all five folds. (18x5=90)


    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start
    print(f"Grid search done in {elapsed:.1f}s")
    print("Best params:", search.best_params_)
    print("Best CV ROC AUC:", search.best_score_)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(OUTPUT_DIR / "rf_cv_results.csv", index=False)

    joblib.dump(search.best_estimator_, MODELS_DIR / "random_forest.joblib")

    with open(OUTPUT_DIR / "rf_best_params.json", "w") as f:
        json.dump(
            {"best_params": search.best_params_, "best_cv_roc_auc": search.best_score_},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
