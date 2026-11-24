"""Column groups and preprocessing pipelines shared by every model script.

Two preprocessing strategies are provided:

- `random_forest_preprocessor()`: classic impute + one-hot pipeline, needed
  because RandomForestClassifier cannot handle NaN or strings natively.
- `hgb_preprocessor()`: much lighter pipeline that only turns categorical
  columns into pandas "category" dtype and leaves everything else (incl.
  NaNs) untouched, because HistGradientBoostingClassifier natively supports
  missing values and categorical features. This matters here: most of our
  missingness is *structural* (section 1.2 of the subject), and letting the
  boosting model learn its own split for "missing" is exactly the kind of
  handling the subject warns a blind imputation would get wrong.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder

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
        raise ValueError(f"Unclassified columns, update common/modeling.py: {leftover}")
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
