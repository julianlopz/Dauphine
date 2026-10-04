"""Raw CSV loaders.

All identifier / code columns are read as strings on purpose: several of
them (INSEE_CODE, department codes, ...) contain leading zeros that pandas
would silently strip if the column were parsed as an integer (e.g. the
commune "01001" would become 1001), see section 1.2 of the project subject.
"""

import pandas as pd

from . import config

# Columns that must never be inferred as numeric.
_STRING_COLS = {
    "Pkey",
    "INSEE_CODE",
    "dep",
    "Job_dep",
    "former_dep",
    "Reg",
}


def _read(filename, **kwargs):
    # Plain `str` (object dtype with float("nan") for missing cells), not
    # pandas' nullable "string" dtype: the latter's pd.NA trips up some
    # scikit-learn internals (SimpleImputer, HistGradientBoosting) that
    # expect ordinary NaN-marked missing values.
    dtype = {c: str for c in _STRING_COLS}
    dtype.update(kwargs.pop("dtype", {}))
    return pd.read_csv(config.DATA_DIR / filename, dtype=dtype, **kwargs)


def load_persons(split):
    """Main table: one row per individual (learn has the target, test doesn't)."""
    return _read(f"{split}_dataset.csv")


def load_emp_contract(split):
    return _read(f"{split}_dataset_EMP_CONTRACT.csv")


def load_job(split):
    return _read(f"{split}_dataset_job.csv")


def load_retired_former(split):
    return _read(f"{split}_dataset_retired_former.csv")


def load_retired_jobs(split):
    return _read(f"{split}_dataset_retired_jobs.csv")


def load_retired_pension(split):
    return _read(f"{split}_dataset_retired_pension.csv")


def load_club(split):
    return _read(f"{split}_dataset_club.csv")


def load_city_adm():
    return _read("city_adm.csv")


def load_city_loc():
    return _read("city_loc.csv")


def load_city_pop():
    return _read("city_pop.csv")


def load_departments():
    return _read("departments.csv")


def load_regions():
    return _read("regions.csv")


def load_code_club():
    return _read("code_club.csv")


def load_job_description_map():
    return _read("code_Job_description_map.csv")
