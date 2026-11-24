"""Build one feature table per split (learn / test) by enriching the main
person-level file with employment, retirement, sport-club and geographic
information, as suggested in section 2 of the project subject.

Design choices (see report for the full discussion):

- Every side table is joined with a left join on Pkey, and its absence is
  kept as an explicit boolean flag (``has_job_detail``, ``has_club``, ...)
  rather than being treated as an ordinary missing value. Per section 1.2
  of the subject, most of these "holes" are *structural* (e.g. a person
  who is not a wage-earner simply has no row in learn_dataset_job.csv) and
  imputing them would be misleading. Genuinely sparse fields inside a
  present block (e.g. ``working_hours``) are left as NaN and handled by
  the model's own missing-value support / the imputer in the sklearn
  pipeline.
- High-cardinality codes are folded into the coarser official
  nomenclatures provided with the data (Job_description: 408 codes -> its
  N2 (29) and N1 (6) aggregation levels from code_Job_description_map.csv).
- A handful of elementary consistency checks (section 1.2) are logged so
  that the data audit in the report can reference concrete numbers; the
  main dataset is treated as more reliable than EMP_CONTRACT, which is in
  turn more reliable than the detailed job description, so we never let a
  side table overwrite a value coming from a more reliable one.
"""

import numpy as np
import pandas as pd

from . import config, data_loading as dl


def _flag(df, col):
    return df[col].notna()


def _add_job_description_levels(df, code_col, map_df, prefix):
    """Map a PCS-ESE N3 job-description code to its N2/N1 aggregates.

    The raw N3 code (`code_col`) has 408 possible values, far too many for
    most models (see section 2, recommendation on modality reduction). We
    keep the two coarser official levels instead and drop the raw code.
    """
    merged = df.merge(
        map_df.rename(columns={"N3": code_col, "N2": f"{prefix}_N2", "N1": f"{prefix}_N1"}),
        on=code_col,
        how="left",
    )
    return merged.drop(columns=[code_col])


def _merge_geography(persons):
    city_adm = dl.load_city_adm()
    city_loc = dl.load_city_loc()
    city_pop = dl.load_city_pop()
    departments = dl.load_departments()
    regions = dl.load_regions()

    geo = (
        city_adm.merge(city_loc, on="INSEE_CODE", how="left")
        .merge(city_pop, on="INSEE_CODE", how="left")
        .merge(departments, on="dep", how="left")
        .merge(regions, on="Reg", how="left")
    )

    out = persons.merge(geo, on="INSEE_CODE", how="left")

    out["distance_to_paris_km"] = (
        np.sqrt(
            (out["X"] - config.PARIS_LAMBERT93_X) ** 2
            + (out["Y"] - config.PARIS_LAMBERT93_Y) ** 2
        )
        / 1000.0
    )
    out["log_inhabitants"] = np.log1p(out["INHABITANTS"])
    return out


def _merge_club(persons, split):
    club = dl.load_club(split)
    code_club = dl.load_code_club().rename(
        columns={
            "Code": "club",
            "Categorie": "club_category",
            "Nom fédération": "club_federation",
            "Nom catégorie": "club_category_name",
        }
    )
    club = club.merge(code_club, on="club", how="left")
    out = persons.merge(
        club[["Pkey", "club_category", "club_federation"]], on="Pkey", how="left"
    )
    out["has_club"] = _flag(out, "club_category")
    return out


def _consistency_checks(df, split):
    """Log a few elementary cross-file consistency checks (section 1.2).

    These are diagnostics only: we never overwrite the main dataset or
    EMP_CONTRACT with information coming from a less reliable table.
    """
    print(f"[{split}] consistency checks")

    # A person with detailed job data should also have an EMP_CONTRACT row
    # (job details only exist for currently salaried people).
    mismatch = df["has_job_detail"] & ~df["has_emp_contract"]
    print(f"  job detail without EMP_CONTRACT: {mismatch.sum()}")

    # Someone flagged as "in employment" (Act == tact1-1) is expected to
    # have an EMP_CONTRACT row; deviations are kept as-is (Act, the main
    # dataset, is the most reliable source) but reported.
    if "Act" in df.columns:
        employed = df["Act"] == "tact1-1"
        mismatch = employed & ~df["has_emp_contract"]
        print(f"  Act=employed without EMP_CONTRACT: {mismatch.sum()}")

    # A retired person (Act == tact2-1) should have a retired_former row.
    if "Act" in df.columns:
        retired = df["Act"] == "tact2-1"
        mismatch = retired & ~df["has_retired_former"]
        print(f"  Act=retired without retired_former data: {mismatch.sum()}")


def build_dataset(split):
    """Return the fully merged, feature-engineered table for `split`.

    `split` is either "learn" or "test". The returned dataframe always
    starts with Pkey and, for "learn" only, also carries `target`.
    """
    assert split in ("learn", "test")

    df = dl.load_persons(split)

    # --- employment (all people currently working, incl. self-employed) --
    emp = dl.load_emp_contract(split)
    df = df.merge(emp, on="Pkey", how="left")
    df["has_emp_contract"] = _flag(df, "EMP_CONTRACT")

    # --- detailed job description (wage-earners only) ---------------------
    job = dl.load_job(split)
    job_map = dl.load_job_description_map()
    job = _add_job_description_levels(job, "Job_description", job_map, prefix="Job_description")
    df = df.merge(job, on="Pkey", how="left")
    df["has_job_detail"] = _flag(df, "Eco_sect")

    # --- last job before retirement (type of former employment) -----------
    retired_former = dl.load_retired_former(split)
    df = df.merge(retired_former, on="Pkey", how="left")
    df["has_retired_former"] = _flag(df, "Former_job_42")

    # --- last job before retirement (detailed, wage-earners only) ---------
    retired_jobs = dl.load_retired_jobs(split)
    retired_jobs = _add_job_description_levels(
        retired_jobs, "Job_description", job_map, prefix="Former_job_description"
    )
    df = df.merge(retired_jobs, on="Pkey", how="left", suffixes=("", "_former_job"))
    df["has_retired_job_detail"] = _flag(df, "Eco_sect_former_job")

    # --- pension amount -----------------------------------------------------
    pension = dl.load_retired_pension(split)
    df = df.merge(pension, on="Pkey", how="left")
    df["has_pension"] = _flag(df, "retirement_pay")

    # --- sport club membership ----------------------------------------------
    df = _merge_club(df, split)

    # --- geography ------------------------------------------------------------
    df = _merge_geography(df)

    _consistency_checks(df, split)

    return df


def save_datasets():
    for split in ("learn", "test"):
        df = build_dataset(split)
        out_path = config.OUTPUT_DIR / f"{split}_features.parquet"
        df.to_parquet(out_path, index=False)
        print(f"{split}: {df.shape} -> {out_path}")


if __name__ == "__main__":
    save_datasets()
