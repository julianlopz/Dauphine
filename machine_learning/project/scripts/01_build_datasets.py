"""Step 1: merge every raw CSV into one feature table per split.

Run once (or whenever a raw file / the feature engineering changes):

    python scripts/01_build_datasets.py

Produces outputs/learn_features.parquet and outputs/test_features.parquet,
consumed by every script downstream. Building this once and caching it to
parquet avoids re-running ~10 merges every time a model script is launched.
"""

from common import features

if __name__ == "__main__":
    features.save_datasets()
