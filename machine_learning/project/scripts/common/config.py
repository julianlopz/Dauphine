"""Shared paths and constants for the whole pipeline.

Expected layout (unchanged if you just unzip the submission archive):

    <project_root>/
        project-7-files/   <- raw CSVs provided for the project
        scripts/           <- this file lives in scripts/common/
        outputs/           <- everything the pipeline writes (created automatically)
"""

from pathlib import Path

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
