# ---------------------------------------------------------------------------
# config.py — Central configuration for all hyperparameters and constants
# ---------------------------------------------------------------------------

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Dataset URLs
DATA_URL = "http://db.csail.mit.edu/labdata/data.txt.gz"
LOCS_URL = "http://db.csail.mit.edu/labdata/mote_locs.txt"
DATA_FILE = os.path.join(RAW_DIR, "data.txt.gz")
LOCS_FILE = os.path.join(RAW_DIR, "mote_locs.txt")

# Cleaning thresholds
TEMP_MIN = -10.0
TEMP_MAX = 60.0
VALID_MOTE_RANGE = (1, 54)
MAX_GAP_FILL = 3  # forward-fill up to 3 consecutive NaNs

# Experiment
N_VALUES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_REPS = 500
TEST_FRACTION = 0.20
RANDOM_SEED = 42

# Model
SVM_C = 1.0
SVM_MAX_ITER = 10000

# Sensor selection thresholds
MISSING_FRAC_MAX_REF = 0.05
MISSING_FRAC_MAX_FEAT = 0.10
CORR_A_MIN = 0.85
CORR_B_HIGH_A_MIN = 0.95
CORR_B_HIGH_R_MIN = 0.80
CORR_B_MID_R_RANGE = (0.50, 0.80)
CORR_B_MID_A_MAX = 0.70
CORR_B_LOW_R_MAX = 0.40
CORR_B_LOW_A_MAX = 0.50
