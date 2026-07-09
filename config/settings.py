"""
Central configuration: paths, sampling rates, window params, feature list.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project roots
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

EPHNOGRAM_RAW_DIR = RAW_DIR / "ephnogram"
WRIST_RAW_DIR = RAW_DIR / "wrist"

# Processed feature CSVs
EPHNOGRAM_FEATURES_CSV = PROCESSED_DIR / "ephnogram_hrv_features.csv"
WRIST_FEATURES_CSV = PROCESSED_DIR / "wrist_hrv_features.csv"

# Spreadsheet shipped with EPHNOGRAM (place under raw/ephnogram/)
EPHNOGRAM_SPREADSHEET = EPHNOGRAM_RAW_DIR / "ECGPCGSpreadsheet.csv"

# ---------------------------------------------------------------------------
# Sampling rates
# ---------------------------------------------------------------------------
EPHNOGRAM_FS_ORIGINAL = 8000  # Hz
EPHNOGRAM_FS_TARGET = 500  # Hz (downsample before HeartPy)
WRIST_FS = 256  # Hz (PPG)

# ---------------------------------------------------------------------------
# Filtering (HeartPy bandpass)
# ---------------------------------------------------------------------------
ECG_BANDPASS = (0.5, 45.0)  # Hz
PPG_BANDPASS = (0.75, 3.5)  # Hz

# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 60
WINDOW_OVERLAP = 0.0  # non-overlapping

# ---------------------------------------------------------------------------
# HeartPy HRV feature names (order matches hp.process() / working_data keys)
# ---------------------------------------------------------------------------
HRV_FEATURES = [
    "bpm",
    "ibi",
    "sdnn",
    "sdsd",
    "rmssd",
    "pnn20",
    "pnn50",
    "hr_mad",
    "lf",
    "hf",
    "lf/hf",
    "breathingrate",
    "sd1",
    "sd2",
    "s",
]

# Friendly column names used in processed CSVs
HRV_FEATURE_COLUMNS = [
    "BPM",
    "IBI",
    "SDNN",
    "SDSD",
    "RMSSD",
    "pNN20",
    "pNN50",
    "HR_MAD",
    "LF",
    "HF",
    "LF_HF",
    "Breathing_Rate",
    "SD1",
    "SD2",
    "S",
]

# Map HeartPy measure keys → CSV column names
HRV_KEY_TO_COLUMN = dict(zip(HRV_FEATURES, HRV_FEATURE_COLUMNS))

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABEL_REST = 0
LABEL_STRESS = 1

# Metadata columns stored alongside features
META_COLUMNS = ["record_id", "subject_id", "window_idx", "label", "sensor"]

# ---------------------------------------------------------------------------
# ML defaults
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
SMOTE_K_NEIGHBORS = 5
N_JOBS = -1

# Classifier hyperparams (sensible defaults for LOSO)
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}
SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "class_weight": "balanced",
    "probability": True,
    "random_state": RANDOM_STATE,
}
KNN_PARAMS = {
    "n_neighbors": 5,
    "weights": "distance",
    "n_jobs": N_JOBS,
}
LR_PARAMS = {
    "max_iter": 2000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}
