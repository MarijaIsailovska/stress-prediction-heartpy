# Stress Prediction with HeartPy HRV

Decision-support research prototype for **mission withdrawal** of soldiers and
firefighters, based on physiological overload detection from wearable cardiac
signals (ECG / PPG) and HeartPy HRV features.

University course: **Intelligent Systems**.

## Research questions

| ID | Question |
|----|----------|
| RQ1 | Can HRV features classify physical stress vs rest from ECG? |
| RQ2 | ECG (chest) vs PPG (wrist) — which gives better classification? |
| RQ3 | Which HRV features are strongest predictors regardless of sensor? |
| EXTRA | Train on EPHNOGRAM ECG → test on Wrist PPG (cross-sensor) |

Methodology decisions are documented in [`research_notes.md`](research_notes.md).

## Datasets

Download and place files as follows (PhysioNet account may be required):

1. **EPHNOGRAM** — https://physionet.org/content/ephnogram/1.0.0/  
   → `data/raw/ephnogram/` (`ECGPCG00XY.mat` or `.dat`/`.hea`, plus spreadsheet)
2. **Wrist PPG During Exercise** — https://physionet.org/content/wrist/1.0.0/  
   → `data/raw/wrist/` (WFDB `.dat`/`.hea`)

Quality-filtered EPHNOGRAM IDs and labels live in `config/ephnogram_records.py`.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Pipeline

```bash
python scripts/01_process_ephnogram.py   # ECG → HRV CSV
python scripts/02_process_wrist.py       # PPG → HRV CSV
python scripts/03_train_evaluate.py      # LOSO + SMOTE (RQ1, RQ2)
python scripts/04_cross_sensor_test.py   # ECG→PPG transfer
python scripts/05_feature_importance.py  # RF importance + SHAP (RQ3)
```

Outputs:

- Features: `data/processed/*.csv`
- Metrics / JSON: `results/metrics/`
- Figures: `results/figures/`
- Fitted models: `results/models/`

## Project layout

```
stress_prediction_heartpy/
├── config/           # paths, record lists, labels
├── data/raw/         # PhysioNet downloads (not committed)
├── data/processed/   # HRV feature tables
├── src/              # loaders, preprocessing, HeartPy, ML
├── scripts/          # runnable experiment pipeline
├── results/          # metrics, figures, models
├── research_notes.md
└── requirements.txt
```

## Models & evaluation

- **Required:** Random Forest, SVM (RBF), k-NN  
- **Extra:** Logistic Regression, soft Voting (RF+SVM+k-NN)  
- **CV:** Leave-One-Subject-Out (LOSO)  
- **Imbalance:** SMOTE on each **train** fold only  
- **Metrics:** Accuracy, F1-macro, ROC-AUC, confusion matrix, SHAP

## HRV features (HeartPy `hp.process`)

BPM, IBI, SDNN, SDSD, RMSSD, pNN20, pNN50, HR_MAD, LF, HF, LF/HF,
Breathing Rate, SD1, SD2, S

## Citation / data

Please cite the original PhysioNet datasets and HeartPy if you publish results
from this pipeline. See `research_notes.md` for methodological references.
