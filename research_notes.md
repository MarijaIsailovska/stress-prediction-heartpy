# Research Notes — Physiological Overload Detection via HRV

Decision-support system for mission withdrawal of soldiers and firefighters,
based on physiological overload detection using wearable sensors and HRV analysis.

Intelligent Systems (university course) — methodology log for the paper.

---

## SECTION 1 — Dataset Selection

**Decision:** EPHNOGRAM (ECG) + Wrist PPG During Exercise (PPG)

**Justification:** No public dataset exists with soldiers/firefighters under
operational load. These two PhysioNet corpora are the closest available proxies:
physically active adult subjects, continuous cardiac signals from wearable-
relevant modalities, and clear rest vs. exertion protocol labels suitable for
supervised classification of physiological overload.

| Dataset | Modality | Rate | Subjects | Role |
|---------|----------|------|----------|------|
| [EPHNOGRAM 1.0.0](https://physionet.org/content/ephnogram/1.0.0/) | Chest ECG (+PCG) | 8000 Hz → 500 Hz | 24 healthy adult males (23–29 y) | Primary ECG / RQ1 |
| [Wrist PPG During Exercise 1.0.0](https://physionet.org/content/wrist/1.0.0/) | Wrist PPG | 256 Hz | 8 subjects | PPG comparison / RQ2 + cross-sensor |

**Reference:** Cursor analysis based on PhysioNet database review (2025)

**Paper section:** Methods — Data Sources

---

## SECTION 2 — Quality Filtering

**Decision:** Use only recordings with ECG Notes = Good and 30 min duration.

**Method:** Filter candidate EPHNOGRAM files via `ECGPCGSpreadsheet.csv`
(quality notes and duration fields). Retained record IDs (XY in `ECGPCG00XY`):

- **REST (label=0):** 10, 11, 13, 14, 15, 16, 21, 22, 23  
- **STRESS (label=1):** 01, 25, 27, 29, 30, 32, 33, 34, 36, 38, 47, 52, 55, 61, 62, 64, 66, 67, 68  

**Total:** 28 recordings × ~30 non-overlapping 60 s windows ≈ 840 HRV samples
(before HeartPy failure discard).

**Reference:**
- https://arxiv.org/html/2506.10212v1 (used the same 28 recordings)
- https://arxiv.org/pdf/2410.19667 (23 stress-test recordings, same quality criteria)

**Paper section:** Methods — Data Preprocessing

---

## SECTION 3 — Labeling Methodology

**Decision:** Protocol-based labeling (whole recording = one label).

- `label = 0`: Rest recordings  
- `label = 1`: Bruce treadmill / bicycle stress-test recordings  

Wrist PPG multi-class activities are mapped to the same binary scheme:
`rest → 0`; `walk / run / bike → 1` (physical exertion as stress proxy).

**Justification:** Periodically labeled (protocol) data achieves significantly
higher detection accuracy than self-report affective labels for stress/affect
detection from physiological signals.

**Reference:**
- Schmidt et al. (2018) — WESAD  
- Vos et al. (2023) systematic review — *“models trained on periodically-labeled
  data achieved significantly higher levels of detection accuracy”*

**Paper section:** Methods — Labeling Strategy

---

## SECTION 4 — Signal Processing

**Decision:** Downsample EPHNOGRAM 8000 → 500 Hz; bandpass filter; 60 s
non-overlapping windows.

| Step | ECG (EPHNOGRAM) | PPG (Wrist) |
|------|-----------------|-------------|
| Resample | 8000 → 500 Hz (polyphase) | native 256 Hz |
| Bandpass | 0.5–45 Hz | 0.75–3.5 Hz |
| Window | 60 s, overlap = 0 | 60 s, overlap = 0 |
| Discard | Windows where HeartPy raises | same |

**Justification:** A minimum of ~60 s is required for stable estimation of
frequency-domain HRV features (LF, HF, LF/HF) under Task Force guidelines.

**Reference:**
- Task Force of the European Society of Cardiology (1996) — HRV standards  
- Shaffer & Ginsberg (2017) *Frontiers in Public Health*
  doi:10.3389/fpubh.2017.00258

**Paper section:** Methods — Signal Processing

---

## SECTION 5 — Feature Extraction

**Decision:** All 15 HeartPy features via `hp.process()`.

| # | Feature | Domain |
|---|---------|--------|
| 1 | BPM | Time |
| 2 | IBI | Time |
| 3 | SDNN | Time |
| 4 | SDSD | Time |
| 5 | RMSSD | Time |
| 6 | pNN20 | Time |
| 7 | pNN50 | Time |
| 8 | HR_MAD | Time |
| 9 | LF | Frequency |
| 10 | HF | Frequency |
| 11 | LF/HF | Frequency |
| 12 | Breathing Rate | Derived |
| 13 | SD1 | Poincaré |
| 14 | SD2 | Poincaré |
| 15 | S | Poincaré |

Failed windows (exceptions or non-finite measures) are discarded and never
imputed into the feature matrix.

**Reference:**
- van Gent et al. (2019) HeartPy — Heart Rate Analysis for Human Factors,
  doi:10.1016/j.trf.2019.09.010  
- Shaffer & Ginsberg (2017) for LF/HF frequency-band definitions

**Paper section:** Methods — Feature Extraction

---

## SECTION 6 — ML Evaluation Strategy

**Decision:** Leave-One-Subject-Out (LOSO) cross-validation.

**Models (required):** Random Forest, SVM (RBF), k-NN  
**Models (extra):** Logistic Regression; soft Voting Ensemble (RF + SVM + k-NN)

**Metrics:** Accuracy, F1-macro, ROC-AUC, Confusion Matrix; Feature Importance + SHAP

**Justification:** LOSO prevents data leakage between windows from the same
subject. A random train/test split would leak subject-specific HRV baselines
into the test set and inflate apparent generalization. SMOTE is applied on each
**training fold only**, never on the held-out subject (test fold).

**Reference:**
- Gjoreski et al. (2016)  
- Vos et al. (2023)  
- Chawla et al. (2002) — SMOTE

**Paper section:** Methods — Model Evaluation

---

## SECTION 7 — Cross-Sensor Generalization Experiment

**Decision:** Train on EPHNOGRAM (ECG chest) → test on Wrist PPG.

**Research question:** Can a model trained on clinical-grade chest ECG
generalize to wrist PPG as used in soldier/firefighter smartwatches?

Shared representation: the same 15 HeartPy HRV features and binary rest/stress
labels, enabling direct transfer without modality-specific deep encoders.

**Reference:**
- Vos et al. (2023) — cross-dataset generalizability  
- Schmidt et al. (2022) — cross-dataset ECG stress detection

**Paper section:** Results + Discussion

---

## Research Questions (summary)

| ID | Question | Primary data |
|----|----------|--------------|
| RQ1 | Can HRV features classify physical stress vs rest from ECG? | EPHNOGRAM |
| RQ2 | ECG (chest) vs PPG (wrist) — which yields better classification? | Both (LOSO each) |
| RQ3 | Which HRV features are strongest predictors regardless of sensor? | Combined + SHAP |
| EXTRA | Train ECG → test PPG (cross-sensor generalization) | Both |

---

## Pipeline checklist

1. Place raw PhysioNet files under `data/raw/ephnogram/` and `data/raw/wrist/`
2. `python scripts/01_process_ephnogram.py`
3. `python scripts/02_process_wrist.py`
4. `python scripts/03_train_evaluate.py`   → RQ1, RQ2
5. `python scripts/04_cross_sensor_test.py` → EXTRA
6. `python scripts/05_feature_importance.py` → RQ3
