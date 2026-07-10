# Research Notes — Physiological Overload Detection via HRV

Decision-support system for mission withdrawal of soldiers and firefighters,
based on physiological overload detection using wearable sensors and HRV analysis.

Intelligent Systems (university course) — methodology log for the paper.

> **Detailed Methods drafts** (evidence-tagged decisions, advantages/disadvantages,
> alternatives, uncertainty markers) live under [`notes/`](notes/README.md).
> This file remains a high-level overview. Prefer **appending** to `notes/` over
> rewriting historical decisions here.

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

---

## Preprocessing Pipeline — EPHNOGRAM — 2026-07-09

Implemented in `scripts/01_process_ephnogram.py`. Output:
`data/processed/ephnogram_features.csv`. Cohort: **25** usable recordings
(9 rest + 16 stress; IDs 55/61/62 excluded — unavailable locally).

Detailed evidence-tagged notes: `notes/04_signal_processing.md`,
`notes/05_feature_extraction.md` (appended decisions).

### Loading
- **Decision:** `wfdb.rdrecord` from `data/raw/ephnogram/WFDB/ECGPCG00XY`; ECG = channel 0; use header `fs`.
- **Justification:** Official PhysioNet WFDB layout for EPHNOGRAM.
- **Reference:** PhysioNet EPHNOGRAM 1.0.0 dataset documentation (`dataset-docs`).
- **Alternatives:** MATLAB `.mat` load; WFDB via `rdsamp`.
- **Limitations:** Assumes ECG is always channel 0 — verify against `.hea` signal names if channel order changes.

### Downsampling (8000 → 500 Hz)
- **Decision:** `scipy.signal.resample_poly` to 500 Hz before HeartPy.
- **Justification:** Peak-detection HRV does not benefit from 8000 Hz; 500 Hz is within the practical range used with HeartPy and cuts compute ~16×.
- **Reference:** van Gent et al. (2019), HeartPy / noisy PPG–oriented HR analysis literature (doi:10.1016/j.trf.2019.09.015) — sampling adequacy for peak detection (`heartpy-docs` / `peer-reviewed`; exact “100–500 Hz optimal” wording **[NEEDS VERIFICATION]** against the paper text).
- **Alternatives:** Keep 8000 Hz; downsample to 250 Hz; anti-alias FIR then decimate.
- **Limitations:** Any resampling can slightly shift peak timing vs native 8000 Hz.

### Bandpass (0.5–45 Hz)
- **Decision:** `hp.filter_signal(..., cutoff=[0.5, 45.0], sample_rate=500, filtertype='bandpass')`.
- **Justification:** High-pass ~0.5 Hz reduces baseline wander; low-pass ~45 Hz reduces EMG/noise while retaining QRS content for R-peak detection.
- **References:** Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, 32(3), 230–236. Sörnmo, L., & Laguna, P. (2005). *Bioelectrical Signal Processing in Cardiac and Neurological Applications*. Elsevier (`peer-reviewed` / textbook).
- **Alternatives:** 5–15 Hz QRS emphasis; 0.5–40 Hz; notch 50/60 Hz only.
- **Limitations:** Fixed cutoffs may be suboptimal for unusual morphologies or strong motion.

### Segmentation (60 s, non-overlapping)
- **Decision:** Fixed 60 s windows, overlap = 0; discard incomplete tail.
- **Justification:** Classical short-term HRV guidance treats ~60 s as a practical lower bound when frequency-related / respiratory estimates are of interest; aligns breathing-rate estimation needs.
- **References:** Task Force of the ESC and NASPE (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation*, 93(5), 1043–1065. Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. *Frontiers in Public Health*. doi:10.3389/fpubh.2017.00258.
- **Alternatives:** 5 min Task Force short-term; 50% overlap; ultra-short windows.
- **Limitations:** Non-overlap reduces N; 60 s may still mix Bruce stages within one stress recording.

### HRV features (HeartPy)
- **Decision:** `hp.process(segment, sample_rate=500, calc_freq=True)`; export bpm, ibi, sdnn, rmssd, sdsd, pnn20, pnn50, sd1, sd2, breathing_rate; **HR_MAD** = median absolute deviation of `wd['RR_list']`.
- **Justification:** Compact time-domain + Poincaré + respiratory estimate set for classical ML; MAD is robust to outlier IBIs.
- **References:** van Gent et al. (2019) HeartPy. Antczak, K. (2017) — MAD/robust HRV context as cited in project brief (**full bibliographic details [NEEDS VERIFICATION]**). Shaffer & Ginsberg (2017) for metric definitions.
- **Alternatives:** NeuroKit2 / Kubios feature sets; omit breathing_rate; include LF/HF explicitly in the CSV (computed via `calc_freq=True` but not exported in this CSV schema).
- **Limitations:** Library defaults differ from clinical Kubios pipelines; breathing_rate is an estimate, not spirometry.

### Quality control
- **Decision:** Discard if HeartPy exception; beats &lt; 20; BPM ∉ [30, 220]; RMSSD &gt; 300 ms; any required measure non-finite.
- **Justification:** Remove physiologically implausible or artifact-dominated windows before ML.
- **Reference:** Clifford, G. D., Azuaje, F., & McSharry, P. E. (Eds.). (2006). *Advanced Methods and Tools for ECG Data Analysis*. Artech House (`peer-reviewed` / edited volume — exact chapter thresholds are **project-chosen** adaptations).
- **Alternatives:** Signal quality indices (SQI); looser BPM bounds; no RMSSD cap.
- **Limitations:** Thresholds can bias toward cleaner rest segments; stress exercise may have higher true RMSSD/BPM near bounds.

### Labeling
- **Decision:** Whole-recording protocol label — rest = 0, Bruce/bicycle stress test = 1 (`config/ephnogram_records.py`).
- **Justification:** Protocol labels are more consistent than self-report for physical-load proxies.
- **References:** Schmidt et al. (2018) WESAD; Vos et al. (2023) systematic review on stress detection — **bibliographic details [NEEDS VERIFICATION]** for exact quotations.
- **Alternatives:** Stage-wise Bruce labels; self-report affect.
- **Limitations:** Physical exertion ≠ operational/psychological stress; within-recording transitions ignored.

---

## Data Quality Corrections — 2026-07-09

Applied after the Signal Quality Investigation (Ask-mode diagnosis).
Cohort is now **24 recordings (8 rest + 16 stress)**.

### Removal of ECGPCG0014 from REST_RECORDS
- **Decision:** Exclude record `14` from the analysis cohort.
- **Justification:** PhysioNet `ECGPCGSpreadsheet.csv` lists ECG Notes =
  **Powerline noise** for ECGPCG0014 (Rest: laying on bed), not **Good**.
  The recording had previously been included under an incorrect “Good / 30 min”
  assumption; preprocessing kept **0/30** windows, consistent with severe
  interference for automated R-peak / HRV extraction.
- **Reference:** PhysioNet EPHNOGRAM 1.0.0 —
  https://physionet.org/content/ephnogram/1.0.0/ECGPCGSpreadsheet.csv
  (`dataset-docs`).
- **Code:** `config/ephnogram_records.py`
- **Limitations:** Exclusion is quality-metadata driven; if a cleaned re-recording
  existed it is not used here.

### Rest BPM > 120 filter (targets artifactual windows, e.g. ECGPCG0021)
- **Decision:** For windows with protocol `label=0` (rest), discard if
  HeartPy BPM **> 120**. Counted as rejection reason `rest_bpm_too_high`.
- **Justification:** Sustained HR > 120 bpm is physiologically implausible for
  the EPHNOGRAM rest protocols (sitting / laying). Investigation found a kept
  ECGPCG0021 window with BPM ≈ 176 and RMSSD ≈ 281 at rest — consistent with
  false R-peaks / noise rather than true resting tachycardia. The filter removes
  such survivors without relaxing global BPM/RMSSD caps used for stress windows.
- **Reference:** Protocol scenarios in `ECGPCGSpreadsheet.csv` (`dataset-docs`);
  filter threshold is a **project-implementation** QC rule informed by that
  investigation (not a Task Force numeric standard).
- **Alternatives considered:** Drop entire ECGPCG0021; raise global RMSSD
  strictness; ECG SQI before HeartPy.
- **Limitations:** A true pathological resting tachycardia (unexpected in this
  healthy young-adult cohort) would also be discarded; stress windows are
  unaffected by this rule.

### Rejection logging
- **Decision:** Script 01 now logs per-window reject reasons
  (`heartpy_exception`, `beats_too_few`, `bpm_out_of_range`, `rmssd_too_high`,
  `rest_bpm_too_high`, plus `nonfinite_or_empty_rr` for incomplete measures)
  per recording and in a global summary.
- **Code:** `scripts/01_process_ephnogram.py`

---

## QC Threshold Revision — 2026-07-09

### Problem
After Data Quality Corrections, **`rmssd_too_high` dominated rejections**
(104 / 124). Rest recordings were hit hardest (e.g. ECGPCG0011/0015: 25/30;
ECGPCG0021: 26/30; ECGPCG0023: 14/30). Discarding high-but-plausible RMSSD at
rest in young healthy adults risks removing valid vagal HRV rather than artifacts.

### Threshold change
| Rule | Old | New |
|------|-----|-----|
| Absolute RMSSD cap | Reject if RMSSD **> 300 ms** | Reject if RMSSD **> 400 ms** (extreme outlier only) |
| Relative artifact check | *(none)* | Reject if RMSSD **> 3 × IBI** (`rmssd_exceeds_ibi`) |

### Scientific justification
- **Absolute cap 400 ms:** Shaffer & Ginsberg (2017) discuss RMSSD as a
  short-term HRV metric with substantial inter-individual range; values in the
  tens to low hundreds of ms are commonly reported in healthy adults. A hard
  300 ms cut was **too conservative** for this young male rest cohort and was
  discarding many rest windows. Raising to **400 ms** retains rare but
  possible high-RMSSD rest segments while still flagging extreme outliers
  almost certainly driven by peak-detection failure.
  - Reference: Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart
    rate variability metrics and norms. *Frontiers in Public Health*.
    doi:10.3389/fpubh.2017.00258 (`peer-reviewed`). Exact numeric “20–200 ms
    normal range” wording should be checked against the paper tables before
    thesis citation — **[NEEDS VERIFICATION]** of the precise range statement
    used in the project brief.
- **RMSSD > 3 × IBI:** Successive-difference dispersion on this scale relative
  to mean RR is not physiologically coherent and indicates false/missed peaks.
  Clifford et al. (2006) emphasize artifact-aware ECG/HRV processing; the
  3×IBI rule is a **project-implementation** mathematical sanity check inspired
  by that quality-control mindset (not a verbatim Task Force formula).
  - Reference: Clifford, G. D., Azuaje, F., & McSharry, P. E. (Eds.). (2006).
    *Advanced Methods and Tools for ECG Data Analysis*. Artech House.

### Expected impact
- **REST window yield should increase** for 0011, 0015, 0021, 0023 where most
  rejects were RMSSD in (300, 400] ms.
- Windows with RMSSD still > 400 ms, or RMSSD > 3×IBI, remain discarded.
- Rest BPM > 120 and other QC rules unchanged.

### Observed impact (re-run 2026-07-09)
| Metric | Before (RMSSD>300) | After (RMSSD>400 + 3×IBI) |
|--------|--------------------|---------------------------|
| Windows kept | 596 | **607** (+11) |
| Rest / Stress | 142 / 454 | **142 / 465** |
| `rmssd_too_high` | 104 | **28** |
| `rest_bpm_too_high` | 7 | **72** |
| `rmssd_exceeds_ibi` | — | **0** |

**Finding:** Raising the RMSSD cap did **not** increase REST yield. Windows
previously counted as `rmssd_too_high` on 0011/0015/0021/0023 largely reappear
as `rest_bpm_too_high` (BPM > 120 at rest) — i.e. they combine inflated RMSSD
**and** implausible resting BPM, consistent with false-peak artifact rather than
healthy high vagal tone. Stress gained 11 windows (RMSSD in (300, 400] with
plausible exercise BPM). Dominant remaining rest reject reason is now
`rest_bpm_too_high`, not RMSSD.

### Code
`scripts/01_process_ephnogram.py` (`RMSSD_MAX_MS = 400`, `RMSSD_IBI_RATIO_MAX = 3`)

---

## EDA Visualizations — 2026-07-09

Script (not executed in this commit step): `scripts/04_eda_visualizations.py`  
Input: `data/processed/ephnogram_features.csv`  
Output directory: `results/figures/` (DPI=300; REST=`#2196F3`, STRESS=`#F44336`)

| Plot | File | What it shows | Paper section | Reference |
|------|------|---------------|---------------|-----------|
| 1 | `01_hrv_boxplots.png` | Side-by-side REST/STRESS boxplots for all 11 HRV features + Mann–Whitney stars | Results — Descriptive HRV | Mann & Whitney (1947); Shaffer & Ginsberg (2017) |
| 2 | `02_correlation_heatmap.png` | Pearson correlation matrices REST vs STRESS (multicollinearity) | Methods/Results — Feature structure | Pearson correlation; HRV collinearity context (Shaffer & Ginsberg 2017) |
| 3 | `03_poincare_plot.png` | SD1 vs SD2 scatter + 95% covariance ellipses | Results — Nonlinear / Poincaré HRV | Brennan et al. (2001) doi:10.1109/10.959330 |
| 4 | `04_sd1_sd2_ratio.png` | Violin of SD1/SD2 (ANS balance proxy), line at ratio=1 | Results — Autonomic balance proxy | Shaffer & Ginsberg (2017) *Front. Public Health* |
| 5 | `05_rmssd_bpm_scatter.png` | RMSSD vs BPM with REST/STRESS quadrant annotations | Introduction / Results — Intuitive separation | Shaffer & Ginsberg (2017) (RMSSD/HR context) |
| 6 | `06_pca_biplot.png` | PC1–PC2 scores + loading arrows for 11 features | Results — Multivariate separability | PCA (standard); HRV feature space exploration |
| 7 | `07_subject_profiles.png` | Per-subject mean BPM vs mean RMSSD (LOSO context) | Methods — Evaluation / LOSO rationale | Subject-wise dependence (see note 06) |
| 8 | `08_breathing_rate.png` | Breathing rate (breaths/min) by class + 12–20 /min guides | Results — Respiratory estimate | Task Force ESC/NASPE (1996) *Circulation* 93(5) |
| 9 | `09_class_distribution.png` | Windows per recording + overall REST/STRESS pie | Methods — Imbalance / SMOTE justification | Chawla et al. (2002) SMOTE (usage rationale) |
| 10 | `10_pairplot.png` | Pairwise bpm/RMSSD/SD1/SD2 with KDE diagonals | Results — Pairwise separability | Exploratory multivariate EDA |

### Interpretation guidance

| Plot | Look for | Expected (HRV / exertion literature) | Unexpected finding would suggest |
|------|----------|--------------------------------------|----------------------------------|
| 1 | Direction and significance of REST vs STRESS shifts | Higher BPM, often lower vagal indices (e.g. RMSSD) under exertion — **not guaranteed** for all features in this protocol-labeled set | Non-significant or reversed RMSSD (already noted in QC notes) → artifact, non-stationarity, or construct mismatch |
| 2 | Block structure / changing r under stress | Strong bpm–ibi anti-correlation; RMSSD–SD1 near-redundancy | Correlation collapse/inflation under stress → noise or regime change |
| 3 | Ellipse separation along SD1/SD2 | Rest: relatively higher short-term (SD1) contribution; stress: shift with HR | Complete overlap → weak Poincaré discriminability |
| 4 | SD1/SD2 distribution vs line at 1 | Higher ratio at rest (parasympathetic proxy) vs lower under stress | Reversed ratio → revisit peak quality / SD2 instability |
| 5 | Mass in “REST zone” vs “STRESS zone” | Rest lower-left/top-left; stress bottom-right | Many rest points in stress zone → residual false peaks (see rest BPM QC) |
| 6 | Class clouds + loading directions | Partial linear separation along HR-related loadings | Total overlap → need nonlinear models or better features |
| 7 | Spread of subject means | Large between-subject gaps → LOSO harder than random split | One subject dominating a class cloud → leakage risk if not LOSO |
| 8 | Breaths/min vs 12–20 band | Mild increase under exertion possible | Extreme rates → HeartPy breathing estimate failure |
| 9 | Uneven bars / pie imbalance | More stress windows (16 vs 8 recordings) | Extreme single-recording dominance → reweight or drop |
| 10 | Off-diagonal class mixing | Clearer separation on bpm than on SD2 alone | No pairwise separation → multivariate / nonlinear methods |

**Evidence tags:** plot design is largely `project-implementation`; physiological expectations cite `peer-reviewed` sources above. Exact numeric norms (e.g. RMSSD ranges) remain subject to **[NEEDS VERIFICATION]** against primary tables when drafting the paper.

**Code:** `scripts/04_eda_visualizations.py`  
**Note:** Filename `04_eda_visualizations.py` coexists with `scripts/04_cross_sensor_test.py` (cross-sensor experiment). Run EDA explicitly by path.

---

## EDA Findings and Interpretation — 2026-07-09

Based on figures in `results/figures/` generated by `scripts/04_eda_visualizations.py`
from `data/processed/ephnogram_features.csv` (**n = 607**; REST = 142, STRESS = 465;
24 recordings). Quantitative checks (Mann–Whitney U, correlations, PCA variance)
were recomputed from the same CSV to support interpretation.

**Literature framing (expected under physical exertion / sympathetic shift):**
higher BPM / lower IBI; often lower vagal time-domain indices (RMSSD, related
SD1); possible SD1/SD2 decrease as a rough autonomic-balance proxy
(`peer-reviewed`: Shaffer & Ginsberg, 2017; Task Force ESC/NASPE, 1996).
Protocol labels here are **physical stress-test / bike**, not psychometric stress.

---

### Plot 1 — `01_hrv_boxplots.png`

**Observation:** BPM and IBI show the clearest class separation (REST BPM
≈ 68 ± 7; STRESS ≈ 128 ± 30; Mann–Whitney *p* ≪ 0.001). pNN20 is higher at
rest (*p* ≪ 0.001). SD2 is higher at rest (*p* ≪ 0.001). HR_MAD is higher
under stress (*p* ≪ 0.001). **RMSSD, SDSD, and SD1 are not significant**
(RMSSD *p* ≈ 0.077; stress mean RMSSD **higher** than rest: ≈ 85 vs 48 ms)
with a heavy stress right-tail / outliers. Breathing rate distributions nearly
overlap (*p* ≈ 0.58).

**Expected?** Partially. BPM↑ / IBI↓ under exertion is **expected**. Higher
mean RMSSD/SD1 under stress is **counterintuitive** vs classic “vagal withdrawal
→ lower RMSSD” narratives (Shaffer & Ginsberg, 2017).

**Unexpected / flags:** Non-significant RMSSD with stress > rest mean; extreme
stress outliers on RMSSD/SD1/HR_MAD; breathing rate non-discriminative.

**Paper wording (suggestion):** Report BPM/IBI as primary univariate separators;
state that short-term variability indices (RMSSD/SD1) did **not** show the
textbook decrease under protocol stress, discuss artifact / non-stationarity /
RSA under Bruce–bike as competing explanations, and avoid claiming “validated
psychological stress” from these labels.

---

### Plot 2 — `02_correlation_heatmap.png`

**Observation:** At REST, BPM–IBI *r* ≈ −0.99; BPM correlates **negatively**
with variability features (e.g. BPM–RMSSD ≈ −0.78). Under STRESS, BPM–IBI
remains strongly negative (≈ −0.97), but BPM–variability correlations **flip
sign** (BPM–RMSSD ≈ **+0.69**). RMSSD–SD1 remains near-perfect (≈ 1.00 rest,
≈ 0.99 stress). Breathing is weakly correlated at rest and moderately positive
with several features under stress. Strong multicollinearity among SDNN / RMSSD
/ SDSD / SD1 / SD2 / pNN* blocks.

**Expected?** Near-perfect BPM–IBI and RMSSD–SD1 redundancy are **expected**
(mathematical relationships). Sign-flip of BPM vs RMSSD under stress is
**not** the usual resting physiology picture.

**Unexpected / flags:** Stress-regime positive BPM–RMSSD coupling suggests
either (a) artifactual peak errors that worsen at high HR, or (b) true
exercise non-stationarity / RSA — both undermine naive “RMSSD = vagal tone”
interpretation in this cohort.

**Paper wording:** Emphasize multicollinearity → need for regularization,
feature pruning, or PCA; report correlation-structure **change by class** as
evidence that a single linear model of “HRV meaning” does not transfer from
rest to exertion; justify not treating all 11 features as independent predictors.

---

### Plot 3 — `03_poincare_plot.png`

**Observation:** REST forms a compact SD1–SD2 cloud; STRESS shows a dense
low-variability cluster **plus** a long high-SD1/SD2 tail. The STRESS 95%
ellipse is much larger and overlaps REST heavily.

**Expected?** Lower short-term variability under sympathetic load would predict
stress points toward lower SD1 — only partly true (low-SD1 stress cluster).
The high-SD1 stress tail is **unexpected** for clean exertion HRV.

**Unexpected / flags:** Bimodal / heavy-tailed stress Poincaré cloud; ellipse
overlap → weak linear separability from SD1/SD2 alone.

**Paper wording:** Present Poincaré as exploratory; note Brennan et al. (2001)
geometry for SD1/SD2 interpretation; explicitly discuss the high-SD1 stress
subpopulation as a quality / physiology ambiguity rather than as “increased
parasympathetic tone.”

---

### Plot 4 — `04_sd1_sd2_ratio.png`

**Observation:** Mean SD1/SD2 ≈ **0.42** (REST) vs ≈ **1.26** (STRESS). REST
mass sits **below** the balance line (ratio = 1); STRESS has a long upper tail
(extreme ratios).

**Expected?** Literature often associates relatively higher short-term
(parasympathetic-related) contribution with rest — here REST ratios are lower
and STRESS ratios higher on average → **counterintuitive** if SD1/SD2 is read
as a simple “parasympathetic/sympathetic balance” meter (Shaffer & Ginsberg,
2017 — use cautiously; ratio is a **proxy**, not a gold-standard ANS assay).

**Unexpected / flags:** Extreme STRESS ratio outliers (non-physiological if
taken literally); contradicts the “high ratio = rest” expectation stated in
the EDA plan.

**Paper wording:** Do **not** claim ANS balance from SD1/SD2 alone in this
dataset; report the reversed class pattern and link it to the same stress
high-SD1 tail seen in Plots 1/3/5; treat as a limitation of Poincaré summaries
under motion-rich exercise ECG.

---

### Plot 5 — `05_rmssd_bpm_scatter.png`

**Observation:** REST points cluster at BPM ≲ 100. Many STRESS points fall in
the annotated “STRESS zone” (high BPM, low RMSSD: **183** stress vs **0** rest
windows). A large STRESS subpopulation occupies high BPM **and** high RMSSD
(up to ~400 ms). REST-zone (BPM&lt;100, RMSSD&gt;50): **43** rest vs **7** stress.

**Expected?** High-BPM / low-RMSSD stress cloud is **expected**. High-BPM /
high-RMSSD stress cloud is **counterintuitive** for vagal RMSSD.

**Unexpected / flags:** Dual stress regimes (clean low-RMSSD vs artifactual /
non-stationary high-RMSSD).

**Paper wording:** Use this figure in Introduction/Results as the most
intuitive display of **partial** class separation by BPM, while captioning the
upper-right stress cloud as a known QC/physiology caveat (ties to earlier
RMSSD investigation).

---

### Plot 6 — `06_pca_biplot.png`

**Observation:** PC1 ≈ **61.7%**, PC2 ≈ **16.7%** variance (≈ 78% combined).
REST occupies a relatively compact region; STRESS spans a much larger area
along PC1. Loading arrows: `ibi` toward the rest-associated side; `bpm`
roughly opposite; variability features (`rmssd`, `sdnn`, …) load toward the
region populated mainly by stress points with high PC1 — consistent with
stress high-variability outliers.

**Expected?** Partial linear separability driven by rate features is
**plausible**. Variability loadings pointing into “stress-high-RMSSD” space
is **unexpected** under a pure vagal-withdrawal model.

**Unexpected / flags:** Separation is incomplete; stress variance dominates
PC1; feature arrows confirm redundancy among variability metrics.

**Paper wording:** State that HRV feature space is **partially** linearly
separable, with rate features as primary axes; note that PC1 also captures
stress-side variability inflation — motivating robust models and LOSO rather
than claiming clean linear discrimination.

---

### Plot 7 — `07_subject_profiles.png`

**Observation:** Eight REST subject means at BPM ≈ 60–85 with moderate RMSSD;
sixteen STRESS subject means at BPM ≈ 100–155 with RMSSD from very low to
~200 ms. Several stress subjects (e.g. high-BPM / high-RMSSD profiles) sit far
from low-RMSSD stress peers.

**Expected?** Large between-subject spread under exercise is **expected** and
is exactly why LOSO is required (note 06).

**Unexpected / flags:** Subject-level confirmation that “stress” is
heterogeneous in RMSSD space — some subjects look textbook (high BPM, low
RMSSD), others look like high-RMSSD outliers.

**Paper wording:** Use to justify LOSO and to warn that pooled metrics hide
subject-specific artifact/physiology mixtures; do not treat window-level i.i.d.
splits as valid.

---

### Plot 8 — `08_breathing_rate.png`

**Observation:** HeartPy breathing estimates convert to ≈ **10 breaths/min**
mean in **both** classes (medians ~8), mostly **below** the 12–20 /min guide
band; class distributions nearly identical; stress has a longer high tail.

**Expected?** Mild respiratory increase under exertion might be expected;
**near-identical class means** and systematically low rates are **unexpected**
relative to clinical resting norms (12–20 /min often cited in clinical
teaching — Task Force 1996 is about HRV standards, not a hard breaths/min
law; treat the band as a **display reference**, not a dataset ground truth).

**Unexpected / flags:** Likely **ECG-derived respiration underestimation** or
mismatch of HeartPy breathingrate units/assumptions under this pipeline —
feature may be weakly informative for classification.

**Paper wording:** Report breathing_rate as exploratory only; note lack of
class separation and possible systematic bias; do not interpret as spirometry.

---

### Plot 9 — `09_class_distribution.png`

**Observation:** Pie ≈ **23.4% REST (142)** vs **76.6% STRESS (465)**. Bar
chart: most stress recordings retain ~27–30 windows; several rest recordings
contribute very few (0011/0015 ≈ 5; 0021 ≈ 3); 0014 absent (excluded).

**Expected?** Imbalance from 8 vs 16 recordings **plus** rest QC losses is
**expected** (prior notes).

**Unexpected / flags:** Extreme under-representation of some rest subjects →
LOSO folds with tiny rest test sets for those subjects.

**Paper wording:** Quantify imbalance; justify **SMOTE on training folds only**;
acknowledge that rest-class learning is dominated by high-yield rest recordings
(0010/0013/0016/0022).

---

### Plot 10 — `10_pairplot.png`

**Observation:** BPM KDE shows clear bimodal class structure. RMSSD and SD1
KDEs are near-duplicates (mathematical redundancy). SD2 stress mass sits lower
than rest for many points (more literature-aligned) while RMSSD/SD1 show a
stress secondary high mode. Pairwise scatters: best visual separation along
BPM; RMSSD–SD1 essentially a line.

**Expected?** BPM separability and RMSSD≡SD1 redundancy are **expected**.
Stress high-RMSSD secondary mode remains **counterintuitive**.

**Unexpected / flags:** Confirms feature set can be pruned (drop SD1 or RMSSD
duplicate) without losing information.

**Paper wording:** Recommend reporting a reduced feature subset for models;
highlight BPM as strongest marginal discriminator; discuss RMSSD/SD1 only with
the dual-regime caveat.

---

### Cross-cutting conclusions for the paper

1. **Primary expected finding:** Protocol REST vs STRESS is strongly reflected
   in **heart rate level** (BPM/IBI).
2. **Primary unexpected finding:** Short-term variability (RMSSD/SD1) is **not**
   cleanly reduced under stress; a stress subpopulation shows **inflated**
   variability and flips BPM–RMSSD correlation — interpret as artifact and/or
   exercise non-stationarity, not as increased vagal tone.
3. **Modeling implications:** Prefer LOSO; SMOTE train-only; account for
   multicollinearity; consider BPM-aware models or explicit artifact features;
   treat breathing_rate and SD1/SD2-ratio claims cautiously.
4. **Evidence tags:** Observations = `project-implementation` (this CSV/figures);
   physiological expectations = `peer-reviewed` (Shaffer & Ginsberg 2017;
   Task Force 1996; Brennan et al. 2001 for Poincaré geometry). Exact clinical
   “normal RMSSD range” statements remain **[NEEDS VERIFICATION]** against
   primary tables when citing numbers in the thesis.

**Code / artifacts:** `scripts/04_eda_visualizations.py`, `results/figures/01_*.png`–`10_*.png`

---

## Modeling — Binary Classification LOSO — 2026-07-09

Script (not executed in this step): `scripts/03_train_evaluate.py`  
Data: `data/processed/ephnogram_features.csv` (11 HRV features; `label`; LOSO groups = `subject_id`)  
Outputs: `results/metrics/ephnogram_binary_loso.json`;
`results/figures/confusion_matrix_[best_model].png`;
`results/figures/per_subject_accuracy.png`

### EDA-informed modeling context
1. BPM and IBI are the strongest univariate separators (EDA Plot 1/5/10).
2. RMSSD/SD1 show dual-regime behavior under exercise — not reliable alone.
3. BPM–RMSSD correlation flips sign between rest and stress.
4. Breathing rate is weakly discriminative.
5. Class imbalance ≈ **142 rest (23%) vs 465 stress (77%)** → majority baseline
   ≈ **77% accuracy** (always predict stress). Models must beat this baseline.

### Models and hyperparameters

| Model | Key hyperparameters | Why this choice |
|-------|---------------------|-----------------|
| Random Forest | `n_estimators=200`, `random_state=42`, `class_weight='balanced'` | Strong tabular baseline; handles nonlinear interactions and multicollinear HRV features; feature importance later (RQ3) |
| SVM (RBF) | `C=1.0`, `gamma='scale'`, `probability=True`, `class_weight='balanced'` + `StandardScaler` in pipeline | Margin classifier for moderate-dimensional continuous features; RBF captures nonlinear boundaries in HRV space |
| KNN | `n_neighbors=5`, `weights='distance'` + `StandardScaler` | Simple local baseline; distance weighting softens equal-vote noise |
| Soft Voting Ensemble | RF + SVM + KNN, `voting='soft'` | Combines complementary inductive biases; soft voting uses predicted probabilities |

**Evidence:** (`project-implementation`) hyperparameter defaults for the course study;
classical ML practice for imbalanced tabular biosignal features.

### Why LOSO (not random k-fold)
- **Decision:** Leave-One-Subject-Out CV (`subject_id` as group).
- **Justification:** Windows from the same subject share baseline HRV, fitness, and
  recording conditions. Random splits leak subject identity into the test set and
  inflate accuracy.
- **References:**
  - Gjoreski et al. (2016) — wearable stress / evaluation context
    (**full bibliographic details [NEEDS VERIFICATION]**).
  - Vos et al. (2023) systematic review on stress detection / generalizability
    (**[NEEDS VERIFICATION]** of exact recommendations cited in the thesis).

### Why SMOTE on the training fold only
- **Decision:** `SMOTE(k_neighbors≤5, random_state=42)` fit on **X_train, y_train**
  after each LOSO split; **never** applied to the held-out subject.
- **Justification:** Oversampling the test fold would leak synthetic structure
  derived from test labels/distribution and invalidate generalization estimates.
- **Reference:** Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.
  (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.

### Why `class_weight='balanced'` in addition to SMOTE
- **Decision:** Use `class_weight='balanced'` for RF and SVM (and ensemble members)
  **together with** train-fold SMOTE.
- **Justification:** SMOTE addresses sample counts; class weights further penalize
  majority-class errors during optimization — complementary under severe imbalance
  (23/77). KNN has no class_weight; relies on SMOTE + distance weights.
- **Reference:** He, H., & Garcia, E. A. (2009). Learning from imbalanced data.
  *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.

### Scaling
- StandardScaler is fit **inside** SVM/KNN (and their ensemble copies) pipelines on
  each training fold only, then applied to that fold’s test features.
- Random Forest is left unscaled (tree splits are scale-invariant).

### Success criterion vs majority baseline
- Always predicting STRESS yields ≈ **77%** accuracy given the current class prior.
- Report Accuracy, F1-macro, F1-weighted, and ROC-AUC (mean ± std across LOSO folds).
- **F1-macro** and **ROC-AUC** are more informative than accuracy under imbalance;
  a model can beat 77% accuracy yet still fail the minority (rest) class.

### Limitations (pre-run)
- Some rest subjects contribute very few windows (EDA Plot 9) → unstable LOSO folds.
- Dual-regime RMSSD may confuse margin/distance models unless BPM dominates.
- Soft voting increases compute; no nested hyperparameter search in this script.

**Code:** `scripts/03_train_evaluate.py`  
**Paper section:** Methods — Model Evaluation / Results — Binary Classification

---

## Decision — 2026-07-09 — ROC-AUC: pooled when LOSO folds are single-class

**Problem:** Per-fold `roc_auc_score` returned nan for every fold →
`np.nanmean` raised “Mean of empty slice”.

**Root cause (confirmed):** Default `SUBJECT_MAP` assigns **one subject_id per
recording** (`S{record_id}`). Each recording has a single protocol label, so
every LOSO test fold has **only one class** in `y_test`. ROC-AUC is undefined
for a single-class test set (not a `predict_proba` bug).

**Inventory:** 24/24 subjects are single-class (8 rest-only, 16 stress-only).

**Fix (`project-implementation`):**
1. Debug prints: per-subject class inventory before LOSO; per-fold
   `y_test` classes / `proba_shape`.
2. Fold AUC: set nan if `len(np.unique(y_test)) < 2` (explicit try/except).
3. Avoid empty-slice warning when all fold AUCs are nan.
4. Report **pooled ROC-AUC** over concatenated test predictions (both classes
   present globally) as the table ROC-AUC when fold-wise mean is undefined.

**Advantages:** Honest metrics; still reports ranking quality across the study.
**Disadvantages:** Pooled AUC is not a mean±std over independent subject folds;
true subject-wise AUC needs subjects with **both** rest and stress sessions
(PhysioNet spreadsheet Subject IDs — future improvement).

**Code:** `scripts/03_train_evaluate.py`  
**Paper section:** Methods — Model Evaluation

---

## Modeling Findings — Binary LOSO Results — 2026-07-10

Metrics from `results/metrics/ephnogram_binary_loso.json`  
Figures from `scripts/05_model_visualizations.py` → `results/figures/11_*.png`–`15_*.png`

### Performance summary (LOSO, train-fold SMOTE)

| Model | Accuracy (mean±std) | F1-macro (mean±std) | ROC-AUC (pooled) | Confusion [TN FP / FN TP] |
|-------|---------------------|---------------------|------------------|---------------------------|
| Random Forest | 0.965 ± 0.083 | 0.844 ± 0.244 | 0.993 | 132 / 10 / 6 / 459 |
| SVM RBF | 0.966 ± 0.083 | 0.843 ± 0.245 | 0.993 | 127 / 15 / 6 / 459 |
| KNN | 0.939 ± 0.117 | 0.773 ± 0.280 | 0.982 | 127 / 15 / 16 / 449 |
| **Ensemble (best F1m)** | 0.955 ± 0.114 | **0.881 ± 0.230** | **0.993** | 127 / 15 / 5 / 460 |

Majority baseline (always STRESS) ≈ **76.6%** accuracy. All models **beat** this baseline on accuracy; F1-macro remains the fairer metric under imbalance.

### Finding 1 — Strong discrimination, BPM/IBI-driven
- Pooled ROC curves are near-ceiling (AUC ≈ 0.98–0.99) for all models.
- RF impurity importance: **bpm ≈ 0.37**, **ibi ≈ 0.34**, then pnn20 ≈ 0.08, rmssd/sd1 ≈ 0.05–0.07; breathing_rate near-bottom.
- Aligns with EDA: rate features dominate REST vs protocol-STRESS separation (`project-implementation` + EDA notes).

**Paper:** State that binary physical-stress classification is highly accurate in this cohort but is largely **heart-rate level discrimination**, not a pure autonomic “stress” biomarker.

### Finding 2 — Ensemble wins F1-macro; KNN is weakest
- Soft voting yields the best F1-macro (0.881), slightly trading accuracy vs RF/SVM.
- KNN has more false negatives (16) and lower F1 / AUC — sensitive to the dual-regime RMSSD geometry and scale locally.

**Paper:** Report Ensemble as primary result if F1-macro is the designated metric; keep RF for interpretability (importance / SHAP).

### Finding 3 — Hard subjects drive F1 fold variance
Per-subject accuracy bars show near-perfect folds for most stress subjects; largest errors on rest/low-n or atypical subjects (e.g. **S16**, **S21**, **S13** depending on model). High F1 **std** (~0.23–0.28) reflects single-class LOSO folds and heterogeneous rest quality.

**Paper:** Emphasize LOSO variance and low rest-window yield for some subjects as a limitation; do not over-claim generalizability from mean accuracy alone.

### Finding 4 — Confusion structure
- Errors are mostly REST→STRESS false positives (10–15) plus few STRESS→REST false negatives (5–16).
- Consistent with class prior and BPM overlap / rest artifact tails from EDA.

### Visualizations generated
| Figure | File | Use in paper |
|------|------|--------------|
| Confusion 2×2 grid | `11_confusion_matrices_grid.png` | Results |
| Pooled ROC (4 models) | `12_roc_curves_pooled.png` | Results |
| Per-subject accuracy | `13_per_subject_accuracy.png` | Results / Discussion |
| F1-macro comparison | `14_f1_macro_comparison.png` | Results |
| RF feature importance | `15_rf_feature_importance.png` | Results (RQ3 preview) |

### Limitations (unchanged)
- `subject_id` = per-recording → single-class folds; ROC reported as **pooled**.
- Construct = protocol exertion, not operational/psych stress.
- Near-ceiling AUC partly reflects BPM separability.

**Code:** `scripts/05_model_visualizations.py`, `scripts/03_train_evaluate.py`  
**Paper section:** Results — Binary Classification


## Feature Engineering and Selection — 2026-07-10

### Engineered Features
| Feature | Formula | Reference |
|---------|---------|-----------|
| sd1_sd2_ratio | SD1/SD2 | Castaldo et al. (2015); Frontiers Digital Health (2021) doi:10.3389/fdgth.2021.639444 — SD1/SD2 as Cardiac Sympathetic Index; RF ~80% stress accuracy (peer-reviewed) |
| baevsky_proxy | HR_MAD / (RMSSD + 1e-6) | Inspired by Baevsky Stress Index (Baevsky & Chernikova 2017); Kubios HRV docs — proxy for sympathetic load, **not** identical to original Baevsky SI (project-implementation) |
| pnn20_normalized | pNN20 / (IBI / 1000) | Task Force ESC (1996) — pNN metrics are heart-rate dependent; IBI normalization reduces that dependency (peer-reviewed) |

Dataset: `data/processed/ephnogram_features.csv` → `data/processed/ephnogram_features_engineered.csv` (n=607 windows).

### Feature Selection Results (LOSO RF, SMOTE train-only, class_weight=balanced)

| Experiment | N features | Accuracy | F1-macro | ROC-AUC (pooled) |
|------------|------------|----------|----------|------------------|
| A (all 11) | 11 | 0.965±0.083 | 0.844±0.244 | 0.993 |
| B (reduced 7) | 7 | 0.975±0.068 | 0.868±0.229 | 0.993 |
| C (minimal 5) | 5 | 0.977±0.067 | 0.869±0.228 | 0.993 |
| D (engineered 9) | 9 | 0.977±0.067 | 0.869±0.228 | 0.990 |

**Feature sets**
- A: bpm, ibi, sdnn, rmssd, sdsd, pnn20, pnn50, hr_mad, sd1, sd2, breathing_rate
- B: bpm, ibi, pnn20, rmssd, sd1, hr_mad, sdnn — drop breathing_rate (~0 importance), pnn50 (redundant with pnn20), sd2 (redundant with sdnn), sdsd (redundant with rmssd)
- C: bpm, ibi, pnn20, rmssd, sd1
- D: bpm, ibi, sdnn, rmssd, pnn20, hr_mad, sd1_sd2_ratio, baevsky_proxy, pnn20_normalized

Winner (F1-macro): **C (minimal 5)** = 0.869 (identical hard predictions to D; D has slightly lower pooled AUC 0.990 vs 0.993).

Practical conclusion (threshold: F1-macro drop vs A < 0.02): **C (minimal 5)** — F1 rises vs A (+0.025), not a drop; fewest features among all sets within threshold. B/D also exceed A; engineered features did not improve over the raw top-5 set on this cohort.

### Conclusion
- **Recommended feature set for final models:** C (minimal 5) — bpm, ibi, pnn20, rmssd, sd1. Matches or exceeds the full 11-feature RF under LOSO; removing redundant/low-importance features reduces noise.
- **Wearable deployment:** Five Time/Poincaré features are enough for binary rest vs stress on this EPHNOGRAM setup; engineered ratios (D) do not beat the minimal raw set here — useful as optional features if deployment can compute them cheaply, but not required for performance.
- Selection rationale aligns with Kohavi & John (1997) feature subset selection and Breiman (2001) RF importance-guided pruning (peer-reviewed methodology).

**Code:** `scripts/06_feature_selection.py`  
**Metrics:** `results/metrics/feature_selection_comparison.json`

### Paper section: Methods — Feature Engineering and Selection

## Final Models — Optimal 5 Features — 2026-07-10

**Main result section.** Feature selection (Experiment C) selected five HRV features; all four classifiers were re-run under the same LOSO + SMOTE (train-only) protocol.

### Optimal feature set
`bpm`, `ibi`, `pnn20`, `rmssd`, `sd1` (n=5)

Rationale: Experiment C achieved best F1-macro among A–D (0.869) with no performance drop vs the full 11-feature set; see Feature Engineering and Selection section.

### Final LOSO results

| Model | Features | Accuracy | F1-macro | F1-weighted | ROC-AUC (pooled) |
|-------|----------|----------|----------|-------------|------------------|
| RF | 5 | 0.977±0.067 | 0.869±0.228 | 0.987±0.040 | 0.993 |
| SVM | 5 | 0.981±0.031 | 0.828±0.243 | 0.990±0.016 | 0.994 |
| KNN | 5 | 0.966±0.070 | 0.803±0.255 | 0.981±0.041 | 0.983 |
| Ens | 5 | 0.972±0.070 | 0.846±0.240 | 0.984±0.041 | 0.994 |

Majority-class baseline (always stress) ≈ 76.6% accuracy — all models exceed this.

**Best by F1-macro:** Random Forest (0.869).  
**Best by Accuracy / F1-weighted:** SVM RBF (Acc 0.981, F1w 0.990).  
**Best pooled ROC-AUC:** SVM / Ensemble (0.994), RF close (0.993).

### RF feature importance (5 features, SMOTE-balanced full set)
| Feature | Importance |
|---------|------------|
| ibi | 0.422 |
| bpm | 0.419 |
| pnn20 | 0.083 |
| sd1 | 0.038 |
| rmssd | 0.037 |

Heart-rate level (bpm/ibi) still dominates; pnn20 contributes secondary discriminative signal; sd1/rmssd are smaller but retained as physiologically motivated HRV markers.

### Figures (paper Results)
| Figure | File | Use |
|--------|------|-----|
| Confusion matrices 2×2 | `16_final_confusion_matrices_5features.png` | Results |
| Pooled ROC (4 models) | `17_final_roc_curves_5features.png` | Results |
| RF importance (5 feat.) | `18_final_rf_importance_5features.png` | Results / Discussion |

### Interpretation
- With only five features, RF remains the strongest balanced classifier (F1-macro), matching the feature-selection RF result exactly.
- SVM slightly edges accuracy and AUC but trails RF on F1-macro (harder rest class under LOSO single-class folds).
- Ensemble does not beat RF on F1-macro on this set; soft voting helps AUC but not minority-class balance relative to RF alone.
- Practical takeaway for wearables: deploy RF (or SVM if maximizing accuracy/AUC) on `{bpm, ibi, pnn20, rmssd, sd1}` — no need for the full HeartPy measure vector.

### Limitations (unchanged)
- `subject_id` = per-recording → single-class LOSO folds; ROC reported as pooled.
- Construct = protocol exertion, not operational/psych stress.
- Near-ceiling metrics partly reflect bpm/ibi separability.

**Code:** `scripts/07_final_models_5features.py`  
**Metrics:** `results/metrics/final_models_5features.json`  
**Importance CSV:** `results/metrics/rf_feature_importance_5features.csv`

### Paper section: Results — Final Classification (Optimal Feature Set)
