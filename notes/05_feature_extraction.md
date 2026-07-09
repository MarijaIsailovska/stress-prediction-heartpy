# 05 — Feature extraction (HeartPy HRV)

**Paper section:** Methods — Feature Extraction  
**Related overview:** `research_notes.md` § SECTION 5

---

## Decision — 2026-07-09 — Extract 15 HeartPy measures per 60 s window

**Decision:** For each valid window, call HeartPy `hp.process()` and retain the following measures (CSV column names in parentheses):

| # | HeartPy key | Column | Domain |
|---|-------------|--------|--------|
| 1 | bpm | BPM | Time |
| 2 | ibi | IBI | Time |
| 3 | sdnn | SDNN | Time |
| 4 | sdsd | SDSD | Time |
| 5 | rmssd | RMSSD | Time |
| 6 | pnn20 | pNN20 | Time |
| 7 | pnn50 | pNN50 | Time |
| 8 | hr_mad | HR_MAD | Time |
| 9 | lf | LF | Frequency |
| 10 | hf | HF | Frequency |
| 11 | lf/hf | LF_HF | Frequency |
| 12 | breathingrate | Breathing_Rate | Derived |
| 13 | sd1 | SD1 | Poincaré |
| 14 | sd2 | SD2 | Poincaré |
| 15 | s | S | Poincaré |

Windows where `hp.process` raises, or any required measure is missing/non-finite, are **discarded** (no imputation).

**Why:** HeartPy provides a documented, dependency-light HRV feature set spanning time, frequency, and Poincaré domains suitable for classical ML baselines in a course project.

**Advantages:**
- Reproducible open-source pipeline
- Mix of domains supports RQ3 (feature importance / SHAP)
- Failure-as-discard is conservative for data quality

**Disadvantages:**
- Library defaults (peak detection, frequency estimation) may differ from Kubios / NeuroKit2 / custom RR pipelines
- Discarding failures can bias class balance and subject coverage
- LF/HF interpretation under exercise is physiologically contested in parts of the HRV literature

**Alternatives in literature:**
- NeuroKit2, HRVAnalysis, Kubios-style feature sets
- Deep learning on raw ECG/PPG without handcrafted HRV
- RR-interval-only time-domain features when spectral estimates are unstable

**Evidence:**
- (`heartpy-docs` / `peer-reviewed`) van Gent, P., Farah, H., van Nes, N., & van Arem, B. (2019). HeartPy: A novel heart rate algorithm for the analysis of noisy signals. *Transportation Research Part F*. doi:10.1016/j.trf.2019.09.015 — **[NEEDS VERIFICATION]** of exact DOI/title against the published record (overview used `...09.010`; confirm before citing in the paper)
- (`peer-reviewed`) Shaffer & Ginsberg (2017) for HRV metric definitions / norms (see note 04)
- (`project-implementation`) Feature list and key→column map in `config/settings.py`; extraction in `src/features/hrv_extractor.py`

**Uncertainty:**
- Exact HeartPy measure names and units should be confirmed against the installed HeartPy version used in experiments
- van Gent et al. bibliographic details marked for verification above

**Code:** `config/settings.py` (`HRV_FEATURES`, `HRV_FEATURE_COLUMNS`), `src/features/hrv_extractor.py`  
**Paper section:** Methods — Feature Extraction

---

## Decision — 2026-07-09 — EPHNOGRAM CSV feature schema (11 HRV + meta)

**Decision:** Export exactly: `bpm`, `ibi`, `sdnn`, `rmssd`, `sdsd`, `pnn20`, `pnn50`, `hr_mad`, `sd1`, `sd2`, `breathing_rate`, plus `recording_id`, `subject_id`, `window_id`, `label`.  
`hr_mad` is computed as \(\mathrm{median}(|RR_i - \mathrm{median}(RR)|)\) from HeartPy `wd['RR_list']`, not taken from `measures` alone.  
LF/HF are enabled via `calc_freq=True` (supports breathing-rate path) but **not** written to this CSV.

**Why:** Aligns the supervised table with the course-required feature list for classical ML while keeping MAD robust to outlier IBIs.

**Advantages:** Fixed schema for LOSO scripts; fewer missing spectral columns when frequency estimation is unstable.  
**Disadvantages:** Drops LF/HF/S from the earlier 15-feature design for this CSV.  
**Alternatives:** Full 15-feature export; Kubios/NeuroKit2.

**Evidence:**
- (`project-implementation`) `scripts/01_process_ephnogram.py`
- (`heartpy-docs`) `hp.process` measure keys
- (`peer-reviewed`) van Gent et al. (2019); Antczak (2017) for MAD framing — **[NEEDS VERIFICATION]** of Antczak full citation

**Code:** `scripts/01_process_ephnogram.py` → `data/processed/ephnogram_features.csv`  
**Paper section:** Methods — Feature Extraction
