# 04 — Signal processing

**Paper section:** Methods — Signal Processing  
**Related overview:** `research_notes.md` § SECTION 4

---

## Decision — 2026-07-09 — Resample ECG, bandpass, 60 s non-overlapping windows

**Decision:**

| Step | ECG (EPHNOGRAM) | PPG (Wrist) |
|------|-----------------|-------------|
| Resample | 8000 → 500 Hz (polyphase `scipy.signal.resample_poly`) | native 256 Hz |
| Bandpass | 0.5–45 Hz | 0.75–3.5 Hz |
| Window | 60 s, overlap = 0 | 60 s, overlap = 0 |
| Failed windows | Discard if HeartPy raises / non-finite measures | same |

**Why:**
- Downsampling reduces compute while remaining adequate for HR/HRV peak timing at human heart rates (`project-implementation`; adequacy vs literature **[NEEDS VERIFICATION]** for 500 Hz specifically)
- Bandpass limits baseline wander and high-frequency noise before peak detection
- ~60 s windows are a common minimum for frequency-domain HRV (LF/HF) under classical HRV guidance

**Advantages:**
- Standardized pipeline across recordings
- Non-overlapping windows avoid trivial autocorrelation between adjacent samples in CV (still subject to within-subject dependence — addressed by LOSO in note 06)
- Explicit discard avoids imputing invalid HRV rows

**Disadvantages:**
- Non-overlap reduces sample count vs overlapping windows
- Fixed 60 s may mix protocol stages inside one window
- Aggressive PPG band (0.75–3.5 Hz) assumes heart-rate band of interest; very high HR may be attenuated (**[NEEDS VERIFICATION]** against HeartPy PPG guidance and exercise HR ranges)
- Discarding failures may bias toward cleaner segments

**Alternatives in literature:**
- Overlapping windows (e.g. 50%) for more samples
- Shorter windows for time-domain-only HRV (with acknowledged LF/HF limitations)
- Different ECG band edges (e.g. 0.5–40 Hz, 5–15 Hz for QRS emphasis)
- Resample to 250 Hz or keep higher rates for research-grade timing

**Evidence:**
- (`peer-reviewed`) Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. *European Heart Journal* / circulation standards document — commonly cited for HRV measurement standards including recording-length considerations. **Confirm page-level claims for “minimum 60 s for LF/HF” when drafting the paper [NEEDS VERIFICATION].**
- (`peer-reviewed`) Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. *Frontiers in Public Health*. doi:10.3389/fpubh.2017.00258
- (`heartpy-docs`) Bandpass applied via `heartpy.filter_signal` when available (`src/preprocessing/filter.py`), else SciPy Butterworth — HeartPy API behavior is an implementation detail; preferred cutoffs above are **project choices** unless matched to a HeartPy tutorial recommendation (**[NEEDS VERIFICATION]**)
- (`project-implementation`) Parameters in `config/settings.py`; code in `src/preprocessing/{resample,filter,segment}.py`

**Uncertainty:**
- Exact Task Force wording on minimum duration for spectral HRV should be quoted from the 1996 document before the paper submission
- PPG cutoff pair (0.75–3.5 Hz) needs a cited source or explicit “project choice” framing

**Code:** `config/settings.py` (`EPHNOGRAM_FS_*`, `ECG_BANDPASS`, `PPG_BANDPASS`, `WINDOW_*`), `src/preprocessing/`, `scripts/01_process_ephnogram.py`, `scripts/02_process_wrist.py`  
**Paper section:** Methods — Signal Processing

---

## Decision — 2026-07-09 — EPHNOGRAM implemented pipeline (script 01)

**Decision:** Production preprocessing for the 25 local EPHNOGRAM recordings follows, in order: `wfdb.rdrecord` (ECG ch0 from `WFDB/`) → `resample_poly` 8000→500 Hz → `hp.filter_signal` bandpass **[0.5, 45]** Hz → **60 s** non-overlapping segments → HeartPy `process(..., calc_freq=True)` with QC (exception / &lt;20 beats / BPM∉[30,220] / RMSSD&gt;300 ms / non-finite measures).

**Why:** Matches the locked study specification for RQ1 ECG HRV classification; see `research_notes.md` § Preprocessing Pipeline — EPHNOGRAM — 2026-07-09 for full citations (Pan & Tompkins 1985; Sörnmo & Laguna 2005; Task Force 1996; Shaffer & Ginsberg 2017; Clifford et al. 2006).

**Advantages / Disadvantages / Alternatives:** Same as earlier section; additionally, QC thresholds are **project-chosen** adaptations of ECG quality practice and may discard more stress windows.

**Evidence:**
- (`project-implementation`) `scripts/01_process_ephnogram.py`; output `data/processed/ephnogram_features.csv`
- (`dataset-docs`) PhysioNet EPHNOGRAM WFDB paths
- (`heartpy-docs`) `filter_signal`, `process`
- (`peer-reviewed`) citations listed in `research_notes.md` appendix (verify page-level quotes before thesis submission)

**Code:** `scripts/01_process_ephnogram.py`  
**Paper section:** Methods — Signal Processing
