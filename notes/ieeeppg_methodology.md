# IEEEPPG — dataset analysis and preprocessing methodology (no code yet)

**Date:** 2026-07-09  
**Status:** Analysis only — **preprocessing not implemented**  
**Paper section:** Methods — Data Sources / Signal Processing (candidate dataset)  
**Local presence:** **`[NEEDS VERIFICATION]` / not found** — as of this note, no `IEEEPPG`, `IEEEPPG_TRAIN.ts`, `IEEEPPG_TEST.ts`, or SPC `DATA_*_TYPE*.mat` files were present under `stress_prediction_heartpy/` (including `data/raw/`). Findings below are from **dataset documentation** and **peer-reviewed / preprint sources**, not from a local file inventory.

---

## 1. What “IEEEPPG” refers to

Two related but **not identical** packages appear in the literature under this name:

| Packaging | Source | Typical files | Role |
|-----------|--------|---------------|------|
| **A. Original IEEE SPC 2015 / TROIKA corpus** | Zhang lab / IEEE Signal Processing Cup 2015 | MATLAB `DATA_##_TYPE##.mat` + `..._BPMtrace.mat` | Raw multi-channel recordings + ECG-derived BPM traces |
| **B. Monash / UEA / UCR TSER “IEEEPPG”** | Zenodo + TSER archive | `IEEEPPG_TRAIN.ts`, `IEEEPPG_TEST.ts` | Pre-windowed **5-D** series for **heart-rate regression** benchmarks |

**Evidence:**
- (`dataset-docs`) Zenodo IEEEPPG: https://zenodo.org/records/3902710  
- (`dataset-docs`) TSER site listing: http://tseregression.org/  
- (`peer-reviewed`) Zhang, Z., Pi, Z., & Liu, B. (2015). TROIKA: A general framework for heart rate monitoring using wrist-type photoplethysmographic signals during intensive physical exercise. *IEEE Transactions on Biomedical Engineering*, 62(2), 522–531. https://doi.org/10.1109/TBME.2014.2359372 (arXiv HTML: https://ar5iv.labs.arxiv.org/html/1409.5181)  
- (`peer-reviewed` / archive paper) Tan, C. W., Bergmeir, C., Petitjean, F., & Webb, G. I. (2020/2021). Monash University, UEA, UCR Time Series Extrinsic Regression Archive. arXiv:2006.10996; journal version *Data Mining and Knowledge Discovery* (2021). https://doi.org/10.1007/s10618-021-00745-9  

**Project implication (`project-implementation`):** Before any loader is written, confirm which packaging will be placed in `data/raw/` (A, B, or both). Channel counts and whether ECG is available differ (see §2).

---

## 2. Dataset structure and channels

### 2.1 Original SPC / TROIKA recordings (packaging A)

**Simultaneously recorded channels** (`dataset-docs` / TROIKA paper):

| # | Channel | Sensor / placement | Notes |
|---|---------|-------------------|--------|
| 1 | PPG channel 1 | Wrist pulse oximeter, green LED (~515 nm) | Two LEDs, ~2 cm center-to-center |
| 2 | PPG channel 2 | Second green LED on same wristband | Often averaged or used jointly in later papers |
| 3–5 | Accel X, Y, Z | Wrist 3-axis accelerometer | Motion-artifact reference |
| 6 | ECG | Chest, wet electrodes | **Ground-truth cardiac reference** (not always kept in TSER export) |

**Sampling frequency:** **125 Hz** for all of the above (`dataset-docs`, TROIKA).

**Subjects / protocol (summary):** Young adults (commonly cited age range **18–35**); treadmill / intensive exercise with strong wrist motion artifact. Exact subject counts differ between “training” treadmill set (~12 sessions in TROIKA experiments) and later SPC test sets with arm-motion variants — **[NEEDS VERIFICATION]** against the official SPC readme when files are obtained.

**MATLAB layout (commonly reported in community code/docs):**  
- `sig`: multi-row array (ECG + 2×PPG + 3×ACC in a fixed row order — **confirm row order from the accompanying readme**, do not assume without the file).  
- Companion `BPM0` / BPMtrace: ground-truth BPM per sliding window.

### 2.2 Monash TSER IEEEPPG (packaging B) — likely what “IEEEPPG dataset” means in ML archives

From the TSER archive description (`peer-reviewed` arXiv:2006.10996):

- **Dimensions:** **5** (2× PPG + 3× accelerometer)  
- **ECG is not included** as an input channel in this packaging (ECG was used upstream to build the BPM target).  
- **Instances:** 1768 train + 1328 test = **3096** windows (matches Zenodo text).  
- **Series length:** **1000 samples** = 8 s × 125 Hz.  
- **Train/test:** original SPC split retained.

**Zenodo text inconsistency to flag:** Zenodo’s abstract still mentions “one-channel ECG signals were simultaneously recorded,” but the **released `.ts` tensors are described as 5-D (PPG+ACC only)** in the TSER paper. Treat ECG as **available only in packaging A**, not in B, until a local file check confirms otherwise. **`[NEEDS VERIFICATION]`**

---

## 3. What the target labels represent

### 3.1 Official task label

**Target = continuous heart rate in beats per minute (BPM)** estimated from reference **ECG** over each analysis window — **not** a categorical stress / rest / activity label.

**Windowing of the label (TROIKA / SPC convention):**
- Window length **T = 8 s**  
- Step **S = 2 s** → **6 s overlap** (75% overlap)  
- Each BPM value is the ECG-derived mean HR for that 8 s window  

**Evidence:** (`peer-reviewed`) Zhang et al. (2015) TROIKA, §III (T=8, S=2); community dataset notes describing `BPM0` as BPM every 8 s with 6 s overlap.

### 3.2 Implications for *this* project’s stress-classification goal

| Intended project construct | IEEEPPG provides? |
|----------------------------|-------------------|
| Binary rest vs physical stress / overload | **No** native labels |
| Multi-class activity (rest/walk/run/bike) | **No** (unlike PhysioNet Wrist PPG) |
| Continuous HR during exercise | **Yes** (regression target) |

**Decision framing (`project-implementation` — proposal, not yet adopted):**  
IEEEPPG is **misaligned as a direct supervised stress classifier dataset**. It is appropriate as:
1. a **PPG-under-motion** stress-*proxy* only if labels are **derived** (e.g. HR thresholds, speed stages) — that derivation must be justified separately and is **not** ground-truth “stress”; or  
2. a **cross-sensor / HR estimation** auxiliary experiment; or  
3. excluded from RQ1-style classification in favor of EPHNOGRAM + Wrist PPG.

**Uncertainty:** Any claim that IEEEPPG “has stress labels” would be **incorrect** based on current documentation.

---

## 4. Sampling frequency and window duration

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Sampling rate | **125 Hz** | TROIKA; Zenodo; TSER |
| Analysis window | **8 s** (1000 samples) | TROIKA; TSER IEEEPPG length 1000 |
| Hop / overlap | **2 s hop**, **6 s overlap** | TROIKA; SPC BPM traces |
| Nyquist | 62.5 Hz | — |

**Contrast with current project defaults** (`config/settings.py`, note 04): EPHNOGRAM/Wrist pipeline uses **60 s non-overlapping** windows for HRV. IEEEPPG’s native **8 s** windows are an order of magnitude shorter — this dominates feature reliability (§7).

---

## 5. How published papers commonly preprocess IEEE SPC / IEEEPPG

Common pipeline elements (HR **estimation** literature, not HRV-stress classification):

1. **Segment** 8 s windows, 2 s step (dataset-native).  
2. **Bandpass** PPG (and often ACC) roughly in the cardiac band. TROIKA uses **0.4–5 Hz** before decomposition (`peer-reviewed` Zhang et al., 2015, Fig. 1 / §III). Later works often use similar bands, e.g. **0.4–5 Hz** or **0.2–5 Hz** Butterworth (`peer-reviewed` example: Biswas et al.-style / ACCESS 2019 PPG-HR papers citing 0.2–5 Hz — confirm per paper when citing a specific method).  
3. **Optional downsample** PPG to **25 Hz** for HR spectral methods (computational; still ≥ theoretical minimum discussed for HR from PPG in some works) — e.g. reported in PPG-during-exercise HR papers (`peer-reviewed` context: https://doi.org/10.1109/ACCESS.2019.2913148 — **verify downsample rationale in that paper before citing numbers in the thesis**).  
4. **Channel fusion:** z-score each PPG channel then **average** the two PPG channels to reduce uncorrelated noise (reported in multiple SPC follow-ups).  
5. **Motion artifact handling:** use **accelerometer** via spectrum subtraction, adaptive filtering, SSA/EMD decomposition, sparse reconstruction (TROIKA), or deep models (e.g. Deep PPG, Sensors 2019).  
6. **Target:** regress or track **BPM**, evaluate MAE / Pearson vs ECG-BPM — **not** F1 for stress classes.

**Deep PPG / TSER usage:** treat windows as multivariate series → predict scalar BPM (`peer-reviewed` Reiss et al., Deep PPG, *Sensors* 2019, https://doi.org/10.3390/s19143079 — describes IEEE_Training at 125 Hz, 8 s / 2 s).

**Alternative approaches in literature:**
- Classical TROIKA / JOSS / SPECTRAP family (signal decomposition + spectral peak tracking)  
- Adaptive noise cancellation with ACC reference  
- End-to-end CNN/LSTM HR estimators  
- Quality indexing + reject windows instead of hard MA removal  

None of these pipelines are designed primarily to output **HeartPy-style HRV feature vectors for stress ML**.

---

## 6. Do HeartPy preprocessing recommendations apply?

### 6.1 What HeartPy recommends (PPG)

**Evidence (`heartpy-docs`):**
- Bandpass example for PPG-like signals: **`cutoff=[0.75, 3.5]` Hz**, `filtertype='bandpass'` — Python Heart Rate Analysis Toolkit docs: https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/heartpy.filtering.html  
- Peak detection oriented to **PPG pulse peaks** with adaptive moving-average threshold (~0.75 s window) — van Gent et al. / HeartPy papers.  
- `hp.process(segment, sample_rate=...)` expects a **1-D** cardiac waveform (single PPG or ECG), not a 5-D PPG+ACC tensor.  
- Default BPM clip often **40–180** (`bpmmin`/`bpmmax`) — may be **too low for intensive running** peaks in SPC (TROIKA reports speeds up to ~15 km/h); raising `bpmmax` is a likely **project** necessity **`[NEEDS VERIFICATION]` on empirical HR range in BPM traces**.

**Evidence (`peer-reviewed` / HeartPy):**
- van Gent, P., Farah, H., van Nes, N., & van Arem, B. (2019). Analysing noisy driver physiology… *Journal of Open Research Software* / related HeartPy publications; TRF paper doi:10.1016/j.trf.2019.09.015 — **confirm exact bibliographic mapping when citing**.

### 6.2 Fit / mismatch with IEEEPPG

| HeartPy assumption | IEEEPPG reality | Verdict |
|--------------------|-----------------|--------|
| Relatively wearable PPG with moderate noise | **Severe motion artifact** during intensive exercise | HeartPy peak detection **often fails** without MA reduction — expected high discard rate |
| Band 0.75–3.5 Hz (~45–210 BPM) | TROIKA uses **0.4–5 Hz**; exercise HR may exceed 3.5 Hz (210 BPM) | **0.75–3.5 may clip high HR**; prefer wider band for this corpus or raise upper cutoff |
| Single channel | Two PPG + ACC available | Must **select/fuse PPG**; ACC not an input to `hp.process` |
| Longer stable segments for HRV | Native **8 s** windows | Many HRV metrics **under-supported** (§7) |
| Designed partly for noisy PPG | Still not a TROIKA-level MA remover | HeartPy ≠ substitute for ACC-aware denoising |

**Conclusion:** HeartPy **can** be applied to **cleaned 1-D PPG** segments at 125 Hz, but **stock HeartPy preprocessing alone is not best practice** for IEEEPPG. Literature-standard SPC preprocessing (bandpass 0.4–5 Hz, dual-PPG fusion, ACC-aware MA handling) should precede HeartPy, and HRV interpretation must respect 8 s limits.

---

## 7. Feature-by-feature reliability (requested HeartPy set)

Context for beat counts: at HR = 60–180 BPM, an **8 s** window contains roughly **8–24** beats (fewer after artifact rejection). Ultra-short HRV validity is contested; conservative norms differ from permissive ones.

**Key references for duration:**
- (`peer-reviewed`) Task Force of the ESC/NASPE (1996). Heart rate variability standards. *European Heart Journal* / *Circulation* — classical **~5 min** short-term HRV; spectral measures need adequate length (LF especially).  
- (`peer-reviewed`) Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. *Frontiers in Public Health*. https://doi.org/10.3389/fpubh.2017.00258  
- (`peer-reviewed`) Shaffer, F., Meehan, Z. M., & Zerr, C. L. (2020). A critical review of ultra-short-term heart rate variability norms research. *Frontiers in Neuroscience*. https://doi.org/10.3389/fnins.2020.594880 — summarizes conservative UST minima (adapted from Shaffer et al., 2019): e.g. **HR ~10 s**; **RMSSD/SDNN/pNN50 ~60 s**; **SD1/SD2 ~90 s** under strict agreement criteria.  
- (`peer-reviewed`) Munoz, M. L., et al. (2015). Validity of (ultra-)short recordings for HRV. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0138921 — more permissive for **RMSSD** (even **10 s** in large sample); **SDNN** often needs longer / multiple shorts; **does not endorse frequency-domain** on &lt;60–120 s.

**Stress-task appropriateness:** Even if a number is computable, IEEEPPG labels are **BPM**, not stress. Features may correlate with **exercise intensity / HR level** (construct confounding) rather than “mission stress.”

| Feature | Computable by HeartPy on 8 s PPG? | Scientifically reliable on 8 s? | Appropriate for *stress classification* on IEEEPPG? | Notes |
|---------|-----------------------------------|--------------------------------|------------------------------------------------------|-------|
| **BPM** | Yes (primary strength) | **Yes** (UST HR often OK from ~10 s; here 8 s is borderline but matches dataset design) | **Indirect only** — BPM **is the label**, using BPM as a feature to predict BPM is leakage; for stress classes, BPM is an exertion proxy, not stress | Prefer ECG-BPM target for HR tasks; for classification, BPM dominates “intensity” |
| **IBI** (mean) | Yes if ≥2 peaks | **Mostly yes** (mean IBI ≈ 60000/BPM) | Same caveats as BPM | Highly redundant with BPM |
| **SDNN** | Numerically yes with few IBIs | **Poor / contested** on 8 s; conservative norms favor **≥60 s** | **Weak** for stress claims on this set | Unstable with 8–20 IBIs; MA → false SDNN inflation |
| **RMSSD** | Numerically yes | **Contested**: Munoz et al. support ultra-short RMSSD; Shaffer conservative line wants **~60 s** | **Limited** — may reflect noise/MA as much as vagal tone under running | Best *candidate* among variability features if peaks are clean |
| **SDSD** | Yes (closely related to RMSSD) | Same as RMSSD | Same as RMSSD | Often near-redundant with RMSSD |
| **pNN20** | Yes | **Unreliable** with few intervals; pNN family sensitive to length | **Poor** | Sparse counts → unstable percentages |
| **pNN50** | Yes | Conservative UST: **~60 s**; 8 s inadequate | **Poor** | Even worse than pNN20 at high HR (few diffs &gt;50 ms) |
| **HR_MAD** | Yes (HeartPy-specific robust spread) | **Limited evidence** as standardized HRV biomarker | **Unclear** for stress; treat as **exploratory** (`heartpy-docs` / project) | Not a Task Force standard metric |
| **SD1** | Yes (Poincaré) | Conservative UST often **~90 s**; SD1≈RMSSD/√2 mathematically related | **Limited** (similar to RMSSD) | Short Poincaré plots are sparse |
| **SD2** | Yes | Conservative UST **~90 s**; needs more intervals than SD1 | **Poor** on 8 s | Long-axis of Poincaré poorly estimated |
| **Breathing Rate** | HeartPy may return a value | **Not reliable** on 8 s exercise PPG | **No** for this dataset/task | Respiratory estimation needs longer, cleaner segments; exercise + MA violate assumptions |

**LF / HF / LF/HF** (not in the user list, but relevant to project-wide HeartPy set): **Not appropriate** on 8 s windows per Task Force / Shaffer guidance — do **not** add them for IEEEPPG native windows.

### 7.1 Summary recommendation on the feature set

**Relatively justifiable on native 8 s IEEEPPG PPG (with strong QC):** BPM, IBI, possibly RMSSD/SDSD/SD1 **as exploratory ultra-short features**, HR_MAD exploratory.

**Not scientifically appropriate to treat as standard HRV for stress inference on 8 s IEEEPPG:** pNN20, pNN50, SDNN (strict interpretation), SD2, Breathing Rate.

**Fundamental task mismatch:** Without derived or external stress labels, **none** of these features should be framed as solving RQ1 “stress vs rest” **on IEEEPPG alone**.

---

## 8. Proposed preprocessing pipeline (best suited — **not implemented**)

Goal options must be chosen explicitly:

### Option α — Stay compatible with dataset-native HR regression (literature-aligned)

1. Load packaging A or B; keep subject/session IDs if available (A better for LOSO).  
2. For each 8 s window (or use pre-cut TSER instances):  
   - Bandpass PPG **0.4–5 Hz** (TROIKA) — **not** 0.75–3.5 without checking max HR.  
   - Z-score and **average dual PPG** (if 2 channels).  
   - Optional: ACC-based MA attenuation (TROIKA-style or simpler spectrum subtraction) **before** peak detection.  
3. **Do not downsample below ~25–50 Hz** if using HeartPy peaks; **prefer keep 125 Hz** for peak timing.  
4. Run HeartPy `process` on 1-D PPG; set `bpmmax` to cover exercise (e.g. ≥200) after checking BPM traces.  
5. Discard windows with peak-detection failure or extreme IBI outliers.  
6. Supervise with **ECG-BPM** (regression) — metrics MAE/RMSE, not stress F1.

### Option β — Force HRV feature table for ML (only if project insists)

1. Prefer **packaging A** continuous recordings; build **≥60 s** (better 120–300 s) segments where protocol allows, **or** accept ultra-short features with explicit limitation language.  
2. MA reduction **mandatory** (ACC-aware).  
3. HeartPy on cleaned PPG; export only features justified in §7.  
4. Labels: **must define** a transparent proxy (e.g. speed stage / HR quintile) and state it is **not** psychometric stress (`project-implementation`).  
5. Evaluation: LOSO by subject; never random-split overlapping 8 s windows (severe leakage due to 75% overlap).

### Option γ — Do not use IEEEPPG for stress classification

Keep EPHNOGRAM + Wrist PPG for RQ1–RQ3; optionally use IEEEPPG only for a **PPG HR-under-motion** appendix. **This is the most coherent with current RQs.**

**Proposed default for this repository (`project-implementation` recommendation):** **Option γ** for main RQs; if IEEEPPG is ingested, implement **Option α** first. Do **not** silently reuse the 60 s EPHNOGRAM pipeline on 8 s TSER clips.

---

## 9. Preprocessing decisions (pending implementation)

| Decision | Choice | Status |
|----------|--------|--------|
| Role of IEEEPPG in project | Prefer **HR-under-motion / optional**; not primary stress labels | Proposed |
| Packaging to support | Detect A vs B; document in loader | Pending files |
| Windowing | Native **8 s / 2 s** for HR; **≥60 s** only if rebuilding from continuous A | Proposed |
| Bandpass | **0.4–5 Hz** (SPC/TROIKA) as default for this dataset; HeartPy 0.75–3.5 only if HR&lt;~210 and empirically validated | Proposed |
| Channels into HeartPy | Fused or best single **PPG**; ACC for MA only | Proposed |
| Feature set | Restrict per §7; exclude Breathing Rate, pNN50, SD2 as “standard HRV” claims | Proposed |
| Overlap handling in ML | Group by subject; ban i.i.d. splits on 75%-overlap windows | Proposed |

---

## 10. Advantages / disadvantages of using IEEEPPG here

**Advantages:**
- Gold-standard benchmark for **wrist PPG + motion**  
- Dual PPG + ACC enable realistic wearable denoising  
- Large number of windows (3096 in TSER form)  
- Strong published baselines for **BPM estimation**

**Disadvantages:**
- **No stress/rest labels**  
- **8 s** windows hostile to many HRV features  
- Extreme MA → HeartPy failure / biased HRV  
- TSER form drops ECG and session structure details  
- Overlapping windows → leakage if CV is naive  
- Population/protocol ≠ soldiers/firefighters operational stress  

---

## 11. Limitations and uncertainty log

1. **Local files absent** — structure not verified by `scipy.io.loadmat` / `.ts` parse in this repo.  
2. Zenodo blurb vs TSER **5-D** description — ECG presence in downloadable IEEEPPG **[NEEDS VERIFICATION]**.  
3. Exact MATLAB `sig` row order **[NEEDS VERIFICATION]** from official readme.  
4. Ultra-short HRV: literature **disagrees** (Munoz vs Shaffer conservative criteria); paper text must present both.  
5. HeartPy breathing-rate algorithm assumptions under running **[NEEDS VERIFICATION]** against HeartPy docs/source — until then, mark **inappropriate**.  
6. LED wavelength: some secondary pages say 515 nm, one GitHub mirror text said 609 nm — trust TROIKA/Zenodo **515 nm** unless contradicted by primary readme.

---

## 12. References (do not invent beyond these; verify PDFs when writing the thesis)

1. Zhang, Z., Pi, Z., & Liu, B. (2015). TROIKA… *IEEE TBME*, 62(2), 522–531. https://doi.org/10.1109/TBME.2014.2359372  
2. Tan, C. W., Bergmeir, C., Petitjean, F., & Webb, G. I. (2020). Monash University, UEA, UCR Time Series Extrinsic Regression Archive. arXiv:2006.10996; (2021) *Data Min. Knowl. Disc.* https://doi.org/10.1007/s10618-021-00745-9  
3. Zenodo IEEEPPG record: https://zenodo.org/records/3902710  
4. TSER IEEEPPG listing: http://tseregression.org/  
5. Reiss, A., et al. (2019). Deep PPG… *Sensors*, 19(14), 3079. https://doi.org/10.3390/s19143079  
6. Task Force ESC/NASPE (1996). Heart rate variability standards.  
7. Shaffer, F., & Ginsberg, J. P. (2017). HRV metrics and norms. *Front. Public Health*. https://doi.org/10.3389/fpubh.2017.00258  
8. Shaffer, F., Meehan, Z. M., & Zerr, C. L. (2020). Ultra-short-term HRV norms review. *Front. Neurosci*. https://doi.org/10.3389/fnins.2020.594880  
9. Munoz, M. L., et al. (2015). Validity of (ultra-)short HRV recordings. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0138921  
10. HeartPy filtering docs: https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/heartpy.filtering.html  
11. van Gent et al. HeartPy / noisy PPG analysis literature (confirm DOI 10.1016/j.trf.2019.09.015 when citing).  

**Optional follow-up citations when implementing MA methods** (locate before use): JOSS, SPECTRAP, and ACCESS 2019 PPG-HR exercise papers — mark **[NEEDS VERIFICATION]** until PDF-checked.

---

## 13. Code linkage

**None yet** (by request). Future implementation should update this note and add loaders under `src/data/` only after packaging A/B is chosen and files exist under `data/raw/ieeeppg/`.

---

## 14. Bottom line

IEEEPPG is a **125 Hz, 8 s-window, PPG+ACC (ECG in original form), continuous BPM** benchmark for **heart rate during intense motion**. It is **not** a stress-classification dataset. HeartPy’s PPG bandpass/peak tools are only partially applicable; **TROIKA-style banding + MA handling** fit the corpus better. Of the requested features, only **BPM/IBI** (and cautiously **RMSSD/SDSD/SD1**) are defensible on native windows; **pNN\*, SDNN (strict), SD2, Breathing Rate** are not appropriate to claim as reliable HRV for stress ML without longer segments and cleaner peaks.
