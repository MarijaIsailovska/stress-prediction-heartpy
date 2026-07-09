# 02 — Quality filtering (EPHNOGRAM)

**Paper section:** Methods — Data Preprocessing  
**Related overview:** `research_notes.md` § SECTION 2

---

## Decision — 2026-07-09 — Retain Good / 30 min EPHNOGRAM recordings only

**Decision:** Use only EPHNOGRAM recordings with **ECG Notes = Good** and **30 min** duration, as listed in the project record lists. Retained IDs (`ECGPCG00XY` → XY):

- **REST (label=0):** 10, 11, 13, 14, 15, 16, 21, 22, 23  
- **STRESS (label=1):** 01, 25, 27, 29, 30, 32, 33, 34, 36, 38, 47, 52, 55, 61, 62, 64, 66, 67, 68  

**Nominal sample budget:** 28 recordings × ~30 non-overlapping 60 s windows ≈ 840 HRV windows before HeartPy discard (`project-implementation` estimate).

**Why:** Restricting to high-quality, fixed-duration recordings reduces artifact-driven HRV failures and aligns the cohort with prior analyses that used the same spreadsheet criteria.

**Advantages:**
- Improves signal usability for automated peak detection
- Fixed duration simplifies window counts across recordings
- Facilitates comparison with prior work using the same filter

**Disadvantages:**
- Reduces available data and may bias toward “easier” recordings
- Spreadsheet-based notes are subjective quality labels
- Skipping incomplete downloads (e.g. historically problematic IDs) further shrinks the set (`project-implementation`)

**Alternatives in literature:**
- Automated signal-quality indices (SQI) instead of spreadsheet notes
- Using all recordings with per-window quality rejection only
- Manual annotation of usable segments

**Evidence:**
- (`dataset-docs`) Filtering via `ECGPCGSpreadsheet.csv` fields (ECG Notes, duration) — confirm column names against the file when available
- (`project-implementation`) Hard-coded lists in `config/ephnogram_records.py`
- Prior work cited in overview (same 28-recording / stress-test criteria):
  - https://arxiv.org/html/2506.10212v1  
  - https://arxiv.org/pdf/2410.19667  
  **[NEEDS VERIFICATION]** — treat as preprints unless/until peer-reviewed status and exact filtering claims are checked in the PDFs

**Uncertainty:**
- Exact spreadsheet column names and whether every listed ID still meets Good/30 min on the current PhysioNet release — **[NEEDS VERIFICATION]** after placing `ECGPCGSpreadsheet.csv` under `data/raw/ephnogram/`

**Code:** `config/ephnogram_records.py` (`REST_RECORDS`, `STRESS_RECORDS`), `scripts/01_process_ephnogram.py`  
**Paper section:** Methods — Data Preprocessing

---

## Decision — 2026-07-09 — Exclude stress records 55, 61, 62 (unavailable locally)

**Decision:** Remove `55`, `61`, and `62` from `STRESS_RECORDS`. Usable cohort is now **25 recordings (9 rest + 16 stress)**.

**Updated stress IDs:** 01, 25, 27, 29, 30, 32, 33, 34, 36, 38, 47, 52, 64, 66, 67, 68

**Why:** These three Good/30 min stress recordings from the original quality-filtered list are not available in the local `data/raw/ephnogram/` download set, so including them would cause processing failures or empty contributions.

**Advantages:**
- Pipeline runs only on files that exist locally
- Avoids repeated FileNotFound errors in `scripts/01_process_ephnogram.py`

**Disadvantages:**
- Shrinks the stress class vs the original 19-stress / 28-recording design (~840 → ~750 nominal 60 s windows before HeartPy discard)
- Slightly reduces comparability with prior work that used the full 28-recording Good/30 min set
- Availability-based exclusion is an acquisition constraint, not an additional clinical quality criterion

**Alternatives in literature / practice:**
- Re-download or obtain the missing PhysioNet files and restore the three IDs
- Keep IDs in config but skip missing files at runtime with logging (softer exclusion)

**Evidence:**
- (`project-implementation`) Local file availability check; lists updated in `config/ephnogram_records.py`
- (`dataset-docs`) Original Good/30 min inclusion of 55/61/62 remains valid on PhysioNet in principle — exclusion is local-only unless confirmed missing upstream **[NEEDS VERIFICATION]** if re-download is attempted

**Uncertainty:** Whether 55/61/62 failed permanently on PhysioNet vs incomplete local download — retry download before claiming permanent exclusion in the paper.

**Code:** `config/ephnogram_records.py`  
**Paper section:** Methods — Data Preprocessing

---

## Decision — 2026-07-09 — Exclude ECGPCG0014 (powerline noise) + rest BPM>120 QC

**Decision:**
1. Remove `"14"` from `REST_RECORDS` (cohort **8 rest + 16 stress = 24**).
2. Reject rest windows with BPM > 120 (`rest_bpm_too_high`).
3. Log per-window rejection reasons in `scripts/01_process_ephnogram.py`.

**Why:** Spreadsheet ECG Notes for 0014 = Powerline noise (not Good). Rest
BPM>120 removes artifactual survivors (e.g. ECGPCG0021) without changing
stress BPM bounds.

**Evidence:** (`dataset-docs`) ECGPCGSpreadsheet.csv; (`project-implementation`)
config + script 01. Full write-up: `research_notes.md` § Data Quality Corrections.

**Code:** `config/ephnogram_records.py`, `scripts/01_process_ephnogram.py`  
**Paper section:** Methods — Data Preprocessing

---

## Decision — 2026-07-09 — Relax RMSSD cap to 400 ms + RMSSD>3×IBI rule

**Decision:** Change absolute RMSSD reject from **>300 ms** to **>400 ms**;
add reject if **RMSSD > 3 × IBI** (`rmssd_exceeds_ibi`).

**Why:** 300 ms was discarding large fractions of valid-looking rest windows
in young adults; 400 ms targets extreme outliers only. The IBI-relative rule
catches mathematically implausible peak-detection failures.

**Evidence:** (`peer-reviewed`) Shaffer & Ginsberg (2017); Clifford et al.
(2006). Full text: `research_notes.md` § QC Threshold Revision.
Exact “20–200 ms normal” claim **[NEEDS VERIFICATION]** against Shaffer tables.

**Code:** `scripts/01_process_ephnogram.py`  
**Paper section:** Methods — Data Preprocessing
