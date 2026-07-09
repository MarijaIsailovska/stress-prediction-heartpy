# 01 — Dataset selection

**Paper section:** Methods — Data Sources  
**Related overview:** `research_notes.md` § SECTION 1

---

## Decision — 2026-07-09 — Use EPHNOGRAM (ECG) + Wrist PPG During Exercise

**Decision:** Primary datasets are PhysioNet **EPHNOGRAM 1.0.0** (chest ECG) and **Wrist PPG During Exercise 1.0.0** (wrist PPG). No public soldier/firefighter operational-load corpus is used.

**Why:** The application scenario (mission withdrawal under physiological overload) lacks an open labeled dataset with soldiers/firefighters. These corpora provide physically active adults, continuous cardiac signals in wearable-relevant modalities, and protocol conditions that can be mapped to rest vs exertion for supervised learning.

| Dataset | Modality | Sampling | Subjects (as documented) | Role in study |
|---------|----------|----------|--------------------------|---------------|
| [EPHNOGRAM 1.0.0](https://physionet.org/content/ephnogram/1.0.0/) | Chest ECG (+ PCG) | 8000 Hz (we resample; see note 04) | 24 healthy adult males, age 23–29 (`dataset-docs`) | RQ1 primary ECG |
| [Wrist PPG During Exercise 1.0.0](https://physionet.org/content/wrist/1.0.0/) | Wrist PPG | 256 Hz | 8 subjects (`dataset-docs`) | RQ2 + cross-sensor |

**Advantages:**
- Open access via PhysioNet; reproducible downloads
- Clear protocol structure suitable for binary rest/exertion labels (see note 03)
- Dual modality supports ECG vs PPG and cross-sensor questions (RQ2, EXTRA)

**Disadvantages:**
- **Population mismatch:** young healthy males / lab exercise ≠ soldiers/firefighters under mission load (`project-implementation` limitation; ecological validity gap)
- **Construct mismatch:** physical exertion is a **proxy** for “physiological overload / stress,” not validated psychological or operational stress
- Wrist set is small (8 subjects), limiting LOSO stability

**Alternatives in literature:**
- Affective/stress corpora with wearables (e.g. WESAD) — stronger for *psychological* stress, weaker for heavy physical load protocols (`peer-reviewed` context: Schmidt et al., 2018 WESAD — full citation details **[NEEDS VERIFICATION]** against the published paper)
- Other PhysioNet exercise/ECG sets — not selected in the initial project design; a systematic PhysioNet comparison was noted as “Cursor analysis … (2025)” in `research_notes.md` and is **not** a peer-reviewed source (`project-implementation` / informal review)

**Evidence:**
- (`dataset-docs`) PhysioNet EPHNOGRAM: https://physionet.org/content/ephnogram/1.0.0/
- (`dataset-docs`) PhysioNet Wrist PPG During Exercise: https://physionet.org/content/wrist/1.0.0/
- (`project-implementation`) Choice of these two datasets as proxies for the course application scenario; recorded in `research_notes.md` and `config/`

**Uncertainty:**
- Claim “no public dataset exists with soldiers/firefighters” is a project working assumption — **[NEEDS VERIFICATION]** against a documented literature/database search log (none stored in-repo yet)
- Subject counts/ages above should be confirmed against the current PhysioNet dataset descriptions when writing the paper

**Code:** `config/settings.py`, `config/ephnogram_records.py`, `config/wrist_records.py`, `download_only_needed.py`, `download_wrist.py`  
**Paper section:** Methods — Data Sources
