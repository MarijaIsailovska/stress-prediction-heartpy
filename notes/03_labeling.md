# 03 — Labeling strategy

**Paper section:** Methods — Labeling Strategy  
**Related overview:** `research_notes.md` § SECTION 3

---

## Decision — 2026-07-09 — Protocol-level binary labels (recording = one label)

**Decision:**
- EPHNOGRAM: entire recording labeled from protocol — rest → `0`; Bruce treadmill / bicycle stress test → `1`
- Wrist PPG: map multi-class activity to binary — `rest → 0`; `walk / run / bike → 1` (exertion as stress **proxy**)

**Why:** Protocol (periodically defined) labels avoid noisy self-report affect labels and match the study focus on **physical** overload rather than subjective stress ratings.

**Advantages:**
- Simple, reproducible labeling without per-window annotation
- Literature on affect/stress detection often reports better performance with protocol/periodic labels than self-report alone (**[NEEDS VERIFICATION]** of exact wording against Vos et al. 2023)

**Disadvantages:**
- Ignores within-recording transitions (warm-up, recovery, stage changes in Bruce protocol)
- Equating walk/run/bike with “stress” conflates metabolic load with stress/overload constructs
- Label noise if protocol metadata are wrong for a file

**Alternatives in literature:**
- Self-report / VAS / STAI-style labels (common in affective computing)
- Stage-wise labels within Bruce protocol
- Multi-class activity recognition (keep walk/run/bike separate) instead of binary collapse
- Physiological ground truth (cortisol, etc.) — rarely available in open wearable sets

**Evidence:**
- (`dataset-docs`) EPHNOGRAM protocol conditions (rest vs stress test) — PhysioNet EPHNOGRAM documentation
- (`dataset-docs`) Wrist activity classes rest/walk/run/bike — PhysioNet wrist dataset documentation
- (`project-implementation`) Binary maps in `config/ephnogram_records.py`, `config/wrist_records.py`
- (`peer-reviewed`) Schmidt et al. (2018), WESAD — cited for protocol-oriented wearable affect/stress benchmarking context; **full bibliographic details [NEEDS VERIFICATION]**
- (`peer-reviewed`) Vos et al. (2023) systematic review — overview quotes higher accuracy for periodically labeled data; **exact quotation and bibliographic details [NEEDS VERIFICATION]** against the published review

**Uncertainty:**
- Do not claim clinical validity of “stress” labels for operational withdrawal decisions without additional validation literature

**Code:** `config/ephnogram_records.py` (`RECORD_LABELS`), `config/wrist_records.py` (`WRIST_BINARY_MAP`, `to_binary_label`), processing scripts `01`/`02`  
**Paper section:** Methods — Labeling Strategy
