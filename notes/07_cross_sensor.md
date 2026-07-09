# 07 — Cross-sensor generalization

**Paper section:** Results + Discussion (experiment design also belongs in Methods)  
**Related overview:** `research_notes.md` § SECTION 7

---

## Decision — 2026-07-09 — Train on EPHNOGRAM ECG features → test on Wrist PPG features

**Decision:** Fit classifiers on the EPHNOGRAM HRV feature table; evaluate on the Wrist PPG HRV feature table using the **same 15 features** and binary rest/exertion labels. No raw-signal domain adaptation.

**Why:** Addresses whether a chest-ECG-trained overload detector can transfer to wrist-PPG smartwatch-like sensing relevant to field wearables (soldiers/firefighters scenario), using HRV as a shared representation.

**Advantages:**
- Directly tests modality shift under a common feature space
- No need for paired ECG–PPG from the same subjects
- Aligns with EXTRA research question

**Disadvantages:**
- Dataset shift confounds sensor shift (different people, protocols, environments)
- Binary label semantics differ slightly (Bruce/bike stress test vs walk/run/bike)
- Negative transfer is likely; interpretation must separate sensor vs population effects

**Alternatives in literature:**
- Train/test both directions (PPG→ECG)
- Multi-dataset training
- Unsupervised domain adaptation / CORAL / adversarial alignment on feature space
- Same-subject multi-sensor recordings (ideal but not available here)

**Evidence:**
- (`project-implementation`) Experiment wiring in `src/models/cross_sensor.py`, `scripts/04_cross_sensor_test.py`
- (`peer-reviewed`) Vos et al. (2023) — overview cites cross-dataset generalizability discussion; **[NEEDS VERIFICATION]**
- (`peer-reviewed`) Schmidt et al. (2022) — overview cites cross-dataset ECG stress detection; **full citation [NEEDS VERIFICATION]** — do not assert findings until the paper is located

**Uncertainty:**
- Cross-dataset citations in the overview are placeholders until verified; frame results as an **engineering transfer experiment** until literature alignment is confirmed

**Code:** `src/models/cross_sensor.py`, `scripts/04_cross_sensor_test.py`  
**Paper section:** Methods (experiment design) + Results/Discussion
