# Methodology notes (Methods draft material)

These notes are the working **Methods** log for the Intelligent Systems project:
physiological overload detection from wearable ECG/PPG via HeartPy HRV features.

They are intended to be expanded into the paper Methods section. Style: precise,
decision-oriented, evidence-tagged. Do not invent citations.

## Conventions

### Evidence tags (required on claims)

| Tag | Meaning |
|-----|---------|
| `peer-reviewed` | Journal/conference paper or systematic review |
| `dataset-docs` | PhysioNet page, dataset paper, spreadsheet, README |
| `heartpy-docs` | HeartPy publication or library documentation |
| `project-implementation` | Choice made for this codebase only |

### Uncertainty

- Mark incomplete or unchecked statements: **`[UNVERIFIED]`** or **`[NEEDS VERIFICATION]`**
- Prefer omitting a citation over fabricating one

### Editing policy

- **Append** new decisions; do not overwrite prior rationale unless correcting an error
- New topic → new file (`09_....md`) and link it here
- Keep `research_notes.md` (repo root) as a short overview; detail lives here

### Entry template

```markdown
## Decision — YYYY-MM-DD — Short title

**Decision:** ...
**Why:** ...
**Advantages:** ...
**Disadvantages:** ...
**Alternatives in literature:** ...
**Evidence:**
- (peer-reviewed) ...
- (dataset-docs) ...
- (heartpy-docs) ...
- (project-implementation) ...
**Uncertainty:** ...
**Code:** `path/to/file.py` — symbol
**Paper section:** Methods — ...
```

## Index

| # | File | Paper section |
|---|------|----------------|
| 01 | [Dataset selection](01_dataset_selection.md) | Methods — Data Sources |
| 02 | [Quality filtering](02_quality_filtering.md) | Methods — Data Preprocessing |
| 03 | [Labeling](03_labeling.md) | Methods — Labeling Strategy |
| 04 | [Signal processing](04_signal_processing.md) | Methods — Signal Processing |
| 05 | [Feature extraction](05_feature_extraction.md) | Methods — Feature Extraction |
| 06 | [ML evaluation](06_ml_evaluation.md) | Methods — Model Evaluation |
| 07 | [Cross-sensor experiment](07_cross_sensor.md) | Results + Discussion |
| 08 | [Data download / storage](08_data_download.md) | Methods — Data Acquisition (supplementary) |
| — | [IEEEPPG analysis (no code yet)](ieeeppg_methodology.md) | Methods — candidate dataset / PPG-under-motion |
| — | EDA figures via `scripts/04_eda_visualizations.py` | Results — Exploratory analysis (see `research_notes.md`) |

## Research questions (context)

| ID | Question |
|----|----------|
| RQ1 | Can HRV features classify physical stress vs rest from ECG? |
| RQ2 | ECG (chest) vs PPG (wrist) — which yields better classification? |
| RQ3 | Which HRV features are strongest predictors regardless of sensor? |
| EXTRA | Train on EPHNOGRAM ECG → test on Wrist PPG |
