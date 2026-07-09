# 06 — ML models and evaluation

**Paper section:** Methods — Model Evaluation  
**Related overview:** `research_notes.md` § SECTION 6

---

## Decision — 2026-07-09 — LOSO CV, SMOTE on train only, classical model zoo

**Decision:**
- **Cross-validation:** Leave-One-Subject-Out (LOSO)
- **Imbalance:** SMOTE on each **training** fold only (never on the held-out subject)
- **Required models:** Random Forest, SVM (RBF), k-NN
- **Extra models:** Logistic Regression; soft Voting ensemble (RF + SVM + k-NN)
- **Metrics:** Accuracy, F1-macro, ROC-AUC, confusion matrix; later RF impurity importance + SHAP (script 05)

**Why:** Windows from the same subject are dependent (baseline HRV, fitness). Random splits leak subject identity into the test set and inflate metrics. SMOTE on the test fold would leak synthetic structure derived from test labels/distribution.

**Advantages:**
- Subject-wise generalization estimate closer to deployment on a new person
- Transparent classical baselines appropriate for small tabular HRV feature sets
- Ensemble provides a simple non-neural combination baseline

**Disadvantages:**
- LOSO variance is high with few subjects (especially wrist, n=8)
- SMOTE assumes local density in feature space; can distort physiology-derived manifolds
- Default hyperparameters may be suboptimal (`project-implementation` defaults in `config/settings.py`)

**Alternatives in literature:**
- Leave-one-recording-out or group k-fold by session
- Class weights instead of SMOTE
- Nested CV for hyperparameter tuning
- Deep models on raw signals
- Domain-adversarial / subject-invariant representation learning

**Evidence:**
- (`peer-reviewed`) Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357. — oversampling method; **application “train fold only” is standard ML practice** (`project-implementation` enforcement in code)
- (`peer-reviewed`) Gjoreski et al. (2016) — cited in overview for wearable stress / evaluation context; **full citation and relevance claim [NEEDS VERIFICATION]**
- (`peer-reviewed`) Vos et al. (2023) systematic review — evaluation / generalizability context; **[NEEDS VERIFICATION]** of bibliographic details
- (`project-implementation`) `src/models/evaluation.py` (`loso_evaluate`, SMOTE helper), `src/models/classifiers.py`, `scripts/03_train_evaluate.py`

**Uncertainty:**
- Subject IDs for EPHNOGRAM: default per-recording IDs unless spreadsheet refresh merges true subjects (`config/ephnogram_records.py`) — incorrect subject mapping would **weaken** LOSO validity **[NEEDS VERIFICATION]** once spreadsheet is parsed

**Code:** `src/models/classifiers.py`, `src/models/evaluation.py`, `scripts/03_train_evaluate.py`, `scripts/05_feature_importance.py`  
**Paper section:** Methods — Model Evaluation
