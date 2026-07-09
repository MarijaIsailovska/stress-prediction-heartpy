"""
Classifier factory: Random Forest, SVM (RBF), KNN, Logistic Regression,
and a soft-voting ensemble (RF + SVM + KNN).
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config.settings import KNN_PARAMS, LR_PARAMS, RF_PARAMS, SVM_PARAMS


def _scale_pipeline(estimator) -> Pipeline:
    """Wrap estimator with StandardScaler (needed for SVM/KNN/LR)."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )


def build_classifiers(include_extra: bool = True) -> dict[str, Any]:
    """
    Build the model zoo for the study.

    Required: Random Forest, SVM (RBF), KNN
    Extra: Logistic Regression, Voting Ensemble (RF+SVM+KNN)
    """
    rf = RandomForestClassifier(**RF_PARAMS)
    svm = _scale_pipeline(SVC(**SVM_PARAMS))
    knn = _scale_pipeline(KNeighborsClassifier(**KNN_PARAMS))

    models: dict[str, Any] = {
        "RandomForest": rf,
        "SVM_RBF": svm,
        "KNN": knn,
    }

    if include_extra:
        lr = _scale_pipeline(LogisticRegression(**LR_PARAMS))
        # Fresh estimators for the ensemble (sklearn clones internally on fit)
        rf_e = RandomForestClassifier(**RF_PARAMS)
        svm_e = _scale_pipeline(SVC(**SVM_PARAMS))
        knn_e = _scale_pipeline(KNeighborsClassifier(**KNN_PARAMS))
        voting = VotingClassifier(
            estimators=[
                ("rf", rf_e),
                ("svm", svm_e),
                ("knn", knn_e),
            ],
            voting="soft",
            n_jobs=RF_PARAMS.get("n_jobs", -1),
        )
        models["LogisticRegression"] = lr
        models["VotingEnsemble"] = voting

    return models


def get_model(name: str, include_extra: bool = True):
    """Return a single named classifier instance."""
    models = build_classifiers(include_extra=include_extra)
    if name not in models:
        raise KeyError(f"Unknown model '{name}'. Available: {list(models)}")
    return models[name]
