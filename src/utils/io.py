"""Filesystem and serialization helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config.settings import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


def ensure_dirs(*paths: Path) -> None:
    """Create directories if they do not exist."""
    defaults = (PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR)
    for p in paths or defaults:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path | str, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False, **kwargs)
    else:
        df.to_csv(path, index=False, **kwargs)
    logger.info("Saved DataFrame (%d rows) → %s", len(df), path)
    return path


def load_dataframe(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_json(obj: Any, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any):
        if hasattr(o, "tolist"):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)
    logger.info("Saved JSON → %s", path)
    return path


def load_json(path: Path | str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_model(model: Any, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model → %s", path)
    return path


def load_model(path: Path | str) -> Any:
    return joblib.load(path)
