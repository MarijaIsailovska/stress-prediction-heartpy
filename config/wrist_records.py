"""
Wrist PPG During Exercise dataset configuration.

Source: PhysioNet wrist/1.0.0
Signal: PPG @ 256 Hz, 8 subjects
Activities: rest, walk, run, bike
"""

from __future__ import annotations

# Original multi-class activity labels
WRIST_ACTIVITY_LABELS: dict[str, int] = {
    "rest": 0,
    "walk": 1,
    "run": 2,
    "bike": 3,
}

# Binary mapping for RQ1/RQ2-style rest vs physical stress
# rest → 0; walk/run/bike → 1 (physical exertion / stress proxy)
WRIST_BINARY_MAP: dict[int, int] = {
    0: 0,  # rest
    1: 1,  # walk
    2: 1,  # run
    3: 1,  # bike
}

WRIST_SUBJECTS: list[str] = [
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "s8",
]

# Common WFDB record naming patterns for this dataset
# Actual filenames vary; loaders discover *.hea under data/raw/wrist/
WRIST_ACTIVITY_NAMES: list[str] = ["rest", "walk", "run", "bike"]


def to_binary_label(activity_label: int) -> int:
    """Map multi-class activity label to binary rest(0) / stress(1)."""
    if activity_label not in WRIST_BINARY_MAP:
        raise ValueError(f"Unknown activity label: {activity_label}")
    return WRIST_BINARY_MAP[activity_label]


def activity_name_to_label(name: str) -> int:
    """Map activity name string to multi-class label."""
    key = name.lower().strip()
    if key not in WRIST_ACTIVITY_LABELS:
        raise ValueError(f"Unknown activity name: {name}")
    return WRIST_ACTIVITY_LABELS[key]
