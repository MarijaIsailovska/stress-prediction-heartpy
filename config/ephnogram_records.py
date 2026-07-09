"""
EPHNOGRAM quality-filtered recordings and subject mapping.

Source: PhysioNet EPHNOGRAM 1.0.0
Filter: ECG Notes = Good, duration = 30 min (via ECGPCGSpreadsheet.csv)

Usable local cohort: 24 recordings (8 rest + 16 stress).
Records 55, 61, 62 were in the original Good/30 min stress list but are
excluded here because the corresponding files are not available locally.
Record 14 excluded: ECG Notes = Powerline noise (ECGPCGSpreadsheet.csv),
not Good quality.

REST  (label=0): XY 10,11,13,15,16,21,22,23
STRESS(label=1): XY 01,25,27,29,30,32,33,34,36,38,47,52,64,66,67,68
"""

from __future__ import annotations

# Record IDs as zero-padded two-digit strings matching ECGPCG00XY filenames
# 8 rest + 16 stress = 24 recordings
REST_RECORDS: list[str] = [
    "10",
    "11",
    "13",
    # 0014 excluded: ECG Notes = Powerline noise
    # (ECGPCGSpreadsheet.csv), not Good quality
    "15",
    "16",
    "21",
    "22",
    "23",
]

# 16 stress recordings (55, 61, 62 omitted — not available locally)
STRESS_RECORDS: list[str] = [
    "01",
    "25",
    "27",
    "29",
    "30",
    "32",
    "33",
    "34",
    "36",
    "38",
    "47",
    "52",
    "64",
    "66",
    "67",
    "68",
]

ALL_RECORDS: list[str] = REST_RECORDS + STRESS_RECORDS

# Recording → binary label (0=rest, 1=stress)
RECORD_LABELS: dict[str, int] = {
    **{r: 0 for r in REST_RECORDS},
    **{r: 1 for r in STRESS_RECORDS},
}


def record_filename(record_id: str, ext: str = "") -> str:
    """Return ECGPCG00XY basename (optionally with extension)."""
    base = f"ECGPCG00{record_id}"
    return f"{base}{ext}" if ext else base


# Default subject mapping for LOSO (one subject ID per recording).
# Prefer overriding via load_subject_map_from_spreadsheet() when
# ECGPCGSpreadsheet.csv is present — that file has the true Subject IDs
# and correctly links rest/stress sessions from the same person.
SUBJECT_MAP: dict[str, str] = {rid: f"S{rid}" for rid in ALL_RECORDS}


def load_subject_map_from_spreadsheet(csv_path) -> dict[str, str]:
    """
    Build record_id → subject_id from ECGPCGSpreadsheet.csv if available.

    Looks for columns resembling Record / Filename and Subject / Volunteer.
    Returns an empty dict if the file is missing or unparseable.
    """
    from pathlib import Path

    import pandas as pd

    path = Path(csv_path)
    if not path.exists():
        return {}

    try:
        sheet = pd.read_csv(path)
    except Exception:
        return {}

    cols = {c.lower().strip(): c for c in sheet.columns}
    rec_col = next(
        (cols[k] for k in cols if "record" in k or "file" in k or "name" in k),
        None,
    )
    sub_col = next(
        (cols[k] for k in cols if "subject" in k or "volunteer" in k or "participant" in k),
        None,
    )
    if rec_col is None or sub_col is None:
        return {}

    mapping: dict[str, str] = {}
    for _, row in sheet.iterrows():
        raw = str(row[rec_col])
        # Extract trailing digits from ECGPCG00XY / XY / 1
        digits = "".join(ch for ch in raw if ch.isdigit())
        if len(digits) >= 2:
            rid = digits[-2:]
        elif digits:
            rid = digits.zfill(2)
        else:
            continue
        if rid in RECORD_LABELS:
            mapping[rid] = f"S{row[sub_col]}"
    return mapping


def get_subject(record_id: str) -> str:
    """Return subject ID for a record, falling back to record-based ID."""
    return SUBJECT_MAP.get(record_id, f"S{record_id}")


def refresh_subject_map(csv_path=None) -> dict[str, str]:
    """Update module-level SUBJECT_MAP from spreadsheet when possible."""
    global SUBJECT_MAP
    if csv_path is None:
        from config.settings import EPHNOGRAM_SPREADSHEET

        csv_path = EPHNOGRAM_SPREADSHEET
    loaded = load_subject_map_from_spreadsheet(csv_path)
    if loaded:
        SUBJECT_MAP = {**SUBJECT_MAP, **loaded}
    return SUBJECT_MAP
