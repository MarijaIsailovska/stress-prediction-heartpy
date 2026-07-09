# 08 — Data download and local storage layout

**Paper section:** Methods — Data Acquisition (supplementary / reproducibility)  
**Related code:** `download_only_needed.py`, `download_wrist.py`

---

## Decision — 2026-07-09 — Selective EPHNOGRAM download + streaming HTTP

**Decision:**
1. Download **only** the quality-filtered EPHNOGRAM record IDs (not the full database) into `data/raw/ephnogram/WFDB/`.
2. Use **streaming** HTTP downloads (`urllib`) with progress and `.part` temp files, instead of `wfdb.dl_files` / `dl_full_file` for large `.dat` files.
3. Skip a record if `data/raw/ephnogram/WFDB/ECGPCGXXXX.dat` already exists and is non-empty.
4. Wrist data: `wfdb.dl_database('wrist', ...)` via `download_wrist.py` (full DB helper).
5. Raw data are **gitignored**; only `.gitkeep` placeholders are committed.

**Why:**
- Full EPHNOGRAM is large; the study only needs 28 recordings (`project-implementation`, tied to note 02).
- Runtime evidence showed `wfdb.dl_full_file` reads the **entire remote file into memory** before writing, so large ~144 MB stress recordings appeared “stuck” at 0 bytes on disk for a long time while buffering. Streaming writes chunks immediately and reports progress.
- An earlier skip path checked `data/raw/ephnogram/*.dat` while `wfdb.dl_files(..., keep_subdirs=True)` stored files under `WFDB/`, causing repeated re-downloads of existing files.

**Advantages:**
- Faster iteration; less disk/network than full DB
- Visible progress; lower peak RAM
- Correct skip path avoids redundant transfers
- Reproducible record list matches analysis cohort

**Disadvantages:**
- Custom downloader diverges from one-line `wfdb.dl_files` examples in PhysioNet tutorials
- Must keep URL layout (`https://physionet.org/files/ephnogram/1.0.0/WFDB/...`) in sync with PhysioNet (`dataset-docs` layout)
- Partial `.part` files possible after interrupt (user should delete and retry)
- Wrist script still uses WFDB bulk download (different tradeoff)

**Alternatives in literature / tooling:**
- `wfdb.dl_database('ephnogram', ...)` full mirror
- Manual browser/wget/rsync from PhysioNet
- Globus or AWS mirrors where provided by PhysioNet for a given dataset (`dataset-docs` — availability **[NEEDS VERIFICATION]** per dataset page)

**Evidence:**
- (`dataset-docs`) PhysioNet file layout for EPHNOGRAM under `files/ephnogram/1.0.0/` (WFDB subdirectory)
- (`project-implementation`) Skip-path bug and streaming fix validated during local download debugging (2026-07-09): wrong path `ephnogram/ECGPCG*.dat` vs actual `ephnogram/WFDB/ECGPCG*.dat`; `wfdb` `dl_full_file` uses `readfile.read()` then write
- (`heartpy-docs`) N/A for download
- (`peer-reviewed`) N/A — engineering reproducibility choice

**Uncertainty:**
- PhysioNet may require login/credentials for some bots; if downloads fail with HTTP 401/403, document auth steps — **[NEEDS VERIFICATION]** against current PhysioNet access policy

**Code:** `download_only_needed.py`, `download_wrist.py`, `.gitignore` (`data/raw/ephnogram/**`, `data/raw/wrist/**`)  
**Paper section:** Reproducibility appendix / Methods — Data Acquisition
