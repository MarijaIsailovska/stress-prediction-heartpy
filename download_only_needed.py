import os
import time
import urllib.request

REST = [10, 11, 13, 14, 15, 16, 21, 22, 23]
STRESS = [1, 25, 27, 29, 30, 32, 33, 34, 36, 38, 47, 52, 55, 61, 62, 64, 66, 67, 68]

all_records = REST + STRESS
dl_dir = 'data/raw/(Get-ChildItem -Path data\raw\ephnogram -Recurse -Filter "*.dat").Count'
wfdb_dir = os.path.join(dl_dir, 'WFDB')
os.makedirs(wfdb_dir, exist_ok=True)

# Problematic records (optional manual skip)
SKIP = []

PN_BASE = "https://physionet.org/files/ephnogram/1.0.0"


def stream_download(url, dest_path, timeout=60, chunk_size=1024 * 256):
    """Stream a file to disk with progress (avoids buffering the entire file in RAM)."""
    tmp_path = dest_path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "stress_prediction_heartpy/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = resp.headers.get("Content-Length")
        total = int(total) if total is not None else None
        downloaded = 0
        t0 = time.time()
        last_print = 0.0
        with open(tmp_path, "wb") as out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print >= 2.0:
                    last_print = now
                    mb = downloaded / (1024 * 1024)
                    if total:
                        pct = 100.0 * downloaded / total
                        print(f"    {mb:.1f}/{total/(1024*1024):.1f} MB ({pct:.0f}%)", flush=True)
                    else:
                        print(f"    {mb:.1f} MB...", flush=True)
        os.replace(tmp_path, dest_path)
        return downloaded, total, round(time.time() - t0, 1)


for xy in all_records:
    if xy in SKIP:
        print(f"⚠ ECGPCG{xy:04d} — прескокнувам (проблематичен)")
        continue

    record_name = f"ECGPCG{xy:04d}"
    # wfdb stores files under WFDB/ when keep_subdirs=True
    dat_file = os.path.join(wfdb_dir, f"{record_name}.dat")
    hea_file = os.path.join(wfdb_dir, f"{record_name}.hea")

    if os.path.exists(dat_file) and os.path.getsize(dat_file) > 0:
        print(f"✓ {record_name} already exists — skipping")
        continue

    print(f"Downloading {record_name}...")
    try:
        if not os.path.exists(hea_file):
            hea_url = f"{PN_BASE}/WFDB/{record_name}.hea"
            stream_download(hea_url, hea_file, timeout=60)

        dat_url = f"{PN_BASE}/WFDB/{record_name}.dat"
        downloaded, total, elapsed = stream_download(dat_url, dat_file, timeout=120)
        print(f"  ✓ Done ({downloaded/(1024*1024):.1f} MB in {elapsed}s)")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\nDone!")
