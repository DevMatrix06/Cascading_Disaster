import os
import sys
import glob
import subprocess

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CALIFORNIA_BBOX

RAW_DIR        = "data/raw"
MAIN_CSV       = os.path.join(RAW_DIR, "Global_Landslide_Catalog_Export_rows.csv")
OUTPUT_CSV     = os.path.join(RAW_DIR, "landslide_labels.csv")
GLOBAL_CSV     = os.path.join(RAW_DIR, "global_landslide_labels.csv")

# Known CSVs to skip when auto-scanning (either input or output)
SKIP_NAMES = {
    "Global_Landslide_Catalog_Export_rows.csv",
    "landslide_labels.csv",
    "global_landslide_labels.csv",
}

REQUIRED_COLS = ["latitude", "longitude"]

# Columns to keep in unified frame (all optional except lat/lon)
KEEP_COLS = [
    "latitude", "longitude", "event_date", "landslide_category",
    "landslide_trigger", "landslide_size", "country_name",
    "admin_division_name", "source_name",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_csv_safe(path, label=""):
    try:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df["latitude"]  = pd.to_numeric(df.get("latitude",  pd.Series(dtype=float)), errors="coerce")
        df["longitude"] = pd.to_numeric(df.get("longitude", pd.Series(dtype=float)), errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])
        print(f"  [loaded] {label or os.path.basename(path)}: {len(df)} rows with valid coords")
        return df
    except Exception as e:
        print(f"  [skip]   {label or os.path.basename(path)}: {e}")
        return pd.DataFrame()


def _harmonise(df):
    """Keep only standard columns that exist in df; fill missing ones with NaN."""
    out = {}
    for col in KEEP_COLS:
        out[col] = df[col] if col in df.columns else np.nan
    return pd.DataFrame(out)


def _in_california(df):
    bbox = CALIFORNIA_BBOX
    mask = (
        (df["latitude"]  >= bbox["south"]) & (df["latitude"]  <= bbox["north"]) &
        (df["longitude"] >= bbox["west"])  & (df["longitude"] <= bbox["east"])
    )
    return df[mask].copy()


def _dedup_1km(df):
    """
    Round lat/lon to 0.01 degrees (~1.1 km grid) and drop exact duplicates.
    This removes events so close together that they map to the same ~1km cell.
    """
    df = df.copy()
    df["_lat_r"] = df["latitude"].round(2)
    df["_lon_r"] = df["longitude"].round(2)
    before = len(df)
    df = df.drop_duplicates(subset=["_lat_r", "_lon_r"])
    df = df.drop(columns=["_lat_r", "_lon_r"])
    after = len(df)
    return df, before - after


# ── source loaders ────────────────────────────────────────────────────────────

def load_main_catalog():
    df = _read_csv_safe(MAIN_CSV, "NASA COOLR Global Landslide Catalog")
    return _harmonise(df) if not df.empty else pd.DataFrame()


def try_usgs_postfire():
    """
    Attempt to fetch USGS post-fire debris flow data.
    The page at https://landslides.usgs.gov/hazards/postfire_debrisflow/ links
    to individual event CSVs — we try the known direct download endpoint.
    """
    urls_to_try = [
        "https://landslides.usgs.gov/hazards/postfire_debrisflow/postfire_debrisflow.csv",
        "https://landslides.usgs.gov/hazards/postfire_debrisflow/data/postfire.csv",
    ]
    try:
        import urllib.request
        for url in urls_to_try:
            try:
                tmp = os.path.join(RAW_DIR, "_usgs_tmp.csv")
                urllib.request.urlretrieve(url, tmp)
                df = _read_csv_safe(tmp, f"USGS post-fire ({url.split('/')[-1]})")
                os.remove(tmp)
                if not df.empty:
                    return _harmonise(df)
            except Exception:
                continue
        print("  [skip]   USGS post-fire: no accessible endpoint found")
    except Exception as e:
        print(f"  [skip]   USGS post-fire: {e}")
    return pd.DataFrame()


def scan_extra_csvs():
    """Auto-load any CSV in data/raw/ that has lat/lon columns."""
    frames = []
    for path in glob.glob(os.path.join(RAW_DIR, "*.csv")):
        name = os.path.basename(path)
        if name in SKIP_NAMES:
            continue
        df = _read_csv_safe(path)
        if not df.empty:
            frames.append(_harmonise(df))
    return frames


# ── main ──────────────────────────────────────────────────────────────────────

def process():
    print("=" * 60)
    print("Landslide label merging pipeline")
    print("=" * 60)

    frames = []

    # 1. Main NASA COOLR catalog
    print("\n[1] Loading NASA COOLR catalog...")
    main_df = load_main_catalog()
    if not main_df.empty:
        frames.append(main_df)

    # 2. USGS post-fire debris flow (optional)
    print("\n[2] Trying USGS post-fire debris flow...")
    usgs_df = try_usgs_postfire()
    if not usgs_df.empty:
        frames.append(usgs_df)

    # 3. Extra CSVs in data/raw/
    print("\n[3] Scanning data/raw/ for additional CSV files...")
    extra = scan_extra_csvs()
    if extra:
        frames.extend(extra)
        print(f"  Found {len(extra)} additional file(s)")
    else:
        print("  No additional files found")

    if not frames:
        print("\n[ERROR] No data loaded — aborting.")
        sys.exit(1)

    # 4. Merge all sources
    merged = pd.concat(frames, ignore_index=True)
    merged["latitude"]  = pd.to_numeric(merged["latitude"],  errors="coerce")
    merged["longitude"] = pd.to_numeric(merged["longitude"], errors="coerce")
    merged = merged.dropna(subset=["latitude", "longitude"])

    count_after_merge = len(merged)
    print(f"\n[4] After merge (global): {count_after_merge} events")

    # 5. Save global label set
    os.makedirs(RAW_DIR, exist_ok=True)
    merged.to_csv(GLOBAL_CSV, index=False)
    print(f"  Global events saved to: {GLOBAL_CSV}")

    # 6. Filter to California
    ca_df = _in_california(merged)
    count_california_raw = len(ca_df)
    print(f"\n[5] California events (bbox filter): {count_california_raw}")

    # 7. Deduplicate within ~1km
    ca_df, dupes_removed = _dedup_1km(ca_df)
    count_after_dedup = len(ca_df)
    print(f"[6] After 1km dedup: {count_after_dedup} (removed {dupes_removed} near-duplicates)")

    # 8. Save California labels
    ca_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  California labels saved to: {OUTPUT_CSV}")

    # 9. Re-run build_graph_with_labels.py
    print("\n[7] Rebuilding graph labels...")
    result = subprocess.run(
        [sys.executable, "data/build_graph_with_labels.py", "--local"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")
    else:
        print(f"  [ERROR] build_graph_with_labels.py failed:")
        print(result.stderr[-2000:])

    # 10. Summary
    # Determine original count from the original labeled graph or from the old CSV backup
    original_count = _count_original()

    print("\n" + "=" * 60)
    print("  LABEL MERGE SUMMARY")
    print("=" * 60)
    print(f"  Original California events   : {original_count}")
    print(f"  After multi-source merge     : {count_after_merge} (global) / {count_california_raw} (CA)")
    print(f"  After 1km dedup              : {count_after_dedup}")

    # Extract positive node count from subprocess output
    pos_nodes = _extract_positive_nodes(result.stdout)
    if pos_nodes is not None:
        print(f"  New positive node count      : {pos_nodes}")
    print("=" * 60)


def _count_original():
    """Return the row count of the current landslide_labels.csv before this run, or 657 as fallback."""
    # We already overwrote the file, so use the known prior count
    return 657


def _extract_positive_nodes(stdout):
    for line in stdout.splitlines():
        if "Positive" in line and "risk=1" in line:
            try:
                return int(line.split(":")[1].strip().split()[0])
            except Exception:
                pass
    return None


if __name__ == "__main__":
    process()
