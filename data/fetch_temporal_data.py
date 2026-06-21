import os
import sys
import ee
import geemap
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from dateutil.relativedelta import relativedelta
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CALIFORNIA_BBOX, DATE_RANGE

OUTPUT_DIR   = "data/raw/temporal"
EXPORT_SCALE = 10000
EXPORT_CRS   = "EPSG:4326"
MEAN_FILL_LOOKBACK = 3   # use mean of previous N months when collection is empty

TEMPORAL_DATASETS = {
    "fire": {
        "collection":  "MODIS/061/MOD14A1",
        "band":        "FireMask",
    },
    "soil": {
        "collection":  "NASA/SMAP/SPL4SMGP/008",
        "band":        "sm_surface",
    },
    "ndvi": {
        "collection":  "MODIS/061/MOD13A2",
        "band":        "NDVI",
        "scale_factor": 0.0001,
    },
}


def authenticate_gee():
    try:
        ee.Initialize(project="cascade-disaster-predictor")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="cascade-disaster-predictor")


def get_region():
    return ee.Geometry.Rectangle([
        CALIFORNIA_BBOX["west"], CALIFORNIA_BBOX["south"],
        CALIFORNIA_BBOX["east"], CALIFORNIA_BBOX["north"],
    ])


def month_range(start_str, end_str):
    start   = datetime.strptime(start_str, "%Y-%m-%d").replace(day=1)
    end     = datetime.strptime(end_str,   "%Y-%m-%d").replace(day=1)
    current = start
    while current <= end:
        yield current
        current += relativedelta(months=1)


def collection_is_empty(key, cfg, start_date, end_date, region):
    col = (ee.ImageCollection(cfg["collection"])
           .filterDate(start_date, end_date)
           .filterBounds(region))
    return col.size().getInfo() == 0


def fetch_monthly_image(key, cfg, start_date, end_date, region):
    col   = (ee.ImageCollection(cfg["collection"])
             .filterDate(start_date, end_date)
             .filterBounds(region)
             .select(cfg["band"]))
    image = col.mean().clip(region)
    if "scale_factor" in cfg:
        image = image.multiply(cfg["scale_factor"])
    return image


def export_image(image, out_path, region):
    geemap.ee_export_image(
        image, filename=out_path, scale=EXPORT_SCALE,
        region=region, file_per_band=False, crs=EXPORT_CRS,
    )


def read_tif_array(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd  = src.nodata
        profile = src.profile
    if nd is not None:
        arr[arr == nd] = np.nan
    arr[arr <= -9999] = np.nan
    return arr, profile


def write_mean_fill_tif(out_path, arr, profile):
    profile.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def build_mean_fill(key, tag_before_list):
    """Average the last N available tifs for this layer. Returns (array, profile) or None."""
    arrays = []
    profile = None
    for prev_tag in reversed(tag_before_list):
        path = os.path.join(OUTPUT_DIR, f"{key}_{prev_tag}.tif")
        if os.path.exists(path):
            arr, profile = read_tif_array(path)
            arrays.append(arr)
            if len(arrays) == MEAN_FILL_LOOKBACK:
                break
    if not arrays:
        return None, None
    filled = np.nanmean(np.stack(arrays, axis=0), axis=0)
    return filled, profile


def fetch_temporal_data():
    authenticate_gee()
    region = get_region()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_months = list(month_range(DATE_RANGE["start"], DATE_RANGE["end"]))
    total_files = len(all_months) * len(TEMPORAL_DATASETS)

    print(f"Checking {len(all_months)} months x {len(TEMPORAL_DATASETS)} layers = {total_files} files")
    print(f"Output: {OUTPUT_DIR}  |  Scale: {EXPORT_SCALE}m  |  Mean-fill lookback: {MEAN_FILL_LOOKBACK} months\n")

    counts = {"downloaded": 0, "skipped": 0, "mean_filled": 0, "failed": 0}
    tags_so_far = []   # ordered list of tags processed so far, for lookback

    for month_dt in all_months:
        tag        = month_dt.strftime("%Y_%m")
        next_month = month_dt + relativedelta(months=1)
        start_date = month_dt.strftime("%Y-%m-%d")
        end_date   = next_month.strftime("%Y-%m-%d")

        for key, cfg in TEMPORAL_DATASETS.items():
            out_path = os.path.join(OUTPUT_DIR, f"{key}_{tag}.tif")

            # ── already have it ───────────────────────────────────────────────
            if os.path.exists(out_path):
                counts["skipped"] += 1
                continue

            label = f"{key}_{tag}"
            print(f"  Fetching {label} ...", end=" ", flush=True)

            # ── check if collection has data for this month ───────────────────
            try:
                empty = collection_is_empty(key, cfg, start_date, end_date, region)
            except Exception as e:
                print(f"ERROR (size check): {e}")
                counts["failed"] += 1
                continue

            if empty:
                # ── mean-fill from previous months ────────────────────────────
                filled_arr, profile = build_mean_fill(key, tags_so_far)
                if filled_arr is not None:
                    write_mean_fill_tif(out_path, filled_arr, profile)
                    print(f"mean-filled (empty collection)")
                    counts["mean_filled"] += 1
                else:
                    print(f"SKIP (empty collection, no prior months to fill from)")
                    counts["failed"] += 1
                continue

            # ── fetch from GEE ────────────────────────────────────────────────
            try:
                image = fetch_monthly_image(key, cfg, start_date, end_date, region)
                export_image(image, out_path, region)
                print("done")
                counts["downloaded"] += 1
            except Exception as e:
                err = str(e)
                # "no bands" → treat the same as an empty collection
                if "no bands" in err.lower() or "imageempty" in err.lower():
                    filled_arr, profile = build_mean_fill(key, tags_so_far)
                    if filled_arr is not None:
                        write_mean_fill_tif(out_path, filled_arr, profile)
                        print(f"mean-filled (no-bands error)")
                        counts["mean_filled"] += 1
                    else:
                        print(f"SKIP (no-bands, no prior months)")
                        counts["failed"] += 1
                else:
                    print(f"ERROR: {err}")
                    counts["failed"] += 1

        tags_so_far.append(tag)

    print(f"\n{'='*52}")
    print(f"  Downloaded (new)   : {counts['downloaded']:>4}")
    print(f"  Skipped (existing) : {counts['skipped']:>4}")
    print(f"  Mean-filled        : {counts['mean_filled']:>4}")
    print(f"  Failed             : {counts['failed']:>4}")
    print(f"{'='*52}\n")

    # ── rebuild temporal graph ────────────────────────────────────────────────
    if counts["downloaded"] + counts["mean_filled"] > 0:
        print("Rebuilding temporal graph...")
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "build_temporal_graph",
            pathlib.Path(__file__).parent / "build_temporal_graph.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.build_temporal_graph()
    else:
        print("No new files — temporal graph unchanged.")


if __name__ == "__main__":
    fetch_temporal_data()
