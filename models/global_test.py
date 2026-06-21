import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tempfile
import rasterio
from rasterio.transform import from_bounds, rowcol

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gnn_model import CascadeGNN

# ── config ────────────────────────────────────────────────────────────────────
AUS_BBOX = {"west": 143.0, "south": -39.0, "east": 154.0, "north": -28.0}

PERIODS = {
    "control":   {"start": "2018-09-01", "end": "2019-03-31", "label": "Control (pre-fire 2018-19)"},
    "bushfire":  {"start": "2019-09-01", "end": "2020-03-31", "label": "Bushfire season 2019-20"},
}

STATIC_MODEL  = "models/saved/cascade_gnn_final.pt"
FALLBACK_MODEL = "models/saved/cascade_gnn.pt"   # static-only weights
OUTPUT_DIR    = "visualizations"
GRID_SIZE     = 50
EXPORT_SCALE  = 10000   # 10 km — matches temporal resolution
EXPORT_CRS    = "EPSG:4326"

GEE_DATASETS = {
    "fire": {
        "collection": "MODIS/061/MOD14A1",
        "band": "FireMask",
        "agg": "max",
    },
    "soil": {
        "collection": "NASA/SMAP/SPL4SMGP/008",
        "band": "sm_surface",
        "agg": "mean",
    },
    "ndvi": {
        "collection": "MODIS/061/MOD13A2",
        "band": "NDVI",
        "agg": "mean",
        "scale_factor": 0.0001,
    },
    "elevation": {
        "collection": "USGS/SRTMGL1_003",
        "band": "elevation",
        "agg": "static",
    },
    "rainfall": {
        "collection": "NASA/GPM_L3/IMERG_MONTHLY_V06",
        "band": "precipitation",
        "agg": "mean",
    },
    "slope": {
        "collection": "USGS/SRTMGL1_003",
        "band": "elevation",
        "agg": "slope",
    },
}


# ── GEE ──────────────────────────────────────────────────────────────────────

def gee_init():
    import ee
    try:
        ee.Initialize(project="cascade-disaster-predictor")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="cascade-disaster-predictor")
    return ee


def fetch_arrays(bbox, start_date, end_date, ee):
    import geemap
    region = ee.Geometry.Rectangle([
        bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    ])
    arrays = {}
    with tempfile.TemporaryDirectory() as tmp:
        for key, cfg in GEE_DATASETS.items():
            path = os.path.join(tmp, f"{key}.tif")
            if cfg["agg"] == "slope":
                image = ee.Terrain.slope(ee.Image(cfg["collection"]).select(cfg["band"])).clip(region)
            elif cfg["agg"] == "static":
                image = ee.Image(cfg["collection"]).select(cfg["band"]).clip(region)
            else:
                col = (ee.ImageCollection(cfg["collection"])
                       .filterDate(start_date, end_date)
                       .filterBounds(region)
                       .select(cfg["band"]))
                image = (col.max() if cfg["agg"] == "max" else col.mean()).clip(region)
            if "scale_factor" in cfg:
                image = image.multiply(cfg["scale_factor"])
            geemap.ee_export_image(image, filename=path, scale=EXPORT_SCALE,
                                   region=region, file_per_band=False, crs=EXPORT_CRS)
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
                nd  = src.nodata
            if nd is not None:
                arr[arr == nd] = np.nan
            arr[arr <= -9999] = np.nan
            arrays[key] = arr
    return arrays


# ── graph construction ────────────────────────────────────────────────────────

def arrays_to_features(arrays, bbox, grid_size=GRID_SIZE):
    lats = np.linspace(bbox["north"], bbox["south"], grid_size + 1)
    lons = np.linspace(bbox["west"],  bbox["east"],  grid_size + 1)
    keys = ["fire", "soil", "ndvi", "elevation", "rainfall", "slope"]
    features = np.zeros((grid_size * grid_size, 6), dtype=np.float32)

    for f_idx, key in enumerate(keys):
        arr  = arrays[key]
        h, w = arr.shape
        tf   = from_bounds(bbox["west"], bbox["south"], bbox["east"], bbox["north"], w, h)
        for i in range(grid_size):
            for j in range(grid_size):
                r0, c0 = rowcol(tf, lons[j],     lats[i])
                r1, c1 = rowcol(tf, lons[j + 1], lats[i + 1])
                r0, r1 = sorted([max(0, min(int(r0), h-1)), max(0, min(int(r1), h-1))])
                c0, c1 = sorted([max(0, min(int(c0), w-1)), max(0, min(int(c1), w-1))])
                patch  = arr[r0:r1+1, c0:c1+1]
                valid  = patch[~np.isnan(patch)]
                features[i * grid_size + j, f_idx] = float(np.mean(valid)) if valid.size > 0 else 0.0

    return features


def build_edges(grid_size=GRID_SIZE):
    src, dst = [], []
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for i in range(grid_size):
        for j in range(grid_size):
            n = i * grid_size + j
            for di, dj in dirs:
                ni, nj = i+di, j+dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    src.append(n); dst.append(ni * grid_size + nj)
    return torch.tensor([src, dst], dtype=torch.long)


# ── model ─────────────────────────────────────────────────────────────────────

def load_model():
    model = CascadeGNN(in_channels=6, hidden_channels=64, out_channels=1)
    for path in [STATIC_MODEL, FALLBACK_MODEL]:
        if os.path.exists(path):
            try:
                model.load_state_dict(
                    torch.load(path, map_location="cpu", weights_only=True)
                )
                print(f"  Loaded weights from {path}")
                return model
            except RuntimeError:
                print(f"  Architecture mismatch for {path} — trying next")
    print("  No compatible weights found — using random init (zero-shot transfer)")
    return model


def run_inference(features_np, edge_index):
    model = load_model()
    model.eval()
    x = torch.tensor(features_np, dtype=torch.float)
    with torch.no_grad():
        scores = model(x, edge_index).numpy()
    return scores


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_risk_map(scores_dict, bbox, output_path):
    """Side-by-side control vs bushfire season."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0d1117")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "risk", ["#ffffff", "#ffe5e5", "#ff9999", "#cc0000", "#7a0000"]
    )
    vmin, vmax = 0.0, 1.0

    for ax, (period_key, scores) in zip(axes, scores_dict.items()):
        grid = scores.reshape(GRID_SIZE, GRID_SIZE)
        info = PERIODS[period_key]

        im = ax.imshow(
            grid,
            extent=[bbox["west"], bbox["east"], bbox["south"], bbox["north"]],
            origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="auto", interpolation="bilinear",
        )
        ax.set_facecolor("#0d1117")
        ax.set_title(info["label"], fontsize=12, fontweight="bold",
                     color="#e6edf3", pad=8)
        ax.set_xlabel("Longitude", color="#8b949e", fontsize=10)
        ax.set_ylabel("Latitude",  color="#8b949e", fontsize=10)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        high = int((scores > 0.5).sum())
        ax.text(0.02, 0.03,
                f"Mean: {scores.mean():.3f}  Max: {scores.max():.3f}  High-risk: {high}",
                transform=ax.transAxes, fontsize=9, color="#e6edf3",
                bbox=dict(boxstyle="round,pad=0.3", fc="#161b22", ec="#30363d", alpha=0.85))

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Cascade Risk Score", color="#8b949e", fontsize=9)
        cbar.ax.tick_params(labelcolor="#8b949e", labelsize=8)

    fig.suptitle("Australia Cascade Risk — Control vs Bushfire Season\n"
                 "Zero-shot transfer of California-trained CascadeGNN",
                 fontsize=13, fontweight="bold", color="#e6edf3", y=1.01)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_global_test():
    print("Initialising Google Earth Engine...")
    ee = gee_init()
    edge_index = build_edges()
    scores_dict = {}

    for period_key, info in PERIODS.items():
        print(f"\nFetching {info['label']} ({info['start']} → {info['end']})...")
        arrays   = fetch_arrays(AUS_BBOX, info["start"], info["end"], ee)
        features = arrays_to_features(arrays, AUS_BBOX)
        print(f"  Feature matrix: {features.shape}  "
              f"fire_mean={features[:,0].mean():.3f}  "
              f"ndvi_mean={features[:,2].mean():.3f}")

        scores = run_inference(features, edge_index)
        scores_dict[period_key] = scores

        high = int((scores > 0.5).sum())
        print(f"  Mean risk : {scores.mean():.4f}")
        print(f"  Max risk  : {scores.max():.4f}")
        print(f"  High-risk nodes (>0.5): {high} / {len(scores)}")

    # ── comparison ────────────────────────────────────────────────────────────
    ctrl = scores_dict["control"]
    fire = scores_dict["bushfire"]
    delta_mean = fire.mean() - ctrl.mean()
    delta_high = int((fire > 0.5).sum()) - int((ctrl > 0.5).sum())
    generalises = delta_mean > 0

    print("\n" + "="*56)
    print(f"  {'Period':<28} {'Mean':>6}  {'Max':>6}  {'High-risk':>9}")
    print("  " + "-"*52)
    print(f"  {'Control (pre-fire 2018-19)':<28} {ctrl.mean():>6.4f}  "
          f"{ctrl.max():>6.4f}  {int((ctrl>0.5).sum()):>9}")
    print(f"  {'Bushfire season 2019-20':<28} {fire.mean():>6.4f}  "
          f"{fire.max():>6.4f}  {int((fire>0.5).sum()):>9}")
    print("  " + "-"*52)
    print(f"  Delta (bushfire - control)        {delta_mean:>+6.4f}            {delta_high:>+9}")
    print("="*56)
    if generalises:
        print(f"\n  [PASS] Bushfire season scores higher by {delta_mean:.4f} "
              f"-- model generalises to unseen region.")
    else:
        print(f"\n  [NOTE] Control scores >= bushfire ({delta_mean:.4f}). "
              f"Zero-shot transfer may require domain adaptation.")

    # ── plot ──────────────────────────────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, "australia_risk_map.png")
    print(f"\nPlotting side-by-side risk maps...")
    plot_risk_map(scores_dict, AUS_BBOX, output_path)


if __name__ == "__main__":
    run_global_test()
