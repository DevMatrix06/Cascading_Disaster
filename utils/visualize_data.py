import os
import sys
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASETS, OUTPUT_DIR

PLOT_CONFIG = {
    "modis_fire":         {"title": "MODIS Fire (FireMask)",          "cmap": "hot"},
    "smap_soil_moisture": {"title": "SMAP Soil Moisture (sm_surface)", "cmap": "Blues"},
    "modis_ndvi":         {"title": "MODIS NDVI",                     "cmap": "RdYlGn"},
    "srtm_elevation":     {"title": "SRTM Elevation (m)",             "cmap": "terrain"},
}


def load_band(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    data[data <= -9999] = np.nan
    return data


def plot_overview(raw_dir=None, output_path="visualizations/data_overview.png"):
    raw_dir = raw_dir or OUTPUT_DIR

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cascading Disaster — California Dataset Overview", fontsize=15, fontweight="bold", y=1.01)

    for ax, (key, cfg) in zip(axes.flat, PLOT_CONFIG.items()):
        export_name = DATASETS[key]["export_name"]
        tif_path = os.path.join(raw_dir, f"{export_name}.tif")

        if not os.path.exists(tif_path):
            ax.set_title(cfg["title"])
            ax.text(0.5, 0.5, f"File not found:\n{tif_path}", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=9)
            ax.axis("off")
            continue

        data = load_band(tif_path)
        vmin = np.nanpercentile(data, 2)
        vmax = np.nanpercentile(data, 98)

        im = ax.imshow(data, cmap=cfg["cmap"], vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(cfg["title"], fontsize=11, fontweight="bold")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_overview()
