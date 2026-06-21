import os
import sys
import re
import numpy as np
import rasterio
from rasterio.transform import rowcol
import torch
from datetime import datetime
from dateutil.relativedelta import relativedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CALIFORNIA_BBOX, DATE_RANGE

TEMPORAL_DIR  = "data/raw/temporal"
GRAPH_PATH    = "data/processed/california_graph_labeled.pt"
OUTPUT_PATH   = "data/processed/california_graph_temporal.pt"
GRID_SIZE     = 50
LAYERS        = ["fire", "soil", "ndvi"]


def all_months(start_str, end_str):
    start   = datetime.strptime(start_str, "%Y-%m-%d").replace(day=1)
    end     = datetime.strptime(end_str,   "%Y-%m-%d").replace(day=1)
    months  = []
    current = start
    while current <= end:
        months.append(current.strftime("%Y_%m"))
        current += relativedelta(months=1)
    return months


def scan_available(temporal_dir):
    available = {}
    if not os.path.isdir(temporal_dir):
        return available
    for fname in os.listdir(temporal_dir):
        m = re.match(r"^(fire|soil|ndvi)_(\d{4}_\d{2})\.tif$", fname)
        if m:
            layer, tag = m.group(1), m.group(2)
            available.setdefault(tag, set()).add(layer)
    return available


def load_raster_grid(path, grid_size=GRID_SIZE):
    lats = np.linspace(CALIFORNIA_BBOX["north"], CALIFORNIA_BBOX["south"], grid_size + 1)
    lons = np.linspace(CALIFORNIA_BBOX["west"],  CALIFORNIA_BBOX["east"],  grid_size + 1)

    with rasterio.open(path) as src:
        data    = src.read(1).astype(np.float32)
        nodata  = src.nodata
        tf      = src.transform
        h, w    = src.shape

    if nodata is not None:
        data[data == nodata] = np.nan
    data[data <= -9999] = np.nan

    grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            row_top, col_left  = rowcol(tf, lons[j],     lats[i])
            row_bot, col_right = rowcol(tf, lons[j + 1], lats[i + 1])

            row_top   = max(0, min(int(row_top),   h - 1))
            row_bot   = max(0, min(int(row_bot),   h - 1))
            col_left  = max(0, min(int(col_left),  w - 1))
            col_right = max(0, min(int(col_right), w - 1))

            if row_bot < row_top:
                row_top, row_bot = row_bot, row_top
            if col_right < col_left:
                col_left, col_right = col_right, col_left

            patch = data[row_top:row_bot + 1, col_left:col_right + 1]
            valid = patch[~np.isnan(patch)]
            grid[i, j] = np.mean(valid) if valid.size > 0 else np.nan

    return grid.reshape(grid_size * grid_size)


def build_temporal_graph():
    months    = all_months(DATE_RANGE["start"], DATE_RANGE["end"])
    available = scan_available(TEMPORAL_DIR)

    num_nodes   = GRID_SIZE * GRID_SIZE
    num_months  = len(months)
    num_layers  = len(LAYERS)

    # shape: (num_nodes, num_months, num_layers)
    temporal = np.full((num_nodes, num_months, num_layers), np.nan, dtype=np.float32)

    complete_months = 0
    partial_months  = 0

    for t, tag in enumerate(months):
        present = available.get(tag, set())
        has_all = all(layer in present for layer in LAYERS)

        if has_all:
            complete_months += 1
        elif present:
            partial_months += 1

        for f_idx, layer in enumerate(LAYERS):
            if layer not in present:
                continue
            path = os.path.join(TEMPORAL_DIR, f"{layer}_{tag}.tif")
            try:
                temporal[:, t, f_idx] = load_raster_grid(path)
                print(f"  Loaded  {layer}_{tag}.tif")
            except Exception as e:
                print(f"  ERROR   {layer}_{tag}.tif  — {e}")

    print(f"\nMonths fully loaded   : {complete_months} / {num_months}")
    print(f"Months partially loaded: {partial_months}")
    print(f"Months missing entirely: {num_months - complete_months - partial_months}")

    # Fill NaNs: forward-fill along time axis, then fill remaining with column mean
    nan_before = int(np.isnan(temporal).sum())

    for node in range(num_nodes):
        for f_idx in range(num_layers):
            series = temporal[node, :, f_idx]
            # forward fill
            last_valid = np.nan
            for t in range(num_months):
                if not np.isnan(series[t]):
                    last_valid = series[t]
                elif not np.isnan(last_valid):
                    series[t] = last_valid
            temporal[node, :, f_idx] = series

    # fill remaining NaNs with per-feature column mean across all nodes & time
    nan_after_ffill = int(np.isnan(temporal).sum())
    for f_idx in range(num_layers):
        col_mean = np.nanmean(temporal[:, :, f_idx])
        if np.isnan(col_mean):
            col_mean = 0.0
        mask = np.isnan(temporal[:, :, f_idx])
        temporal[:, :, f_idx][mask] = col_mean

    nan_filled = nan_before - int(np.isnan(temporal).sum())

    graph = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    graph.temporal_x = torch.tensor(temporal, dtype=torch.float)

    torch.save(graph, OUTPUT_PATH)

    print(f"\nTemporal feature shape : {tuple(graph.temporal_x.shape)}  (nodes x months x features)")
    print(f"NaN values filled      : {nan_filled}  ({nan_before} before -> 0 after)")
    print(f"Graph saved to         : {OUTPUT_PATH}")

    return graph


if __name__ == "__main__":
    build_temporal_graph()
