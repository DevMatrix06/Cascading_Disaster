import os
import sys
import numpy as np
import rasterio
from rasterio.transform import rowcol
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASETS, OUTPUT_DIR, CALIFORNIA_BBOX

GRID_SIZE = 50
OUTPUT_PATH = "data/processed/california_graph.pt"

LAYER_ORDER = ["modis_fire", "smap_soil_moisture", "modis_ndvi", "srtm_elevation", "gpm_rainfall", "srtm_slope"]


def load_raster(key):
    export_name = DATASETS[key]["export_name"]
    path = os.path.join(OUTPUT_DIR, f"{export_name}.tif")
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        shape = src.shape
    if nodata is not None:
        data[data == nodata] = np.nan
    data[data <= -9999] = np.nan
    return data, transform, shape


def extract_grid_features(grid_size=GRID_SIZE):
    lats = np.linspace(CALIFORNIA_BBOX["north"], CALIFORNIA_BBOX["south"], grid_size + 1)
    lons = np.linspace(CALIFORNIA_BBOX["west"], CALIFORNIA_BBOX["east"], grid_size + 1)

    rasters = {}
    transforms = {}
    shapes = {}
    for key in LAYER_ORDER:
        rasters[key], transforms[key], shapes[key] = load_raster(key)
        print(f"  Loaded {key}: shape={rasters[key].shape}")

    features = np.full((grid_size, grid_size, len(LAYER_ORDER)), np.nan, dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            lat_top    = lats[i]
            lat_bottom = lats[i + 1]
            lon_left   = lons[j]
            lon_right  = lons[j + 1]

            for f_idx, key in enumerate(LAYER_ORDER):
                data = rasters[key]
                tf   = transforms[key]
                h, w = shapes[key]

                row_top, col_left  = rowcol(tf, lon_left,  lat_top)
                row_bot, col_right = rowcol(tf, lon_right, lat_bottom)

                row_top   = max(0, min(row_top,   h - 1))
                row_bot   = max(0, min(row_bot,   h - 1))
                col_left  = max(0, min(col_left,  w - 1))
                col_right = max(0, min(col_right, w - 1))

                if row_bot < row_top:
                    row_top, row_bot = row_bot, row_top
                if col_right < col_left:
                    col_left, col_right = col_right, col_left

                patch = data[row_top:row_bot + 1, col_left:col_right + 1]
                valid = patch[~np.isnan(patch)]
                features[i, j, f_idx] = np.mean(valid) if valid.size > 0 else 0.0

    return features


def build_edges(grid_size=GRID_SIZE):
    src_nodes, dst_nodes = [], []
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]

    for i in range(grid_size):
        for j in range(grid_size):
            node = i * grid_size + j
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    neighbor = ni * grid_size + nj
                    src_nodes.append(node)
                    dst_nodes.append(neighbor)

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return edge_index


def build_graph():
    print("Extracting node features from rasters...")
    features = extract_grid_features()

    x = torch.tensor(
        features.reshape(GRID_SIZE * GRID_SIZE, len(LAYER_ORDER)),
        dtype=torch.float
    )

    print("Building edges (Moore neighborhood)...")
    edge_index = build_edges()

    graph = Data(x=x, edge_index=edge_index)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(graph, OUTPUT_PATH)

    print(f"\nGraph saved to: {OUTPUT_PATH}")
    print(f"  Nodes        : {graph.num_nodes}")
    print(f"  Edges        : {graph.num_edges}")
    print(f"  Feature shape: {tuple(graph.x.shape)}  (nodes x features)")
    print(f"  Features     : {LAYER_ORDER}")

    return graph


if __name__ == "__main__":
    build_graph()
