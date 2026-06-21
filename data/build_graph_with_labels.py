import os
import sys
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CALIFORNIA_BBOX

GRAPH_PATH  = "data/processed/california_graph.pt"
LABELS_PATH = "data/raw/landslide_labels.csv"
OUTPUT_PATH = "data/processed/california_graph_labeled.pt"
GRID_SIZE   = 50


def latlon_to_node(lat, lon, grid_size=GRID_SIZE):
    col = int((lon - CALIFORNIA_BBOX["west"])  / (CALIFORNIA_BBOX["east"]  - CALIFORNIA_BBOX["west"])  * grid_size)
    row = int((CALIFORNIA_BBOX["north"] - lat) / (CALIFORNIA_BBOX["north"] - CALIFORNIA_BBOX["south"]) * grid_size)
    col = max(0, min(col, grid_size - 1))
    row = max(0, min(row, grid_size - 1))
    return row * grid_size + col


def build_labeled_graph():
    graph = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    df    = pd.read_csv(LABELS_PATH)

    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    num_nodes = graph.num_nodes
    labels    = torch.zeros(num_nodes, dtype=torch.float)

    hit_nodes = set()
    for _, row in df.iterrows():
        node_idx = latlon_to_node(row["latitude"], row["longitude"])
        labels[node_idx] = 1.0
        hit_nodes.add(node_idx)

    graph.y = labels

    pos   = int(labels.sum().item())
    neg   = num_nodes - pos
    ratio = neg / pos if pos > 0 else float("inf")

    torch.save(graph, OUTPUT_PATH)

    print(f"Graph saved to  : {OUTPUT_PATH}")
    print(f"Total nodes     : {num_nodes}")
    print(f"Positive (risk=1): {pos}  ({pos / num_nodes * 100:.1f}%)")
    print(f"Negative (risk=0): {neg}  ({neg / num_nodes * 100:.1f}%)")
    print(f"Class imbalance  : 1 : {ratio:.1f}  (neg:pos)")
    print(f"\nNote: {len(df)} landslide events mapped to {pos} unique grid nodes "
          f"({len(df) - pos} events share a node with another event)")

    return graph


if __name__ == "__main__":
    build_labeled_graph()
