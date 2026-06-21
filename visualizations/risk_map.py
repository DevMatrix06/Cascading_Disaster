import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CALIFORNIA_BBOX
from models.gnn_model import TemporalCascadeGNN

GRAPH_PATH  = "data/processed/california_graph_temporal.pt"
MODEL_PATH  = "models/saved/cascade_gnn_temporal.pt"
LABELS_PATH = "data/raw/landslide_labels.csv"
OUTPUT_PATH = "visualizations/cascade_risk_map.png"
GRID_SIZE   = 50


def load_model(model_path, device):
    model = TemporalCascadeGNN(static_in=4, temporal_in=3, lstm_hidden=32, hidden_channels=64)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def run_inference(graph, model, device):
    x          = graph.x.to(device)
    temporal_x = graph.temporal_x.to(device)
    edge_index = graph.edge_index.to(device)
    with torch.no_grad():
        scores = model(x, temporal_x, edge_index)
    return scores.cpu().numpy()


def plot_risk_map(scores, landslide_df, output_path):
    risk_grid = scores.reshape(GRID_SIZE, GRID_SIZE)

    west, east   = CALIFORNIA_BBOX["west"],  CALIFORNIA_BBOX["east"]
    south, north = CALIFORNIA_BBOX["south"], CALIFORNIA_BBOX["north"]

    fig, ax = plt.subplots(figsize=(9, 11))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "risk", ["#ffffff", "#ffe5e5", "#ff9999", "#cc0000", "#7a0000"]
    )

    im = ax.imshow(
        risk_grid,
        extent=[west, east, south, north],
        origin="upper",
        cmap=cmap,
        vmin=0, vmax=1,
        aspect="auto",
        interpolation="bilinear",
        zorder=1,
    )

    # overlay landslide points
    ax.scatter(
        landslide_df["longitude"],
        landslide_df["latitude"],
        c="#1a6faf", s=18, alpha=0.75, linewidths=0.4,
        edgecolors="#0d3d6b", zorder=3, label="Actual landslide events",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Cascade Risk Score", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude",  fontsize=11)
    ax.set_title(
        "California Cascade Disaster Risk Map\n"
        "TemporalCascadeGNN — Predicted Risk vs. Actual Landslide Events (2020–2023)",
        fontsize=12, fontweight="bold", pad=12,
    )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1a6faf",
               markeredgecolor="#0d3d6b", markersize=7, label=f"Actual landslide events (n={len(landslide_df)})"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.85)

    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4, color="gray", zorder=2)

    # cardinal grid lines for readability
    for lon in np.arange(-124, -114, 2):
        ax.axvline(lon, color="gray", linewidth=0.3, alpha=0.3, zorder=2)
    for lat in np.arange(33, 43, 2):
        ax.axhline(lat, color="gray", linewidth=0.3, alpha=0.3, zorder=2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Risk map saved to: {output_path}")


def generate_risk_map():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    graph = torch.load(GRAPH_PATH, map_location=device, weights_only=False)
    model = load_model(MODEL_PATH, device)

    print("Running inference on 2500 nodes ...")
    scores = run_inference(graph, model, device)

    print(f"Risk score range: {scores.min():.4f} – {scores.max():.4f}  |  "
          f"Mean: {scores.mean():.4f}  |  "
          f"Nodes > 0.5: {(scores > 0.5).sum()}")

    landslide_df = pd.read_csv(LABELS_PATH)
    landslide_df["latitude"]  = pd.to_numeric(landslide_df["latitude"],  errors="coerce")
    landslide_df["longitude"] = pd.to_numeric(landslide_df["longitude"], errors="coerce")
    landslide_df = landslide_df.dropna(subset=["latitude", "longitude"])

    plot_risk_map(scores, landslide_df, OUTPUT_PATH)


if __name__ == "__main__":
    generate_risk_map()
