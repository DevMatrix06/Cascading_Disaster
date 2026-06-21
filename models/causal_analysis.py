import os
import sys
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRAPH_PATH    = "data/processed/california_graph_temporal.pt"
SUMMARY_PATH  = "data/processed/causal_summary.json"
PLOT_PATH     = "visualizations/causal_graph.png"
VAR_NAMES     = ["fire", "soil_moisture", "ndvi"]
MAX_LAG       = 3
SAMPLE_SIZE   = 50
RANDOM_SEED   = 42


def sample_nodes(y, n=SAMPLE_SIZE, seed=RANDOM_SEED):
    rng       = random.Random(seed)
    pos_idx   = (y == 1).nonzero(as_tuple=True)[0].tolist()
    neg_idx   = (y == 0).nonzero(as_tuple=True)[0].tolist()
    sampled_pos = rng.sample(pos_idx, min(n, len(pos_idx)))
    sampled_neg = rng.sample(neg_idx, min(n, len(neg_idx)))
    return sampled_pos, sampled_neg


def run_pcmci_on_node(series):
    # series: (T, 3) numpy array
    if np.isnan(series).any():
        col_means = np.nanmean(series, axis=0)
        for j in range(series.shape[1]):
            mask = np.isnan(series[:, j])
            series[mask, j] = col_means[j]

    # z-score normalise
    std = series.std(axis=0)
    std[std == 0] = 1.0
    series = (series - series.mean(axis=0)) / std

    dataframe = pp.DataFrame(series, var_names=VAR_NAMES)
    pcmci     = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results   = pcmci.run_pcmci(tau_max=MAX_LAG, pc_alpha=0.05)
    return results


def aggregate_results(all_results, n_vars=3):
    # accumulate MCI test statistics and p-values across nodes
    # shape per entry: (n_vars, n_vars, max_lag+1)
    val_sum   = np.zeros((n_vars, n_vars, MAX_LAG + 1))
    val_count = np.zeros((n_vars, n_vars, MAX_LAG + 1))

    for res in all_results:
        vals   = res["val_matrix"]    # (n_vars, n_vars, max_lag+1)
        pvals  = res["p_matrix"]
        sig    = pvals < 0.05
        val_sum   += np.abs(vals) * sig
        val_count += sig.astype(float)

    mean_strength = np.where(val_count > 0, val_sum / val_count, 0.0)
    freq          = val_count / max(len(all_results), 1)
    return mean_strength, freq


def build_summary(mean_strength, freq):
    links = []
    for cause in range(len(VAR_NAMES)):
        for effect in range(len(VAR_NAMES)):
            for lag in range(1, MAX_LAG + 1):   # lag 0 = contemporaneous, skip
                strength = float(mean_strength[effect, cause, lag])
                frequency = float(freq[effect, cause, lag])
                if strength > 0:
                    links.append({
                        "cause":     VAR_NAMES[cause],
                        "effect":    VAR_NAMES[effect],
                        "lag":       lag,
                        "strength":  round(strength, 4),
                        "frequency": round(frequency, 4),
                    })
    links.sort(key=lambda x: x["strength"], reverse=True)
    return links


def plot_causal_graph(links, output_path):
    n       = len(VAR_NAMES)
    angles  = [90, 210, 330]   # degrees for fire, soil_moisture, ndvi
    radius  = 1.4
    pos     = {
        name: (radius * np.cos(np.radians(a)), radius * np.sin(np.radians(a)))
        for name, a in zip(VAR_NAMES, angles)
    }
    colors  = {"fire": "#e74c3c", "soil_moisture": "#3498db", "ndvi": "#2ecc71"}
    labels  = {"fire": "Fire", "soil_moisture": "Soil\nMoisture", "ndvi": "NDVI"}

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Causal Graph — Fire / Soil Moisture / NDVI\n(PCMCI, max lag 3 months)",
                 fontsize=13, fontweight="bold", pad=14)

    # draw nodes
    for name, (px, py) in pos.items():
        circle = plt.Circle((px, py), 0.38, color=colors[name], zorder=3, alpha=0.9)
        ax.add_patch(circle)
        ax.text(px, py, labels[name], ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=4)

    # keep only top links (strength > 0) and deduplicate by (cause, effect) keeping strongest
    seen   = {}
    for lk in links:
        key = (lk["cause"], lk["effect"])
        if key not in seen or lk["strength"] > seen[key]["strength"]:
            seen[key] = lk

    max_str = max((lk["strength"] for lk in seen.values()), default=1.0)

    for (cause, effect), lk in seen.items():
        if cause == effect:
            continue
        x1, y1 = pos[cause]
        x2, y2 = pos[effect]
        # shorten arrow so it doesn't overlap nodes
        dx, dy   = x2 - x1, y2 - y1
        length   = np.hypot(dx, dy)
        ux, uy   = dx / length, dy / length
        start    = (x1 + ux * 0.40, y1 + uy * 0.40)
        end      = (x2 - ux * 0.42, y2 - uy * 0.42)

        alpha    = 0.4 + 0.6 * (lk["strength"] / max_str)
        lw       = 1.5 + 3.0 * (lk["strength"] / max_str)

        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color="#2c3e50",
                                    lw=lw, alpha=alpha,
                                    mutation_scale=18))

        mid_x = (start[0] + end[0]) / 2 + uy * 0.18
        mid_y = (start[1] + end[1]) / 2 - ux * 0.18
        ax.text(mid_x, mid_y, f"lag {lk['lag']}\n({lk['strength']:.2f})",
                ha="center", va="center", fontsize=7.5, color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

    legend_els = [mpatches.Patch(color=colors[n], label=labels[n]) for n in VAR_NAMES]
    ax.legend(handles=legend_els, loc="lower right", fontsize=9, framealpha=0.8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Causal graph saved to: {output_path}")


def run_causal_analysis():
    graph      = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    temporal_x = graph.temporal_x.numpy()   # (2500, 48, 3)
    y          = graph.y

    pos_nodes, neg_nodes = sample_nodes(y)
    sampled = pos_nodes + neg_nodes
    print(f"Sampled {len(pos_nodes)} high-risk + {len(neg_nodes)} low-risk nodes")
    print(f"Running PCMCI on {len(sampled)} nodes (max_lag={MAX_LAG}) ...\n")

    all_results = []
    for i, node in enumerate(sampled):
        series = temporal_x[node].copy()   # (48, 3)
        try:
            res = run_pcmci_on_node(series)
            all_results.append(res)
        except Exception as e:
            print(f"  Node {node} failed: {e}")
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(sampled)} nodes ...")

    print(f"\nAggregating results across {len(all_results)} successful nodes ...")
    mean_strength, freq = aggregate_results(all_results)
    links               = build_summary(mean_strength, freq)

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(links, f, indent=2)
    print(f"Causal summary saved to: {SUMMARY_PATH}")

    print("\n--- Top 5 Causal Links ---")
    print(f"  {'Cause':<16} {'Effect':<16} {'Lag':>4}  {'Strength':>9}  {'Frequency':>10}")
    print(f"  {'-'*16} {'-'*16} {'-'*4}  {'-'*9}  {'-'*10}")
    for lk in links[:5]:
        print(f"  {lk['cause']:<16} {lk['effect']:<16} {lk['lag']:>4}  "
              f"{lk['strength']:>9.4f}  {lk['frequency']:>10.4f}")

    plot_causal_graph(links, PLOT_PATH)
    return links


if __name__ == "__main__":
    run_causal_analysis()
