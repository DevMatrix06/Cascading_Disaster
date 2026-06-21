import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gnn_model import CascadeGNN, TemporalCascadeGNN

GRAPH_PATH   = "data/processed/california_graph_temporal.pt"
FINAL_MODEL  = "models/saved/cascade_gnn_temporal.pt"
OUTPUT_PATH  = "data/processed/ablation_results.json"
TRAIN_MONTHS = 36
EPOCHS       = 100
LR           = 1e-3
SEED         = 42


# ── shared utilities ──────────────────────────────────────────────────────────

def get_splits(y):
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=y.numpy().astype(int)
    )
    return torch.tensor(train_idx), torch.tensor(test_idx)


def evaluate(model, x, temporal_x, edge_index, y, idx, use_temporal=True, use_static=True):
    model.eval()
    with torch.no_grad():
        if use_temporal and use_static:
            out = model(x, temporal_x, edge_index)
        elif use_temporal and not use_static:
            out = model(x, temporal_x, edge_index)
        else:
            out = model(x, edge_index)
    preds = (out[idx] > 0.5).float().cpu().numpy()
    truth = y[idx].cpu().numpy()
    return {
        "precision": float(precision_score(truth, preds, zero_division=0)),
        "recall":    float(recall_score(truth, preds,    zero_division=0)),
        "f1":        float(f1_score(truth, preds,        zero_division=0)),
        "params":    sum(p.numel() for p in model.parameters()),
    }


def train_loop(model, x, temporal_x, edge_index, y, train_idx, use_temporal=True):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss, best_state = float("inf"), None

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        if use_temporal:
            out = model(x, temporal_x, edge_index)
        else:
            out = model(x, edge_index)
        logits = torch.logit(out.clamp(1e-6, 1 - 1e-6))
        loss   = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


# ── experiment 1: static only ────────────────────────────────────────────────

def run_static(graph, train_idx, test_idx):
    print("  Training Static CascadeGNN (4 features, no time series)...")
    x          = graph.x
    edge_index = graph.edge_index
    y          = graph.y

    model = CascadeGNN(in_channels=4, hidden_channels=64, out_channels=1)
    model = train_loop(model, x, None, edge_index, y, train_idx, use_temporal=False)
    result = evaluate(model, x, None, edge_index, y, test_idx,
                      use_temporal=False, use_static=True)
    print(f"  Done. Test F1={result['f1']:.4f}")
    return result


# ── experiment 2: temporal only (zero out static) ────────────────────────────

def run_temporal_only(graph, train_idx, test_idx):
    print("  Training TemporalCascadeGNN with zeroed static features...")
    x_zeros    = torch.zeros_like(graph.x)          # static features zeroed
    temporal_x = graph.temporal_x[:, :TRAIN_MONTHS, :]   # train window only
    test_temp  = graph.temporal_x[:, TRAIN_MONTHS:, :]
    edge_index = graph.edge_index
    y          = graph.y

    model = TemporalCascadeGNN(static_in=4, temporal_in=3, lstm_hidden=32, hidden_channels=64)
    model = train_loop(model, x_zeros, temporal_x, edge_index, y, train_idx, use_temporal=True)

    # evaluate on test temporal window with zeroed static
    model.eval()
    with torch.no_grad():
        out   = model(x_zeros, test_temp, edge_index)
        preds = (out[test_idx] > 0.5).float().cpu().numpy()
        truth = y[test_idx].cpu().numpy()
    result = {
        "precision": float(precision_score(truth, preds, zero_division=0)),
        "recall":    float(recall_score(truth, preds,    zero_division=0)),
        "f1":        float(f1_score(truth, preds,        zero_division=0)),
        "params":    sum(p.numel() for p in model.parameters()),
    }
    print(f"  Done. Test F1={result['f1']:.4f}")
    return result


# ── experiment 3: full model (load trained weights, no retrain) ───────────────

def run_full(graph, train_idx, test_idx):
    print(f"  Loading pretrained TemporalCascadeGNN from {FINAL_MODEL}...")
    x          = graph.x
    temporal_x = graph.temporal_x[:, TRAIN_MONTHS:, :]   # test temporal window
    edge_index = graph.edge_index
    y          = graph.y

    model = TemporalCascadeGNN(static_in=4, temporal_in=3, lstm_hidden=32, hidden_channels=64)
    state = torch.load(FINAL_MODEL, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    model.eval()
    with torch.no_grad():
        out   = model(x, temporal_x, edge_index)
        preds = (out[test_idx] > 0.5).float().cpu().numpy()
        truth = y[test_idx].cpu().numpy()
    result = {
        "precision": float(precision_score(truth, preds, zero_division=0)),
        "recall":    float(recall_score(truth, preds,    zero_division=0)),
        "f1":        float(f1_score(truth, preds,        zero_division=0)),
        "params":    sum(p.numel() for p in model.parameters()),
    }
    print(f"  Done. Test F1={result['f1']:.4f}")
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def run_ablation():
    print("Loading graph...")
    graph = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    train_idx, test_idx = get_splits(graph.y)

    pos = int(graph.y[test_idx].sum().item())
    print(f"Node split: {len(train_idx)} train / {len(test_idx)} test ({pos} positives in test)\n")

    print("Experiment 1/3 — Static GNN")
    r_static = run_static(graph, train_idx, test_idx)

    print("\nExperiment 2/3 — Temporal Only (zeroed static features)")
    r_temporal = run_temporal_only(graph, train_idx, test_idx)

    print("\nExperiment 3/3 — Static + Temporal (pretrained full model)")
    r_full = run_full(graph, train_idx, test_idx)

    # ── print comparison table ────────────────────────────────────────────────
    rows = [
        ("Static GNN",        r_static),
        ("Temporal Only",     r_temporal),
        ("Static + Temporal", r_full),
    ]

    w_name = max(len(r[0]) for r in rows) + 2
    header = f"{'Model':<{w_name}} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'Params':>8}"
    sep    = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, r in rows:
        print(f"{name:<{w_name}} | {r['precision']:>9.4f} | {r['recall']:>6.4f} "
              f"| {r['f1']:>6.4f} | {r['params']:>8,}")
    print(sep)

    # best F1
    best = max(rows, key=lambda x: x[1]["f1"])
    print(f"\nBest F1: {best[0]} ({best[1]['f1']:.4f})")

    # recall improvement: full vs static
    recall_gain = (r_full["recall"] - r_static["recall"]) / max(r_static["recall"], 1e-9) * 100
    print(f"Recall gain (Full vs Static): {recall_gain:+.1f}%")

    # ── save JSON ─────────────────────────────────────────────────────────────
    output = {
        "static_gnn":         {**r_static,   "name": "Static GNN"},
        "temporal_only":      {**r_temporal, "name": "Temporal Only"},
        "static_temporal":    {**r_full,     "name": "Static + Temporal"},
        "meta": {
            "train_months":  TRAIN_MONTHS,
            "test_months":   48 - TRAIN_MONTHS,
            "train_nodes":   len(train_idx),
            "test_nodes":    len(test_idx),
            "epochs":        EPOCHS,
            "seed":          SEED,
        },
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_ablation()
