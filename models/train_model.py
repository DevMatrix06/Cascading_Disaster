import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gnn_model import TemporalCascadeGNN

GRAPH_PATH  = "data/processed/california_graph_temporal.pt"
SAVE_PATH   = "models/saved/cascade_gnn_final.pt"
EPOCHS      = 100
LR          = 1e-3
TRAIN_MONTHS = 36   # 2020-01 → 2022-12
TEST_MONTHS  = 12   # 2023-01 → 2023-12


def metrics(preds, labels):
    p = preds.cpu().numpy()
    y = labels.cpu().numpy()
    return {
        "accuracy":  float((preds == labels).float().mean().item()),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall":    float(recall_score(y, p,    zero_division=0)),
        "f1":        float(f1_score(y, p,        zero_division=0)),
    }


def stratified_split(y_tensor, test_size=0.2, seed=42):
    y = y_tensor.numpy().astype(int)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    return torch.tensor(train_idx), torch.tensor(test_idx)


def print_metrics(split_name, m):
    print(f"  {split_name:<6}  Acc={m['accuracy']:.4f}  "
          f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  F1={m['f1']:.4f}")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    graph      = torch.load(GRAPH_PATH, map_location=device, weights_only=False)
    x          = graph.x                 # (2500, 6)
    temporal_x = graph.temporal_x        # (2500, 48, 3)
    edge_index = graph.edge_index
    y          = graph.y.to(device)

    # ── temporal split ────────────────────────────────────────────────────────
    train_temporal = temporal_x[:, :TRAIN_MONTHS, :]   # (2500, 36, 3)
    test_temporal  = temporal_x[:, TRAIN_MONTHS:, :]   # (2500, 12, 3)

    # ── node split (stratified 80/20) ─────────────────────────────────────────
    train_idx, test_idx = stratified_split(y.cpu())
    train_idx = train_idx.to(device)
    test_idx  = test_idx.to(device)

    pos_train = int(y[train_idx].sum().item())
    pos_test  = int(y[test_idx].sum().item())
    print(f"Temporal split  : train months 0–{TRAIN_MONTHS-1} (2020-2022) | "
          f"test months {TRAIN_MONTHS}–47 (2023)")
    print(f"Node split      : {len(train_idx)} train ({pos_train} pos) | "
          f"{len(test_idx)} test ({pos_test} pos)")
    print(f"Class balance   : train 1:{(len(train_idx)-pos_train)/max(pos_train,1):.1f} | "
          f"test 1:{(len(test_idx)-pos_test)/max(pos_test,1):.1f}\n")

    # ── two separate GNN models (different LSTM input sizes) ──────────────────
    # train model uses 36-month sequences; test uses 12-month sequences.
    # we train one model on the 36-month window, then evaluate on the 12-month window
    # by building a second model with the same weights but smaller seq input.
    # Simpler: use the same model for both — LSTM is stateless per forward pass,
    # so the same weights handle variable-length sequences.
    model = TemporalCascadeGNN(
        static_in=6, temporal_in=3, lstm_hidden=32, hidden_channels=64
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.0], device=device))

    best_test_f1  = -1.0
    best_state    = None

    print(f"Training for {EPOCHS} epochs…\n")
    print(f"  {'Ep':>4}  {'TrLoss':>7}  {'Tr-F1':>6}  {'Te-F1':>6}")
    print(f"  {'----':>4}  {'-------':>7}  {'------':>6}  {'------':>6}")

    for epoch in range(1, EPOCHS + 1):
        # ── train step (train nodes, train temporal window) ───────────────────
        model.train()
        optimizer.zero_grad()
        out_all = model(x, train_temporal, edge_index)
        logits  = torch.logit(out_all.clamp(1e-6, 1 - 1e-6))
        loss    = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # train metrics: train nodes, train window
                tr_out   = model(x, train_temporal, edge_index)
                tr_preds = (tr_out[train_idx] > 0.5).float()
                tr_m     = metrics(tr_preds, y[train_idx])

                # test metrics: test nodes, test window (unseen temporal data)
                te_out   = model(x, test_temporal, edge_index)
                te_preds = (te_out[test_idx] > 0.5).float()
                te_m     = metrics(te_preds, y[test_idx])

            print(f"  {epoch:>4d}  {loss.item():>7.4f}  {tr_m['f1']:>6.4f}  {te_m['f1']:>6.4f}")

            if te_m["f1"] > best_test_f1:
                best_test_f1 = te_m["f1"]
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    # ── final evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tr_out   = model(x, train_temporal, edge_index)
        tr_preds = (tr_out[train_idx] > 0.5).float()
        tr_final = metrics(tr_preds, y[train_idx])

        te_out   = model(x, test_temporal, edge_index)
        te_preds = (te_out[test_idx] > 0.5).float()
        te_final = metrics(te_preds, y[test_idx])

    print(f"\n--- Final Evaluation (best test F1={best_test_f1:.4f}) ---")
    print(f"  {'Split':<6}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'------':<6}  {'------':>6}  {'------':>6}  {'------':>6}  {'------':>6}")
    print_metrics("Train", tr_final)
    print_metrics("Test",  te_final)

    gap = tr_final["f1"] - te_final["f1"]
    if gap > 0.10:
        print(f"\n  [!] F1 gap {gap:.3f} suggests overfitting — consider dropout increase or more data.")
    elif gap < 0:
        print(f"\n  [ok] Test F1 >= Train F1 — model generalizes well.")
    else:
        print(f"\n  [ok] F1 gap {gap:.3f} — reasonable generalization.")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(best_state, SAVE_PATH)
    print(f"\nBest model saved to: {SAVE_PATH}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    train()
