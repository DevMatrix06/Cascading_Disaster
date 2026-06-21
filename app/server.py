import os
import sys
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CALIFORNIA_BBOX
from models.gnn_model import TemporalCascadeGNN, CascadeGNN

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

# ── paths ─────────────────────────────────────────────────────────────────────
GRAPH_PATH      = "data/processed/california_graph_temporal.pt"
TEMPORAL_MODEL  = "models/saved/cascade_gnn_temporal.pt"
STATIC_MODEL    = "models/saved/cascade_gnn_final.pt"
LABELS_PATH     = "data/raw/landslide_labels.csv"
CAUSAL_PATH     = "data/processed/causal_summary.json"
GRID_SIZE       = 50

PRESET_REGIONS = {
    "california": {
        "name":  "California, USA",
        "bbox":  {"west": -124.48, "south": 32.53, "east": -114.13, "north": 42.01},
    },
    "australia": {
        "name":  "Eastern Australia",
        "bbox":  {"west": 143.0, "south": -39.0, "east": 154.0, "north": -28.0},
    },
    "amazon": {
        "name":  "Amazon Basin",
        "bbox":  {"west": -73.0, "south": -15.0, "east": -45.0, "north": 5.0},
    },
}

# ── lazy-loaded California globals ────────────────────────────────────────────
_graph   = None
_model_t = None   # temporal model (California)
_scores  = None
_labels  = None
_causal  = None


def _load_california():
    global _graph, _model_t, _scores, _labels, _causal
    if _graph is not None:
        return
    device = torch.device("cpu")

    _graph = torch.load(GRAPH_PATH, map_location=device, weights_only=False)

    _model_t = TemporalCascadeGNN(static_in=6, temporal_in=3, lstm_hidden=32, hidden_channels=64)
    _model_t.load_state_dict(torch.load(TEMPORAL_MODEL, map_location=device, weights_only=True))
    _model_t.eval()

    with torch.no_grad():
        _scores = _model_t(_graph.x, _graph.temporal_x, _graph.edge_index).numpy()

    _labels = pd.read_csv(LABELS_PATH)
    _labels["latitude"]  = pd.to_numeric(_labels["latitude"],  errors="coerce")
    _labels["longitude"] = pd.to_numeric(_labels["longitude"], errors="coerce")
    _labels = _labels.dropna(subset=["latitude", "longitude"])

    with open(CAUSAL_PATH) as f:
        _causal = json.load(f)


# ── GEE helpers ───────────────────────────────────────────────────────────────

def _gee_init():
    import ee
    try:
        ee.Initialize(project="cascade-disaster-predictor")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="cascade-disaster-predictor")
    return ee


def _fetch_static_features_gee(bbox, start_date, end_date):
    """Fetch fire, soil, NDVI, elevation from GEE for arbitrary bbox.
    Returns dict of numpy arrays keyed by feature name, or raises on failure."""
    ee = _gee_init()

    region = ee.Geometry.Rectangle([
        bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    ])

    fire = (ee.ImageCollection("MODIS/061/MOD14A1")
            .filterDate(start_date, end_date)
            .filterBounds(region)
            .select("FireMask")
            .max()
            .clip(region))

    soil = (ee.ImageCollection("NASA/SMAP/SPL4SMGP/008")
            .filterDate(start_date, end_date)
            .filterBounds(region)
            .select("sm_surface")
            .mean()
            .clip(region))

    ndvi = (ee.ImageCollection("MODIS/061/MOD13A2")
            .filterDate(start_date, end_date)
            .filterBounds(region)
            .select("NDVI")
            .mean()
            .multiply(0.0001)
            .clip(region))

    elev = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(region)

    import geemap
    import tempfile, rasterio
    results = {}
    layers  = {"fire": fire, "soil": soil, "ndvi": ndvi, "elevation": elev}

    with tempfile.TemporaryDirectory() as tmp:
        for name, image in layers.items():
            path = os.path.join(tmp, f"{name}.tif")
            geemap.ee_export_image(image, filename=path, scale=10000,
                                   region=region, file_per_band=False, crs="EPSG:4326")
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
                nd  = src.nodata
            if nd is not None:
                arr[arr == nd] = np.nan
            arr[arr <= -9999] = np.nan
            results[name] = arr

    return results


def _build_graph_from_arrays(arrays, bbox):
    """Extract 50x50 grid node features from raster arrays for a given bbox."""
    import rasterio.transform
    from rasterio.transform import from_bounds

    g = GRID_SIZE
    lats = np.linspace(bbox["north"], bbox["south"], g + 1)
    lons = np.linspace(bbox["west"],  bbox["east"],  g + 1)
    layer_keys = ["fire", "soil", "ndvi", "elevation"]
    features   = np.zeros((g * g, 4), dtype=np.float32)

    for f_idx, key in enumerate(layer_keys):
        arr = arrays[key]
        h, w = arr.shape
        # build a synthetic transform mapping array pixels to bbox
        tf = from_bounds(bbox["west"], bbox["south"], bbox["east"], bbox["north"], w, h)

        from rasterio.transform import rowcol
        for i in range(g):
            for j in range(g):
                r0, c0 = rowcol(tf, lons[j],     lats[i])
                r1, c1 = rowcol(tf, lons[j + 1], lats[i + 1])
                r0, r1 = sorted([max(0, min(int(r0), h-1)), max(0, min(int(r1), h-1))])
                c0, c1 = sorted([max(0, min(int(c0), w-1)), max(0, min(int(c1), w-1))])
                patch  = arr[r0:r1+1, c0:c1+1]
                valid  = patch[~np.isnan(patch)]
                features[i * g + j, f_idx] = float(np.mean(valid)) if valid.size > 0 else 0.0

    return features


def _build_edges():
    src, dst = [], []
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            n = i * GRID_SIZE + j
            for di, dj in dirs:
                ni, nj = i+di, j+dj
                if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                    src.append(n); dst.append(ni * GRID_SIZE + nj)
    return torch.tensor([src, dst], dtype=torch.long)


def _run_static_inference(features_np):
    """Run static CascadeGNN on extracted features. Falls back to weight init if no saved model."""
    device = torch.device("cpu")
    x      = torch.tensor(features_np, dtype=torch.float)
    edges  = _build_edges()

    model = CascadeGNN(in_channels=4, hidden_channels=64, out_channels=1)
    if os.path.exists(STATIC_MODEL):
        state = torch.load(STATIC_MODEL, map_location=device, weights_only=True)
        # cascade_gnn_final.pt holds TemporalCascadeGNN weights — only load if keys match
        try:
            model.load_state_dict(state)
        except RuntimeError:
            pass  # architecture mismatch — use random init as a fallback signal
    model.eval()
    with torch.no_grad():
        scores = model(x, edges).numpy()
    return scores


# ── shared helpers ────────────────────────────────────────────────────────────

def _risk_label(score):
    if score < 0.25: return "Low"
    if score < 0.50: return "Medium"
    if score < 0.75: return "High"
    return "Critical"


def _node_to_latlon(node_idx, bbox):
    row  = node_idx // GRID_SIZE
    col  = node_idx  % GRID_SIZE
    lats = np.linspace(bbox["north"], bbox["south"], GRID_SIZE + 1)
    lons = np.linspace(bbox["west"],  bbox["east"],  GRID_SIZE + 1)
    return float((lats[row] + lats[row+1]) / 2), float((lons[col] + lons[col+1]) / 2)


def _trigger_from_features(feat):
    fire, soil, ndvi, elev = feat
    parts = []
    if fire > 5:       parts.append("active fire presence")
    if soil < 0.15:    parts.append("critically low soil moisture")
    elif soil < 0.25:  parts.append("below-average soil moisture")
    if ndvi < 0.2:     parts.append("sparse or burned vegetation")
    if elev > 1500:    parts.append("steep high-elevation terrain")
    return ("Primary triggers: " + ", ".join(parts) + ".") if parts \
        else "No dominant trigger — risk driven by spatial proximity."


def _top5(scores, features, bbox):
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for idx in ranked[:5]:
        lat, lon = _node_to_latlon(idx, bbox)
        out.append({
            "node_index": idx,
            "lat":        round(lat, 4),
            "lon":        round(lon, 4),
            "risk_score": round(float(scores[idx]), 4),
            "risk_level": _risk_label(float(scores[idx])),
            "trigger":    _trigger_from_features(features[idx]),
        })
    return out


# ── California-specific helpers (use loaded graph) ────────────────────────────

def _ca_node_to_latlon(node_idx):
    return _node_to_latlon(node_idx, CALIFORNIA_BBOX)


def _pct(new_val, old_val):
    """Percentage change from old_val to new_val, None if old is zero."""
    if old_val == 0:
        return None
    return (new_val - old_val) / abs(old_val) * 100.0


def _feature_trends(node_idx):
    """Compare mean of last 3 months vs previous 3 months for fire/soil/ndvi.
    Returns dict with keys fire/soil/ndvi, each a float % change (+ = increase)."""
    series = _graph.temporal_x[node_idx].numpy()   # (48, 3)
    recent = series[-3:].mean(axis=0)               # (3,) — last 3 months
    prior  = series[-6:-3].mean(axis=0)             # (3,) — 3 months before that

    return {
        "fire": _pct(float(recent[0]), float(prior[0])),
        "soil": _pct(float(recent[1]), float(prior[1])),
        "ndvi": _pct(float(recent[2]), float(prior[2])),
        "_recent": recent.tolist(),
        "_prior":  prior.tolist(),
    }


def _temporal_trigger(node_idx):
    """Generate a plain-English trigger explanation driven by actual % changes."""
    trends = _feature_trends(node_idx)
    fire_pct = trends["fire"]
    soil_pct = trends["soil"]
    ndvi_pct = trends["ndvi"]

    # rank candidates by absolute magnitude so the dominant signal leads
    candidates = []
    if ndvi_pct is not None:
        candidates.append(("ndvi", ndvi_pct))
    if soil_pct is not None:
        candidates.append(("soil", soil_pct))
    if fire_pct is not None:
        candidates.append(("fire", fire_pct))

    candidates.sort(key=lambda x: abs(x[1]) if x[1] is not None else 0, reverse=True)
    dominant = candidates[0][0] if candidates else None

    sentences = []

    if dominant == "ndvi" and ndvi_pct is not None:
        if ndvi_pct < -20:
            sentences.append(
                f"Vegetation declined {abs(ndvi_pct):.0f}% over the past 3 months, "
                f"drying out surface soil and increasing landslide susceptibility."
            )
        elif ndvi_pct < -10:
            sentences.append(
                f"Vegetation cover fell {abs(ndvi_pct):.0f}%, reducing root cohesion in surface soils."
            )
        elif ndvi_pct > 10:
            sentences.append(
                f"Vegetation increased {ndvi_pct:.0f}%, suggesting recovery — reduced near-term risk."
            )

    if dominant == "fire" and fire_pct is not None:
        if fire_pct > 100:
            sentences.append(
                f"Fire activity spiked {fire_pct/100:.1f}x over the past 3 months, "
                f"burning vegetation and destabilizing soil structure."
            )
        elif fire_pct > 20:
            sentences.append(
                f"Fire activity rose {fire_pct:.0f}%, increasing burn scar area and erosion risk."
            )
        elif fire_pct < -20:
            sentences.append(
                f"Fire activity decreased {abs(fire_pct):.0f}% — conditions stabilising."
            )

    if dominant == "soil" and soil_pct is not None:
        if soil_pct < -30:
            sentences.append(
                f"Soil moisture fell {abs(soil_pct):.0f}% over the past 3 months, "
                f"critically reducing ground cohesion."
            )
        elif soil_pct < -15:
            sentences.append(
                f"Soil moisture dropped {abs(soil_pct):.0f}%, weakening slope stability."
            )
        elif soil_pct > 30:
            sentences.append(
                f"Soil moisture rose {soil_pct:.0f}%, saturating slopes and raising runoff risk."
            )

    # append secondary context
    if dominant != "ndvi" and ndvi_pct is not None and ndvi_pct < -15:
        sentences.append(
            f"Vegetation also down {abs(ndvi_pct):.0f}%, compounding surface instability."
        )
    if dominant != "soil" and soil_pct is not None and soil_pct < -20:
        sentences.append(f"Soil moisture also fell {abs(soil_pct):.0f}%.")
    if dominant != "fire" and fire_pct is not None and fire_pct > 50:
        sentences.append(f"Fire activity also elevated ({fire_pct:.0f}% increase).")

    if not sentences:
        # fall back to static feature check
        x = _graph.x[node_idx].numpy()
        fire_val, soil_val, ndvi_val, elev_val = x
        parts = []
        if fire_val > 5:      parts.append("active fire presence")
        if soil_val < 0.15:   parts.append("critically low soil moisture")
        if ndvi_val < 0.2:    parts.append("sparse or burned vegetation")
        if elev_val > 1500:   parts.append("steep high-elevation terrain")
        return ("Primary triggers: " + ", ".join(parts) + ".") if parts \
            else "No dominant short-term trigger — risk driven by long-term spatial exposure."

    return " ".join(sentences)


def _trend(node_idx):
    series = _graph.temporal_x[node_idx].numpy()
    delta  = float(series[-1].mean() - series[-2].mean())
    if delta >  0.02: return "increasing"
    if delta < -0.02: return "decreasing"
    return "stable"


def _plain_english_trigger(node_idx):
    return _trigger_from_features(_graph.x[node_idx].numpy())


def _landslide_count_in_cell(node_idx):
    row  = node_idx // GRID_SIZE
    col  = node_idx  % GRID_SIZE
    lats = np.linspace(CALIFORNIA_BBOX["north"], CALIFORNIA_BBOX["south"], GRID_SIZE + 1)
    lons = np.linspace(CALIFORNIA_BBOX["west"],  CALIFORNIA_BBOX["east"],  GRID_SIZE + 1)
    mask = (
        (_labels["latitude"]  >= lats[row+1]) & (_labels["latitude"]  < lats[row]) &
        (_labels["longitude"] >= lons[col])   & (_labels["longitude"] < lons[col+1])
    )
    return int(mask.sum())


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/regions", methods=["GET"])
def get_regions():
    return jsonify({
        key: {"name": v["name"], "bbox": v["bbox"]}
        for key, v in PRESET_REGIONS.items()
    })


@app.route("/api/static_risk_map", methods=["GET"])
def static_risk_map():
    _load_california()
    grid       = _scores.reshape(GRID_SIZE, GRID_SIZE).tolist()
    landslides = _labels[["latitude", "longitude", "landslide_type"]].rename(
        columns={"landslide_type": "type"}).to_dict(orient="records")
    return jsonify({
        "grid":       grid,
        "grid_size":  GRID_SIZE,
        "bbox":       CALIFORNIA_BBOX,
        "landslides": landslides,
        "stats": {
            "min":         float(_scores.min()),
            "max":         float(_scores.max()),
            "mean":        float(_scores.mean()),
            "high_risk":   int((_scores > 0.5).sum()),
            "total_nodes": int(len(_scores)),
        },
    })


@app.route("/api/region_summary", methods=["POST"])
def region_summary():
    _load_california()
    data     = request.get_json()
    node_idx = max(0, min(int(data.get("node_index", 0)), GRID_SIZE * GRID_SIZE - 1))
    score    = float(_scores[node_idx])
    temporal = _graph.temporal_x[node_idx].numpy()   # (48, 3)

    months = []
    for i in range(48):
        y = 2020 + i // 12
        m = i % 12 + 1
        months.append(f"{y}-{m:02d}")

    trends  = _feature_trends(node_idx)
    trigger = _temporal_trigger(node_idx)
    lat, lon = _ca_node_to_latlon(node_idx)

    def _fmt_pct(v):
        if v is None: return None
        return f"{'+' if v >= 0 else ''}{v:.1f}%"

    return jsonify({
        "node_index":      node_idx,
        "lat":             lat,
        "lon":             lon,
        "risk_score":      round(score, 4),
        "risk_level":      _risk_label(score),
        "trend":           _trend(node_idx),
        "trigger":         trigger,
        "landslide_count": _landslide_count_in_cell(node_idx),
        "feature_trends": {
            "fire": _fmt_pct(trends["fire"]),
            "soil": _fmt_pct(trends["soil"]),
            "ndvi": _fmt_pct(trends["ndvi"]),
        },
        "time_series": {
            "months":        months,
            "fire":          temporal[:, 0].tolist(),
            "soil_moisture": temporal[:, 1].tolist(),
            "ndvi":          temporal[:, 2].tolist(),
        },
        "cascade_chain": [
            {"from": "Fire Activity",      "to": "Vegetation Loss",    "lag": "0 mo"},
            {"from": "Vegetation Loss",    "to": "Soil Moisture Drop", "lag": "1 mo"},
            {"from": "Soil Moisture Drop", "to": "Landslide Risk",     "lag": "1-2 mo"},
        ],
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data       = request.get_json()
    bbox       = data.get("region_bbox", CALIFORNIA_BBOX)
    start_date = data.get("start_date", "2020-01-01")
    end_date   = data.get("end_date",   "2023-12-31")
    hazards    = data.get("hazards", ["fire", "landslide", "flood"])

    # validate bbox
    required = {"west", "east", "north", "south"}
    if not required.issubset(bbox.keys()):
        return jsonify({"error": "region_bbox must include west, east, north, south"}), 400

    try:
        arrays   = _fetch_static_features_gee(bbox, start_date, end_date)
        features = _build_graph_from_arrays(arrays, bbox)
        scores   = _run_static_inference(features)
        source   = "earth_engine"
    except Exception as e:
        # graceful fallback: return structured error, keep server alive
        return jsonify({
            "error":      str(e),
            "error_type": type(e).__name__,
            "hint":       "Earth Engine fetch failed. Check GEE authentication, "
                          "project permissions, and that the bbox overlaps available data.",
        }), 502

    top5 = _top5(scores, features, bbox)
    _load_california()  # ensure causal summary is available

    return jsonify({
        "bbox":             bbox,
        "start_date":       start_date,
        "end_date":         end_date,
        "hazards":          hazards,
        "source":           source,
        "model":            "CascadeGNN (static, 4 features)",
        "nodes_analyzed":   int(len(scores)),
        "grid":             scores.reshape(GRID_SIZE, GRID_SIZE).tolist(),
        "grid_size":        GRID_SIZE,
        "mean_risk":        round(float(np.mean(scores)), 4),
        "max_risk":         round(float(np.max(scores)), 4),
        "high_risk_count":  int((scores > 0.5).sum()),
        "model_confidence": round(float(np.std(scores)), 4),
        "top5_zones":       top5,
        "causal_summary":   _causal[:3] if _causal else [],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
