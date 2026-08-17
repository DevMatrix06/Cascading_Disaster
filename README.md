<div align="center">

# 🌋 Cascade Disaster Prediction

**Spatiotemporal Graph Neural Network for predicting cascading natural disasters** (wildfire → landslide) using NASA satellite data and causal inference.

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch%20Geometric-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?style=flat-square&logo=googleearth&logoColor=white" />
<img src="https://img.shields.io/badge/tigramite-causal%20discovery-7C3AED?style=flat-square" />
<img src="https://img.shields.io/badge/license-MIT-00C2FF?style=flat-square" />

</div>

---

## What this does

Most disaster prediction systems model hazards in isolation. This project models the **cascade** — wildfire burns vegetation, dry soil loses cohesion, rainfall triggers landslide. The full chain, predicted from satellite data alone.

Built on 4 years of NASA/ESA observations over California (2020–2023), validated against **657 real landslide events** from the NASA COOLR catalog, with a live interactive dashboard.

<p align="center">
  <img src="visualizations/cascade_risk_map.png" width="600" alt="Cascade risk heatmap over California" />
</p>

## ⚡ Quickstart (use the pretrained model)

Already have a trained checkpoint in `models/saved/`? Skip straight to the dashboard:

```bash
git clone https://github.com/DevMatrix06/Cascading_Disaster.git
cd Cascading_Disaster
pip install -r requirements.txt
python app/run.py
# Open http://localhost:5000
```

For full setup (fetching satellite data, training from scratch), see [Full Setup](#full-setup) below.

## Dashboard

<p align="center">
  <img src="visualizations/dashboard_demo.gif" width="700" alt="Interactive dashboard demo" />
</p>

- Live satellite data via Google Earth Engine for any region on Earth
- 50×50 risk heatmap overlaid on OpenStreetMap
- Per-cell cascade chain: Fire → Vegetation Loss → Soil Moisture Drop → Landslide Risk
- 48-month time series per node with trend detection
- PCMCI causal graph visualization inline

> 💡 Don't have a GIF yet? Record one with [Peek](https://github.com/phw/peek) (Linux) or [ScreenToGif](https://www.screentogif.com/) (Windows) — a 5–10s loop of the heatmap + causal graph sells this project far better than a static screenshot.

## Results

| Model | Precision | Recall | F1 | Params |
|---|---|---|---|---|
| Static GNN (baseline) | 0.128 | 0.840 | 0.223 | 8,961 |
| Temporal only (LSTM) | 0.344 | 0.660 | 0.452 | 17,793 |
| **Static + Temporal (full)** | 0.305 | 0.780 | 0.439 | 17,793 |

*Test set: held-out 2023 temporal window + stratified 80/20 node split. Model catches 6 in 10 real landslide zones on unseen data.*

The full model trades a small amount of precision for materially higher recall than temporal-only (0.78 vs. 0.66) at nearly the same F1 — for an early-warning use case, missing fewer real landslides matters more than a few extra false alarms, so the full model is the one shipped in the dashboard.

**Key causal finding:** NDVI decline Granger-causes soil moisture loss at 1-month lag (PCMCI strength 0.450, frequency 24%) — the drought-to-landslide precursor chain confirmed from satellite data alone.

## Architecture

```
NASA Satellite Data
(MODIS fire, SMAP soil moisture, MODIS NDVI, SRTM elevation)
                    ↓
        Google Earth Engine (data pipeline)
                    ↓
    50×50 Spatial Graph — 2,500 nodes, 19,404 edges
         (Moore neighborhood, 21km × 21km cells)
                    ↓
    LSTM encoder — 48-month time series per node
         (hidden dim 32, 3 features per timestep)
                    ↓
    GraphSAGE × 3 layers — spatial message passing
         (static 4-dim + temporal 32-dim input)
                    ↓
    Cascade Risk Score per node (0–1)
                    ↓
    PCMCI Causal Discovery
         (tigramite, ParCorr, max lag 3 months)
```

## Data Sources

| Dataset | Source | Resolution | Use |
|---|---|---|---|
| Fire activity (FireMask) | MODIS MOD14A1 | 1 km | Primary hazard trigger |
| Soil moisture | NASA SMAP SPL4SMGP | 9 km | Cascade signal |
| Vegetation index (NDVI) | MODIS MOD13A2 | 1 km | Precursor feature |
| Elevation | USGS SRTM | 30 m | Terrain context |
| Landslide events | NASA COOLR Catalog | Point | Ground truth labels |

All data fetched via Google Earth Engine. 657 California landslide events → 251 unique high-risk grid nodes (1:9 class imbalance).

## Full Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Authenticate Google Earth Engine**
```bash
earthengine authenticate
```
You need a Google Earth Engine account — sign up free at [earthengine.google.com](https://earthengine.google.com).

**3. Fetch satellite data**
```bash
python data/fetch_data.py
python data/fetch_temporal_data.py
```

**4. Build graph and labels**
```bash
python data/process_landslide_labels.py
python data/build_graph_with_labels.py
python data/build_temporal_graph.py
```

**5. Train model**
```bash
python models/train_model.py
```

**6. Run causal analysis**
```bash
python models/causal_analysis.py
```

**7. Launch dashboard**
```bash
python app/run.py
```

## Project Structure

```
cascade-disaster-prediction/
├── app/
│   ├── server.py
│   ├── static/index.html
│   └── run.py
├── data/
│   ├── fetch_data.py
│   ├── fetch_temporal_data.py
│   ├── build_graph_with_labels.py
│   ├── build_temporal_graph.py
│   └── raw/
├── models/
│   ├── gnn_model.py
│   ├── train_model.py
│   ├── causal_analysis.py
│   ├── ablation_study.py
│   ├── global_test.py
│   └── saved/
├── utils/
│   ├── build_graph.py
│   └── visualize_data.py
├── visualizations/
│   ├── cascade_risk_map.png
│   ├── australia_risk_map.png
│   ├── causal_graph.png
│   └── data_overview.png
├── config.py
├── requirements.txt
└── PAPER_DRAFT.md
```

## Causal Discovery Results

| Cause | Effect | Lag | Strength | Frequency |
|---|---|---|---|---|
| Soil moisture | Soil moisture | 1 mo | 0.749 | 90% |
| NDVI | NDVI | 1 mo | 0.697 | 85% |
| Fire | Fire | 1 mo | 0.486 | 76% |
| **NDVI** | **Soil moisture** | **1 mo** | **0.450** | **24%** |

The NDVI → soil moisture link at lag 1 month is the key cross-variable finding. Vegetation stress precedes soil moisture loss — a known drought-to-landslide precursor confirmed from observational satellite data.

<p align="center">
  <img src="visualizations/causal_graph.png" width="500" alt="PCMCI causal graph" />
</p>

## Limitations

- 21 km spatial resolution too coarse for individual hillslope prediction
- NASA COOLR catalog underreports events in rural areas
- Zero-shot transfer to Australia shows no score separation — domain adaptation required for global deployment
- Spatial block cross-validation not implemented (nearby grid cells are spatially autocorrelated, so reported metrics may be optimistic — this is the top priority fix)

## Tech Stack

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch%20Geometric-3C2179?style=flat-square" />
<img src="https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?style=flat-square&logo=googleearth&logoColor=white" />
<img src="https://img.shields.io/badge/tigramite-7C3AED?style=flat-square" />
<img src="https://img.shields.io/badge/rasterio-00A3E0?style=flat-square" />
<img src="https://img.shields.io/badge/geopandas-139C5A?style=flat-square" />
<img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white" />
<img src="https://img.shields.io/badge/Leaflet.js-199900?style=flat-square&logo=leaflet&logoColor=white" />
<img src="https://img.shields.io/badge/Chart.js-FF6384?style=flat-square&logo=chart.js&logoColor=white" />

## License

MIT
