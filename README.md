# Cascading Disaster Prediction

Predicts cascading natural disasters (wildfire, drought, flood) in California using multi-source remote sensing data, Graph Neural Networks, and causal inference.

## Project Structure

```
Cascading_Disaster/
├── config.py                  # Global configuration (bounding box, dates, dataset IDs)
├── requirements.txt           # Python dependencies
├── data/
│   ├── fetch_data.py          # GEE authentication and dataset download
│   └── raw/                   # Downloaded GeoTIFFs (auto-created on fetch)
├── models/                    # GNN and prediction model code
├── notebooks/                 # Exploratory analysis notebooks
├── utils/                     # Preprocessing and helper functions
└── visualizations/            # Map and plot outputs
```

## Datasets

| Key | Source | Description |
|-----|--------|-------------|
| `modis_fire` | MODIS/006/MOD14A1 | Thermal anomalies & active fire (FireMask) |
| `smap_soil_moisture` | NASA_USDA/HSL/SMAP10KM_soil_moisture | Surface soil moisture (ssm) |
| `modis_ndvi` | MODIS/006/MOD13A2 | Vegetation index (NDVI) |
| `srtm_elevation` | USGS/SRTMGL1_003 | Digital elevation model |

**Region:** California (bbox: -124.48W, 32.53S, -114.13E, 42.01N)  
**Period:** 2020-01-01 to 2023-12-31

## Setup

```bash
pip install -r requirements.txt
```

Authenticate with Google Earth Engine (first run only):

```bash
earthengine authenticate
```

## Fetch Data

```bash
python data/fetch_data.py
```

GeoTIFFs are saved to `data/raw/`. The script:
1. Authenticates with GEE (prompts on first run)
2. Fetches all 4 datasets clipped to the California bounding box
3. Exports each as a GeoTIFF at 1 km resolution in EPSG:4326

## Configuration

Edit [config.py](config.py) to change:
- `CALIFORNIA_BBOX` — spatial extent
- `DATE_RANGE` — temporal window
- `DATASETS` — collection IDs, bands, export names
- `EXPORT_SCALE` — output resolution in meters (default 1000)
