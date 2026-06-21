CALIFORNIA_BBOX = {
    "west": -124.48,
    "south": 32.53,
    "east": -114.13,
    "north": 42.01,
}

DATE_RANGE = {
    "start": "2020-01-01",
    "end": "2023-12-31",
}

DATASETS = {
    "modis_fire": {
        "collection": "MODIS/061/MOD14A1",
        "band": "FireMask",
        "description": "MODIS Terra Thermal Anomalies & Fire Daily Global 1km",
        "export_name": "modis_fire_california",
    },
    "smap_soil_moisture": {
        "collection": "NASA/SMAP/SPL4SMGP/008",
        "band": "sm_surface",
        "description": "SMAP Level-4 Global Surface Soil Moisture 9km",
        "export_name": "smap_soil_moisture_california",
    },
    "modis_ndvi": {
        "collection": "MODIS/061/MOD13A2",
        "band": "NDVI",
        "description": "MODIS Terra Vegetation Indices 16-Day Global 1km",
        "export_name": "modis_ndvi_california",
        "scale_factor": 0.0001,
    },
    "srtm_elevation": {
        "collection": "USGS/SRTMGL1_003",
        "band": "elevation",
        "description": "NASA SRTM Digital Elevation 30m",
        "export_name": "srtm_elevation_california",
    },
    "gpm_rainfall": {
        "collection": "NASA/GPM_L3/IMERG_MONTHLY_V06",
        "band": "precipitation",
        "description": "NASA GPM IMERG Monthly Mean Precipitation",
        "export_name": "gpm_rainfall_california",
    },
    "srtm_slope": {
        "collection": "USGS/SRTMGL1_003",
        "band": "elevation",
        "description": "SRTM-derived Slope in Degrees",
        "export_name": "srtm_slope_california",
    },
}

EXPORT_SCALE = 1000  # meters per pixel for raster exports
EXPORT_CRS = "EPSG:4326"
OUTPUT_DIR = "data/raw"
