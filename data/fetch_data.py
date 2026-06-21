import os
import sys
import ee
import geemap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CALIFORNIA_BBOX, DATE_RANGE, DATASETS, EXPORT_SCALE, EXPORT_CRS, OUTPUT_DIR


def authenticate_gee():
    try:
        ee.Initialize(project='cascade-disaster-predictor')
        print("Google Earth Engine initialized successfully.")
    except Exception:
        print("Authenticating with Google Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project='cascade-disaster-predictor')
        print("Google Earth Engine authenticated and initialized.")


def get_california_geometry():
    return ee.Geometry.Rectangle(
        [
            CALIFORNIA_BBOX["west"],
            CALIFORNIA_BBOX["south"],
            CALIFORNIA_BBOX["east"],
            CALIFORNIA_BBOX["north"],
        ]
    )


def fetch_modis_fire(region, start_date, end_date):
    cfg = DATASETS["modis_fire"]
    collection = (
        ee.ImageCollection(cfg["collection"])
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .select(cfg["band"])
    )
    return collection.max().clip(region)


def fetch_smap_soil_moisture(region, start_date, end_date):
    cfg = DATASETS["smap_soil_moisture"]
    collection = (
        ee.ImageCollection(cfg["collection"])
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .select(cfg["band"])
    )
    return collection.mean().clip(region)


def fetch_modis_ndvi(region, start_date, end_date):
    cfg = DATASETS["modis_ndvi"]
    collection = (
        ee.ImageCollection(cfg["collection"])
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .select(cfg["band"])
    )
    ndvi = collection.mean().multiply(cfg["scale_factor"]).clip(region)
    return ndvi


def fetch_srtm_elevation(region):
    cfg = DATASETS["srtm_elevation"]
    image = ee.Image(cfg["collection"]).select(cfg["band"]).clip(region)
    return image


def export_to_geotiff(image, export_name, region, output_dir, scale=EXPORT_SCALE, crs=EXPORT_CRS):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{export_name}.tif")
    print(f"Exporting {export_name} to {output_path} ...")
    geemap.ee_export_image(
        image,
        filename=output_path,
        scale=scale,
        region=region,
        file_per_band=False,
        crs=crs,
    )
    print(f"Saved: {output_path}")
    return output_path


def fetch_all(start_date=None, end_date=None, output_dir=None):
    start_date = start_date or DATE_RANGE["start"]
    end_date = end_date or DATE_RANGE["end"]
    output_dir = output_dir or OUTPUT_DIR

    authenticate_gee()
    region = get_california_geometry()

    fetchers = {
        "modis_fire": lambda: fetch_modis_fire(region, start_date, end_date),
        "smap_soil_moisture": lambda: fetch_smap_soil_moisture(region, start_date, end_date),
        "modis_ndvi": lambda: fetch_modis_ndvi(region, start_date, end_date),
        "srtm_elevation": lambda: fetch_srtm_elevation(region),
    }

    exported = {}
    for key, fetcher in fetchers.items():
        export_name = DATASETS[key]["export_name"]
        print(f"\nFetching {DATASETS[key]['description']} ...")
        image = fetcher()
        path = export_to_geotiff(image, export_name, region, output_dir)
        exported[key] = path

    print("\nAll datasets exported:")
    for key, path in exported.items():
        print(f"  {key}: {path}")

    return exported


if __name__ == "__main__":
    fetch_all()
