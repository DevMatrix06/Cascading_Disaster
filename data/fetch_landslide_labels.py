import os
import sys
import json
import urllib.request
import urllib.parse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CALIFORNIA_BBOX

BASE_URL = "https://maps.nccs.nasa.gov/arcgis/rest/services/landslide_v2/MapServer/0/query"
OUTPUT_PATH = "data/raw/landslide_labels.csv"
MAX_RECORDS = 2000  # ArcGIS REST default page size cap


def build_url(offset=0):
    params = urllib.parse.urlencode({
        "where": "country_name='United States'",
        "outFields": "latitude,longitude,event_date,landslide_type",
        "f": "json",
        "resultOffset": offset,
        "resultRecordCount": MAX_RECORDS,
        "orderByFields": "event_date ASC",
    })
    return f"{BASE_URL}?{params}"


def fetch_page(offset=0):
    url = build_url(offset)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def fetch_all_records():
    records = []
    offset = 0
    print("Fetching NASA COOLR Global Landslide Catalog (US records)...")
    while True:
        print(f"  Requesting records {offset}–{offset + MAX_RECORDS - 1} ...")
        data = fetch_page(offset)

        if "error" in data:
            raise RuntimeError(f"API error: {data['error']}")

        features = data.get("features", [])
        if not features:
            break

        for feat in features:
            attrs = feat.get("attributes", {})
            records.append({
                "latitude":       attrs.get("latitude"),
                "longitude":      attrs.get("longitude"),
                "event_date":     attrs.get("event_date"),
                "landslide_type": attrs.get("landslide_type"),
            })

        if not data.get("exceededTransferLimit", False):
            break

        offset += MAX_RECORDS

    print(f"Total US records fetched: {len(records)}")
    return records


def filter_california(records):
    df = pd.DataFrame(records)
    df = df.dropna(subset=["latitude", "longitude"])
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    mask = (
        (df["longitude"] >= CALIFORNIA_BBOX["west"])  &
        (df["longitude"] <= CALIFORNIA_BBOX["east"])  &
        (df["latitude"]  >= CALIFORNIA_BBOX["south"]) &
        (df["latitude"]  <= CALIFORNIA_BBOX["north"])
    )
    return df[mask].reset_index(drop=True)


def load_from_local_csv(path):
    print(f"Loading from local file: {path}")
    df = pd.read_csv(path, low_memory=False)
    records = []
    for col_lat in ("latitude", "lat"):
        if col_lat in df.columns:
            break
    for col_lon in ("longitude", "lon"):
        if col_lon in df.columns:
            break
    for _, row in df.iterrows():
        records.append({
            "latitude":       row.get(col_lat),
            "longitude":      row.get(col_lon),
            "event_date":     row.get("event_date", row.get("date")),
            "landslide_type": row.get("landslide_category", row.get("landslide_type", row.get("type"))),
        })
    print(f"Total records in local file: {len(records)}")
    return records


def fetch_landslide_labels(local_csv=None):
    if local_csv and os.path.exists(local_csv):
        records = load_from_local_csv(local_csv)
    else:
        records = fetch_all_records()
    ca_df   = filter_california(records)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    ca_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nCalifornia landslide events found : {len(ca_df)}")
    print(f"Saved to                          : {OUTPUT_PATH}")
    if not ca_df.empty:
        print(f"\nSample records:")
        print(ca_df.head(5).to_string(index=False))

    return ca_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=str, default=None,
                        help="Path to a locally downloaded landslide CSV (skips API fetch)")
    args = parser.parse_args()
    fetch_landslide_labels(local_csv=args.local)
