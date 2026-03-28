import requests
import pandas as pd

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def fetch_skysat_metadata_by_ids(api_key, scene_ids, chunk_size=100):
    url = "https://api.planet.com/data/v1/quick-search"
    all_features = []

    for batch in chunks(scene_ids, chunk_size):
        payload = {
            "item_types": ["SkySatScene"],  # can also take PSScene for Planetscope imagery
            "filter": {
                "type": "StringInFilter",
                "field_name": "id",
                "config": batch
            }
        }

        r = requests.post(url, json=payload, auth=(api_key, ""))
        if r.status_code != 200:
            print(f"Batch failed ({r.status_code}): {r.text[:300]}")
            continue

        all_features.extend(r.json().get("features", []))

    if not all_features:
        return pd.DataFrame()

    df = pd.json_normalize(
        all_features,
        sep="."
    )

    if "id" in df.columns:
        df = df.rename(columns={"id": "scene_id"})

    # Track missing IDs
    found_ids = set(df["scene_id"]) if "scene_id" in df.columns else set()
    missing = [sid for sid in scene_ids if sid not in found_ids]

    if missing:
        df_missing = pd.DataFrame({"scene_id": missing})
        df = pd.concat([df, df_missing], ignore_index=True)

    return df