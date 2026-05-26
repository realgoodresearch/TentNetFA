import os
import random
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box
import matplotlib.pyplot as plt
import sys

# ==========================
# CONFIGURATION
# ==========================

TIF_DIR = "tif_files/historic"
PRED_DIR = "tif_files/historic/predictions"
GAZA_BOUNDARY_SHP = "gaza_boundaries/GazaStrip_MunicipalBoundaries.shp"
PREWAR_TIF = "tif_files/prewar_gaza.tif"

OUTPUT_CSV = "evaluation/manual_annotation_results.csv"

TIF_LIST = [
    # DONE "abasan_al_kabira_khuzaa_20251014_122107_ssc10_u0001_visual_clip.tif"
    # DONE "al_mawasi_20250427_120438_ssc9_u0002_visual_clip.tif"
    # DONE "al_mawasi_khan_yunis_20241014_120709_ssc10_u0001_visual_clip.tif"
    # DONE "deir_el_balah_nuseirat_20241024_120909_ssc8_u0001_visual_clip.tif"
    # "deir_el_balah_nuseirat_20250119_121636_ssc10_u0002_visual_clip.tif"
    # "deir_el_balah_nuseirat_20250124_060314_ssc4_u0002_visual_clip.tif"
    # "deir_el_balah_nuseirat_20250331_121449_ssc10_u0002_visual_clip.tif"
    # "deir_el_balah_nuseirat_20250717_122424_ssc7_u0001_visual_clip.tif"
    # "deir_el_balah_nuseirat_20250822_122241_ssc9_u0002_visual_clip.tif"
    # DONE  "deir_el_balah_nuseirat_gaza_city_20251014_121159_ssc7_u0002_visual_clip.tif"
    # "deir_el_balah_nuseirat_gaza_city_20251202_053316_ssc12_u0002_visual_clip.tif"
    # "deir_el_blah_nuseirat_20250717_063437_ssc13_u0002_visual_clip.tif"
    # DONE  "gaza_city_20250610_062803_ssc1_u0001_visual_clip.tif"
    # "gaza_city_20250717_122418_ssc7_u0002_visual_clip.tif"
    # DONE  "gaza_city_and_jabalia_20250315_121122_ssc10_u0001_visual_clip_file_format.tif"
    # "gaza_city_and_jabalia_20250319_121502_ssc8_u0001_visual_clip_file_format.tif"
    # "gaza_city_beit_hanoun_20251014_122137_ssc10_u0001_visual_clip.tif"
    # DONE  "gaza_city_beit_lahiya_20241024_120902_ssc8_u0001_visual_clip.tif"
    # "gaza_city_jabalia_20241014_120703_ssc10_u0001_visual_clip.tif"
    # "gaza_city_jabalia_beit_lahiya_20250124_060314_ssc4_u0001_visual_clip.tif"
    # "gaza_city_jabalia_beit_lahiya_20250427_120432_ssc9_u0001_visual_clip.tif"
    # "gaza_city_jabalia_beit_lahiya_20250610_120700_ssc8_u0001_visual_clip.tif"
    # DONE  "gaza_city_jabalia_bit_lahiya_20250717_063437_ssc13_u0001_visual_clip.tif"
    # "gaza_city_jabalia_beit_lahiya_20250822_122241_ssc9_u0001_visual_clip.tif"
    # "gaza_city_jabalia_beit_lahiya_20251014_121159_ssc7_u0001_visual_clip.tif"
    # "gaza_city_jabalia_beit_lahiya_20251202_053316_ssc12_u0001_visual_clip.tif"
    # "gaza_jabalia_beit_hanoun_20251202_052652_ssc2_u0001_visual_clip.tif"
    # DONE  "jabalia_beit_hanoun_20251014_122855_ssc6_u0001_visual_clip.tif"
    # "jabalia_beit_lahiya_20250717_122418_ssc7_u0001_visual_clip.tif"
    # DONE  "kerem_salom_crossing_20250124_060928_ssc2_u0002_visual_clip.tif"
    # "khan_yunis_20250121_121534_ssc6_u0002_visual_clip.tif"
    # "khan_yunis_20250717_122424_ssc7_u0002_visual_clip.tif"
    # "khan_yunis_20250822_053233_ssc2_u0001_visual_clip.tif"
    # DONE  "khan_yunis_al_maghazi_bureij_20251202_052652_ssc2_u0002_visual_clip.tif"
    # "khan_yunis_al_mawasi_deir_el_balah_20250610_120711_ssc8_u0001_visual_clip.tif"
    # "khan_yunis_deir_el_balah_20250113_120653_ssc8_u0001_visual_clip_file_format.tif"
    # "khan_yunis_deir_el_balah_nuseirat_20250427_120438_ssc9_u0001_visual_clip.tif"
    # DONE  "khan_yunis_deir_el_balah_nuseirat_20251014_122137_ssc10_u0002_visual_clip.tif"
    # DONE  "nuseirat_netzarim_corridor_gaza_city_20250610_120704_ssc8_u0001_visual_clip.tif"
    # DONE  "rafah_al_mawasi_20241014_120506_ssc8_u0001_visual.tif"
    "rafah_al_mawasi_20250121_060038_ssc2_u0001_visual_clip.tif"
    # "rafah_al_mawasi_20250124_060928_ssc2_u0001_visual_clip.tif"
    # "rafah_al_mawasi_khan_yunis_20251014_121159_ssc7_u0003_visual_clip.tif"
    # DONE  "rafah_al_mawasi_khan_yunis_20251202_053316_ssc12_u0003_visual_clip.tif"
    # "rafah_khan_yunis_20251014_122137_ssc10_u0003_visual_clip.tif"
    # "rafah_rafah_crossing_20251202_052652_ssc2_u0003_visual_clip.tif"
    # DONE  "tel_al-sultan_al_mawasi_20250331_055243_ssc4_u0001_visual_clip.tif"
    # DONE  "wadi_gaza_20251014_122855_ssc6_u0002_visual_clip.tif"
]

N_TILES_PER_IMAGE = 3
TILE_SIZE_METERS = 100


# ==========================
# HELPER FUNCTIONS
# ==========================

def parse_tif_name(filename):
    """
    Parse filename into:
    region, date, time, satellite
    """
    name = os.path.splitext(os.path.basename(filename))[0]

    # Expected format:
    # region_YYYYMMDD_HHMMSS_satellite_...
    parts = name.split("_")

    # region is everything before date
    date_idx = None
    for i, p in enumerate(parts):
        if re.fullmatch(r"\d{8}", p):
            date_idx = i
            break

    if date_idx is None:
        raise ValueError(f"Could not parse date in filename: {filename}")

    region = "_".join(parts[:date_idx])
    date_raw = parts[date_idx]
    time_raw = parts[date_idx + 1]
    satellite = parts[date_idx + 2]

    date = f"{date_raw[0:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
    time = f"{time_raw[0:2]}:{time_raw[2:4]}:{time_raw[4:6]}"

    return region, date, time, satellite


def random_tile_within_polygon(src, polygon_gdf):
    """
    Sample a random 100m x 100m tile fully inside
    intersection of raster bounds and Gaza boundary.
    """
    raster_bounds = box(*src.bounds)

    # Reproject polygon to raster CRS
    poly = polygon_gdf.to_crs(src.crs)
    intersection = poly.geometry.union_all().intersection(raster_bounds)

    if intersection.is_empty:
        raise ValueError("No overlap between raster and Gaza boundary.")

    minx, miny, maxx, maxy = intersection.bounds

    for _ in range(1000):
        x = random.uniform(minx, maxx - TILE_SIZE_METERS)
        y = random.uniform(miny, maxy - TILE_SIZE_METERS)

        tile_geom = box(x, y, x + TILE_SIZE_METERS, y + TILE_SIZE_METERS)

        if intersection.contains(tile_geom):
            return tile_geom

    raise RuntimeError("Failed to sample valid tile after many attempts.")


def count_predictions_in_tile(tile_geom, geojson_path, raster_crs):
    """
    Count number of prediction points inside tile.
    Ensures CRS alignment and verifies spatial overlap.
    """

    if not os.path.exists(geojson_path):
        print("GeoJSON not found:", geojson_path)
        return 0

    preds = gpd.read_file(geojson_path)

    if preds.empty:
        return 0

    # If CRS missing, assume WGS84 (very common for GeoJSON)
    if preds.crs is None:
        preds.set_crs("EPSG:4326", inplace=True)

    # Reproject predictions to raster CRS
    preds = preds.to_crs(raster_crs)

    # Quick sanity check: do predictions overlap raster at all?
    raster_bounds_geom = box(*rasterio.open(tif_path).bounds)

    if not preds.total_bounds.any():
        return 0

    # Filter predictions to raster extent first
    preds = preds[preds.intersects(raster_bounds_geom)]

    if preds.empty:
        print("Predictions do not overlap raster after reprojection.")
        return 0

    # Now count points inside tile
    count = preds.within(tile_geom).sum()

    return int(count)


def show_tile_and_get_count(tile_array, prewar_array):
    """
    Display current and prewar tiles side by side.
    Assumes first three bands are RGB.
    """

    def prepare_rgb(arr):
        rgb = arr[:3]
        rgb = np.transpose(rgb, (1, 2, 0))

        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.float32)
            p2 = np.percentile(rgb, 2)
            p98 = np.percentile(rgb, 98)
            if p98 > p2:
                rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
            else:
                rgb = np.zeros_like(rgb)
        else:
            rgb = rgb / 255.0

        return rgb

    rgb_current = prepare_rgb(tile_array)
    rgb_prewar = prepare_rgb(prewar_array)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(rgb_prewar)
    axes[0].set_title("Prewar")
    axes[0].axis("off")

    axes[1].imshow(rgb_current)
    axes[1].set_title("Current")
    axes[1].axis("off")

    plt.show(block=True)
    plt.close(fig)

    while True:
        try:
            val = int(input("Number of tents you count in this tile: "))
            return val
        except ValueError:
            print("Enter a valid integer.")


# ==========================
# MAIN PROCESS
# ==========================

def main():

    if not TIF_LIST:
        raise ValueError("TIF_LIST is empty. Add filenames to process.")

    gaza_boundary = gpd.read_file(GAZA_BOUNDARY_SHP)

    rows = []

    for tif_name in TIF_LIST:
        tif_path = os.path.join(TIF_DIR, tif_name)
        base = os.path.splitext(tif_name)[0]
        geojson_path = None
        for ext in (".geojson", ".json"):
            candidate = os.path.join(PRED_DIR, base + ext)
            if os.path.exists(candidate):
                geojson_path = candidate
                break

        # optional sanity check (remove if unnecessary)
        if geojson_path is None:
            raise FileNotFoundError(f"No .geojson/.json found for {tif_name}")

        print(f"\nProcessing: {tif_name}")

        region, date, time, satellite = parse_tif_name(tif_name)

        with rasterio.open(tif_path) as src:
            with rasterio.open(PREWAR_TIF) as prewar_src:

                # ---- Load and align predictions ONCE ----
                if os.path.exists(geojson_path):
                    preds = gpd.read_file(geojson_path)

                    if preds.empty:
                        sys.exit("Prediction file is empty.")
                    else:
                        # If GeoJSON has no CRS, assume WGS84
                        if preds.crs is None:
                            preds.set_crs("EPSG:4326", inplace=True)

                        # Reproject to raster CRS
                        preds = preds.to_crs(src.crs)

                        # Keep only predictions inside raster bounds
                        raster_bounds_geom = box(*src.bounds)
                        preds = preds[preds.intersects(raster_bounds_geom)]
                else:
                    preds = gpd.GeoDataFrame(geometry=[], crs=src.crs)

                # ---- sample until we have N_TILES_PER_IMAGE valid tiles ----
                accepted = 0
                attempts = 0
                max_attempts = max(N_TILES_PER_IMAGE * 20, 500)

                while accepted < N_TILES_PER_IMAGE and attempts < max_attempts:
                    attempts += 1

                    tile_geom = random_tile_within_polygon(src, gaza_boundary)

                    window = from_bounds(
                        *tile_geom.bounds,
                        transform=src.transform
                    )

                    # prepare prewar window, reproject tile_geom to prewar CRS if necessary
                    if prewar_src.crs == src.crs:
                        prewar_geom = tile_geom
                    else:
                        prewar_geom = gpd.GeoSeries([tile_geom], crs=src.crs).to_crs(prewar_src.crs).iloc[0]

                    prewar_window = from_bounds(
                        *prewar_geom.bounds,
                        transform=prewar_src.transform
                    )

                    # safe reads: skip tile if reading fails (out-of-range)
                    try:
                        tile_array = src.read(window=window)
                        prewar_array = prewar_src.read(window=prewar_window)
                    except Exception as e:
                        print("Read error for sampled window, trying another tile:", e)
                        continue

                    # Ensure we have at least 3 bands
                    if tile_array.shape[0] < 3:
                        print("Tile has fewer than 3 bands, skipping")
                        continue

                    # ---- Reject empty / nodata tiles ----
                    nodata = src.nodata
                    rgb = tile_array[:3]

                    # Condition 1: all zeros
                    all_zero = np.all(rgb == 0)

                    # Condition 2: all nodata (if defined)
                    all_nodata = (nodata is not None) and np.all(rgb == nodata)

                    # Condition 3: extremely low variance (visually black)
                    low_variance = np.std(rgb) < 1

                    # Condition 4: too few nonzero pixels (coverage)
                    valid_pixels = np.sum(rgb > 0)
                    total_pixels = rgb.size
                    low_coverage = (valid_pixels / total_pixels) < 0.05

                    if all_zero or all_nodata or low_variance or low_coverage:
                        continue

                    # accept this tile
                    print(f"TILE {accepted + 1}/{N_TILES_PER_IMAGE}")
                    manual_count = show_tile_and_get_count(tile_array, prewar_array)
                    model_count = preds.within(tile_geom).sum()

                    # Centroid in raster CRS
                    centroid = tile_geom.centroid

                    # Convert centroid to lat/lon
                    centroid_gdf = gpd.GeoSeries(
                        [centroid],
                        crs=src.crs
                    ).to_crs("EPSG:4326")

                    lon = centroid_gdf.x.values[0]
                    lat = centroid_gdf.y.values[0]

                    rows.append({
                        "region": region,
                        "date": date,
                        "time": time,
                        "satellite": satellite,
                        "tif_name": tif_name,
                        "latitude": lat,
                        "longitude": lon,
                        "manual_tent_count": manual_count,
                        "model_tent_count": model_count
                    })

                    accepted += 1

                if accepted < N_TILES_PER_IMAGE:
                    print(f"Warning: only collected {accepted}/{N_TILES_PER_IMAGE} tiles for {tif_name} after {attempts} attempts.")

    df = pd.DataFrame(rows)

    if os.path.exists(OUTPUT_CSV):
        df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved results to {OUTPUT_CSV}")

    print(f"\nSaved results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()