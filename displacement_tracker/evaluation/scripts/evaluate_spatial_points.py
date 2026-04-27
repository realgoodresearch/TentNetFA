import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ==========================
# CONFIGURATION
# ==========================

ANNOTATION_CSV = "displacement_tracker/evaluation/manual_annotation_results.csv"
OUTPUT_PATH = "displacement_tracker/evaluation/results/spatial_tile_error_hexbin.png"

os.makedirs("displacement_tracker/evaluation/results", exist_ok=True)

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(ANNOTATION_CSV)

required_cols = {
    "latitude",
    "longitude",
    "manual_tent_count",
    "model_tent_count"
}

if not required_cols.issubset(df.columns):
    raise ValueError("Annotation CSV missing required columns.")

# Tile-level error
df["tile_error"] = df["model_tent_count"] - df["manual_tent_count"]

lon = df["longitude"].values
lat = df["latitude"].values
error = df["tile_error"].values

# ==========================
# HEXBIN PLOT
# ==========================

plt.figure(figsize=(10, 10))

hb = plt.hexbin(
    lon,
    lat,
    C=error,
    reduce_C_function=np.mean,
    gridsize=60,
    cmap="RdBu_r"
)

# Get aggregated mean values per hex
# Get aggregated mean values per hex
hex_means = hb.get_array()

if len(hex_means) == 0:
    raise ValueError("No hex bins were created. Check data.")

vmin = np.min(hex_means)
vmax = np.max(hex_means)

# Round bounds to nearest 10
tick_min = int(np.floor(vmin / 10.0) * 10)
tick_max = int(np.ceil(vmax / 10.0) * 10)

# Zero-centered normalization using rounded bounds
norm = TwoSlopeNorm(vmin=tick_min, vcenter=0, vmax=tick_max)
hb.set_norm(norm)
hb.set_clim(tick_min, tick_max)

cbar = plt.colorbar(hb, label="Mean Tile-Level Prediction Error")

# Create ticks in increments of 10
ticks = np.arange(tick_min, tick_max + 10, 10)
cbar.set_ticks(ticks)
cbar.set_ticklabels([str(int(t)) for t in ticks])

plt.title("Local Mean Prediction Error (Hexbin Aggregation)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.savefig(OUTPUT_PATH)
plt.close()

print(f"Saved hexbin plot to {OUTPUT_PATH}")

### TODO
# Spatial bootstrap to local regional hex, get hex level CIs