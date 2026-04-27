import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ==========================
# CONFIGURATION
# ==========================

ANNOTATION_CSV = "displacement_tracker/evaluation/manual_annotation_results.csv"

OUTPUT_LINEAR = "displacement_tracker/evaluation/results/tile_prediction_correlation_linear.png"
OUTPUT_LOG = "displacement_tracker/evaluation/results/tile_prediction_correlation_log.png"
OUTPUT_LINEAR_NONZERO = "displacement_tracker/evaluation/results/tile_prediction_correlation_linear_nonzero.png"
OUTPUT_LOG_NONZERO = "displacement_tracker/evaluation/results/tile_prediction_correlation_log_nonzero.png"

os.makedirs("displacement_tracker/evaluation/results", exist_ok=True)

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(ANNOTATION_CSV)

required_cols = {"manual_tent_count", "model_tent_count"}

if not required_cols.issubset(df.columns):
    raise ValueError("Annotation CSV missing required columns.")

x = df["manual_tent_count"].values
y = df["model_tent_count"].values

# ==========================
# LINEAR CORRELATION (ALL)
# ==========================

r_linear, p_linear = pearsonr(x, y)

plt.figure(figsize=(8, 8))
plt.scatter(x, y, alpha=0.6)

max_val = max(np.max(x), np.max(y))
plt.plot([0, max_val], [0, max_val])

plt.xlabel("Manual Tent Count")
plt.ylabel("Model Tent Count")
plt.title(f"Tile-Level Prediction Correlation (Linear)\nPearson r = {r_linear:.3f}")

plt.tight_layout()
plt.savefig(OUTPUT_LINEAR)
plt.close()

# ==========================
# LOG CORRELATION (ALL)
# ==========================

x_log = np.log1p(x)
y_log = np.log1p(y)

r_log, p_log = pearsonr(x_log, y_log)

plt.figure(figsize=(8, 8))
plt.scatter(x_log, y_log, alpha=0.6)

max_log = max(np.max(x_log), np.max(y_log))
plt.plot([0, max_log], [0, max_log])

plt.xlabel("log(1 + Manual Tent Count)")
plt.ylabel("log(1 + Model Tent Count)")
plt.title(f"Tile-Level Prediction Correlation (Log Scale)\nPearson r = {r_log:.3f}")

plt.tight_layout()
plt.savefig(OUTPUT_LOG)
plt.close()

# ==========================
# FILTER: MANUAL COUNT > 0
# ==========================

df_nonzero = df[df["manual_tent_count"] > 0]

x_nz = df_nonzero["manual_tent_count"].values
y_nz = df_nonzero["model_tent_count"].values

# ==========================
# LINEAR CORRELATION (NONZERO)
# ==========================

r_linear_nz, p_linear_nz = pearsonr(x_nz, y_nz)

plt.figure(figsize=(8, 8))
plt.scatter(x_nz, y_nz, alpha=0.6)

max_val_nz = max(np.max(x_nz), np.max(y_nz))
plt.plot([0, max_val_nz], [0, max_val_nz])

plt.xlabel("Manual Tent Count")
plt.ylabel("Model Tent Count")
plt.title(f"Tile-Level Prediction Correlation (Linear, Manual > 0)\nPearson r = {r_linear_nz:.3f}")

plt.tight_layout()
plt.savefig(OUTPUT_LINEAR_NONZERO)
plt.close()

# ==========================
# LOG CORRELATION (NONZERO)
# ==========================

x_log_nz = np.log1p(x_nz)
y_log_nz = np.log1p(y_nz)

r_log_nz, p_log_nz = pearsonr(x_log_nz, y_log_nz)

plt.figure(figsize=(8, 8))
plt.scatter(x_log_nz, y_log_nz, alpha=0.6)

max_log_nz = max(np.max(x_log_nz), np.max(y_log_nz))
plt.plot([0, max_log_nz], [0, max_log_nz])

plt.xlabel("log(1 + Manual Tent Count)")
plt.ylabel("log(1 + Model Tent Count)")
plt.title(f"Tile-Level Prediction Correlation (Log, Manual > 0)\nPearson r = {r_log_nz:.3f}")

plt.tight_layout()
plt.savefig(OUTPUT_LOG_NONZERO)
plt.close()

# ==========================
# PRINT RESULTS
# ==========================

print("Saved plots:")
print(" - Linear (all):", OUTPUT_LINEAR)
print(" - Log (all):", OUTPUT_LOG)
print(" - Linear (manual > 0):", OUTPUT_LINEAR_NONZERO)
print(" - Log (manual > 0):", OUTPUT_LOG_NONZERO)

print("\nCorrelations:")
print(f"Linear (all)        r={r_linear:.4f}, p={p_linear:.4g}")
print(f"Log (all)           r={r_log:.4f}, p={p_log:.4g}")
print(f"Linear (manual>0)   r={r_linear_nz:.4f}, p={p_linear_nz:.4g}")
print(f"Log (manual>0)      r={r_log_nz:.4f}, p={p_log_nz:.4g}")