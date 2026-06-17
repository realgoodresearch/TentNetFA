#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# CORE FUNCTION
# ==========================================================

def evaluate_error_by_month(
    annotation_csv: str = "manual_annotation_results_with_new_model.csv",
    output_dir: str = "results",
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
):
    """
    Evaluate model prediction error by year-month.

    Outputs:
        - error_by_month.csv
        - error_by_month_plot.png

    Returns:
        results_df
    """
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "error_by_month.csv")
    output_plot = os.path.join(output_dir, "error_by_month_plot.png")

    # ==========================
    # LOAD DATA
    # ==========================

    df = pd.read_csv(annotation_csv)

    required_cols = {
        "date",
        manual_column,
        model_column,
    }

    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Annotation CSV missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    df["tile_error"] = df[model_column] - df[manual_column]
    
    # ==========================
    # METRICS
    # ==========================

    results = []

    for ym, group in df.groupby("year_month"):
        errors = group["tile_error"].dropna().values
        n = len(errors)

        if n == 0:
            continue

        mean_error = float(np.mean(errors))

        if n > 1:
            std_error = float(np.std(errors, ddof=1))
            ci_margin = 1.96 * (std_error / np.sqrt(n))
        else:
            ci_margin = 0.0

        lower = mean_error - ci_margin
        upper = mean_error + ci_margin

        results.append({
            "year_month": ym,
            "mean_tile_error": mean_error,
            "ci_lower": lower,
            "ci_upper": upper,
            "num_tiles": n,
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values("year_month")
    results_df.to_csv(output_csv, index=False)

    # ==========================
    # BAR PLOT
    # ==========================

    if not results_df.empty:
        plt.figure(figsize=(10, 5))

        means = results_df["mean_tile_error"].values
        ci_lower = results_df["ci_lower"].values
        ci_upper = results_df["ci_upper"].values

        lower_err = np.maximum(0, means - ci_lower)
        upper_err = np.maximum(0, ci_upper - means)
        yerr = np.vstack((lower_err, upper_err))

        x = np.arange(len(results_df))

        plt.bar(x, means, yerr=yerr, capsize=5)

        labels = [
            f"{ym} (n={n})"
            for ym, n in zip(results_df["year_month"], results_df["num_tiles"])
        ]

        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Mean Tile-Level Prediction Error (%)")
        plt.title("Prediction Error by Month (95% CI)")
        plt.axhline(0, linestyle="--")
        plt.tight_layout()
        plt.savefig(output_plot)
        plt.close()
    else:
        print("WARNING: No monthly groups found for plot.")

    return results_df


# ==========================================================
# CLI ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    evaluate_error_by_month(
        annotation_csv="manual_annotation_results_with_new_model.csv",
        output_dir="results",
        manual_column="manual_tent_count",
        model_column="model_tent_count",
    )
