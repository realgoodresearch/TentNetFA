import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# CORE FUNCTION
# ==========================================================

def evaluate_error_by_month(
    annotation_csv: str,
    output_dir: str
):
    """
    Evaluate model prediction error by year-month.

    Outputs:
        - error_by_month.csv
        - error_by_month_plot.png
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
        "manual_tent_count",
        "model_tent_count"
    }

    if not required_cols.issubset(df.columns):
        raise ValueError("Annotation CSV missing required columns.")

    # Ensure date parsed correctly
    df["date"] = pd.to_datetime(df["date"])

    # Year-month grouping
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # Tile error
    df["tile_error"] = df["model_tent_count"] - df["manual_tent_count"]

    # ==========================
    # METRICS
    # ==========================

    results = []

    for ym, group in df.groupby("year_month"):

        errors = group["tile_error"].values
        n = len(errors)

        if n == 0:
            continue

        mean_error = np.mean(errors)

        if n > 1:
            std_error = np.std(errors, ddof=1)
            ci_margin = 1.96 * (std_error / np.sqrt(n))
        else:
            ci_margin = 0

        lower = mean_error - ci_margin
        upper = mean_error + ci_margin

        results.append({
            "year_month": ym,
            "mean_tile_error": mean_error,
            "ci_lower": lower,
            "ci_upper": upper,
            "num_tiles": n
        })

    results_df = pd.DataFrame(results)

    # Sort chronologically
    results_df = results_df.sort_values("year_month")

    results_df.to_csv(output_csv, index=False)

    # ==========================
    # BAR PLOT
    # ==========================

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

    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Mean Tile-Level Prediction Error")
    plt.title("Prediction Error by Month (95% CI)")

    plt.axhline(0, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    return results_df


# ==========================================================
# CLI ENTRYPOINT
# ==========================================================

if __name__ == "__main__":

    evaluate_error_by_month(
        annotation_csv="displacement_tracker/evaluation/manual_annotation_results.csv",
        output_dir="displacement_tracker/evaluation/results"
    )