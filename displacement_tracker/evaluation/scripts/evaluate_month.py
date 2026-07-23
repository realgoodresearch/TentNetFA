"""Evaluate model prediction error by year-month."""

import os

import pandas as pd

from displacement_tracker.evaluation.scripts.plots import plot_error_bars
from displacement_tracker.evaluation.scripts.common import (
    ensure_output_dir,
    group_error_summary,
    load_annotations,
)


def evaluate_error_by_month(
    annotation_csv: str,
    output_dir: str,
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
):
    """
    Evaluate model prediction error by year-month.

    Outputs:
        - error_by_month.csv
        - error_by_month_plot.png
    """
    ensure_output_dir(output_dir)

    df = load_annotations(
        annotation_csv, manual_column, model_column, extra_columns=("date",)
    )
    df["year_month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    results_df = group_error_summary(df, "year_month")
    if not results_df.empty:
        results_df = results_df.sort_values("year_month")
    results_df.to_csv(os.path.join(output_dir, "error_by_month.csv"), index=False)

    plot_error_bars(
        results_df,
        label_column="year_month",
        title="Prediction Error by Month (95% CI)",
        output_plot=os.path.join(output_dir, "error_by_month_plot.png"),
        figsize=(10, 5),
        rotate_labels=True,
    )

    return results_df
