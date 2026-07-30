"""Expose the manual tile annotations as validation reference data.

The evaluation package's ground truth — the manually annotated tiles in
``manual_eval/manual_annotation_results.csv`` (one row per 100 m tile with
its centroid lat/lon, acquisition date and ``manual_tent_count``) — is made
available to the validation/tuning flow: each annotated tile contributes its
count to the master-grid cell containing the tile centroid.

Two ways to consume it:

1. Materialized as a counts raster via the ``annotation-reference`` CLI,
   which resolves one date's annotations onto a master grid and writes a
   GeoTIFF consumable by the validation flow's built-in ``raster``
   reference type — useful for handing one date's counts to external
   tooling, or pinning them as a fixed input.

2. Directly, as reference type ``manual_eval``. Importing this module
   registers the type in ``util/reference_data.py``'s ``SOURCE_TYPES``, and
   ``build_reference_source`` imports this module on demand, so the type is
   reachable from config- and CLI-driven flows without anything importing
   the evaluation package first::

       reference:
         type: manual_eval
         path: displacement_tracker/evaluation/manual_eval/manual_annotation_results.csv
         date: 2024-10-14

   A ``.csv`` reference path infers this type, so ``type`` may be omitted.
   Either route resolves the same counts.

The CSV spans several acquisition dates, so ``date`` must pick one
(``YYYY-MM-DD`` or ``YYYYMMDD``) whenever more than one is present —
references are never inferred from prediction timestamps.

Caveat: the annotations are a sparse sample of tiles, not a census. Cells
without an annotated tile read as zero reference counts, so validation
metrics are only meaningful where the compared window is covered by
annotated tiles.
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg

from displacement_tracker.evaluation.scripts.common import as_points, read_annotations
from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.reference_data import (
    SOURCE_TYPES,
    ReferenceSource,
)

LOGGER = setup_logging("annotation_reference")


class ManualAnnotationReferenceSource(ReferenceSource):
    """Manually annotated tile counts as a reference source.

    Implements the ``ReferenceSource.counts_on_grid`` contract from
    ``util/reference_data.py``. ``path`` is a CSV with one row per annotated
    tile carrying the tile centroid (``latitude``/``longitude``, WGS84), an
    acquisition ``date`` and a count column (``manual_tent_count`` by
    default). Each tile's count lands in the master-grid cell containing its
    centroid, which resolves counts exactly when the master grid matches the
    100 m annotation tiling.
    """

    def __init__(
        self,
        path: str,
        date: str | None = None,
        count_column: str = "manual_tent_count",
        lat_column: str = "latitude",
        lon_column: str = "longitude",
        date_column: str = "date",
    ):
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")

        df = read_annotations(
            csv_path, (lat_column, lon_column, date_column, count_column)
        )
        df = _select_date(df, date, date_column, csv_path)
        df = df.dropna(subset=[count_column])

        self._count_column = count_column
        self._gdf = as_points(df, lat_column=lat_column, lon_column=lon_column)
        LOGGER.info(
            "Loaded %d annotated tiles (%s total: %.0f) from %s",
            len(self._gdf),
            count_column,
            float(self._gdf[count_column].sum()),
            csv_path,
        )

    def counts_on_grid(self, grid_shape, transform, crs, clip_geom=None):
        gdf = self._gdf
        if crs is not None:
            gdf = gdf.to_crs(crs)
        if clip_geom is not None:
            gdf = gdf.clip(clip_geom)
        if gdf.empty:
            return np.zeros(grid_shape, dtype=np.float32)

        shapes = (
            (geom, float(count))
            for geom, count in zip(gdf.geometry, gdf[self._count_column])
        )
        return features.rasterize(
            shapes=shapes,
            out_shape=grid_shape,
            transform=transform,
            merge_alg=MergeAlg.add,
            fill=0,
            dtype="float32",
        )


def _select_date(
    df: pd.DataFrame, date: str | None, date_column: str, csv_path: Path
) -> pd.DataFrame:
    """Restrict the annotations to one acquisition date, required when ambiguous."""
    available = sorted(df[date_column].dropna().unique())

    if date is None:
        if len(available) > 1:
            raise ValueError(
                f"{csv_path} contains annotations for {len(available)} dates; "
                f"set reference.date (YYYY-MM-DD) to pick one explicitly. "
                f"Available: {available}"
            )
        return df

    wanted = str(date).replace("-", "")
    selected = df[
        df[date_column].astype(str).str.replace("-", "", regex=False) == wanted
    ]
    if selected.empty:
        raise ValueError(
            f"No annotations dated {date} in {csv_path}. Available: {available}"
        )
    return selected


def register() -> None:
    """Register ``manual_eval`` in ``reference_data.SOURCE_TYPES``.

    The value is `(factory, options accepted besides path)` — the same tuple
    contract ``build_reference_source`` unpacks for every built-in type. The
    option set is every constructor keyword, so a CSV that does not use the
    default column names is still configurable.

    This is a function rather than a bare assignment at import so that
    ``reference_data._ensure_optional_types`` can *re-establish* the
    registration, not merely trigger it: importing an already-imported
    module runs no side effects, so an import alone cannot restore a
    registry something else has cleared. Calling it repeatedly is a no-op.
    """
    SOURCE_TYPES["manual_eval"] = (
        ManualAnnotationReferenceSource,
        frozenset({"date", "count_column", "lat_column", "lon_column", "date_column"}),
    )


register()


@click.command()
@click.option(
    "--annotation-csv",
    type=click.Path(exists=True, dir_okay=False),
    default="displacement_tracker/evaluation/manual_eval/manual_annotation_results.csv",
    show_default=True,
    help="Manual annotation CSV (one row per annotated tile).",
)
@click.option(
    "--date",
    default=None,
    help="Acquisition date to export (YYYY-MM-DD); required when the CSV "
    "contains more than one date.",
)
@click.option(
    "--master-grid",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Master-grid raster defining the shape, transform and CRS of the output.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    required=True,
    help="Output GeoTIFF of reference counts on the master grid, consumable "
    "by the 'raster' reference type.",
)
@click.option(
    "--count-column",
    default="manual_tent_count",
    show_default=True,
    help="Annotation column holding the per-tile reference counts.",
)
def cli(annotation_csv, date, master_grid, output, count_column):
    """Resolve manual tile annotations onto a master grid as a counts raster."""
    source = ManualAnnotationReferenceSource(
        annotation_csv, date=date, count_column=count_column
    )

    with rasterio.open(master_grid) as src:
        counts = source.counts_on_grid((src.height, src.width), src.transform, src.crs)
        profile = src.profile.copy()

    profile.update(driver="GTiff", count=1, dtype="float32", nodata=0.0)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(counts, 1)

    LOGGER.info(
        "Wrote reference raster %s (%d cells with annotations, total count %.0f)",
        out_path,
        int((counts > 0).sum()),
        float(counts.sum()),
    )


if __name__ == "__main__":
    cli()
