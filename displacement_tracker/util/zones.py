"""Loading polygon zone files used to include or exclude predictions.

Three stages clip predictions against a zone file: the merge stage drops
points inside its exclusion zones and outside its inclusion zone, and the two
validation stages keep only the predictions inside their inclusion zones.
They all need the same thing — one unified geometry, in the CRS of whatever
they are about to clip — so they share this loader rather than each rolling
its own ``read_file(...).geometry.union_all()``.

The ``crs`` argument is what makes that sharing safe. A bare shapely geometry
carries no CRS, and ``GeoDataFrame.clip()`` only validates CRS when the mask
is itself a GeoDataFrame/GeoSeries, so an unreprojected mask clips silently in
whatever space the file happened to be stored in.
"""

from pathlib import Path

import click
import geopandas as gpd

from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("zones")


def load_zone_geometry(zones_path: str | None, label: str, crs: str):
    """Load and unify polygon geometries from a shapefile or GeoPackage file.

    Returns a single geometry in ``crs``, or None when no path is given or the
    file holds no features. ``label`` names the zones in log and error
    messages ("exclusion", "inclusion").

    ``crs`` is deliberately required rather than defaulted: a caller that
    clips in a projected CRS and forgets it would get a silent geographic
    geometry and a mask that selects nothing, which is the exact failure this
    parameter exists to prevent.
    """
    if not zones_path:
        return None

    path = Path(zones_path)
    if not path.exists():
        raise click.ClickException(f"{label} zones file not found: {path}")

    try:
        zones_gdf = gpd.read_file(path)
    except Exception as exc:
        raise click.ClickException(
            f"Failed to read {label} zones file {path}: {exc}"
        ) from exc

    if zones_gdf.empty:
        LOGGER.warning("%s zones file is empty: %s", label, path)
        return None

    if zones_gdf.crs is None:
        LOGGER.warning("%s zones CRS missing, assuming EPSG:4326: %s", label, path)
        zones_gdf = zones_gdf.set_crs("EPSG:4326")
    zones_gdf = zones_gdf.to_crs(crs)

    geom = zones_gdf.geometry.union_all()
    LOGGER.info("Loaded %s zones from %s", label, path)
    return geom
