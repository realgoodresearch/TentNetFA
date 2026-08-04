"""Unit tests for displacement_tracker/util/zones.py.

Covers the zone-file loader shared by the merge and validation stages: the
error and empty-file paths, and the CRS handling that decides whether a clip
against the returned geometry is geometrically meaningful.
"""

import click
import geopandas as gpd
import pytest
from shapely.geometry import Point

from displacement_tracker.util.zones import load_zone_geometry

# A UTM zone (36N) covering Gaza, and the geographic CRS zone files most
# commonly arrive in. The two disagree by hundreds of thousands of units, so
# a mask in the wrong one selects nothing rather than selecting sloppily.
GRID_CRS = "EPSG:32636"
FILE_CRS = "EPSG:4326"


def _write_zone_around(path, points, crs, buffer_deg=0.01):
    """Write a zone file in `crs` covering `points` (given in GRID_CRS)."""
    covered = gpd.GeoSeries(points, crs=GRID_CRS).to_crs(crs)
    gdf = gpd.GeoDataFrame(geometry=[covered.union_all().buffer(buffer_deg)], crs=crs)
    gdf.to_file(path)
    return gdf


def test_no_path_returns_none():
    # Given: no zones configured -- the shipped default for every zone key
    # When: the loader runs on None and on the empty string
    # Then: both mean "no clipping", not "clip to nothing"
    assert load_zone_geometry(None, "inclusion", GRID_CRS) is None
    assert load_zone_geometry("", "inclusion", GRID_CRS) is None


def test_missing_file_raises_naming_the_label(tmp_path):
    # Given: a zones path that does not exist, labelled "exclusion"
    missing = str(tmp_path / "not_here.gpkg")

    # When: the loader tries to load it
    # Then: a ClickException names the missing exclusion zones file
    with pytest.raises(click.ClickException, match="exclusion zones file not found"):
        load_zone_geometry(missing, "exclusion", GRID_CRS)


def test_empty_file_returns_none(tmp_path):
    # Given: a readable zone file holding no features
    path = tmp_path / "empty.gpkg"
    gpd.GeoDataFrame(geometry=[], crs=FILE_CRS).to_file(path)

    # When: the loader runs
    # Then: it degrades to "no clipping" rather than to an empty mask, which
    #       would silently discard every prediction
    assert load_zone_geometry(str(path), "inclusion", GRID_CRS) is None


def test_geometry_is_reprojected_to_the_requested_crs(tmp_path):
    # Given: two points on the master grid, and a zone file covering both but
    #        stored in EPSG:4326 as zone files usually are
    points = [Point(636000, 3475000), Point(636500, 3475500)]
    path = tmp_path / "zones.gpkg"
    _write_zone_around(path, points, FILE_CRS)
    preds = gpd.GeoDataFrame(geometry=points, crs=GRID_CRS)

    # When: the zones are loaded for use against grid-CRS predictions
    geom = load_zone_geometry(str(path), "inclusion", GRID_CRS)

    # Then: clipping keeps both points. The mask is a bare shapely geometry,
    #       which .clip() cannot CRS-check, so an unreprojected mask would
    #       silently keep zero -- the whole reason crs is a parameter.
    assert len(preds.clip(geom)) == 2


def test_reprojects_a_projected_file_to_wgs84(tmp_path):
    # Given: a zone file stored in the projected grid CRS this time
    points = [Point(636000, 3475000), Point(636500, 3475500)]
    path = tmp_path / "zones_utm.gpkg"
    _write_zone_around(path, points, GRID_CRS, buffer_deg=1000)

    # When: EPSG:4326 is requested, as the merge stage does -- merged
    #       prediction points are in lon/lat
    geom = load_zone_geometry(str(path), "exclusion", FILE_CRS)

    # Then: the geometry comes back in EPSG:4326. Gaza sits near 34.4E/31.4N,
    #       so degrees and UTM metres are unmistakable from the bounds.
    minx, miny, maxx, maxy = geom.bounds
    assert 34 < minx < 35 and 34 < maxx < 35
    assert 31 < miny < 32 and 31 < maxy < 32


def test_crs_less_file_is_assumed_wgs84(tmp_path):
    # Given: a shapefile holding lon/lat coordinates whose .prj sidecar is
    #        missing, so it genuinely reads back with crs=None. (A GPKG
    #        cannot stand in here -- it records the CRS internally, so the
    #        assumption branch would never be reached and this test could
    #        not fail.)
    path = tmp_path / "no_crs.shp"
    gpd.GeoDataFrame(geometry=[Point(34.45, 31.41).buffer(0.01)], crs=FILE_CRS).to_file(
        path
    )
    (tmp_path / "no_crs.prj").unlink()
    assert gpd.read_file(path).crs is None

    # When: the loader is asked for it on the master grid's CRS
    geom = load_zone_geometry(str(path), "inclusion", GRID_CRS)

    # Then: the coordinates are assumed to be WGS84 and reprojected, landing
    #       in the hundreds of thousands of metres rather than staying near 34
    minx, _, _, _ = geom.bounds
    assert minx > 100_000
