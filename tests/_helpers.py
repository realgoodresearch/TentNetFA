"""Shared fixture builders and geo constants for the test suite.

Everything here is fixture machinery, not behavior under test: the on-disk
formats the pipelines read (GeoJSON point files, YAML configs, counts
GeoTIFFs, the annotation CSV), the CRSs those fixtures are expressed in,
and the spherical constants expectations are derived from. Keeping one
copy of each means a format that is really a domain contract — the
annotation-CSV header, above all — is written down once.

Test files import from here explicitly, e.g.::

    from _helpers import CRS_UTM, write_geojson

which resolves because pytest puts ``tests/`` on ``sys.path`` (the
directory deliberately has no ``__init__.py``). Helpers carry no
Given/When/Then labels — those belong to tests.
"""

import json
import math

import numpy as np
import rasterio
import yaml
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------

# UTM zone 36N: the projected CRS the imagery and master grids use.
CRS_UTM = "EPSG:32636"
CRS_WGS84 = "EPSG:4326"

_UTM_TO_LONLAT = Transformer.from_crs(CRS_UTM, CRS_WGS84, always_xy=True)


def utm_to_lonlat(x, y):
    """Convert an (easting, northing) in CRS_UTM to (lon, lat) in CRS_WGS84."""
    return _UTM_TO_LONLAT.transform(x, y)


# ---------------------------------------------------------------------------
# Spherical geometry
# ---------------------------------------------------------------------------

# The sphere the distance code works on. Along a meridian the haversine
# distance is exactly R * dphi, so an offset of d metres of latitude is
# d * DEG_PER_M degrees; one degree of arc is ~111194.93 m.
EARTH_RADIUS_M = 6371000.0
DEG_PER_M = 180.0 / (math.pi * EARTH_RADIUS_M)


# ---------------------------------------------------------------------------
# On-disk fixture builders
# ---------------------------------------------------------------------------

# The annotation CSV contract: one row per 100 m tile, centroid in WGS84.
ANNOTATION_COLUMNS = ("date", "latitude", "longitude", "manual_tent_count")


def annotation_header(*extra_columns):
    """The annotation-CSV header line, plus any extra model-count columns."""
    return ",".join((*ANNOTATION_COLUMNS, *extra_columns)) + "\n"


def write_annotation_csv(path, rows, extra_columns=()):
    """Write an annotation CSV from pre-formatted row strings.

    Each row is a complete ``date,latitude,longitude,count[,extra...]\\n``
    line, so a test can spell out exactly what it needs (an empty count, a
    trailing model column) while the header stays defined in one place.
    """
    path.write_text(annotation_header(*extra_columns) + "".join(rows))
    return str(path)


def write_geojson(path, feats):
    """Write (lon, lat, peak_value, adjusted_peak) tuples as a GeoJSON file."""
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"peak_value": peak, "adjusted_peak": adj},
        }
        for lon, lat, peak, adj in feats
    ]
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
    return str(path)


def write_yaml(path, data):
    """Dump a mapping to a YAML file."""
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def write_geotiff(path, data, transform, crs=CRS_UTM):
    """Write `data` as a float32 GeoTIFF.

    `data` is a 2-D array (one band) or a 3-D (band, row, col) stack;
    `crs=None` writes a raster with no CRS attached.
    """
    data = np.asarray(data, dtype="float32")
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
    return str(path)
