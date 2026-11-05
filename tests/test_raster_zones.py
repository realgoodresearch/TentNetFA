import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from displacement_tracker.h_merge_geojsons import filter_points_by_spatial_zones
from displacement_tracker.util.raster_zones import open_raster_zone_matchers


class RasterZoneTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "zones.tif"
        with rasterio.open(
            self.path,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(34, 32, 1, 1),
            nodata=0,
        ) as dataset:
            dataset.write(np.array([[[1, 0], [0, 2]]], dtype="uint8"))

        self.points = [
            (31.5, 34.5, 1.0, 1.0),  # raster value 1
            (30.5, 35.5, 1.0, 1.0),  # raster value 2
            (29.5, 36.5, 1.0, 1.0),  # outside raster
        ]

    def tearDown(self):
        self.tempdir.cleanup()

    def test_exclusion_removes_only_matching_value(self):
        rules = [{"path": str(self.path), "mode": "exclude", "band": 1, "values": [1]}]
        with open_raster_zone_matchers(rules) as matchers:
            kept = filter_points_by_spatial_zones(self.points, None, None, matchers)
        self.assertEqual(kept, self.points[1:])

    def test_inclusion_requires_a_match(self):
        rules = [{"path": str(self.path), "mode": "include", "band": 1, "values": [2]}]
        with open_raster_zone_matchers(rules) as matchers:
            kept = filter_points_by_spatial_zones(self.points, None, None, matchers)
        self.assertEqual(kept, [self.points[1]])


if __name__ == "__main__":
    unittest.main()
