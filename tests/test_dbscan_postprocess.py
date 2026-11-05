import unittest

import geopandas as gpd
from shapely.geometry import Point

from displacement_tracker.j_dbscan_postprocess import dbscan_retain_non_noise


class DbscanPostprocessTests(unittest.TestCase):
    def test_removes_isolated_point_and_preserves_attributes(self):
        points = gpd.GeoDataFrame(
            {"peak_value": [0.1, 0.2, 0.3, 0.4], "tile": ["a", "a", "b", "c"]},
            geometry=[
                Point(34.0, 31.0),
                Point(34.0001, 31.0),
                Point(34.0002, 31.0),
                Point(34.01, 31.0),
            ],
            crs="EPSG:4326",
        )

        retained = dbscan_retain_non_noise(points, eps_m=40, min_samples=3)

        self.assertEqual(list(retained["tile"]), ["a", "a", "b"])
        self.assertEqual(list(retained["peak_value"]), [0.1, 0.2, 0.3])
        self.assertEqual(retained.crs, points.crs)

    def test_empty_layer_is_accepted(self):
        points = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        retained = dbscan_retain_non_noise(points)
        self.assertTrue(retained.empty)
        self.assertEqual(retained.crs, points.crs)


if __name__ == "__main__":
    unittest.main()
