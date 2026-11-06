import unittest

from displacement_tracker.util.deduplication import merge_close_points_global
from displacement_tracker.util.thresholding import filter_points_by_adjusted_peak


class RepresentativeMergeTests(unittest.TestCase):
    def test_retains_original_highest_adjusted_peak_detection(self):
        points = [
            (31.500000, 34.450000, 0.95, 0.60, "image-a.tif"),
            (31.500010, 34.450010, 0.70, 0.91, "image-b.tif"),
        ]

        merged = merge_close_points_global(points, min_distance_m=3.0)

        self.assertEqual(merged, [points[1]])

    def test_exact_adjusted_peak_tie_is_deterministic(self):
        points = [
            (31.500010, 34.450010, 0.95, 0.91, "image-b.tif"),
            (31.500000, 34.450000, 0.70, 0.91, "image-a.tif"),
        ]

        merged = merge_close_points_global(points, min_distance_m=3.0)

        self.assertEqual(merged, [points[1]])

    def test_adjusted_peak_filter_keeps_source_provenance(self):
        points = [(31.5, 34.45, 0.8, 0.9, "image-a.tif")]

        kept = filter_points_by_adjusted_peak(points, 0.5)

        self.assertEqual(kept[0][4], "image-a.tif")


if __name__ == "__main__":
    unittest.main()
