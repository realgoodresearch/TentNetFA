import tempfile
import unittest
from pathlib import Path

import fiona
import geopandas as gpd

from displacement_tracker.util.prediction_coverage import (
    coverage_record,
    save_prediction_coverage,
    write_merged_footprint,
)


class PredictionCoverageTests(unittest.TestCase):
    def test_writes_source_and_priority_footprint_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            coverage_path = tmp_path / "pred_coverage.gpkg"
            save_prediction_coverage(
                [
                    coverage_record(
                        {
                            "origin_image": "older",
                            "origin_date": "20261010",
                            "lon_min": 34.0,
                            "lat_min": 31.0,
                            "lon_max": 34.1,
                            "lat_max": 31.1,
                        }
                    ),
                    coverage_record(
                        {
                            "origin_image": "target",
                            "origin_date": "20261014",
                            "lon_min": 34.05,
                            "lat_min": 31.0,
                            "lon_max": 34.15,
                            "lat_max": 31.1,
                        }
                    ),
                ],
                coverage_path,
            )
            footprint_path = tmp_path / "prediction_footprint.gpkg"
            self.assertTrue(
                write_merged_footprint(
                    [coverage_path],
                    footprint_path,
                    priorities=["target", "older"],
                )
            )
            self.assertEqual(
                set(fiona.listlayers(footprint_path)),
                {"coverage_by_source", "preferred_coverage"},
            )
            preferred = gpd.read_file(footprint_path, layer="preferred_coverage")
            self.assertEqual(preferred.iloc[0].source_image, "target")


if __name__ == "__main__":
    unittest.main()
