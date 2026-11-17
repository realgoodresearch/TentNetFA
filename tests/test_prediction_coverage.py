import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from shapely.geometry import Point

from displacement_tracker.util import prediction_coverage
from displacement_tracker.util.prediction_coverage import (
    coverage_record,
    filter_points_by_priority_coverage,
    save_prediction_coverage,
    write_merged_footprint,
)


class PredictionCoverageTests(unittest.TestCase):
    def test_clips_inference_coverage_to_alpha_footprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raster_path = tmp_path / "target.tif"
            data = np.full((4, 10, 10), 100, dtype=np.uint8)
            data[3, :, :5] = 0
            data[3, :, 5:] = 128
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=10,
                height=10,
                count=4,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_origin(0, 10, 1, 1),
            ) as dst:
                dst.write(data)
                dst.colorinterp = (
                    ColorInterp.red,
                    ColorInterp.green,
                    ColorInterp.blue,
                    ColorInterp.alpha,
                )

            with mock.patch.object(prediction_coverage, "MAX_FOOTPRINT_DIMENSION", 5):
                coarse_geometry = prediction_coverage._valid_raster_footprint(
                    raster_path
                )
            self.assertFalse(coarse_geometry.covers(Point(2, 5)))
            self.assertTrue(coarse_geometry.covers(Point(7, 5)))

            coverage_path = tmp_path / "coverage.gpkg"
            save_prediction_coverage(
                [
                    coverage_record(
                        {
                            "origin_image": "target.tif",
                            "origin_date": "20261014",
                            "raster_path": str(raster_path),
                            "lon_min": 0,
                            "lat_min": 0,
                            "lon_max": 10,
                            "lat_max": 10,
                        }
                    )
                ],
                coverage_path,
            )
            coverage = gpd.read_file(coverage_path, layer="inference_tiles")
            geometry = coverage.geometry.iloc[0]
            self.assertFalse(geometry.covers(Point(2, 5)))
            self.assertTrue(geometry.covers(Point(7, 5)))
            self.assertEqual(coverage.iloc[0].raster_path, str(raster_path))

            invalid_edge_point = (5, 2, 0.9, 0.9, "older.tif")
            valid_coverage_point = (5, 7, 0.9, 0.9, "older.tif")
            self.assertEqual(
                filter_points_by_priority_coverage(
                    [invalid_edge_point, valid_coverage_point],
                    [coverage_path],
                    ["target", "older"],
                ),
                [invalid_edge_point],
            )

    def test_clips_coverage_to_nodata_mask_without_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raster_path = tmp_path / "masked.tif"
            data = np.ones((3, 10, 10), dtype=np.uint8)
            data[:, :, :5] = 0
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=10,
                height=10,
                count=3,
                dtype="uint8",
                nodata=0,
                crs="EPSG:4326",
                transform=from_origin(0, 10, 1, 1),
            ) as dst:
                dst.write(data)

            coverage_path = tmp_path / "coverage.gpkg"
            with mock.patch(
                "displacement_tracker.util.prediction_coverage.MAX_FOOTPRINT_DIMENSION",
                5,
            ):
                self._write_raster_coverage(raster_path, coverage_path)
            geometry = gpd.read_file(
                coverage_path, layer="inference_tiles"
            ).geometry.iloc[0]
            self.assertFalse(geometry.covers(Point(2, 5)))
            self.assertTrue(geometry.covers(Point(7, 5)))

    def test_clips_coverage_to_nonzero_rgb_without_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raster_path = tmp_path / "rgb.tif"
            data = np.ones((3, 10, 10), dtype=np.uint8)
            data[:, :, :5] = 0
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=10,
                height=10,
                count=3,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_origin(0, 10, 1, 1),
            ) as dst:
                dst.write(data)

            coverage_path = tmp_path / "coverage.gpkg"
            with mock.patch(
                "displacement_tracker.util.prediction_coverage.MAX_FOOTPRINT_DIMENSION",
                5,
            ):
                self._write_raster_coverage(raster_path, coverage_path)
            geometry = gpd.read_file(
                coverage_path, layer="inference_tiles"
            ).geometry.iloc[0]
            self.assertFalse(geometry.covers(Point(2, 5)))
            self.assertTrue(geometry.covers(Point(7, 5)))

    @staticmethod
    def _write_raster_coverage(raster_path, coverage_path):
        save_prediction_coverage(
            [
                coverage_record(
                    {
                        "origin_image": raster_path.name,
                        "origin_date": "20261014",
                        "raster_path": str(raster_path),
                        "lon_min": 0,
                        "lat_min": 0,
                        "lon_max": 10,
                        "lat_max": 10,
                    }
                )
            ],
            coverage_path,
        )

    def test_removes_stale_coverage_when_current_run_has_no_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            coverage_path = Path(tmp) / "coverage.gpkg"
            save_prediction_coverage(
                [
                    coverage_record(
                        {
                            "origin_image": "target.tif",
                            "origin_date": "20261014",
                            "lon_min": 0,
                            "lat_min": 0,
                            "lon_max": 1,
                            "lat_max": 1,
                        }
                    )
                ],
                coverage_path,
            )
            self.assertTrue(coverage_path.exists())
            save_prediction_coverage([], coverage_path)
            self.assertFalse(coverage_path.exists())

    def test_writes_source_and_priority_footprint_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            coverage_path = tmp_path / "pred_coverage.gpkg"
            save_prediction_coverage(
                [
                    coverage_record(
                        {
                            "origin_image": "older.tif",
                            "origin_date": "20261010",
                            "lon_min": 34.0,
                            "lat_min": 31.0,
                            "lon_max": 34.1,
                            "lat_max": 31.1,
                        }
                    ),
                    coverage_record(
                        {
                            "origin_image": "target.tif",
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
            self.assertEqual(preferred.iloc[0].source_image, "target.tif")

            # Rerunning merge for the same output must replace both layers,
            # rather than append another copy of preferred coverage.
            self.assertTrue(
                write_merged_footprint(
                    [coverage_path],
                    footprint_path,
                    priorities=["target", "older"],
                )
            )
            by_source = gpd.read_file(footprint_path, layer="coverage_by_source")
            preferred = gpd.read_file(footprint_path, layer="preferred_coverage")
            self.assertEqual(len(by_source), 2)
            self.assertEqual(len(preferred), 2)

    def test_priority_coverage_keeps_only_best_covered_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            coverage_path = Path(tmp) / "coverage.gpkg"
            save_prediction_coverage(
                [
                    coverage_record(
                        {
                            "origin_image": "target.tif",
                            "origin_date": "20261014",
                            "lon_min": 34.0,
                            "lat_min": 31.0,
                            "lon_max": 34.1,
                            "lat_max": 31.1,
                        }
                    ),
                    coverage_record(
                        {
                            "origin_image": "older.tif",
                            "origin_date": "20261010",
                            "lon_min": 34.0,
                            "lat_min": 31.0,
                            "lon_max": 34.2,
                            "lat_max": 31.1,
                        }
                    ),
                ],
                coverage_path,
            )
            points = [
                (31.05, 34.05, 0.8, 0.8, "target.tif"),
                (31.05, 34.05, 0.9, 0.9, "older.tif"),
                (31.05, 34.15, 0.9, 0.9, "older.tif"),
                (31.05, 34.05, 0.9, 0.9, "unlisted"),
            ]
            self.assertEqual(
                filter_points_by_priority_coverage(
                    points, [coverage_path], ["target", "older"]
                ),
                [points[0], points[2]],
            )


if __name__ == "__main__":
    unittest.main()
