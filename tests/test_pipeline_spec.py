import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from displacement_tracker.pipelines.runner import prepare_run
from displacement_tracker.pipelines.spec import PIPELINES
from displacement_tracker.i_zonal_point_sums import configured_points_path


class PipelineSpecTests(unittest.TestCase):
    def test_zonal_point_source_selects_merged_or_dbscan_output(self):
        params = {
            "merge": {"output": "/run/merged/tents.gpkg"},
            "postprocess": {"output": "/run/postprocessing/tents_dbscan.gpkg"},
        }
        self.assertEqual(
            configured_points_path(params, {"point_source": "merged"}),
            "/run/merged/tents.gpkg",
        )
        self.assertEqual(
            configured_points_path(params, {"point_source": "dbscan"}),
            "/run/postprocessing/tents_dbscan.gpkg",
        )
        self.assertEqual(
            configured_points_path(
                params,
                {"point_source": "dbscan", "points_gpkg": "/older/run.gpkg"},
            ),
            "/older/run.gpkg",
        )

    def test_every_ui_parameter_has_help_text(self):
        missing = [
            f"{pipeline.key}:{param.path}"
            for pipeline in PIPELINES.values()
            for param in pipeline.params
            if not param.help.strip()
        ]
        self.assertEqual(missing, [])

    def test_prediction_pipeline_prepares_zonal_stage_paths(self):
        pipeline = PIPELINES["predict"]
        self.assertIn("zonal", [stage.key for stage in pipeline.stages])

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump({"shared": {}, "predict": {"merge": {"output_filename": "tents.gpkg"}}}),
                encoding="utf-8",
            )
            ctx = prepare_run(
                pipeline,
                config_path,
                run_name="test-run",
                run_root=root / "runs",
            )

            self.assertEqual(ctx.config["zonal"]["point_source"], "merged")
            self.assertIsNone(ctx.config["zonal"]["points_gpkg"])
            self.assertEqual(
                Path(ctx.config["merge"]["output"]),
                ctx.run_dir / "merged" / "tents.gpkg",
            )
            self.assertEqual(
                Path(ctx.config["zonal"]["output_dir"]),
                ctx.run_dir / "zonal_summaries",
            )
            self.assertTrue((ctx.run_dir / "zonal_summaries").is_dir())


if __name__ == "__main__":
    unittest.main()
