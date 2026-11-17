import unittest

from displacement_tracker.pipelines.spec import PIPELINES


class PipelineSpecTests(unittest.TestCase):
    def test_every_ui_parameter_has_help_text(self):
        missing = [
            f"{pipeline.key}:{param.path}"
            for pipeline in PIPELINES.values()
            for param in pipeline.params
            if not param.help.strip()
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
