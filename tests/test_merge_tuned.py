"""Unit tests for displacement_tracker/h2_merge_tuned.py (tuned merge stage)."""

import click
import geopandas as gpd
import pytest
from click.testing import CliRunner

from _helpers import write_geojson, write_yaml
from displacement_tracker.h2_merge_tuned import cli, load_best_params

# ---------------------------------------------------------------------------
# load_best_params
# ---------------------------------------------------------------------------


def test_load_best_params_missing_file_raises(tmp_path):
    # Given: a best_params path that was never written
    missing = str(tmp_path / "best_params.yaml")

    # When: load_best_params tries to load it
    # Then: a ClickException points at the scan stage that should have
    #       produced the file
    with pytest.raises(click.ClickException, match="scan-validation"):
        load_best_params(missing)


def test_load_best_params_missing_one_key_raises(tmp_path):
    # Given: a best_params file holding only min_adj_peak
    path = write_yaml(tmp_path / "best_params.yaml", {"min_adj_peak": 0.3})

    # When: load_best_params validates it
    # Then: a ClickException names the missing adjustment_factor key
    with pytest.raises(click.ClickException, match="adjustment_factor"):
        load_best_params(path)


def test_load_best_params_empty_file_lists_both_keys(tmp_path):
    # Given: a best_params file that is empty (YAML loads as None)
    path = tmp_path / "best_params.yaml"
    path.write_text("")

    # When: load_best_params validates it
    # Then: a ClickException reports both required keys as missing
    with pytest.raises(click.ClickException, match=r"min_adj_peak.*adjustment_factor"):
        load_best_params(str(path))


def test_load_best_params_valid_returns_full_mapping(tmp_path):
    # Given: a best_params file with both required keys plus scan metadata
    path = write_yaml(
        tmp_path / "best_params.yaml",
        {
            "min_adj_peak": 0.3,
            "adjustment_factor": 0.8,
            "metric": "f1",
            "value": 0.91,
        },
    )

    # When: load_best_params reads it
    best = load_best_params(path)

    # Then: the complete mapping comes back including the extras
    assert best["min_adj_peak"] == pytest.approx(0.3)
    assert best["adjustment_factor"] == pytest.approx(0.8)
    assert best["metric"] == "f1"


# ---------------------------------------------------------------------------
# cli: config resolution and tuned-threshold precedence (real files, no mocks)
# ---------------------------------------------------------------------------


def make_preds(tmp_path):
    """Two far-apart points: A(peak .5, adj .8) and B(peak .5, adj .2).

    With the tuned pair (min_adj_peak=0.5, adjustment_factor=1.0) only A
    survives; with the config's own thresholds (0.99 / factor 0) or the
    per-file thresholds file (default 0.99) BOTH would be dropped.
    """
    preds = tmp_path / "preds"
    preds.mkdir()
    write_geojson(
        preds / "pred.geojson",
        [(0.0, 0.0, 0.5, 0.8), (0.01, 0.01, 0.5, 0.2)],
    )
    return preds


def test_cli_tuned_params_override_config_and_thresholds_config(tmp_path):
    # Given: predictions where only the tuned pair (0.5 / 1.0) keeps point A,
    #        a merge config whose own min_adj_peak=0.99, adjustment_factor=0
    #        and per-file thresholds_config (default 0.99) would drop
    #        everything, and best_params.yaml discovered via the
    #        tuning.out_dir fallback (tuning.best_params unset)
    preds = make_preds(tmp_path)
    out_dir = tmp_path / "tuning"
    out_dir.mkdir()
    write_yaml(
        out_dir / "best_params.yaml",
        {"min_adj_peak": 0.5, "adjustment_factor": 1.0},
    )
    kill_all = write_yaml(tmp_path / "thresholds.yaml", {"default": 0.99})
    final = tmp_path / "merged_tuned.gpkg"
    config = write_yaml(
        tmp_path / "config.yaml",
        {
            "merge": {
                "input_folder": str(preds),
                "min_distance_m": 3.0,
                "agreement": 1,
                "min_adj_peak": 0.99,
                "adjustment_factor": 0.0,
                "thresholds_config": kill_all,
            },
            "tuning": {"final_output": str(final), "out_dir": str(out_dir)},
        },
    )

    # When: the merge-tuned CLI runs on that config
    result = CliRunner().invoke(cli, [config])

    # Then: it exits 0 and the final output holds exactly point A at the
    #       origin -- proving the tuned pair replaced the config thresholds
    #       and thresholds_config was deliberately not forwarded
    assert result.exit_code == 0, result.output
    gdf = gpd.read_file(final)
    assert len(gdf) == 1
    assert gdf.iloc[0].geometry.x == pytest.approx(0.0)
    assert gdf.iloc[0].geometry.y == pytest.approx(0.0)
    assert gdf.iloc[0]["adjusted_peak"] == pytest.approx(0.8)


def test_cli_explicit_best_params_beats_out_dir_fallback(tmp_path):
    # Given: an out_dir whose best_params.yaml (min_adj_peak 0.99) would
    #        drop every point, and an explicit tuning.best_params file with
    #        min_adj_peak 0.5 that keeps point A
    preds = make_preds(tmp_path)
    out_dir = tmp_path / "tuning"
    out_dir.mkdir()
    write_yaml(
        out_dir / "best_params.yaml",
        {"min_adj_peak": 0.99, "adjustment_factor": 1.0},
    )
    explicit = write_yaml(
        tmp_path / "chosen_params.yaml",
        {"min_adj_peak": 0.5, "adjustment_factor": 1.0},
    )
    final = tmp_path / "merged_tuned.gpkg"
    config = write_yaml(
        tmp_path / "config.yaml",
        {
            "merge": {"input_folder": str(preds)},
            "tuning": {
                "final_output": str(final),
                "out_dir": str(out_dir),
                "best_params": explicit,
            },
        },
    )

    # When: the merge-tuned CLI runs
    result = CliRunner().invoke(cli, [config])

    # Then: the explicit path wins over the out_dir fallback and exactly
    #       point A (origin, adj 0.8 >= 0.5) is written -- had the fallback
    #       0.99 been used, zero points would survive
    assert result.exit_code == 0, result.output
    gdf = gpd.read_file(final)
    assert len(gdf) == 1
    assert gdf.iloc[0].geometry.x == pytest.approx(0.0)
    assert gdf.iloc[0].geometry.y == pytest.approx(0.0)
    assert gdf.iloc[0]["adjusted_peak"] == pytest.approx(0.8)


def test_cli_missing_best_params_and_out_dir_errors(tmp_path):
    # Given: a config with input folder and final_output but neither
    #        tuning.best_params nor tuning.out_dir
    preds = make_preds(tmp_path)
    config = write_yaml(
        tmp_path / "config.yaml",
        {
            "merge": {"input_folder": str(preds)},
            "tuning": {"final_output": str(tmp_path / "final.gpkg")},
        },
    )

    # When: the merge-tuned CLI runs
    result = CliRunner().invoke(cli, [config])

    # Then: it fails and names the missing tuning.best_params key
    assert result.exit_code != 0
    assert "tuning.best_params" in result.output


def test_cli_missing_input_folder_errors(tmp_path):
    # Given: a config whose merge section has no input_folder (and no
    #        prediction fallback is honoured by this stage)
    config = write_yaml(
        tmp_path / "config.yaml",
        {
            "merge": {},
            "tuning": {"final_output": str(tmp_path / "final.gpkg")},
        },
    )

    # When: the merge-tuned CLI runs
    result = CliRunner().invoke(cli, [config])

    # Then: it fails and names merge.input_folder as the missing key
    assert result.exit_code != 0
    assert "merge.input_folder" in result.output
