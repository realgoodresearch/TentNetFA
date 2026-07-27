"""Unit tests for displacement_tracker/util/scan_orchestrator.py.

Covers tif collection and its substring filter, the per-TIFF scan loop and
manifest writer lifecycle, and presence-based config-key validation.

This module is the per-TIFF orchestration shared by the tiling scanners
(b1_annotated_scanner, b2_image_scanner) — its only two consumers.
"""

import pandas as pd
import pytest

from displacement_tracker.util.manifest_writer import MANIFEST_COLUMNS
from displacement_tracker.util.scan_orchestrator import (
    collect_tif_files,
    require_keys,
    run_scans,
)


# ---------------------------------------------------------------------------
# scan_orchestrator.collect_tif_files
# ---------------------------------------------------------------------------


def test_collect_tif_files_returns_all_tifs_without_filter(tmp_path):
    # Given: a directory with two .tif files and one .txt file
    (tmp_path / "a.tif").touch()
    (tmp_path / "b.tif").touch()
    (tmp_path / "c.txt").touch()

    # When: collect_tif_files runs with no params, and with an empty filter
    unfiltered = collect_tif_files(str(tmp_path))
    empty_filter = collect_tif_files(str(tmp_path), {"loading": {"files": []}})

    # Then: exactly the two .tif files are returned either way
    expected = {str(tmp_path / "a.tif"), str(tmp_path / "b.tif")}
    assert set(unfiltered) == expected
    assert set(empty_filter) == expected


def test_collect_tif_files_substring_filter(tmp_path):
    # Given: scene_alpha.tif and scene_beta.tif with loading.files=["alpha",
    #        "gamma"]
    (tmp_path / "scene_alpha.tif").touch()
    (tmp_path / "scene_beta.tif").touch()
    params = {"loading": {"files": ["alpha", "gamma"]}}

    # When: collect_tif_files applies the filter
    matched = collect_tif_files(str(tmp_path), params)

    # Then: only files whose basename contains a search string survive; a
    #       search string matching nothing adds nothing
    assert matched == [str(tmp_path / "scene_alpha.tif")]


# ---------------------------------------------------------------------------
# scan_orchestrator.run_scans: manifest lifecycle on real Parquet files
# ---------------------------------------------------------------------------


def _manifest_row(raster_path: str) -> dict:
    """A complete manifest row describing one tile of ``raster_path``.

    The key set is checked against the writer's own MANIFEST_COLUMNS, so a
    column added or removed upstream fails here naming the difference in
    both directions.
    """
    row = {
        "tile_id": 1,
        "raster_path": raster_path,
        "prewar_path": "pre.tif",
        "labels_path": "labels.json",
        "r0": 0,
        "r1": 256,
        "c0": 0,
        "c1": 256,
        "lon_min": 34.0,
        "lon_max": 34.1,
        "lat_min": 31.0,
        "lat_max": 31.1,
        "origin_image": "scene",
        "origin_date": "2023-01-01",
        "valid_fraction": 1.0,
        "is_complete": True,
        "label_feature_ids": [1, 2],
    }
    assert set(row) == set(MANIFEST_COLUMNS), (
        "manifest row does not match MANIFEST_COLUMNS: "
        f"missing {sorted(set(MANIFEST_COLUMNS) - set(row))}, "
        f"unexpected {sorted(set(row) - set(MANIFEST_COLUMNS))}"
    )
    return row


def test_run_scans_writes_one_parquet_per_tif(tmp_path):
    # Given: two tif paths and a callback that writes one manifest row per tif
    manifest_folder = tmp_path / "manifests"
    tifs = [str(tmp_path / "scene_a.tif"), str(tmp_path / "scene_b.tif")]

    def scan_one(tif_path, writer):
        writer.add_row(_manifest_row(tif_path))

    # When: run_scans drives the loop
    run_scans(tifs, scan_one, manifest_folder=str(manifest_folder))

    # Then: one Parquet file per tif exists, named for that tif's stem, each
    #       holding exactly the row written for that tif
    for tif, expected_name in zip(tifs, ("scene_a.parquet", "scene_b.parquet")):
        out = manifest_folder / expected_name
        assert out.is_file()
        df = pd.read_parquet(out)
        assert len(df) == 1
        assert df.loc[0, "raster_path"] == tif


def test_run_scans_flushes_manifest_when_callback_raises(tmp_path):
    # Given: a callback that writes one row and then raises
    manifest_folder = tmp_path / "manifests"
    tif = str(tmp_path / "scene_x.tif")

    def scan_one(tif_path, writer):
        writer.add_row(_manifest_row(tif_path))
        raise RuntimeError("scan blew up")

    # When: run_scans processes the tif
    # Then: the exception propagates
    with pytest.raises(RuntimeError, match="scan blew up"):
        run_scans([tif], scan_one, manifest_folder=str(manifest_folder))

    # Then: the manifest is still closed and written with the row that made it
    #       in (the finally-block contract)
    out = manifest_folder / "scene_x.parquet"
    assert out.is_file()
    assert len(pd.read_parquet(out)) == 1


def test_run_scans_empty_input_is_a_noop(tmp_path):
    # Given: an empty tif list and a callback that must never be reached
    manifest_folder = tmp_path / "manifests"

    def scan_one(tif_path, writer):
        raise AssertionError("must not be called")

    # When: run_scans runs
    run_scans([], scan_one, manifest_folder=str(manifest_folder))

    # Then: it returns without raising and never creates the manifest folder
    assert not manifest_folder.exists()


# ---------------------------------------------------------------------------
# scan_orchestrator.require_keys
# ---------------------------------------------------------------------------


def test_require_keys_checks_presence_not_truthiness():
    # Given: a config whose keys are present but falsy (None / 0), and one
    #        with a key absent altogether
    falsy_present = {"a": None, "b": 0}
    key_absent = {"a": 1}

    # When: require_keys validates the config whose keys are all present
    # Then: it accepts them — a configured-but-empty value is not a missing
    #       one, so presence alone is the contract
    require_keys(falsy_present, ("a", "b"))

    # When: it validates the config with a key absent
    # Then: KeyError names the absent key
    with pytest.raises(KeyError, match="missing_key"):
        require_keys(key_absent, ("a", "missing_key"))
