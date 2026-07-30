"""Tile geometry derived from the configured pixel size.

Covers the two things `processing.pixel_metres` decides: how the tile margin
becomes the prediction crop band, and which rasters a scan stage will accept.
"""

import pytest
import rasterio
from rasterio.transform import from_origin

from _helpers import CRS_UTM, write_geotiff
from displacement_tracker.util.tile_builder import (
    DEFAULT_PIXEL_METRES,
    derive_selection_geometry,
    pixel_size_mismatch,
    resolve_pixel_metres,
    tile_pixel_size,
)

# The tile span in use: core 70 m + 2 x 15 m margin.
SPAN_M = 100.0


def _raster_at(tmp_path, pixel_size, name="tile.tif"):
    """A small single-band GeoTIFF whose pixels are `pixel_size` metres."""
    transform = from_origin(500000.0, 3500000.0, pixel_size, pixel_size)
    path = write_geotiff(tmp_path / name, [[1.0, 1.0], [1.0, 1.0]], transform, CRS_UTM)
    return rasterio.open(path)


# ---------------------------------------------------------------------------
# Resolving the configured pixel size
# ---------------------------------------------------------------------------


def test_pixel_metres_comes_from_the_processing_section():
    # Given: a processing section naming a half-metre ground sample distance
    processing = {"margin_metres": 15, "pixel_metres": 0.5}

    # When: the pixel size is resolved
    pixel_metres = resolve_pixel_metres(processing)

    # Then: the configured value is used
    assert pixel_metres == 0.5


def test_a_config_without_pixel_metres_keeps_the_historic_pixel_size():
    # Given: a legacy flat config, written before the key existed
    processing = {"margin_metres": 15}

    # When: the pixel size is resolved
    pixel_metres = resolve_pixel_metres(processing)

    # Then: it falls back to the 0.5 m the pipeline was hardcoded to, so an
    #       existing run directory re-executes with the same tile geometry
    assert pixel_metres == DEFAULT_PIXEL_METRES == 0.5


def test_a_non_positive_pixel_metres_is_rejected():
    # Given: a config setting the pixel size to zero
    processing = {"pixel_metres": 0}

    # When: the pixel size is resolved
    # Then: it refuses, naming the key the user has to fix rather than
    #       dividing by it downstream
    with pytest.raises(ValueError, match=r"processing\.pixel_metres"):
        resolve_pixel_metres(processing)


# ---------------------------------------------------------------------------
# Margin -> prediction crop band
# ---------------------------------------------------------------------------


def test_the_margin_becomes_the_crop_band_in_pixels():
    # Given: a 15 m margin on half-metre imagery
    # When: the selection geometry is derived
    crop_pixels, nms_sigma = derive_selection_geometry(15.0, 0.5)

    # Then: the band is 15 / 0.5 = 30 px wide, and the NMS sigma is
    #       three quarters of it: 0.75 * 30 = 22.5
    assert crop_pixels == 30
    assert nms_sigma == pytest.approx(22.5)


def test_a_coarser_pixel_size_shrinks_the_crop_band():
    # Given: the same 15 m margin, but on 1 m imagery
    # When: the selection geometry is derived
    crop_pixels, nms_sigma = derive_selection_geometry(15.0, 1.0)

    # Then: the same ground distance is half as many pixels — 15 / 1.0 = 15,
    #       and 0.75 * 15 = 11.25 — so the crop tracks the imagery rather
    #       than the 0.5 m the code once assumed
    assert crop_pixels == 15
    assert nms_sigma == pytest.approx(11.25)


# ---------------------------------------------------------------------------
# The scan-stage resolution guard
# ---------------------------------------------------------------------------


def test_a_raster_at_the_configured_resolution_is_accepted(tmp_path):
    # Given: a raster whose pixels are exactly the configured 0.5 m
    src = _raster_at(tmp_path, 0.5)

    # When: it is checked against the configured pixel size
    # Then: nothing is reported, so the scan proceeds
    assert pixel_size_mismatch(src, 0.5) is None


def test_a_raster_at_another_resolution_is_reported_with_both_sizes(tmp_path):
    # Given: a 1 m raster where the config expects 0.5 m
    src = _raster_at(tmp_path, 1.0)

    # When: it is checked against the configured pixel size
    message = pixel_size_mismatch(src, 0.5)

    # Then: the mismatch is reported, naming the raster's own resolution and
    #       the key that has to change to accept it
    assert message is not None
    assert "1 m" in message
    assert "processing.pixel_metres=0.5 m" in message


def test_the_guard_admits_exactly_the_rasters_that_keep_the_tile_pixel_size(tmp_path):
    # Given: three rasters off the configured 0.5 m by 0.1 %, 0.5 % and 1 %.
    #        At a 100 m span a tile is round(100 / pixel_size) px, so the
    #        reference tile is 200 px and only the first still rounds to it:
    #        100/0.5005 = 199.80 -> 200, but 100/0.5025 = 199.01 -> 199 and
    #        100/0.505 = 198.02 -> 198.
    within = _raster_at(tmp_path, 0.5005, "within.tif")
    half_percent = _raster_at(tmp_path, 0.5025, "half_percent.tif")
    one_percent = _raster_at(tmp_path, 0.505, "one_percent.tif")

    # When: each is checked against the configured pixel size
    # Then: the guard accepts precisely the raster whose tiles stay 200 px —
    #       a 1 % band would admit all three and let 198 px tiles into a
    #       batch of 200 px ones
    assert tile_pixel_size(within, SPAN_M) == 200
    assert pixel_size_mismatch(within, 0.5) is None

    assert tile_pixel_size(half_percent, SPAN_M) == 199
    assert pixel_size_mismatch(half_percent, 0.5) is not None

    assert tile_pixel_size(one_percent, SPAN_M) == 198
    assert pixel_size_mismatch(one_percent, 0.5) is not None
