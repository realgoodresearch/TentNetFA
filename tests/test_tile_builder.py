"""Which rasters the scan stages will tile.

`pixel_size_mismatch` is the scan-stage guard: it admits a raster only when a
tile off it is the same pixel size as a tile off imagery at the configured
`processing.pixel_metres`, which is what makes tiles batchable downstream.
"""

import contextlib

import pytest
import rasterio
from _helpers import CRS_UTM, write_geotiff
from rasterio.transform import from_origin

from displacement_tracker.util.tile_builder import pixel_size_mismatch, tile_pixel_size

# The tile span in use: core 70 m + 2 x 15 m margin.
SPAN_M = 100.0


@contextlib.contextmanager
def raster_at(tmp_path, x_metres, y_metres=None, name="tile.tif"):
    """A small GeoTIFF whose pixels are `x_metres` x `y_metres` on the ground."""
    transform = from_origin(500000.0, 3500000.0, x_metres, y_metres or x_metres)
    path = write_geotiff(tmp_path / name, [[1.0, 1.0], [1.0, 1.0]], transform, CRS_UTM)
    with rasterio.open(path) as src:
        yield src


def test_a_raster_at_the_configured_resolution_is_accepted(tmp_path):
    # Given: a raster whose pixels are exactly the configured 0.5 m
    with raster_at(tmp_path, 0.5) as src:
        # When: it is checked against the configured pixel size
        # Then: nothing is reported, so the scan proceeds
        assert pixel_size_mismatch(src, 0.5, SPAN_M) is None


def test_a_raster_that_tiles_at_another_pixel_size_is_reported(tmp_path):
    # Given: a 1 m raster where the config expects 0.5 m
    with raster_at(tmp_path, 1.0) as src:
        # When: it is checked against the configured pixel size
        message = pixel_size_mismatch(src, 1.0 / 2, SPAN_M)

    # Then: the mismatch is reported in the terms that matter — a 100 m span
    #       tiles at 100 px here against the expected 200 px — and names the
    #       key the user can change to accept the raster
    assert message is not None
    assert "100x100 px" in message
    assert "200 px" in message
    assert "processing.pixel_metres=0.5 m" in message


def test_a_raster_with_non_square_pixels_is_rejected(tmp_path):
    # Given: a raster at the configured 0.5 m across but 1 m down. `world_window`
    #        sizes both axes from the x resolution, so this would tile 100 m x
    #        200 m of ground into one square 200 px window — geometrically
    #        stretched imagery, with nothing else in the pipeline to catch it.
    with raster_at(tmp_path, 0.5, 1.0) as src:
        # When: it is checked against the configured pixel size
        message = pixel_size_mismatch(src, 0.5, SPAN_M)

    # Then: it is rejected on the y axis alone — 200 px across, 100 px down
    assert message is not None
    assert "200x100 px" in message


def test_the_guard_admits_exactly_the_rasters_that_keep_the_tile_pixel_size(tmp_path):
    # Given: three rasters off the configured 0.5 m by 0.1 %, 0.5 % and 1 %.
    #        A tile is round(span_m / pixel_size) px, so the reference tile is
    #        200 px and only the first still rounds to it: 100/0.5005 = 199.80
    #        -> 200, but 100/0.5025 = 199.01 -> 199 and 100/0.505 = 198.02 -> 198.
    sizes = {0.5005: 200, 0.5025: 199, 0.505: 198}

    # When: each is checked against the configured pixel size
    # Then: the guard admits precisely the raster whose tiles stay 200 px, so
    #       a 198 px tile can never reach a batch of 200 px ones
    for i, (pixel_size, expected_px) in enumerate(sizes.items()):
        with raster_at(tmp_path, pixel_size, name=f"r{i}.tif") as src:
            assert tile_pixel_size(src, SPAN_M) == expected_px
            accepted = pixel_size_mismatch(src, 0.5, SPAN_M) is None
            assert accepted == (expected_px == 200)


def test_the_admitted_band_follows_the_span_rather_than_a_fixed_tolerance(tmp_path):
    # Given: a raster 0.2 % off the configured 0.5 m
    with raster_at(tmp_path, 0.501) as src:
        # When: it is checked at the 100 m span in use today, then at the
        #       130 m span the master-grid migration moves to
        # Then: the same raster is admitted at one span and rejected at the
        #       other — 100/0.501 = 199.60 -> 200 px, but 130/0.501 = 259.48
        #       -> 259 px against the expected 260. A fixed percentage band
        #       cannot express this; the tile size it protects is span-relative.
        assert pixel_size_mismatch(src, 0.5, 100.0) is None

        message = pixel_size_mismatch(src, 0.5, 130.0)
        assert message is not None
        assert "259x259 px" in message


def test_a_pixel_size_far_from_the_configured_one_is_rejected(tmp_path):
    # Given: a 5 m raster, coarse enough that a 100 m span is only 20 px
    with raster_at(tmp_path, 5.0) as src:
        # When: it is checked against the configured 0.5 m
        message = pixel_size_mismatch(src, 0.5, SPAN_M)

    # Then: it is rejected, reporting the 20 px it would actually tile at
    assert message is not None
    assert "20x20 px" in message


@pytest.mark.parametrize("pixel_metres", [0.5, 0.25])
def test_a_raster_matching_the_configured_size_is_accepted_at_any_resolution(
    tmp_path, pixel_metres
):
    # Given: imagery whose resolution is whatever the config says it is
    with raster_at(tmp_path, pixel_metres, name=f"{pixel_metres}.tif") as src:
        # When: it is checked against that same configured pixel size
        # Then: it is admitted — the guard pins agreement with the config, not
        #       a hardcoded half metre
        assert pixel_size_mismatch(src, pixel_metres, SPAN_M) is None
