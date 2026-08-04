"""Unit tests for displacement_tracker/util/selection.py.

Covers the conversion from the tile margin in metres to the peak-selection
band in tile pixels.
"""

import pytest

from displacement_tracker.util.selection import selection_pixels


def test_the_margin_becomes_the_crop_band_in_pixels():
    # Given: a 15 m margin on half-metre imagery
    # When: the selection pixels are derived
    crop_pixels, nms_sigma = selection_pixels(15.0, 0.5)

    # Then: the band is 15 / 0.5 = 30 px wide, and the NMS sigma is three
    #       quarters of it: 0.75 * 30 = 22.5
    assert crop_pixels == 30
    assert nms_sigma == pytest.approx(22.5)


def test_a_coarser_pixel_size_shrinks_the_crop_band():
    # Given: the same 15 m margin, but on 1 m imagery
    # When: the selection pixels are derived
    crop_pixels, nms_sigma = selection_pixels(15.0, 1.0)

    # Then: the same ground distance is half as many pixels — 15 / 1.0 = 15,
    #       and 0.75 * 15 = 11.25 — so the crop tracks the imagery rather than
    #       the 0.5 m the prediction stage once assumed
    assert crop_pixels == 15
    assert nms_sigma == pytest.approx(11.25)
