"""Geometry of the peak-selection step that turns a prediction map into points.

The selection stage (`e_predict_json.extract_tile_nms` /
`extract_tile_centroids`) works in tile pixels, while the tile margin that
governs it is configured in metres. Converting between them lives here so the
arithmetic is reachable without importing torch.
"""

from __future__ import annotations

# NMS blur sigma as a fraction of the crop band, in pixels.
NMS_SIGMA_FRACTION = 0.75


def selection_pixels(margin_metres: float, pixel_metres: float) -> tuple[int, float]:
    """Return ``(crop_pixels, nms_sigma)`` in tile pixels for a tile margin.

    ``crop_pixels`` is the margin itself expressed in pixels: peaks inside
    that band are discarded before emission, so the regions tiles emit into
    cover the map exactly once, with no overlap between neighbours.
    """
    crop_pixels = round(margin_metres / pixel_metres)
    return crop_pixels, NMS_SIGMA_FRACTION * crop_pixels
