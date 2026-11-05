"""Point matching for categorical raster inclusion and exclusion zones."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import click
import rasterio
from pyproj import Transformer


@dataclass(frozen=True)
class RasterZoneRule:
    path: Path
    mode: str
    band: int
    values: tuple[float, ...]


@dataclass
class RasterZoneMatcher:
    rule: RasterZoneRule
    dataset: rasterio.io.DatasetReader
    transformer: Transformer

    def matches(self, lon: float, lat: float) -> bool:
        """True when this WGS84 point samples a configured, non-NoData value."""
        x, y = self.transformer.transform(lon, lat)
        sample = next(self.dataset.sample([(x, y)], indexes=self.rule.band, masked=True))
        if getattr(sample, "mask", False).any():
            return False
        return float(sample.item()) in self.rule.values


def parse_raster_zone_rules(raw_rules: object) -> list[RasterZoneRule]:
    """Validate YAML/UI raster-zone entries into normalized rules."""
    if not raw_rules:
        return []
    if not isinstance(raw_rules, list):
        raise click.ClickException("merge.raster_zones must be a list")

    rules = []
    for index, raw in enumerate(raw_rules, start=1):
        if not isinstance(raw, dict):
            raise click.ClickException(f"Raster zone {index} must be a mapping")
        path_value = raw.get("path")
        mode = str(raw.get("mode", "")).lower()
        band = raw.get("band", 1)
        values = raw.get("values")
        if not isinstance(path_value, str) or not path_value.strip():
            raise click.ClickException(f"Raster zone {index} requires a path")
        if mode not in {"include", "exclude"}:
            raise click.ClickException(
                f"Raster zone {index} mode must be 'include' or 'exclude'"
            )
        if not isinstance(band, int) or band < 1:
            raise click.ClickException(f"Raster zone {index} band must be a positive integer")
        if not isinstance(values, list) or not values:
            raise click.ClickException(f"Raster zone {index} requires non-empty values")
        try:
            numeric_values = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise click.ClickException(
                f"Raster zone {index} values must be numeric"
            ) from exc
        rules.append(
            RasterZoneRule(Path(path_value), mode, band, numeric_values)
        )
    return rules


@contextmanager
def open_raster_zone_matchers(raw_rules: object) -> Iterator[list[RasterZoneMatcher]]:
    """Open all configured raster zones for the duration of a merge operation."""
    rules = parse_raster_zone_rules(raw_rules)
    with ExitStack() as stack:
        matchers = []
        for rule in rules:
            if not rule.path.is_file():
                raise click.ClickException(f"Raster zone not found: {rule.path}")
            try:
                dataset = stack.enter_context(rasterio.open(rule.path))
            except Exception as exc:
                raise click.ClickException(
                    f"Could not open raster zone {rule.path}: {exc}"
                ) from exc
            if dataset.crs is None:
                raise click.ClickException(f"Raster zone has no CRS: {rule.path}")
            if rule.band > dataset.count:
                raise click.ClickException(
                    f"Raster zone band {rule.band} exceeds {rule.path}'s {dataset.count} band(s)"
                )
            matchers.append(
                RasterZoneMatcher(
                    rule=rule,
                    dataset=dataset,
                    transformer=Transformer.from_crs(
                        "EPSG:4326", dataset.crs, always_xy=True
                    ),
                )
            )
        yield matchers


def matches_any(matchers: list[RasterZoneMatcher], mode: str, lon: float, lat: float) -> bool:
    """Return whether a point matches any rule of ``mode``."""
    return any(
        matcher.rule.mode == mode and matcher.matches(lon, lat)
        for matcher in matchers
    )
