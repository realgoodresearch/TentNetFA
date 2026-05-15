"""Train an unsupervised row embedding via SimCLR over tabular features.

Per-row feature vector (21 dims):
  - date    (3): sin/cos of day-of-year, normalized year
  - time    (2): sin/cos of time-of-day, parsed from origin_image
  - spatial (2): bbox center normalized over the Gaza extent
  - ids     (2): ssc and u000 numbers parsed from origin_image, scalar-normalized
  - feature stats  (6): per-channel mean+std of the standardized feature tile
  - prewar  stats  (6): per-channel mean+std of the standardized prewar tile

Stats are taken from PairedImageDataset's outputs, i.e. post per-TIFF
standardization — the same signal a downstream model would consume.
"""
from __future__ import annotations

import datetime
import json
import math
import re
from pathlib import Path
from typing import Any

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from displacement_tracker.paired_image_dataset import PairedImageDataset
from displacement_tracker.util.env_loader import load_yaml_with_env
from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("metadata-embedding")

FEATURE_DIM = 21

_ORIGIN_NAME_PATTERN = re.compile(
    r"^(?P<place>.+)_"
    r"(?P<date>\d{8})_"
    r"(?P<time>\d{4,6})_"
    r"ssc(?P<ssc>\d+)_"
    r"u(?P<u000>\d+)_"
    r"visual"
    r"(?:_[a-z_]+)?"
    r"\.tif$"
)


def parse_origin_image_name(name: str) -> dict[str, Any]:
    """Parse `place_date_time_sscX_u000Y_visual(_clip)?(_file_format)?.tif`."""
    basename = Path(name).name
    m = _ORIGIN_NAME_PATTERN.match(basename)
    if m is None:
        raise ValueError(f"Could not parse origin_image filename: {name!r}")
    return {
        "place": m.group("place"),
        "date": m.group("date"),
        "time": m.group("time"),
        "ssc": int(m.group("ssc")),
        "u000": int(m.group("u000")),
    }


def _date_features(yyyymmdd: str) -> tuple[float, float, float]:
    dt = datetime.datetime.strptime(yyyymmdd, "%Y%m%d").date()
    doy = dt.timetuple().tm_yday
    return (
        math.sin(2 * math.pi * doy / 365.0),
        math.cos(2 * math.pi * doy / 365.0),
        (dt.year - 2020) / 10.0,
    )


def _time_features(hhmmss: str) -> tuple[float, float]:
    if len(hhmmss) == 4:
        h, m, s = int(hhmmss[:2]), int(hhmmss[2:4]), 0
    elif len(hhmmss) == 6:
        h, m, s = int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:6])
    else:
        raise ValueError(f"Unexpected time format in origin_image: {hhmmss!r}")
    frac = (h + m / 60.0 + s / 3600.0) / 24.0
    return math.sin(2 * math.pi * frac), math.cos(2 * math.pi * frac)


def _spatial_features(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    gaza_bbox: list[float],
) -> tuple[float, float]:
    lon_lo, lon_hi, lat_lo, lat_hi = gaza_bbox
    cx = (lon_min + lon_max) / 2.0
    cy = (lat_min + lat_max) / 2.0
    return (cx - lon_lo) / (lon_hi - lon_lo), (cy - lat_lo) / (lat_hi - lat_lo)


def row_to_features(
    item: dict[str, Any],
    gaza_bbox: list[float],
    id_norm: dict[str, float],
) -> torch.Tensor:
    """Convert a PairedImageDataset item to a (FEATURE_DIM,) float32 tensor."""
    meta = json.loads(item["meta"])
    parsed = parse_origin_image_name(meta["origin_image"])

    date = _date_features(meta["origin_date"])
    time = _time_features(parsed["time"])
    spatial = _spatial_features(
        meta["lon_min"], meta["lon_max"], meta["lat_min"], meta["lat_max"], gaza_bbox
    )
    ids = (
        parsed["ssc"] / float(id_norm.get("ssc", 100.0)),
        parsed["u000"] / float(id_norm.get("u000", 100.0)),
    )

    feat = item["feature"]
    pre = item["prewar"]
    feat_mean = feat.mean(dim=(1, 2))
    feat_std = feat.std(dim=(1, 2))
    pre_mean = pre.mean(dim=(1, 2))
    pre_std = pre.std(dim=(1, 2))

    return torch.tensor(
        [
            *date,
            *time,
            *spatial,
            *ids,
            *feat_mean.tolist(),
            *feat_std.tolist(),
            *pre_mean.tolist(),
            *pre_std.tolist(),
        ],
        dtype=torch.float32,
    )


def make_collate(gaza_bbox: list[float], id_norm: dict[str, float]):
    def collate(batch):
        return torch.stack(
            [row_to_features(b, gaza_bbox, id_norm) for b in batch], dim=0
        )

    return collate


def augment(x: torch.Tensor, noise_std: float, mask_prob: float) -> torch.Tensor:
    """Two-view augmentation: additive Gaussian noise + random feature masking."""
    noise = torch.randn_like(x) * noise_std
    mask = (torch.rand_like(x) >= mask_prob).float()
    return (x + noise) * mask


class MetadataEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.config = {
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
        }
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """SimCLR InfoNCE over in-batch negatives. z1, z2: (B, D)."""
    batch_size = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = z @ z.t() / temperature

    diag = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag, float("-inf"))

    targets = torch.arange(2 * batch_size, device=z.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return F.cross_entropy(sim, targets)


@click.command()
@click.argument("config", type=click.Path(exists=True))
def cli(config: str) -> None:
    params = load_yaml_with_env(config)
    if "metadata_embedding" not in params:
        raise click.ClickException("Missing required config key: metadata_embedding")
    manifest_path = params.get("manifest") or params.get("manifest_folder")
    if not manifest_path:
        raise click.ClickException(
            "Missing required config key: manifest (or manifest_folder)"
        )
    train(manifest_path, **params["metadata_embedding"])


def train(
    manifest_path: str,
    embedding_dim: int,
    hidden_dim: int,
    projection_dim: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    temperature: float,
    augment_noise_std: float,
    augment_mask_prob: float,
    gaza_bbox: list[float],
    id_norm: dict[str, float],
    weight_decay: float = 1e-5,
    num_workers: int = 0,
    device: str | None = None,
    checkpoint_path: str | None = None,
) -> None:
    if torch.cuda.is_available() and (device is None or device == "cuda"):
        dev = torch.device("cuda")
        LOGGER.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        LOGGER.info("Using CPU")

    dataset = PairedImageDataset(manifest_path)
    LOGGER.info(f"Loaded dataset with {len(dataset)} rows.")

    collate = make_collate(gaza_bbox, id_norm)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        worker_init_fn=PairedImageDataset.worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True,
    )

    encoder = MetadataEncoder(FEATURE_DIM, hidden_dim, embedding_dim).to(dev)
    proj = ProjectionHead(embedding_dim, projection_dim).to(dev)

    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(proj.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = (
        Path(checkpoint_path) if checkpoint_path else (run_dir / "metadata_embedding.pth")
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(epochs):
        encoder.train()
        proj.train()
        total = 0.0
        n = 0
        for i, x in enumerate(loader):
            x = x.to(dev, non_blocking=True)
            v1 = augment(x, augment_noise_std, augment_mask_prob)
            v2 = augment(x, augment_noise_std, augment_mask_prob)

            z1 = proj(encoder(v1))
            z2 = proj(encoder(v2))
            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bsz = x.size(0)
            total += loss.item() * bsz
            n += bsz
            print(f"Epoch {epoch}: step {i + 1}/{len(loader)}", end="\r")

        avg = total / max(n, 1)
        LOGGER.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg:.4f}")

        if avg < best_loss:
            torch.save(
                {"state_dict": encoder.state_dict(), "model_args": encoder.config},
                save_path,
            )
            LOGGER.info(
                f"Best encoder saved to {save_path} at epoch {epoch} (loss {avg:.4f})."
            )
            best_loss = avg


if __name__ == "__main__":
    cli()
