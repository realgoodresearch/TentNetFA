from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import click
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from displacement_tracker.paired_image_dataset import PairedImageDataset
from displacement_tracker.simple_cnn import SimpleCNN
from displacement_tracker.util.config import flow_option, load_flow_config
from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("train-cnn")


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, num_workers: int = 0):
        loader = DataLoader(
            base_ds,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=lambda b: b[0],
            worker_init_fn=PairedImageDataset.worker_init_fn if num_workers else None,
        )
        self.data = []
        for sample in loader:
            self.data.append(
                {k: v.clone() if torch.is_tensor(v) else v for k, v in sample.items()}
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def custom_collate(batch):
    collated_dict = {}
    for key in batch[0].keys():
        entry = [d[key] for d in batch]
        if key != "meta":
            entry = torch.stack(entry, dim=0)
        collated_dict[key] = entry
    return collated_dict


@click.command()
@click.argument("config", type=click.Path(exists=True))
@flow_option(default="train")
@click.option("--holdout-fold", type=int, default=None, help="Spatial CV test strip 1–5 (north→south).")
@click.option("--spatial-folds-path", type=click.Path(), default=None)
@click.option("--early-stop-patience", type=int, default=None)
@click.option("--epochs", "epochs_opt", type=int, default=None)
def cli(config: str, flow: str, holdout_fold, spatial_folds_path, early_stop_patience, epochs_opt) -> None:
    params = load_flow_config(config, flow)
    if "training" not in params:
        raise click.ClickException("Missing required config key: training")
    manifest_path = params.get("manifest") or params.get("manifest_folder")
    if not manifest_path:
        raise click.ClickException(
            "Missing required config key: manifest (or manifest_folder)"
        )
    training = dict(params["training"])
    holdout_fold = holdout_fold if holdout_fold is not None else training.pop("holdout_fold", None)
    spatial_folds_path = spatial_folds_path or training.pop("spatial_folds_path", None)
    early_stop_patience = (
        early_stop_patience
        if early_stop_patience is not None
        else training.pop("early_stop_patience", 0)
    )
    inner_val_frac = training.pop("inner_val_frac", 0.1)
    if epochs_opt is not None:
        training["epochs"] = epochs_opt
    for key in ("holdout_fold", "spatial_folds_path", "early_stop_patience", "inner_val_frac"):
        training.pop(key, None)
    train(
        manifest_path,
        artifact_dir=params.get("artifact_dir", "runs"),
        holdout_fold=holdout_fold,
        spatial_folds_path=spatial_folds_path,
        early_stop_patience=int(early_stop_patience or 0),
        inner_val_frac=float(inner_val_frac),
        **training,
    )


def train(
    manifest_path: str,
    training_frac: float,
    validation_frac: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    sigma: float = 3.0,
    checkpoint: str | None = None,
    device: str | None = None,
    model_kwargs: dict | None = None,
    memory: bool = False,
    num_workers: int = 0,
    artifact_dir: str = "runs",
    holdout_fold: int | None = None,
    spatial_folds_path: str | None = None,
    early_stop_patience: int = 0,
    inner_val_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    model_kwargs = model_kwargs or {}
    spatial = holdout_fold is not None

    if torch.cuda.is_available() and (device is None or device == "cuda"):
        device = torch.device("cuda")
        LOGGER.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif device is None or device == "cpu":
        device = torch.device("cpu")
        LOGGER.info("Using CPU")
    else:
        raise Exception(f"Could not find device {device}")

    if spatial:
        LOGGER.info(
            "Spatial CV holdout fold %s: training from scratch (checkpoint ignored).",
            holdout_fold,
        )
        model = SimpleCNN(9, 1, **model_kwargs).to(device)
        save_loc = None
    elif checkpoint:
        checkpoint = Path(checkpoint)
        LOGGER.warning("Loading model from checkpoint, ignoring model_kwargs.")
        model = SimpleCNN.from_pth(checkpoint).to(device)
        save_loc = checkpoint.parent
    else:
        model = SimpleCNN(9, 1, **model_kwargs).to(device)
        save_loc = None

    dataset = PairedImageDataset(manifest_path, sigma=sigma)
    holdout_set = None
    fold_info: dict = {}
    if spatial:
        if not spatial_folds_path:
            raise click.ClickException(
                "holdout_fold requires spatial_folds_path (tiles_with_folds.parquet)"
            )
        train_set, val_set, holdout_set, fold_info = dataset.create_fold_subsets(
            spatial_folds_path,
            int(holdout_fold),
            inner_val_frac=inner_val_frac,
            seed=seed,
        )
        splits = [1.0, 0.0, 0.0]
        idcs_list = [
            train_set.indices or [],
            val_set.indices or [],
            holdout_set.indices or [],
        ]
        LOGGER.info(
            "Spatial split fold %s: train=%s inner_val=%s holdout=%s",
            holdout_fold,
            fold_info["n_train"],
            fold_info["n_inner_val"],
            fold_info["n_holdout"],
        )
    else:
        splits = [training_frac, validation_frac, 1 - training_frac - validation_frac]
        (train_set, val_set, _), idcs_list = dataset.create_subsets(
            splits, shuffle=True, save_loc=save_loc
        )

    if memory:
        LOGGER.info("Caching training dataset in RAM...")
        train_set = CachedDataset(train_set, num_workers=int(num_workers))
        val_set = CachedDataset(val_set, num_workers=int(num_workers))
        LOGGER.info(
            f"Cached {len(train_set)} training and {len(val_set)} validation samples."
        )
        loader_workers = 0
    else:
        loader_workers = int(num_workers)

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=custom_collate,
        num_workers=loader_workers,
    )
    if loader_workers > 0:
        loader_kwargs["worker_init_fn"] = PairedImageDataset.worker_init_fn
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_set, pin_memory=True, **loader_kwargs)
    val_loader = DataLoader(val_set, **loader_kwargs)
    LOGGER.info(
        f"Split {len(dataset)} samples into {len(train_set)} train and {len(val_set)} validation samples."
    )

    def criterion(x, y):
        return 1e6 * torch.nn.functional.mse_loss(x, y)

    def eval_loader(loader) -> float:
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for entry in loader:
                diff = entry["feature"] - entry["prewar"]
                feats = torch.cat((entry["feature"], entry["prewar"], diff), axis=1).to(
                    device
                )
                labels = entry["label"].to(device)
                loss = criterion(model(feats), labels)
                bsz = labels.size(0)
                total += loss.item() * bsz
                n += bsz
        return total / n if n else 0.0

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if spatial:
        scheme = fold_info.get("scheme") or "north_south"
        if scheme == "north_south":
            run_dir = os.path.join(artifact_dir, "spatial_cv", f"fold{int(holdout_fold)}")
        else:
            run_dir = os.path.join(artifact_dir, "spatial_cv", scheme, f"fold{int(holdout_fold)}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(artifact_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "splits.csv"), "w") as split_file:
        split_file.write(",".join([str(split) for split in splits]) + "\n")
        for idcs in idcs_list:
            split_file.write(",".join([str(idx) for idx in idcs]) + "\n")
    if fold_info:
        Path(run_dir, "fold_info.json").write_text(json.dumps(fold_info, indent=2) + "\n")

    best_eval = float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    epoch_rows = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_train = 0
        for i, entry in enumerate(train_loader):
            diff = entry["feature"] - entry["prewar"]
            feats = torch.cat((entry["feature"], entry["prewar"], diff), axis=1).to(device)
            labels = entry["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()
            optimizer.step()
            bsz = labels.size(0)
            train_loss += loss.item() * bsz
            n_train += bsz
            print(f"Epoch {epoch}: Completed step {i + 1} / {len(train_loader)}", end="\r")

        train_loss /= n_train if n_train else 1.0
        val_loss = eval_loader(val_loader)
        if val_loss < best_eval:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_args": getattr(model, "config", {}),
                },
                os.path.join(run_dir, "best_model.pth"),
            )
            LOGGER.info(
                "Best model saved to %s at epoch %s with loss %s < %s.",
                os.path.join(run_dir, "best_model.pth"),
                epoch,
                val_loss,
                best_eval,
            )
            best_eval = val_loss
            best_epoch = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        LOGGER.info(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}"
        )
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_eval,
            }
        )
        if early_stop_patience and epochs_without_improve >= early_stop_patience:
            LOGGER.info(
                "Early stopping at epoch %s (patience=%s).",
                epoch + 1,
                early_stop_patience,
            )
            break

    holdout_loss = None
    if holdout_set is not None and len(holdout_set) > 0:
        best_path = Path(run_dir) / "best_model.pth"
        if best_path.exists():
            model = SimpleCNN.from_pth(best_path).to(device)
        holdout_loader = DataLoader(holdout_set, **loader_kwargs)
        holdout_loss = eval_loader(holdout_loader)
        LOGGER.info("Holdout fold %s loss (best ckpt): %.4f", holdout_fold, holdout_loss)

    pd.DataFrame(epoch_rows).to_csv(os.path.join(run_dir, "epoch_log.csv"), index=False)
    summary = {
        "holdout_fold": holdout_fold,
        "n_train": len(train_set),
        "n_inner_val": len(val_set),
        "n_holdout": len(holdout_set) if holdout_set is not None else 0,
        "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
        "inner_val_loss": best_eval if best_eval < float("inf") else None,
        "holdout_loss": holdout_loss,
        "epochs_ran": len(epoch_rows),
        "run_dir": run_dir,
    }
    Path(run_dir, "cv_fold_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if spatial and epoch_rows:
        from displacement_tracker.spatial_cv.plot import plot_epoch_curves

        curves = Path(run_dir) / "epoch_curves.png"
        plot_epoch_curves(Path(run_dir) / "epoch_log.csv", summary, curves)
        LOGGER.info("Wrote %s", curves)
    return summary


if __name__ == "__main__":
    cli()
