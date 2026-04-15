import matplotlib
matplotlib.use("TkAgg")

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import random


class DatasetViewer:
    def __init__(self, dataset):
        self.dataset = dataset

    def show(self, idx: int, overlay: bool = True) -> None:
        if overlay:
            self.show_overlay(idx)
        else:
            self.show_split(idx)

    def _prepare_display_data(self, idx: int):
        sample = self.dataset[idx]

        def to_numpy(t):
            return (
                t.squeeze().cpu().numpy()
                if isinstance(t, torch.Tensor)
                else np.array(t)
            )

        arr_feat = to_numpy(sample["feature"]).astype(np.float32)
        arr_prewar = to_numpy(sample["prewar"]).astype(np.float32)
        arr_label = to_numpy(sample["label"])

        if arr_feat.ndim == 3 and arr_feat.shape[0] == 3:
            arr_feat = np.transpose(arr_feat, (1, 2, 0))

        if arr_prewar.ndim == 3 and arr_prewar.shape[0] == 3:
            arr_prewar = np.transpose(arr_prewar, (1, 2, 0))

        if arr_label.ndim == 3 and arr_label.shape[0] == 1:
            arr_label = arr_label[0]

        meta = sample["meta"]
        if isinstance(meta, bytes):
            meta = meta.decode("utf-8")

        try:
            meta_dict = json.loads(str(meta))
            meta_text = "\n".join(f"{k}: {v}" for k, v in meta_dict.items())
        except json.JSONDecodeError:
            meta_text = str(meta)

        return arr_feat, arr_prewar, arr_label, meta_text

    def _plot_meta(self, ax, text):
        ax.axis("off")
        ax.text(
            0.05,
            0.95,
            text,
            fontsize=11,
            color="black",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

    def _plot_overlay_on_axis(self, ax, base, mask):
        ax.imshow(base, interpolation="none")

        ax.imshow(
            np.ones_like(mask),
            cmap="spring",
            alpha=mask,
            interpolation="none",
        )

        h, w = mask.shape
        for i in range(1, 3):
            ax.axhline(y=i * h // 3, color="red", linestyle="--")
            ax.axvline(x=i * w // 3, color="red", linestyle="--")

        ax.axis("off")

    def show_overlay(self, idx: int) -> None:
        arr_feat, arr_prewar, arr_label, meta_text = self._prepare_display_data(idx)

        fig, axes = plt.subplots(
            1, 5,
            figsize=(30, 8),  # bigger figure
            dpi=150  # higher resolution
        )

        self._plot_meta(axes[0], meta_text)

        axes[1].imshow(arr_prewar, interpolation="none")
        axes[1].set_title("Prewar")
        axes[1].axis("off")

        self._plot_overlay_on_axis(axes[2], arr_feat, arr_label)
        axes[2].set_title("Current + Label")

        plt.tight_layout()
        plt.show()

    def show_split(self, idx: int) -> None:
        arr_feat, arr_prewar, arr_label, meta_text = self._prepare_display_data(idx)

        diff = arr_feat - arr_prewar

        # normalize diff for visualization only
        diff_vis = diff.copy()
        diff_vis -= diff_vis.min()
        if diff_vis.max() > 0:
            diff_vis /= diff_vis.max()

        fig, axes = plt.subplots(
            1, 5,
            figsize=(30, 8),  # bigger figure
            dpi=150  # higher resolution
        )

        self._plot_meta(axes[0], meta_text)

        def enhance(img):
            img = img - img.min()
            if img.max() > 0:
                img = img / img.max()
            return img

        arr_feat = enhance(arr_feat)
        arr_prewar = enhance(arr_prewar)

        axes[1].imshow(arr_prewar, interpolation="none")
        axes[1].set_title("Prewar")
        axes[1].axis("off")

        axes[2].imshow(arr_feat, interpolation="none")
        axes[2].set_title("Current")
        axes[2].axis("off")

        axes[3].imshow(diff_vis, interpolation="none")
        axes[3].set_title("Diff")
        axes[3].axis("off")

        axes[4].imshow(arr_label)
        axes[4].set_title("Label")
        axes[4].axis("off")

        plt.tight_layout()
        plt.show()

    def show_batch(self, indices: list[int]) -> None:
        indices = indices[:18]
        n_plots = len(indices)
        if n_plots == 0:
            return

        best_rows = 1
        best_cols = n_plots
        best_waste = float("inf")
        best_scale = 0.0

        for cols in range(1, min(n_plots, 6) + 1):
            rows = (n_plots + cols - 1) // cols
            waste = rows * cols - n_plots
            scale = min(12 / cols, 8 / rows)

            if waste < best_waste or (waste == best_waste and scale > best_scale):
                best_waste = waste
                best_rows, best_cols = rows, cols
                best_scale = scale

        figsize = (best_cols * best_scale, best_rows * best_scale)
        fig, axes = plt.subplots(best_rows, best_cols, figsize=figsize)

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        axes_flat = axes.flatten()

        for i, idx in enumerate(indices):
            ax = axes_flat[i]
            arr_feat, arr_prewar, arr_label, _ = self._prepare_display_data(idx)

            combined = np.concatenate([arr_prewar, arr_feat], axis=1)
            ax.imshow(combined, interpolation="none")

            # --- ADD LABEL OVERLAY ---
            h, w = arr_label.shape
            overlay = np.zeros((h, w * 2))

            if arr_label.max() > 0:
                scaled_label = arr_label / arr_label.max()
            else:
                scaled_label = arr_label

            overlay[:, w:] = scaled_label

            ax.imshow(
                np.ones_like(overlay),
                cmap="spring",
                alpha=overlay * 0.8,  # increase visibility
                interpolation="none",
            )
            # --- END ADDITION ---

            txt = ax.text(
                0.5,
                0.95,
                str(idx),
                fontsize=10,
                color="lightgreen",
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontweight="bold",
            )
            txt.set_path_effects([pe.withStroke(linewidth=3, foreground="black")])

            ax.axis("off")

        for i in range(n_plots, len(axes_flat)):
            axes_flat[i].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from displacement_tracker.paired_image_dataset import PairedImageDataset

    ds = PairedImageDataset(
        "tif_files/training/khan_yunis_20241005_062045_ssc2_u0001_visual_clip.h5"
        #"tif_files/historic/processed/khan_yunis_20250918_122620_ssc10_u0001_visual_clip.h5"
    )

    print("Dataset length:", len(ds))

    viewer = DatasetViewer(ds)

    if len(ds) > 0:
        indices = random.sample(range(len(ds)), min(12, len(ds)))
        print("Showing indices:", indices)
        for idx in indices:
            viewer.show_split(idx)
    else:
        print("Dataset is empty.")

# poetry run python displacement_tracker/visualization/dataset_viewer.py