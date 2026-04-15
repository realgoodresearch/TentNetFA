import h5py
import numpy as np
import click
import sys

from displacement_tracker.util.env_loader import load_yaml_with_env
import os
from tqdm import tqdm


def resample_and_merge(config_path):
    config = load_yaml_with_env(config_path)

    loading_files = config["loading"]["files"]
    rebal_config = config["rebalancing"]

    hdf5_dir = os.path.dirname(config["hdf5"])
    out_path = rebal_config["out"]
    rng_seed = rebal_config.get("rng_seed", 42)
    null_keep_fraction = 0.25

    rng = np.random.default_rng(rng_seed)

    if not os.path.isdir(hdf5_dir):
        print(f"Error: HDF5 directory not found: {hdf5_dir}", file=sys.stderr)
        sys.exit(1)

    out_f = None
    total_written = 0

    for tif_name in tqdm(loading_files, desc="Merging H5 files"):
        base = os.path.splitext(os.path.basename(tif_name))[0]
        h5_path = os.path.join(hdf5_dir, f"{base}.h5")

        if not os.path.exists(h5_path):
            print(f"Warning: {h5_path} not found, skipping")
            continue

        with h5py.File(h5_path, "r") as f:
            labels = f["label"][:]
            is_null = np.all(labels == 0, axis=(1, 2))

            null_idx = np.where(is_null)[0]
            non_null_idx = np.where(~is_null)[0]

            keep_null_n = int(null_keep_fraction * len(null_idx))
            if keep_null_n > 0:
                keep_null_idx = rng.choice(null_idx, size=keep_null_n, replace=False)
            else:
                keep_null_idx = np.array([], dtype=int)

            keep_idx = np.sort(np.concatenate([non_null_idx, keep_null_idx]))

            n_keep = len(keep_idx)
            if n_keep == 0:
                continue

            # Create output file and datasets on first write
            if out_f is None:
                out_f = h5py.File(out_path, "w")

                for name, d in f.items():
                    shape = (0, *d.shape[1:])
                    maxshape = (None, *d.shape[1:])
                    out_f.create_dataset(
                        name,
                        shape=shape,
                        maxshape=maxshape,
                        chunks=(1, *d.shape[1:]),
                        dtype=d.dtype,
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # Append to output datasets
            for name, d in f.items():
                selected = d[keep_idx]

                ds_out = out_f[name]
                old_size = ds_out.shape[0]
                new_size = old_size + n_keep

                ds_out.resize(new_size, axis=0)
                ds_out[old_size:new_size] = selected

            total_written += n_keep

            if total_written % 5000 == 0:
                out_f.flush()

    if out_f is None:
        print("No tiles selected. Aborting.", file=sys.stderr)
        sys.exit(1)

    out_f.close()
    print(f"Saved merged and balanced dataset to {out_path}")
    print(f"Total tiles written: {total_written}")


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def cli(config_path):
    resample_and_merge(config_path)


if __name__ == "__main__":
    cli()