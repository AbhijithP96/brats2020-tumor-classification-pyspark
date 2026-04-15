"""
BraTS2020 Dataset - Initial Data Analysis
==========================================
Explores the structure of the BraTS2020 training dataset stored as HDF5 slices.
Each file follows the naming convention: volume_<id>_slice_<idx>.h5
"""

import os
import glob
from collections import Counter

import h5py
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "/media/basilisk/Files/New folder/Dataset/archive/BraTS2020_training_data/content/data"
FILE_PATTERN = os.path.join(DATA_DIR, "*.h5")

if __name__ == "__main__":

    # ---------------------------------------------------------------------------
    # 1. Discover files
    # ---------------------------------------------------------------------------

    files = sorted(glob.glob(FILE_PATTERN))

    if not files:
        raise FileNotFoundError(f"No .h5 files found at: {FILE_PATTERN}")

    console.rule("[bold cyan]1. Discover Files")
    console.print(f"Total files found : [bold]{len(files)}[/bold]")
    console.print("Sample filenames  :")
    for f in files[:5]:
        console.print(f"  [dim]{os.path.basename(f)}[/dim]")

    # ---------------------------------------------------------------------------
    # 2. Volume / slice distribution
    #    Filename format: volume_<vol_id>_slice_<slice_idx>.h5
    # ---------------------------------------------------------------------------

    vol_ids = [int(os.path.basename(f).split("_")[1]) for f in files]
    slice_counts = Counter(vol_ids)
    unique_vols = sorted(slice_counts.keys())

    console.rule("[bold cyan]2. Volume / Slice Distribution")
    console.print(f"Unique volumes    : [bold]{len(unique_vols)}[/bold]")
    console.print(
        f"Slice counts (min / max / mean): "
        f"[green]{min(slice_counts.values())}[/green] / "
        f"[green]{max(slice_counts.values())}[/green] / "
        f"[green]{np.mean(list(slice_counts.values())):.1f}[/green]"
    )

    # ---------------------------------------------------------------------------
    # 3. HDF5 dataset keys for one file
    # ---------------------------------------------------------------------------

    INSPECT_VOL = unique_vols[0]
    vol_files = sorted(
        glob.glob(os.path.join(DATA_DIR, f"volume_{INSPECT_VOL}_slice_*.h5"))
    )

    console.rule("[bold cyan]3. HDF5 Dataset Keys")
    with h5py.File(vol_files[0], "r") as h5:
        console.print(f"File: [bold]{os.path.basename(vol_files[0])}[/bold]")
        key_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        key_table.add_column("Key")
        key_table.add_column("Shape")
        key_table.add_column("Dtype")
        for key in h5.keys():
            key_table.add_row(key, str(h5[key].shape), str(h5[key].dtype))
    console.print(key_table)

    # ---------------------------------------------------------------------------
    # 4. Inspect mask labels for a single volume
    # ---------------------------------------------------------------------------

    console.rule(f"[bold cyan]4. Mask Labels -> Volume {INSPECT_VOL}")
    label_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    label_table.add_column("Slice", style="dim")
    label_table.add_column("Unique mask labels")

    for idx, path in enumerate(vol_files):
        with h5py.File(path, "r") as h5:
            labels = np.unique(h5["mask"][:])
        label_table.add_row(str(idx + 1), str(labels))

    console.print(label_table)

    # ---------------------------------------------------------------------------
    # 5. Inspect mask labels per channel for a single slice
    # ---------------------------------------------------------------------------

    for f in vol_files:
        with h5py.File(f, "r") as h5:
            mask_probe = h5["mask"][:]
        if np.any(mask_probe > 0):
            sample_slice = f
            break

    console.rule("[bold cyan]5. Channel-wise Mask Labels")
    console.print(f"File: [bold]{os.path.basename(sample_slice)}[/bold]")

    ch_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    ch_table.add_column("Channel", style="dim")
    ch_table.add_column("Unique labels")

    with h5py.File(sample_slice, "r") as h5:
        mask = h5["mask"][:]
    for ch in range(mask.shape[-1]):
        ch_table.add_row(str(ch), str(np.unique(mask[:, :, ch])))

    console.print(ch_table)

    # ---------------------------------------------------------------------------
    # 6. Active mask pixels per channel across representative slices
    # ---------------------------------------------------------------------------

    SAMPLE_SLICES = [50, 100, 150]
    SAMPLE_VOL = unique_vols[0]

    console.rule(f"[bold cyan]6. Active Mask Pixels per Channel on Volume {SAMPLE_VOL}")

    with h5py.File(vol_files[0], "r") as h5:
        n_channels = h5["mask"].shape[-1]

    pixel_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    pixel_table.add_column("Slice", style="dim")
    for ch in range(n_channels):
        pixel_table.add_column(f"ch{ch} (px)")

    for s in SAMPLE_SLICES:
        path = os.path.join(DATA_DIR, f"volume_{SAMPLE_VOL}_slice_{s}.h5")
        with h5py.File(path, "r") as h5:
            mask = h5["mask"][:]
        pixel_counts = [
            str(int(np.sum(mask[:, :, ch] > 0))) for ch in range(n_channels)
        ]
        pixel_table.add_row(str(s), *pixel_counts)

    console.print(pixel_table)

    # ---------------------------------------------------------------------------
    # 7. Channel overlap analysis
    #    Independent channels -> whole tumour = union of all channels (BraTS convention)
    # ---------------------------------------------------------------------------

    OVERLAP_SLICE = 50
    overlap_path = os.path.join(
        DATA_DIR, f"volume_{SAMPLE_VOL}_slice_{OVERLAP_SLICE}.h5"
    )

    console.rule("[bold cyan]7. Channel Overlap Analysis")
    console.print(f"File: [bold]volume_{SAMPLE_VOL}_slice_{OVERLAP_SLICE}.h5[/bold]")

    with h5py.File(overlap_path, "r") as h5:
        mask = h5["mask"][:]

    channels = [mask[:, :, ch] > 0 for ch in range(mask.shape[-1])]

    overlap_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    overlap_table.add_column("Channel pair")
    overlap_table.add_column("Overlapping pixels", justify="right")

    from itertools import combinations

    for i, j in combinations(range(len(channels)), 2):
        overlap = int(np.sum(channels[i] & channels[j]))
        overlap_table.add_row(f"ch{i} & ch{j}", str(overlap))

    console.print(overlap_table)

    total_union = int(np.sum(np.any(np.stack(channels, axis=-1), axis=-1)))
    total_sum = int(sum(np.sum(ch) for ch in channels))
    console.print(f"Union (any channel active) : [bold]{total_union}[/bold] pixels")
    console.print(f"Sum   (per-channel total)  : [bold]{total_sum}[/bold] pixels")
    if total_union == total_sum:
        console.print(
            "[green]Channels are non-overlapping -> whole tumour = union of all channels.[/green]"
        )
    else:
        console.print(
            "[yellow]Channels overlap -> further investigation needed.[/yellow]"
        )
