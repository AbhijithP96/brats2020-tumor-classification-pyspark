"""
BraTS2020 — Build Final Feature Dataset
========================================
Pipeline:
  1. Aggregate metadata CSV by volume → volume + slice features  (no H5 reads)
  2. Load H5 slices per volume        → intensity + spatial features
  3. Merge both feature sets
  4. Map  volume_X  →  BraTS20_Training_00X  →  Grade  (name_mapping.csv)
  5. Left-merge survival_info.csv for Age (available for ~236 / 369 volumes)
  6. Save to OUTPUT_CSV

Flat column schema (final CSV):
  volume_id
  voxels_ch0, voxels_ch1, voxels_ch2
  total_tumour_voxels
  ratio_ch0, ratio_ch1, ratio_ch2
  tumour_slice_count, start_slice, end_slice
  flair_mean, flair_std, flair_min, flair_max, flair_skewness
  t1_mean,    t1_std,    t1_min,    t1_max,    t1_skewness
  t1ce_mean,  t1ce_std,  t1ce_min,  t1ce_max,  t1ce_skewness
  t2_mean,    t2_std,    t2_min,    t2_max,    t2_skewness
  centroid_x, centroid_y, centroid_slice
  spread_x,   spread_y,   spread_slice
  brats20_id
  grade                    (HGG / LGG)
  age                      (float, NaN where not available)
"""

import os
import sys

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from feature_extraction import (
    _load_slices,
    extract_intensity_features,
    extract_spatial_features,
)

console = Console()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = (
    "/media/basilisk/Files/New folder/Dataset/archive/"
    "BraTS2020_training_data/content/data"
)
METADATA_CSV = (
    "/media/basilisk/Files/New folder/Dataset/archive/"
    "BraTS20 Training Metadata.csv"
)
NAME_MAP_CSV = os.path.join(DATA_DIR, "name_mapping.csv")
SURVIVAL_CSV = os.path.join(DATA_DIR, "survival_info.csv")

OUTPUT_CSV = "/home/basilisk/Documents/brats-spark/brats2020_features.csv"

MODALITY_LABELS = {0: "flair", 1: "t1", 2: "t1ce", 3: "t2"}


# ---------------------------------------------------------------------------
# Step 1 — Aggregate metadata CSV (no H5 reads)
# ---------------------------------------------------------------------------

def aggregate_metadata(metadata_csv: str) -> pd.DataFrame:
    """Return one row per volume with volume + slice features from the CSV."""
    console.log("[bold cyan]Step 1[/] — Aggregating metadata CSV …")

    df = pd.read_csv(metadata_csv)

    # Per-channel voxel counts
    agg = (
        df.groupby("volume")
        .agg(
            voxels_ch0=("label0_pxl_cnt", "sum"),
            voxels_ch1=("label1_pxl_cnt", "sum"),
            voxels_ch2=("label2_pxl_cnt", "sum"),
        )
        .reset_index()
        .rename(columns={"volume": "volume_id"})
    )

    # Total tumour voxels and sub-region ratios
    agg["total_tumour_voxels"] = agg["voxels_ch0"] + agg["voxels_ch1"] + agg["voxels_ch2"]
    for ch in range(3):
        agg[f"ratio_ch{ch}"] = agg[f"voxels_ch{ch}"] / agg["total_tumour_voxels"].replace(0, np.nan)
    agg[["ratio_ch0", "ratio_ch1", "ratio_ch2"]] = agg[
        ["ratio_ch0", "ratio_ch1", "ratio_ch2"]
    ].fillna(0.0)

    # Slice features: tumour present when any label count > 0
    df["has_tumour"] = (
        (df["label0_pxl_cnt"] > 0)
        | (df["label1_pxl_cnt"] > 0)
        | (df["label2_pxl_cnt"] > 0)
    )
    tumour_slices = df[df["has_tumour"]].groupby("volume")["slice"]
    slice_feats = pd.DataFrame(
        {
            "volume_id": tumour_slices.count().index,
            "tumour_slice_count": tumour_slices.count().values,
            "start_slice": tumour_slices.min().values,
            "end_slice": tumour_slices.max().values,
        }
    )
    agg = agg.merge(slice_feats, on="volume_id", how="left")
    agg["tumour_slice_count"] = agg["tumour_slice_count"].fillna(0).astype(int)
    agg["start_slice"] = agg["start_slice"].fillna(-1).astype(int)
    agg["end_slice"] = agg["end_slice"].fillna(-1).astype(int)

    console.log(f"  → {len(agg)} volumes aggregated from metadata CSV.")
    return agg


# ---------------------------------------------------------------------------
# Step 2 — Intensity + spatial features from H5 slices
# ---------------------------------------------------------------------------

def _flatten_intensity(intensity: dict) -> dict:
    row = {}
    for mod_idx, stats in intensity.items():
        prefix = MODALITY_LABELS.get(mod_idx, f"mod{mod_idx}")
        for stat, val in stats.items():
            row[f"{prefix}_{stat}"] = val
    return row


def _flatten_spatial(spatial: dict) -> dict:
    row = {}
    for metric, axes in spatial.items():
        for axis, val in axes.items():
            row[f"{metric}_{axis}"] = val
    return row


def extract_h5_features(volume_ids: list[int], data_dir: str) -> pd.DataFrame:
    """Intensity + spatial features for every volume by reading H5 slices."""
    console.log("[bold cyan]Step 2[/] — Extracting intensity + spatial features from H5 …")

    rows = []
    skipped = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("volumes", total=len(volume_ids))
        for vid in volume_ids:
            progress.update(task, advance=1, description=f"volume {vid:>3}")
            try:
                images, masks, slice_idxs = _load_slices(vid, data_dir)
            except FileNotFoundError:
                skipped.append(vid)
                continue

            row = {"volume_id": vid}
            row.update(_flatten_intensity(extract_intensity_features(images, masks)))
            row.update(_flatten_spatial(extract_spatial_features(masks, slice_idxs)))
            rows.append(row)

    if skipped:
        console.log(f"  [yellow]⚠ Skipped {len(skipped)} volumes (H5 not found): {skipped}")

    console.log(f"  → {len(rows)} volumes processed from H5.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3 — Merge feature sets
# ---------------------------------------------------------------------------

def merge_features(df_meta: pd.DataFrame, df_h5: pd.DataFrame) -> pd.DataFrame:
    console.log("[bold cyan]Step 3[/] — Merging feature sets …")
    df = df_meta.merge(df_h5, on="volume_id", how="inner")
    console.log(f"  → {len(df)} rows after inner merge.")
    return df


# ---------------------------------------------------------------------------
# Step 4 — Map volume_id → brats20_id → Grade
# ---------------------------------------------------------------------------

def attach_grade(df: pd.DataFrame, name_map_csv: str) -> pd.DataFrame:
    console.log("[bold cyan]Step 4[/] — Attaching Grade from name_mapping.csv …")

    df_map = pd.read_csv(name_map_csv)

    # Extract the 3-digit number from BraTS_2020_subject_ID
    df_map["volume_id"] = (
        df_map["BraTS_2020_subject_ID"]
        .str.extract(r"_(\d+)$")[0]
        .astype(int)
    )

    df = df.merge(
        df_map[["volume_id", "BraTS_2020_subject_ID", "Grade"]].rename(
            columns={"BraTS_2020_subject_ID": "brats20_id", "Grade": "grade"}
        ),
        on="volume_id",
        how="left",
    )

    n_missing = df["grade"].isna().sum()
    if n_missing:
        console.log(f"  [yellow]⚠ {n_missing} volumes without a Grade mapping.")
    console.log(f"  → Grade distribution: {df['grade'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Step 5 — Merge Age from survival_info.csv
# ---------------------------------------------------------------------------

def attach_age(df: pd.DataFrame, survival_csv: str) -> pd.DataFrame:
    console.log("[bold cyan]Step 5[/] — Merging Age from survival_info.csv …")

    df_surv = pd.read_csv(survival_csv)[["Brats20ID", "Age"]].rename(
        columns={"Brats20ID": "brats20_id", "Age": "age"}
    )

    df = df.merge(df_surv, on="brats20_id", how="left")
    n_age = df["age"].notna().sum()
    console.log(f"  → Age available for {n_age} / {len(df)} volumes.")

    median_age = df["age"].median()
    n_filled = df["age"].isna().sum()
    df["age"] = df["age"].fillna(median_age)
    console.log(f"  → Filled {n_filled} missing Age values with median ({median_age:.3f}).")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Step 1 — fast metadata aggregation
    df_meta_agg = aggregate_metadata(METADATA_CSV)

    volume_ids = sorted(df_meta_agg["volume_id"].tolist())

    # Step 2 — H5 intensity + spatial
    df_h5 = extract_h5_features(volume_ids, DATA_DIR)

    # Step 3 — merge
    df = merge_features(df_meta_agg, df_h5)

    # Step 4 — grade
    df = attach_grade(df, NAME_MAP_CSV)

    # Step 5 — age
    df = attach_age(df, SURVIVAL_CSV)

    # Step 6 — save
    console.log(f"[bold cyan]Step 6[/] — Saving to {OUTPUT_CSV} …")
    df.to_csv(OUTPUT_CSV, index=False)
    console.log(f"  → Saved [bold green]{len(df)} rows × {len(df.columns)} columns[/].")
    console.log(f"  → Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
