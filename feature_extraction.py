"""
BraTS2020 Dataset - Feature Extraction
=======================================
Extracts four feature groups per volume (patient) by aggregating all slices:

  1. Volume features   — voxel counts per sub-region, total tumour, sub-region ratios
  2. Intensity features — per-modality statistics (mean, std, min, max, skewness)
                          computed over tumour-masked voxels
  3. Spatial features  — 3-D tumour centroid (x, y, slice) and spread (std per axis)
  4. Slice features    — tumour-containing slice count, start slice, end slice

BraTS mask convention (3 binary channels):
  ch0 — necrotic core
  ch1 — peritumoral edema
  ch2 — enhancing tumour
  total — union of all channels (whole tumour)

Image convention (4 modalities, channels 0-3):
  0 — FLAIR  |  1 — T1  |  2 — T1ce  |  3 — T2
"""

import os
import glob

import h5py
import numpy as np
from scipy.stats import skew
from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

MASK_CHANNEL_LABELS = {
    0: "Necrotic core",
    1: "Peritumoral edema",
    2: "Enhancing tumour",
}

MODALITY_LABELS = {
    0: "FLAIR",
    1: "T1",
    2: "T1ce",
    3: "T2",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_slices(
    volume_id: int, data_dir: str
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Load all image and mask arrays for a volume, sorted by slice index.

    Returns:
        images     — list of (H, W, 4) float arrays, one per slice
        masks      — list of (H, W, 3) binary arrays, one per slice
        slice_idxs — list of integer slice indices extracted from filenames
    """
    pattern = os.path.join(data_dir, f"volume_{volume_id}_slice_*.h5")
    slice_files = sorted(glob.glob(pattern))

    if not slice_files:
        raise FileNotFoundError(
            f"No slices found for volume {volume_id} in: {data_dir}"
        )

    images, masks, slice_idxs = [], [], []
    for path in slice_files:
        slice_idx = int(os.path.basename(path).split("_")[3].split(".")[0])
        with h5py.File(path, "r") as h5:
            images.append(h5["image"][:].astype(np.float32))
            masks.append((h5["mask"][:] > 0).astype(np.uint8))
        slice_idxs.append(slice_idx)

    return images, masks, slice_idxs


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------


def extract_intensity_features(
    images: list[np.ndarray], masks: list[np.ndarray]
) -> dict:
    """Per-modality intensity statistics over tumour-masked voxels.

    Args:
        images: List of (H, W, 4) float arrays.
        masks:  List of (H, W, C) binary arrays; whole-tumour mask is
                derived as the union across channels.

    Returns:
        {
          modality_idx: {
            "mean": float, "std": float,
            "min":  float, "max": float,
            "skewness": float,
          }
        }
    """
    n_modalities = images[0].shape[-1]

    # Build whole-tumour mask per slice (union across mask channels)
    tumour_masks = [np.any(m, axis=-1) for m in masks]  # (H, W) bool per slice

    # Concatenate voxels across all slices per modality
    modality_voxels: dict[int, list] = {mod: [] for mod in range(n_modalities)}
    for img, tmask in zip(images, tumour_masks):
        for mod in range(n_modalities):
            modality_voxels[mod].append(img[:, :, mod][tmask])

    stats = {}
    for mod in range(n_modalities):
        vals = (
            np.concatenate(modality_voxels[mod])
            if modality_voxels[mod]
            else np.array([])
        )
        if vals.size == 0:
            stats[mod] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "skewness": 0.0,
            }
        else:
            stats[mod] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "skewness": float(skew(vals)),
            }

    return stats


def extract_spatial_features(masks: list[np.ndarray], slice_idxs: list[int]) -> dict:
    """3-D tumour centroid and spread across spatial axes.

    Centroid is computed as the mean (x, y, slice_index) of all tumour voxels
    (whole-tumour union). Spread is the standard deviation on each axis.

    Args:
        masks:      List of (H, W, C) binary arrays.
        slice_idxs: Slice index corresponding to each mask array.

    Returns:
        {
          "centroid": {"x": float, "y": float, "slice": float},
          "spread":   {"x": float, "y": float, "slice": float},
        }
    """
    xs, ys, ss = [], [], []

    for mask, s_idx in zip(masks, slice_idxs):
        tumour = np.any(mask, axis=-1)  # (H, W)
        ys_slice, xs_slice = np.where(tumour)
        xs.extend(xs_slice.tolist())
        ys.extend(ys_slice.tolist())
        ss.extend([s_idx] * len(xs_slice))

    if not xs:
        zero = {"x": 0.0, "y": 0.0, "slice": 0.0}
        return {"centroid": zero, "spread": zero}

    xs_arr = np.array(xs, dtype=np.float32)
    ys_arr = np.array(ys, dtype=np.float32)
    ss_arr = np.array(ss, dtype=np.float32)

    return {
        "centroid": {
            "x": float(np.mean(xs_arr)),
            "y": float(np.mean(ys_arr)),
            "slice": float(np.mean(ss_arr)),
        },
        "spread": {
            "x": float(np.std(xs_arr)),
            "y": float(np.std(ys_arr)),
            "slice": float(np.std(ss_arr)),
        },
    }


def extract_slice_features(masks: list[np.ndarray], slice_idxs: list[int]) -> dict:
    """Slice-level tumour extent features.

    Args:
        masks:      List of (H, W, C) binary arrays.
        slice_idxs: Slice index corresponding to each mask array.

    Returns:
        {
          "tumour_slice_count": int,
          "start_slice":        int,
          "end_slice":          int,
        }
    """
    tumour_slices = [s_idx for mask, s_idx in zip(masks, slice_idxs) if np.any(mask)]

    if not tumour_slices:
        return {"tumour_slice_count": 0, "start_slice": -1, "end_slice": -1}

    return {
        "tumour_slice_count": len(tumour_slices),
        "start_slice": min(tumour_slices),
        "end_slice": max(tumour_slices),
    }


def extract_all_features(volume_id: int, data_dir: str) -> dict:
    """Extract all feature groups for a single volume.

    Args:
        volume_id: Integer volume identifier.
        data_dir:  Directory containing the HDF5 slice files.

    Returns:
        {
          "volume_id": int,
          "volume":    dict,
          "intensity": dict,
          "spatial":   dict,
          "slice":     dict,
        }
    """
    images, masks, slice_idxs = _load_slices(volume_id, data_dir)

    return {
        "volume_id": volume_id,
        "intensity": extract_intensity_features(images, masks),
        "spatial": extract_spatial_features(masks, slice_idxs),
        "slice": extract_slice_features(masks, slice_idxs),
    }
