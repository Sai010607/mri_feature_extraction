"""
voxel_features.py
=================
Second, independent feature family computed directly from the FreeSurfer
volumes (via nibabel), not from the .stats text tables:

* Intensity distribution inside each ROI: Mean, Std, Min, Max, Median, P5,
  P95, Skewness, Kurtosis (excess / Fisher definition). Std is the
  population standard deviation (ddof=0).
* Shape descriptors: NVoxels, Volume_mm3 (recomputed from the voxel count and
  voxel size), centroid in scanner-RAS millimetres (CentroidX/Y/Z) and the
  bounding-box extent along each voxel axis in millimetres (BBoxDX/DY/DZ_mm).

Subcortical ROIs are read from mri/aseg.mgz, cortical ROIs from
mri/aparc+aseg.mgz. Intensities come from the first existing volume in
config.INTENSITY_VOLUME_CANDIDATES (norm.mgz by default -- the same volume
FreeSurfer uses for the normMean/normStdDev columns of aseg.stats, so the
two families can be cross-checked against each other).

Column names written to the wide table: ``<ROI>_vox_<Feature>``.
Missing-value records use ``vox_ALL`` for a whole ROI (empty label or
unreadable volume) and ``vox_<Feature>`` / ``vox_intensity`` for individual
features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config
from .models import MissingRecord, Subject

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when nibabel is absent
    nib = None
    NIBABEL_AVAILABLE = False

logger = logging.getLogger("fs_pipeline.voxel_features")

NIBABEL_MISSING_MESSAGE = (
    "Voxel-level features requested but nibabel is not installed. "
    "Run `pip install nibabel`, or pass --no-voxel-features."
)

#: Features that need the intensity volume (everything else is geometry only).
_INTENSITY_FEATURES = ("Mean", "Std", "Min", "Max", "Median", "P5", "P95", "Skewness", "Kurtosis")


def empty_feature_dict() -> Dict[str, Optional[float]]:
    """Every ``<ROI>_vox_<Feature>`` key, all set to None."""
    return {
        f"{roi}_vox_{feat}": None
        for roi in config.ALL_ROI_NAMES
        for feat in config.VOX_FEATURES
    }


def load_volume_3d(path: Path):
    """
    Load an .mgz/.nii volume and return ``(data, zooms, affine)`` with the
    data squeezed to exactly three dimensions (some FreeSurfer volumes carry
    a trailing singleton frame axis).
    """
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim > 3:
        if any(s != 1 for s in data.shape[3:]):
            raise ValueError(f"{path} has more than one frame (shape {data.shape}); expected a 3D volume")
        data = data.reshape(data.shape[:3])
    elif data.ndim < 3:
        raise ValueError(f"{path} is not a 3D volume (shape {data.shape})")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, zooms, img.affine


def _as_label_volume(data: np.ndarray) -> np.ndarray:
    """Segmentation volumes are integer labels; round away any float storage noise."""
    if np.issubdtype(data.dtype, np.integer):
        return data
    return np.rint(data).astype(np.int32)


def _find_intensity_volume(subject: Subject) -> Optional[Path]:
    for name in config.INTENSITY_VOLUME_CANDIDATES:
        candidate = subject.mri_dir / name
        if candidate.is_file():
            return candidate
    return None


def _roi_features(
    mask: np.ndarray,
    intensity: Optional[np.ndarray],
    zooms: Tuple[float, float, float],
    affine: np.ndarray,
) -> Tuple[Dict[str, Optional[float]], List[str]]:
    """
    Compute every VOX_FEATURES value for one non-empty boolean ROI mask.
    Returns (features, names of features that are undefined for this ROI).
    """
    feats: Dict[str, Optional[float]] = {f: None for f in config.VOX_FEATURES}
    undefined: List[str] = []

    idx = np.nonzero(mask)
    n = int(idx[0].size)
    feats["NVoxels"] = float(n)
    feats["Volume_mm3"] = float(n * zooms[0] * zooms[1] * zooms[2])

    centroid_vox = np.array([idx[0].mean(), idx[1].mean(), idx[2].mean(), 1.0])
    centroid_world = affine @ centroid_vox
    feats["CentroidX"], feats["CentroidY"], feats["CentroidZ"] = (float(v) for v in centroid_world[:3])

    for axis, key in enumerate(("BBoxDX_mm", "BBoxDY_mm", "BBoxDZ_mm")):
        extent_vox = int(idx[axis].max() - idx[axis].min() + 1)
        feats[key] = float(extent_vox * zooms[axis])

    if intensity is not None:
        values = intensity[mask].astype(np.float64)
        feats["Mean"] = float(values.mean())
        feats["Std"] = float(values.std())
        feats["Min"] = float(values.min())
        feats["Max"] = float(values.max())
        feats["Median"] = float(np.median(values))
        feats["P5"] = float(np.percentile(values, 5))
        feats["P95"] = float(np.percentile(values, 95))

        centred = values - values.mean()
        m2 = float(np.mean(centred ** 2))
        if m2 > 0.0:
            feats["Skewness"] = float(np.mean(centred ** 3) / m2 ** 1.5)
            feats["Kurtosis"] = float(np.mean(centred ** 4) / m2 ** 2 - 3.0)
        else:
            # Constant intensity inside the ROI: skewness/kurtosis are undefined.
            undefined.extend(["Skewness", "Kurtosis"])

    return feats, undefined


def extract_voxel_features(subject: Subject) -> Tuple[Dict[str, Optional[float]], List[MissingRecord]]:
    """
    Compute the voxel-level feature family for one subject.

    The returned dict always contains every ``<ROI>_vox_<Feature>`` key
    (value None when it could not be computed), so every subject yields the
    same wide-table columns. Anything that could not be computed is recorded
    in the returned MissingRecord list instead of being silently dropped.
    """
    if not NIBABEL_AVAILABLE:
        raise RuntimeError(NIBABEL_MISSING_MESSAGE)

    features = empty_feature_dict()
    missing: List[MissingRecord] = []

    def miss(roi: str, feature: str, reason: str) -> None:
        missing.append(MissingRecord(subject.subject_id, subject.group, roi, feature, reason))

    # --- intensity volume (optional: geometry features work without it) ---
    intensity: Optional[np.ndarray] = None
    intensity_problem = ""
    intensity_path = _find_intensity_volume(subject)
    if intensity_path is None:
        intensity_problem = "no intensity volume (%s) found in mri/" % "/".join(config.INTENSITY_VOLUME_CANDIDATES)
    else:
        try:
            intensity, _, _ = load_volume_3d(intensity_path)
        except Exception as exc:  # noqa: BLE001 - corrupt volume must not stop the subject
            intensity_problem = f"could not read {intensity_path.name}: {exc}"
            logger.error("[%s] %s", subject.subject_id, intensity_problem)

    # --- segmentation volumes, loaded once each ---
    seg_cache: Dict[str, Optional[Tuple[np.ndarray, Tuple[float, float, float], np.ndarray]]] = {}
    seg_problem: Dict[str, str] = {}
    for seg_filename in sorted(set(config.ROI_SOURCE_VOLUME.values())):
        seg_path = subject.mri_dir / seg_filename
        if not seg_path.is_file():
            seg_cache[seg_filename] = None
            seg_problem[seg_filename] = f"missing mri/{seg_filename}"
            continue
        try:
            data, zooms, affine = load_volume_3d(seg_path)
            seg_cache[seg_filename] = (_as_label_volume(data), zooms, affine)
        except Exception as exc:  # noqa: BLE001
            seg_cache[seg_filename] = None
            seg_problem[seg_filename] = f"could not read mri/{seg_filename}: {exc}"
            logger.error("[%s] %s", subject.subject_id, seg_problem[seg_filename])

    intensity_shape_warned = set()

    for roi in config.ALL_ROI_NAMES:
        seg_filename = config.ROI_SOURCE_VOLUME[roi]
        loaded = seg_cache[seg_filename]
        if loaded is None:
            miss(roi, "vox_ALL", seg_problem[seg_filename])
            continue

        seg, zooms, affine = loaded
        mask = seg == config.ALL_SEGIDS[roi]
        if not mask.any():
            miss(roi, "vox_ALL", f"label {config.ALL_SEGIDS[roi]} not present in {seg_filename}")
            continue

        roi_intensity = intensity
        if intensity is not None and intensity.shape != seg.shape:
            if seg_filename not in intensity_shape_warned:
                logger.warning(
                    "[%s] %s shape %s != %s shape %s; intensity features skipped for its ROIs",
                    subject.subject_id, seg_filename, seg.shape, intensity_path.name, intensity.shape,
                )
                intensity_shape_warned.add(seg_filename)
            roi_intensity = None
            reason = f"shape mismatch: {seg_filename} {seg.shape} vs {intensity_path.name} {intensity.shape}"
        else:
            reason = intensity_problem

        roi_feats, undefined = _roi_features(mask, roi_intensity, zooms, affine)
        for feat, value in roi_feats.items():
            features[f"{roi}_vox_{feat}"] = value

        if roi_intensity is None:
            miss(roi, "vox_intensity", reason)
        for feat in undefined:
            miss(roi, f"vox_{feat}", "undefined: zero intensity variance inside ROI")

    return features, missing
