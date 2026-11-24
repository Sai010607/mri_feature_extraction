"""
segmentation.py
================
Exports each configured ROI as an individual binary NIfTI mask
(0 = background, 1 = ROI), preserving the source volume's affine, voxel
spacing, orientation, and dimensions -- directly compatible with
PyRadiomics, SimpleITK, nibabel, MONAI, and ITK.

Subcortical masks are read from mri/aseg.mgz. Cortical masks (entorhinal,
insula) are read from mri/aparc+aseg.mgz, which is FreeSurfer's own
volumetric conversion of lh/rh.aparc.annot (produced by mri_aparc2aseg
during recon-all) -- this is the authoritative, already-reproducible way to
get cortical parcels into voxel/NIfTI space, and is far more robust than
re-implementing surface-to-volume ribbon projection from the raw .annot
files by hand.

Output layout: <output_dir>/segmented_regions/<Group>/<SubjectID>/<Region>.nii.gz
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import config
from .models import SegmentationRecord, Subject

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when nibabel is absent
    nib = None
    NIBABEL_AVAILABLE = False

logger = logging.getLogger("fs_pipeline.segmentation")

#: ROI internal name -> exported filename stem (subcortical names pass
#: through unchanged; cortical names get the Left_/Right_ friendly form).
EXPORT_NAME: Dict[str, str] = {
    **{name: name for name in config.ASEG_REGIONS},
    **config.CORTICAL_EXPORT_NAMES,
}


def _load_label_volume(path: Path):
    """Load a segmentation volume as (3D integer label array, zooms, affine)."""
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim > 3 and all(n == 1 for n in data.shape[3:]):
        data = data.reshape(data.shape[:3])
    if data.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {data.shape}")
    if not np.issubdtype(data.dtype, np.integer):
        data = np.rint(data).astype(np.int32)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, zooms, img.affine


def export_subject_segmentations(subject: Subject, output_dir: Path) -> List[SegmentationRecord]:
    """
    Write one binary .nii.gz mask per configured ROI for a single subject.

    Returns a SegmentationRecord per ROI (including failures), suitable for
    writing straight to segmentation_summary.csv. A failure on one ROI -- or
    one unreadable source volume -- does not stop the others from being
    exported.

    Status values: OK, EMPTY (label not present), DIM_MISMATCH (the source
    volume's grid differs from mri/aseg.mgz, so the mask would not line up
    voxel-for-voxel with the subcortical masks), ERROR: <reason>.
    """
    records: List[SegmentationRecord] = []
    dest_dir = output_dir / "segmented_regions" / subject.group / subject.subject_id

    if not NIBABEL_AVAILABLE:
        for region in config.ALL_ROI_NAMES:
            records.append(SegmentationRecord(
                subject.subject_id, subject.group, EXPORT_NAME[region], config.ALL_SEGIDS[region],
                0, 0.0, "", "ERROR: nibabel not installed",
            ))
        return records

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Each source volume is loaded once; an unreadable one is remembered as an
    # error string so every ROI that depends on it gets a proper record.
    seg_cache: Dict[str, object] = {}
    reference_shape = None  # grid of mri/aseg.mgz, used for the DIM_MISMATCH check

    for region in config.ALL_ROI_NAMES:
        export_name = EXPORT_NAME[region]
        seg_filename = config.ROI_SOURCE_VOLUME[region]
        seg_id = config.ALL_SEGIDS[region]
        mask_path = dest_dir / f"{export_name}.nii.gz"

        if seg_filename not in seg_cache:
            seg_path = subject.mri_dir / seg_filename
            if not seg_path.is_file():
                seg_cache[seg_filename] = f"ERROR: missing {seg_filename}"
            else:
                try:
                    seg_cache[seg_filename] = _load_label_volume(seg_path)
                except Exception as exc:  # noqa: BLE001 - corrupt volume must not abort the subject
                    logger.error("[%s] Cannot read %s: %s", subject.subject_id, seg_path, exc)
                    seg_cache[seg_filename] = f"ERROR: cannot read {seg_filename}: {exc}"
            if reference_shape is None and seg_filename == "aseg.mgz" and not isinstance(seg_cache[seg_filename], str):
                reference_shape = seg_cache[seg_filename][0].shape

        loaded = seg_cache[seg_filename]
        if isinstance(loaded, str):
            records.append(SegmentationRecord(
                subject.subject_id, subject.group, export_name, seg_id, 0, 0.0, "", loaded,
            ))
            continue

        try:
            seg_data, zooms, affine = loaded
            mask = (seg_data == seg_id).astype(np.uint8)
            voxel_count = int(mask.sum())
            volume_mm3 = voxel_count * float(zooms[0] * zooms[1] * zooms[2])

            # Build a clean NIfTI header rather than copying the MGH one: same
            # affine, voxel size and grid, uint8 data, scanner-RAS qform/sform.
            mask_img = nib.Nifti1Image(mask, affine)
            mask_img.header.set_data_dtype(np.uint8)
            mask_img.header.set_zooms(zooms)
            mask_img.header.set_xyzt_units("mm")
            mask_img.set_qform(affine, code=1)
            mask_img.set_sform(affine, code=1)
            nib.save(mask_img, str(mask_path))

            if voxel_count == 0:
                status = "EMPTY"
            elif reference_shape is not None and mask.shape != reference_shape:
                status = "DIM_MISMATCH"
            else:
                status = "OK"

            records.append(SegmentationRecord(
                subject.subject_id, subject.group, export_name, seg_id,
                voxel_count, volume_mm3, str(mask_path), status,
            ))
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Failed to export %s: %s", subject.subject_id, export_name, exc)
            records.append(SegmentationRecord(
                subject.subject_id, subject.group, export_name, seg_id, 0, 0.0, "", f"ERROR: {exc}",
            ))

    return records
