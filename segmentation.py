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


def export_subject_segmentations(subject: Subject, output_dir: Path) -> List[SegmentationRecord]:
    """
    Write one binary .nii.gz mask per configured ROI for a single subject.

    Returns a SegmentationRecord per ROI (including failures), suitable for
    writing straight to segmentation_summary.csv. A failure on one ROI does
    not stop the others from being exported.
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
    seg_cache: Dict[str, object] = {}

    for region in config.ALL_ROI_NAMES:
        export_name = EXPORT_NAME[region]
        seg_filename = config.ROI_SOURCE_VOLUME[region]
        seg_id = config.ALL_SEGIDS[region]
        mask_path = dest_dir / f"{export_name}.nii.gz"

        if seg_filename not in seg_cache:
            seg_path = subject.mri_dir / seg_filename
            seg_cache[seg_filename] = nib.load(str(seg_path)) if seg_path.is_file() else None

        seg_img = seg_cache[seg_filename]
        if seg_img is None:
            records.append(SegmentationRecord(
                subject.subject_id, subject.group, export_name, seg_id, 0, 0.0,
                "", f"ERROR: missing {seg_filename}",
            ))
            continue

        try:
            seg_data = seg_img.get_fdata()
            mask = (seg_data == seg_id).astype(np.uint8)
            voxel_count = int(mask.sum())
            zooms = seg_img.header.get_zooms()[:3]
            voxel_vol_mm3 = float(zooms[0] * zooms[1] * zooms[2])
            volume_mm3 = voxel_count * voxel_vol_mm3

            mask_img = nib.Nifti1Image(mask, affine=seg_img.affine, header=seg_img.header.copy())
            mask_img.header.set_data_dtype(np.uint8)
            mask_img.set_qform(seg_img.affine)
            mask_img.set_sform(seg_img.affine)
            nib.save(mask_img, str(mask_path))

            dims_match = mask.shape == seg_img.shape
            if voxel_count == 0:
                status = "EMPTY"
            elif not dims_match:
                status = "DIM_MISMATCH"
            else:
                status = "OK"

            records.append(SegmentationRecord(
                subject.subject_id, subject.group, export_name, seg_id,
                voxel_count, volume_mm3, str(mask_path), status,
            ))
        except Exception as exc:
            logger.error("[%s] Failed to export %s: %s", subject.subject_id, export_name, exc)
            records.append(SegmentationRecord(
                subject.subject_id, subject.group, export_name, seg_id, 0, 0.0, "", f"ERROR: {exc}",
            ))

    return records
