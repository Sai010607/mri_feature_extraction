"""
qc.py
=====
Quality-control reporting: per-subject progress logging, aggregate run
statistics (qc_report.txt), missing-value export (missing_values.csv), and
segmentation summary export (segmentation_summary.csv).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from .models import SegmentationRecord, SubjectResult

logger = logging.getLogger("fs_pipeline.qc")


def log_subject_progress(res: SubjectResult, group_index: int, group_total: int) -> None:
    """Print the per-subject progress block used for interactive/log monitoring."""
    status = "OK" if res.error is None else f"FAILED: {res.error}"
    n_missing_rois = sum(1 for m in res.missing if m.feature in ("ALL", "vox_ALL"))
    logger.info("-" * 50)
    logger.info("Processing Group : %s", res.subject.group)
    logger.info("Subject %d / %d : %s", group_index, group_total, res.subject.subject_id)
    logger.info("Extracted ROIs : %d / %d", res.n_rois_found, res.n_rois_expected)
    logger.info("Missing ROIs : %d", n_missing_rois)
    logger.info("Status : %s", status)
    logger.info("-" * 50)


def write_missing_values(results: List[SubjectResult], path: Path) -> pd.DataFrame:
    """Write missing_values.csv: SubjectID, Group, ROI, Feature, Reason."""
    rows = [
        {"SubjectID": m.subject_id, "Group": m.group, "ROI": m.roi, "Feature": m.feature, "Reason": m.reason}
        for res in results for m in res.missing
    ]
    df = pd.DataFrame(rows, columns=["SubjectID", "Group", "ROI", "Feature", "Reason"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def write_segmentation_summary(records: List[SegmentationRecord], path: Path) -> pd.DataFrame:
    """Write segmentation_summary.csv."""
    rows = [
        {
            "SubjectID": r.subject_id, "Group": r.group, "Region": r.region,
            "ROI_Label": r.roi_label, "Voxel_Count": r.voxel_count,
            "Volume_mm3": r.volume_mm3, "Mask_File": r.mask_file, "Status": r.status,
        }
        for r in records
    ]
    df = pd.DataFrame(rows, columns=[
        "SubjectID", "Group", "Region", "ROI_Label", "Voxel_Count", "Volume_mm3", "Mask_File", "Status",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def generate_qc_report(
    results: List[SubjectResult],
    missing_df: pd.DataFrame,
    total_elapsed_seconds: float,
    path: Path,
) -> None:
    """Write qc_report.txt summarizing the whole run."""
    n_subjects = len(results)
    n_success = sum(1 for r in results if r.error is None)
    n_failed = n_subjects - n_success
    groups = sorted({r.subject.group for r in results})

    if not missing_df.empty:
        roi_level_mask = missing_df["Feature"].isin(["ALL", "vox_ALL"])
        missing_rois = sorted(missing_df.loc[roi_level_mask, "ROI"].unique())
        missing_features = sorted(missing_df.loc[~roi_level_mask, "Feature"].unique())
    else:
        missing_rois, missing_features = [], []

    avg_time = (total_elapsed_seconds / n_subjects) if n_subjects else 0.0

    lines = [
        "FreeSurfer AD Morphometric Extraction -- QC Report",
        "=" * 55,
        f"Number of Groups            : {len(groups)} ({', '.join(groups)})",
        f"Number of Subjects           : {n_subjects}",
        f"Successful Subjects          : {n_success}",
        f"Failed Subjects              : {n_failed}",
        f"Missing ROIs (any subject)   : {len(missing_rois)}",
        f"  -> {', '.join(missing_rois) if missing_rois else 'none'}",
        f"Missing Features (any subj.) : {len(missing_features)}",
        f"  -> {', '.join(missing_features) if missing_features else 'none'}",
        f"Total Processing Time        : {total_elapsed_seconds:.2f} s",
        f"Average Time per Subject     : {avg_time:.2f} s",
        "",
        "Failed subjects detail:",
    ]
    for r in results:
        if r.error is not None:
            lines.append(f"  - {r.subject.group}/{r.subject.subject_id}: {r.error}")
    if n_failed == 0:
        lines.append("  (none)")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    logger.info("QC report written to %s", path)
