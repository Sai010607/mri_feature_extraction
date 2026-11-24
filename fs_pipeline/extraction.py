"""
extraction.py
=============
Per-subject orchestration: parse the stats-table files, validate ROI
coverage, and (optionally) compute voxel-level features from the
segmentation volumes. A failure in one subject never raises past this
function -- it is captured on SubjectResult.error and the run continues.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from . import config
from .models import MissingRecord, Subject, SubjectResult
from .parsing import parse_aparc, parse_aseg
from .validation import validate_subject
from .voxel_features import empty_feature_dict, extract_voxel_features

logger = logging.getLogger("fs_pipeline.extraction")


def extract_subject(subject: Subject, compute_voxel_features: bool = True) -> SubjectResult:
    """
    Extract every configured stats-table feature (and, optionally, every
    voxel-level feature) for one subject, validating ROI coverage along the
    way.

    Parameters
    ----------
    subject
        The subject to process.
    compute_voxel_features
        If True (default), also compute the nibabel-based voxel-level
        feature family from mri/aseg.mgz + mri/aparc+aseg.mgz.
    """
    start = time.time()
    features: Dict[str, Optional[float]] = {}
    missing: List[MissingRecord] = []

    # --- Subcortical (aseg.stats) ---
    aseg_path = subject.stats_dir / "aseg.stats"
    aseg_data: Dict[str, Dict[str, Optional[float]]] = {}
    aseg_missing_features: List[str] = []
    stats_issues: Dict[str, str] = {}
    if aseg_path.is_file():
        try:
            aseg_data, aseg_missing_features = parse_aseg(aseg_path)
        except Exception as exc:
            logger.error("[%s] Failed to parse %s: %s", subject.subject_id, aseg_path, exc)
            stats_issues["aseg"] = f"aseg.stats could not be parsed: {exc}"
    else:
        logger.warning("[%s] Missing aseg.stats", subject.subject_id)
        stats_issues["aseg"] = "aseg.stats file missing"

    for region in config.ASEG_REGIONS:
        region_feats = aseg_data.get(region, {})
        for feat in config.ASEG_FEATURES:
            features[f"{region}_{feat}"] = region_feats.get(feat)

    # --- Cortical (lh/rh.aparc.stats) ---
    aparc_data: Dict[str, Dict[str, Optional[float]]] = {}
    aparc_missing_features: Dict[str, List[str]] = {}
    for hemi in ("lh", "rh"):
        aparc_path = subject.stats_dir / f"{hemi}.aparc.stats"
        if aparc_path.is_file():
            try:
                hemi_data, hemi_missing = parse_aparc(aparc_path, hemi)
                aparc_data.update(hemi_data)
                aparc_missing_features[hemi] = hemi_missing
            except Exception as exc:
                logger.error("[%s] Failed to parse %s: %s", subject.subject_id, aparc_path, exc)
                stats_issues[hemi] = f"{hemi}.aparc.stats could not be parsed: {exc}"
        else:
            logger.warning("[%s] Missing %s.aparc.stats", subject.subject_id, hemi)
            stats_issues[hemi] = f"{hemi}.aparc.stats file missing"

    for hemi in ("lh", "rh"):
        for base_name in config.APARC_REGION_BASENAMES:
            canonical = f"{hemi}_{base_name}"
            region_feats = aparc_data.get(canonical, {})
            for feat in config.APARC_FEATURES:
                features[f"{canonical}_{feat}"] = region_feats.get(feat)

    # --- ROI coverage validation ---
    subject_missing, n_expected, n_found = validate_subject(
        subject, aseg_data, aparc_data, aseg_missing_features, aparc_missing_features, stats_issues,
    )
    missing.extend(subject_missing)

    # --- Voxel-level (.mgz) features ---
    if compute_voxel_features:
        try:
            vox_feats, vox_missing = extract_voxel_features(subject)
            features.update(vox_feats)
            missing.extend(vox_missing)
        except Exception as exc:
            # Keep the wide-table columns consistent across subjects and make
            # the failure visible in missing_values.csv instead of only in the log.
            logger.error("[%s] Voxel-level feature extraction failed: %s", subject.subject_id, exc)
            features.update(empty_feature_dict())
            missing.extend(
                MissingRecord(subject.subject_id, subject.group, roi, "vox_ALL",
                              f"voxel feature extraction failed: {exc}")
                for roi in config.ALL_ROI_NAMES
            )

    elapsed = time.time() - start
    return SubjectResult(
        subject=subject,
        features=features,
        missing=missing,
        n_rois_expected=n_expected,
        n_rois_found=n_found,
        elapsed_seconds=elapsed,
    )
