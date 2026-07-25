"""
validation.py
=============
Post-parse validation: confirms every expected ROI was actually found in a
subject's stats files, and records anything missing (whole ROI or a single
feature column) for the QC / missing-values report. Processing always
continues past a validation failure -- this module only records, it never
raises.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import config
from .models import MissingRecord, Subject


def validate_subject(
    subject: Subject,
    aseg_data: Dict[str, Dict[str, Optional[float]]],
    aparc_data: Dict[str, Dict[str, Optional[float]]],
    aseg_missing_features: List[str],
    aparc_missing_features: Dict[str, List[str]],
) -> Tuple[List[MissingRecord], int, int]:
    """
    Compare parsed ROI data against the full expected ROI list.

    Parameters
    ----------
    aseg_data, aparc_data
        Parsed per-region feature dicts, as returned by parsing.parse_aseg /
        parsing.parse_aparc.
    aseg_missing_features
        Feature columns absent from this subject's aseg.stats (applies to
        every ROI found in the file).
    aparc_missing_features
        {hemi: [feature, ...]} -- columns absent from lh/rh.aparc.stats.

    Returns
    -------
    missing : list of MissingRecord (one row per missing ROI or feature)
    n_expected : total number of ROIs expected (subcortical + cortical)
    n_found : number of ROIs actually present in the parsed data
    """
    missing: List[MissingRecord] = []
    found = 0

    for region in config.ASEG_REGIONS:
        if region in aseg_data:
            found += 1
            for feat in aseg_missing_features:
                missing.append(MissingRecord(
                    subject.subject_id, subject.group, region, feat,
                    "column absent from aseg.stats",
                ))
        else:
            missing.append(MissingRecord(
                subject.subject_id, subject.group, region, "ALL",
                "StructName not found in aseg.stats",
            ))

    for hemi in ("lh", "rh"):
        hemi_missing_feats = aparc_missing_features.get(hemi, [])
        for base_name in config.APARC_REGION_BASENAMES:
            region = f"{hemi}_{base_name}"
            if region in aparc_data:
                found += 1
                for feat in hemi_missing_feats:
                    missing.append(MissingRecord(
                        subject.subject_id, subject.group, region, feat,
                        f"column absent from {hemi}.aparc.stats",
                    ))
            else:
                missing.append(MissingRecord(
                    subject.subject_id, subject.group, region, "ALL",
                    f"StructName not found in {hemi}.aparc.stats",
                ))

    n_expected = len(config.ALL_ROI_NAMES)
    return missing, n_expected, found
