"""
dataset.py
==========
Subject discovery and dataset assembly: builds both the wide (one row per
subject) and long/tidy (one row per subject x region x feature) output
tables, and merges optional clinical metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from . import config
from .models import Subject, SubjectResult

logger = logging.getLogger("fs_pipeline.dataset")


def discover_subjects(data_root: Path, groups: Optional[List[str]]) -> List[Subject]:
    """
    Walk data_root and return one Subject per subdirectory that contains a
    'stats' folder (filters out stray non-subject entries such as
    fsaverage). Group folders are auto-discovered unless `groups` restricts
    processing to a specific subset -- no subject or group names are ever
    hard-coded.
    """
    if groups:
        group_dirs = [data_root / g for g in groups]
    else:
        group_dirs = sorted(p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith("."))

    subjects: List[Subject] = []
    for group_dir in group_dirs:
        if not group_dir.is_dir():
            logger.warning("Group directory not found, skipping: %s", group_dir)
            continue
        for subject_dir in sorted(p for p in group_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            if (subject_dir / "stats").is_dir():
                subjects.append(Subject(subject_dir.name, group_dir.name, subject_dir))
            else:
                logger.debug("Skipping '%s': no stats/ subdirectory", subject_dir)
    return subjects


def build_wide_dataframe(results: List[SubjectResult]) -> pd.DataFrame:
    """One row per subject; every ROI x feature combination becomes one column."""
    records = []
    for res in results:
        if res.error is not None:
            continue
        record = {"SubjectID": res.subject.subject_id, "Group": res.subject.group}
        record.update(res.features)
        for col in config.METADATA_COLUMNS:
            record.setdefault(col, None)
        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    id_cols = ["SubjectID", "Group"] + config.METADATA_COLUMNS
    feature_cols = sorted(c for c in df.columns if c not in id_cols)
    df = df[id_cols + feature_cols]
    return df.sort_values(["Group", "SubjectID"]).reset_index(drop=True)


def _split_region_feature(column: str) -> Tuple[str, str]:
    """
    Split a wide-format column name like "CC_Mid_Posterior_vox_Volume_mm3"
    back into ("CC_Mid_Posterior", "vox_Volume_mm3"). Matches against the
    known feature-name suffix list (longest first) rather than a naive
    rpartition("_"), since several feature names (Volume_mm3, BBoxDX_mm,
    vox_Volume_mm3, ...) contain underscores themselves.
    """
    for suffix in config.FEATURE_NAME_SUFFIXES:
        marker = f"_{suffix}"
        if column.endswith(marker):
            return column[: -len(marker)], suffix
    return column, ""


def build_long_dataframe(results: List[SubjectResult]) -> pd.DataFrame:
    """Tidy long-format table: one row per (subject, region, feature)."""
    rows = []
    for res in results:
        if res.error is not None:
            continue
        for column, value in res.features.items():
            region, feature = _split_region_feature(column)
            hemisphere = ""
            if region.startswith("Left") or region.startswith("lh_"):
                hemisphere = "Left"
            elif region.startswith("Right") or region.startswith("rh_"):
                hemisphere = "Right"
            rows.append({
                "SubjectID": res.subject.subject_id,
                "Group": res.subject.group,
                "Hemisphere": hemisphere,
                "Region": region,
                "Feature": feature,
                "Value": value,
            })

    if not rows:
        return pd.DataFrame(columns=["SubjectID", "Group", "Hemisphere", "Region", "Feature", "Value"])
    return pd.DataFrame.from_records(rows)


def merge_metadata(df: pd.DataFrame, metadata_csv: Optional[Path]) -> pd.DataFrame:
    """
    Left-join optional clinical metadata (Age, Sex, MMSE, Diagnosis, Scanner,
    AcquisitionDate) onto the wide feature table by SubjectID.

    If metadata_csv is None or unreadable, the metadata columns are left as
    the empty placeholders already added by build_wide_dataframe -- this
    call is always safe to make even when no metadata is available yet.
    """
    if metadata_csv is None or df.empty:
        return df
    if not metadata_csv.is_file():
        logger.warning("Metadata file not found, skipping merge: %s", metadata_csv)
        return df

    try:
        meta = pd.read_csv(metadata_csv, dtype={"SubjectID": str})
    except Exception as exc:  # noqa: BLE001 - a bad clinical file must never block extraction
        logger.warning("Could not read metadata file %s (%s), skipping merge", metadata_csv, exc)
        return df
    if "SubjectID" not in meta.columns:
        logger.warning("Metadata file has no 'SubjectID' column, skipping merge: %s", metadata_csv)
        return df

    meta["SubjectID"] = meta["SubjectID"].astype(str).str.strip()
    if meta["SubjectID"].duplicated().any():
        dupes = meta.loc[meta["SubjectID"].duplicated(), "SubjectID"].unique().tolist()
        logger.warning("Metadata has duplicate SubjectIDs (keeping the first row of each): %s", dupes[:10])
        meta = meta.drop_duplicates("SubjectID", keep="first")

    supplied = [c for c in config.METADATA_COLUMNS if c in meta.columns]
    if not supplied:
        logger.warning("Metadata file has none of the expected columns %s", config.METADATA_COLUMNS)
        return df

    # Replace only the placeholder columns the CSV actually supplies; the rest
    # stay as empty placeholders so the output schema never changes.
    n_unmatched = int((~df["SubjectID"].isin(meta["SubjectID"])).sum())
    df = df.drop(columns=supplied).merge(meta[["SubjectID"] + supplied], on="SubjectID", how="left")
    id_cols = ["SubjectID", "Group"] + config.METADATA_COLUMNS
    feature_cols = [c for c in df.columns if c not in id_cols]
    for col in config.METADATA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    if n_unmatched:
        logger.info("%d subject(s) had no matching row in the metadata file", n_unmatched)
    return df[id_cols + feature_cols]
