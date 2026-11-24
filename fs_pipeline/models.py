"""
models.py
=========
Lightweight dataclasses shared across the pipeline modules. Keeping these
separate from config.py (pure constants) and the processing modules avoids
circular imports and gives every module a single, typed vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Subject:
    """One FreeSurfer subject directory discovered under a diagnostic group."""

    subject_id: str
    group: str
    path: Path

    @property
    def mri_dir(self) -> Path:
        return self.path / "mri"

    @property
    def stats_dir(self) -> Path:
        return self.path / "stats"

    @property
    def label_dir(self) -> Path:
        return self.path / "label"


@dataclass
class MissingRecord:
    """One missing ROI or feature value, for missing_values.csv."""

    subject_id: str
    group: str
    roi: str
    feature: str
    reason: str


@dataclass
class SegmentationRecord:
    """One exported ROI mask, for segmentation_summary.csv."""

    subject_id: str
    group: str
    region: str
    roi_label: int
    voxel_count: int
    volume_mm3: float
    mask_file: str
    status: str


@dataclass
class SubjectResult:
    """Everything produced by processing one subject."""

    subject: Subject
    features: Dict[str, Optional[float]]
    missing: List[MissingRecord] = field(default_factory=list)
    segmentation: List[SegmentationRecord] = field(default_factory=list)
    n_rois_expected: int = 0
    n_rois_found: int = 0
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
