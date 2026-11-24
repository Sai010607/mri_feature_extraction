"""
fs_pipeline
===========
Publication-quality FreeSurfer ROI morphometric extraction and segmentation
export pipeline for Alzheimer's Disease classification research.

Modules
-------
config            Central configuration: ROI lists, aliases, feature names, paths.
models            Shared dataclasses (Subject, MissingRecord, SegmentationRecord, SubjectResult).
parsing           .stats text-table parsing (ColHeaders + legacy TableCol formats).
validation        Post-parse ROI coverage validation.
voxel_features    Raw voxel/volume-derived features from mri/*.mgz.
extraction        Per-subject orchestration of parsing + validation + voxel features.
segmentation      Per-ROI binary NIfTI mask export.
dataset           Subject discovery and wide/long tabular assembly.
qc                QC report, missing-value report, segmentation summary.
pipeline          CLI entry point and multiprocessing orchestration.
"""

from .pipeline import main

__all__ = ["main"]
