"""
config.py
=========
Central configuration for the FreeSurfer AD morphometric extraction pipeline.

Every ROI list, alias table, feature list, and output path lives here so the
parsing / extraction / segmentation modules never hard-code domain constants.
Extend the dictionaries below (not the parsing logic) when a new FreeSurfer
naming variant or ROI is discovered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Subcortical ROIs (aseg.stats / mri/aseg.mgz)
# ---------------------------------------------------------------------------

#: Canonical ROI name -> FreeSurferColorLUT segmentation ID (mri/aseg.mgz).
ASEG_SEGIDS: Dict[str, int] = {
    "Left_Hippocampus": 17,
    "Right_Hippocampus": 53,
    "Left_Amygdala": 18,
    "Right_Amygdala": 54,
    "Left_Putamen": 12,
    "Right_Putamen": 51,
    "Left_Thalamus_Proper": 10,
    "Right_Thalamus_Proper": 49,
    "Brain_Stem": 16,
    "Left_Lateral_Ventricle": 4,
    "Right_Lateral_Ventricle": 43,
    "Third_Ventricle": 14,
    "CC_Posterior": 251,
    "CC_Mid_Posterior": 252,
    "CC_Central": 253,
    "CC_Mid_Anterior": 254,
    "CC_Anterior": 255,
    "Left_Cerebellum_Cortex": 8,
    "Right_Cerebellum_Cortex": 47,
}

#: Every raw StructName string that might appear in aseg.stats across
#: FreeSurfer releases, mapped to one canonical ROI name.
ASEG_ALIASES: Dict[str, str] = {
    "Left-Hippocampus": "Left_Hippocampus",
    "Right-Hippocampus": "Right_Hippocampus",
    "Left-Amygdala": "Left_Amygdala",
    "Right-Amygdala": "Right_Amygdala",
    "Left-Putamen": "Left_Putamen",
    "Right-Putamen": "Right_Putamen",
    "Left-Thalamus-Proper": "Left_Thalamus_Proper",
    "Left-Thalamus": "Left_Thalamus_Proper",           # FreeSurfer 7+
    "Right-Thalamus-Proper": "Right_Thalamus_Proper",
    "Right-Thalamus": "Right_Thalamus_Proper",          # FreeSurfer 7+
    "Brain-Stem": "Brain_Stem",
    "Brainstem": "Brain_Stem",                           # seen in some ADNI reprocessing
    "Left-Lateral-Ventricle": "Left_Lateral_Ventricle",
    "Right-Lateral-Ventricle": "Right_Lateral_Ventricle",
    "3rd-Ventricle": "Third_Ventricle",
    "Third-Ventricle": "Third_Ventricle",
    "CC_Posterior": "CC_Posterior",
    "CC-Posterior": "CC_Posterior",
    "CC_Mid_Posterior": "CC_Mid_Posterior",
    "CC-Mid-Posterior": "CC_Mid_Posterior",
    "CC_Central": "CC_Central",
    "CC-Central": "CC_Central",
    "CC_Mid_Anterior": "CC_Mid_Anterior",
    "CC-Mid-Anterior": "CC_Mid_Anterior",
    "CC_Anterior": "CC_Anterior",
    "CC-Anterior": "CC_Anterior",
    "Left-Cerebellum-Cortex": "Left_Cerebellum_Cortex",
    "Right-Cerebellum-Cortex": "Right_Cerebellum_Cortex",
}

#: Ordered list of canonical subcortical ROI names (defines column/report order).
ASEG_REGIONS: List[str] = list(ASEG_SEGIDS.keys())

#: aseg.stats feature columns to extract (modern FreeSurfer 5.x+ naming).
ASEG_FEATURES: List[str] = [
    "NVoxels", "Volume_mm3",
    "normMean", "normStdDev", "normMin", "normMax", "normRange",
]

#: fallback_column -> preferred_column. If the preferred column is missing
#: from a given stats file but the fallback column is present, its value is
#: used instead and stored under the preferred feature name. Some very old
#: or manually re-exported stats files have been seen using bare
#: Mean/StdDev/Min/Max/Range instead of the norm-prefixed names.
ASEG_FEATURE_ALIASES: Dict[str, str] = {
    "Mean": "normMean",
    "StdDev": "normStdDev",
    "Min": "normMin",
    "Max": "normMax",
    "Range": "normRange",
}

# ---------------------------------------------------------------------------
# Cortical ROIs (lh/rh.aparc.stats, mri/aparc+aseg.mgz)
# ---------------------------------------------------------------------------

#: aparc.stats StructName values are lowercase and hemisphere-agnostic; the
#: hemisphere is implied by which file (lh/rh.aparc.stats) the row came from.
APARC_ALIASES: Dict[str, str] = {
    "entorhinal": "Entorhinal",
    "insula": "Insula",
}
APARC_REGION_BASENAMES: List[str] = ["Entorhinal", "Insula"]

APARC_FEATURES: List[str] = [
    "NumVert", "SurfArea", "GrayVol", "ThickAvg", "ThickStd",
    "MeanCurv", "GausCurv", "FoldInd", "CurvInd",
]

#: Desikan-Killiany volumetric parcellation IDs (mri/aparc+aseg.mgz), used
#: for both voxel-level feature extraction and NIfTI mask export of cortical
#: ROIs. ctx-lh-* = 1000 + atlas index, ctx-rh-* = 2000 + atlas index;
#: entorhinal = index 6, insula = index 35 in the standard DK atlas -- these
#: IDs have been stable across every FreeSurfer release since the DK atlas
#: was introduced (FreeSurfer 4.x onward).
APARC_SEGIDS: Dict[str, int] = {
    "lh_Entorhinal": 1006,
    "rh_Entorhinal": 2006,
    "lh_Insula": 1035,
    "rh_Insula": 2035,
}

#: Human-friendly export names for the NIfTI segmentation masks (spec asks
#: for e.g. "Left_Entorhinal.nii.gz" rather than "lh_Entorhinal.nii.gz").
CORTICAL_EXPORT_NAMES: Dict[str, str] = {
    "lh_Entorhinal": "Left_Entorhinal",
    "rh_Entorhinal": "Right_Entorhinal",
    "lh_Insula": "Left_Insula",
    "rh_Insula": "Right_Insula",
}

#: All ROI names, combined subcortical + cortical, in report/output order.
ALL_ROI_NAMES: List[str] = ASEG_REGIONS + ["lh_Entorhinal", "rh_Entorhinal", "lh_Insula", "rh_Insula"]

#: Segmentation ID lookup across BOTH subcortical + cortical ROIs.
ALL_SEGIDS: Dict[str, int] = {**ASEG_SEGIDS, **APARC_SEGIDS}

#: Which mri/*.mgz volume a given ROI's mask should be read from. Cortical
#: ROIs use mri/aparc+aseg.mgz, FreeSurfer's own volumetric conversion of
#: lh/rh.aparc.annot (produced by mri_aparc2aseg during recon-all) -- this is
#: the authoritative way to get cortical parcels in voxel/NIfTI space.
ROI_SOURCE_VOLUME: Dict[str, str] = {
    **{name: "aseg.mgz" for name in ASEG_SEGIDS},
    **{name: "aparc+aseg.mgz" for name in APARC_SEGIDS},
}

# ---------------------------------------------------------------------------
# Voxel-level feature configuration
# ---------------------------------------------------------------------------

#: Intensity volumes to use for voxel-level statistics, in preference order.
INTENSITY_VOLUME_CANDIDATES: List[str] = ["norm.mgz", "brain.mgz", "T1.mgz"]

#: Features computed directly from the segmentation + intensity volumes,
#: independent of the .stats text files (see voxel_features.py).
VOX_FEATURES: List[str] = [
    "NVoxels", "Volume_mm3", "Mean", "Std", "Min", "Max", "Median", "P5", "P95",
    "Skewness", "Kurtosis", "CentroidX", "CentroidY", "CentroidZ",
    "BBoxDX_mm", "BBoxDY_mm", "BBoxDZ_mm",
]

#: Every possible "<Feature>" column-name suffix across both feature
#: families, longest-first, used by dataset.py to split a wide-format column
#: name like "CC_Mid_Posterior_vox_Volume_mm3" back into (region, feature)
#: for the long-format table. Sorting longest-first ensures multi-token
#: names (e.g. "vox_Volume_mm3") are matched before shorter substrings of
#: themselves (e.g. "Volume_mm3", "mm3").
_ALL_FEATURE_NAMES = list(dict.fromkeys(
    ASEG_FEATURES + APARC_FEATURES + [f"vox_{f}" for f in VOX_FEATURES]
))
FEATURE_NAME_SUFFIXES: List[str] = sorted(_ALL_FEATURE_NAMES, key=len, reverse=True)

# ---------------------------------------------------------------------------
# Optional metadata columns
#
# Populated later by dataset.merge_metadata() from a user-supplied CSV.
# Never required for the core extraction to run -- present as empty
# placeholder columns in master_features_wide.csv until supplied.
# ---------------------------------------------------------------------------

METADATA_COLUMNS: List[str] = ["Age", "Sex", "MMSE", "Diagnosis", "Scanner", "AcquisitionDate"]

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Runtime configuration for a single pipeline run."""

    data_root: Path
    output_dir: Path = Path("fs_pipeline_output")
    groups: Optional[List[str]] = None
    workers: int = 1
    export_segmentations: bool = True
    compute_voxel_features: bool = True
    metadata_csv: Optional[Path] = None
    log_level: str = "INFO"

    @property
    def wide_csv(self) -> Path:
        return self.output_dir / "master_features_wide.csv"

    @property
    def long_csv(self) -> Path:
        return self.output_dir / "master_features_long.csv"

    @property
    def qc_report(self) -> Path:
        return self.output_dir / "qc_report.txt"

    @property
    def missing_values_csv(self) -> Path:
        return self.output_dir / "missing_values.csv"

    @property
    def segmentation_summary_csv(self) -> Path:
        return self.output_dir / "segmentation_summary.csv"

    @property
    def segmented_regions_dir(self) -> Path:
        return self.output_dir / "segmented_regions"

    @property
    def log_file(self) -> Path:
        return self.output_dir / "processing.log"
