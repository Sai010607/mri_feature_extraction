# FreeSurfer AD Morphometric Extraction Pipeline (`fs_pipeline`)

Publication-quality, modular extraction of subcortical/cortical ROI morphometry and
per-ROI NIfTI segmentation masks from FreeSurfer subject outputs, for Alzheimer's
Disease classification and downstream radiomics (PyRadiomics / MONAI / SimpleITK).

## 1. Project Overview

Given a directory of FreeSurfer `recon-all` outputs organized by diagnostic group, the
pipeline:

1. Auto-discovers every group folder and every subject (no hard-coded names).
2. Parses `aseg.stats`, `lh.aparc.stats`, `rh.aparc.stats` for a fixed set of 23 ROIs,
   supporting both the modern `# ColHeaders` format and the legacy `# TableCol`
   per-column format used by pre-5.x FreeSurfer.
3. Computes a second, independent feature family directly from the segmentation
   volumes (`mri/aseg.mgz`, `mri/aparc+aseg.mgz`) via `nibabel` — intensity
   distribution (mean/std/median/percentiles/skewness/kurtosis) and shape descriptors
   (recomputed volume, centroid, bounding box).
4. Validates that every expected ROI was actually found, logging and recording
   anything missing instead of silently producing `NaN`.
5. Exports every ROI as an individual binary `.nii.gz` mask, ready for PyRadiomics /
   MONAI / SimpleITK pipelines.
6. Writes a wide (one row per subject) table, a long/tidy table, a QC report, a
   missing-values report, and a segmentation summary.

## 2. Folder Structure

```
mri_feature_extraction/          # this repository
├── fs_pipeline/                 # the package
│   ├── __init__.py
│   ├── config.py                 # ROI lists, aliases, feature names, PipelineConfig
│   ├── models.py                 # Subject, MissingRecord, SegmentationRecord, SubjectResult
│   ├── parsing.py                # .stats parsing (ColHeaders + legacy TableCol)
│   ├── validation.py             # ROI coverage validation
│   ├── voxel_features.py         # nibabel-based voxel-level features
│   ├── extraction.py             # per-subject orchestration
│   ├── segmentation.py           # NIfTI ROI mask export
│   ├── dataset.py                # subject discovery + wide/long assembly
│   ├── qc.py                     # QC / missing-value / segmentation reports
│   ├── pipeline.py               # CLI entry point, multiprocessing
│   └── __main__.py               # enables `python3 -m fs_pipeline ...`
├── tests/                        # pytest suite (runs on small synthetic FreeSurfer subjects)
├── index.html                    # static results dashboard (see section 11)
├── run_pipeline.py               # `python3 run_pipeline.py ...`
├── requirements.txt
├── requirements-dev.txt          # adds pytest
├── .gitignore                    # keeps imaging data and outputs out of git
└── README.md
```

Your FreeSurfer subject data (`SMCI/ EMCI/ MCI/ LMCI/ AD_Preprocessed/ ...`) lives wherever you
like and is passed with `--data-root`; it is never part of this repository.

Input layout expected under `--data-root` (any group names, auto-discovered):

```
<data_root>/<Group>/<Subject>/mri/, stats/, surf/, label/
```

## 3. Installation

```bash
git clone https://github.com/Sai010607/mri_feature_extraction.git
cd mri_feature_extraction
python3 -m venv .venv && source .venv/bin/activate        # optional but recommended
pip install -r requirements.txt
```

Core dependencies (`pandas`, `numpy`) are required always. `nibabel` is required for
voxel-level features and segmentation export (both on by default; disable with
`--no-voxel-features` / `--no-segmentations` if you only want the stats-table numbers —
if `nibabel` is missing and neither is disabled, the run stops immediately with a clear
message instead of producing a table full of error rows). `tqdm` is optional — progress
bars are skipped gracefully if it isn't installed.

## 4. Requirements

* Python 3.9+
* pandas, numpy
* nibabel (only needed for voxel features / segmentation export)
* tqdm (optional, progress bars only)
* pytest (only to run the test suite, see `requirements-dev.txt`)

All of the above except pytest are listed in `requirements.txt`.

## 5. Usage

```bash
python3 run_pipeline.py \
    --data-root /home/sai/freesurfer_subjects \
    --groups SMCI EMCI MCI LMCI AD_Preprocessed \
    --output-dir output \
    --workers 4
```

`python3 -m fs_pipeline ...` is equivalent. `--data-root` is required (there is no built-in
default path).

Common flags:

| Flag | Meaning |
|---|---|
| `--data-root PATH` | **Required.** Root folder containing one subfolder per diagnostic group. |
| `--groups A B C` | Restrict to specific group folders (default: every subfolder). |
| `--output-dir PATH` | Where all outputs are written (default `fs_pipeline_output/`). |
| `--workers N` | Parallel subject processing via `ProcessPoolExecutor` (output files are identical for any `N`). |
| `--no-voxel-features` | Skip the nibabel-based voxel feature family. |
| `--no-segmentations` | Skip NIfTI mask export. |
| `--metadata-csv PATH` | Optional CSV (`SubjectID` + any of `Age, Sex, MMSE, Diagnosis, Scanner, AcquisitionDate`) left-joined onto the wide table. |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. |

Excluding non-diagnostic folders (e.g. a `dump/` staging folder, or `fsaverage`) is
done the same way — either don't include them in `--groups`, or make sure they don't
contain a `stats/` subdirectory (subjects are only picked up if `stats/` exists).

## 6. Expected Outputs

Written to `--output-dir` (default `fs_pipeline_output/`):

| File | Contents |
|---|---|
| `master_features_wide.csv` | One row per subject. Columns: `SubjectID, Group`, metadata placeholders, then `<Region>_<Feature>` (stats-table) and `<Region>_vox_<Feature>` (voxel-level) for every ROI. |
| `master_features_long.csv` | Tidy format: `SubjectID, Group, Hemisphere, Region, Feature, Value` — one row per value. |
| `qc_report.txt` | Groups/subjects counts, success/failure counts, every ROI/feature that was missing for at least one subject, total and average processing time, and a per-subject failure list. |
| `missing_values.csv` | `SubjectID, Group, ROI, Feature, Reason` — every individual missing value, with `Feature="ALL"` (or `"vox_ALL"`) meaning the whole ROI was absent rather than a single column. |
| `segmentation_summary.csv` | `SubjectID, Group, Region, ROI_Label, Voxel_Count, Volume_mm3, Mask_File, Status` for every exported mask. |
| `processing.log` | Full log of the run (mirrors console output). |
| `segmented_regions/<Group>/<SubjectID>/<Region>.nii.gz` | One binary mask per ROI per subject. |

**Voxel-level feature definitions** (`<Region>_vox_<Feature>`): `NVoxels`, `Volume_mm3`
(voxel count × voxel volume), intensity `Mean / Std / Min / Max / Median / P5 / P95`
(`Std` is the population standard deviation, ddof=0), `Skewness`, `Kurtosis` (excess/Fisher,
so a normal distribution is 0), `CentroidX/Y/Z` (scanner-RAS millimetres) and
`BBoxDX/DY/DZ_mm` (bounding-box extent along each voxel axis, in mm). Intensities are read
from the first existing volume among `norm.mgz`, `brain.mgz`, `T1.mgz`; with `norm.mgz`,
`vox_Mean` reproduces the `normMean` column of `aseg.stats`, which is a handy sanity check.
Skewness/Kurtosis are recorded in `missing_values.csv` (not silently `NaN`) for the rare ROI
whose intensities are all identical.

**Column naming note:** stats-table columns keep FreeSurfer's own names exactly
(`Volume_mm3`, `normMean`, ...) for traceability back to the source `.stats` files,
rather than being renamed to generic labels like `"Volume"`.

## 7. ROIs Extracted (23 total)

Subcortical (19, from `aseg.stats` / `aseg.mgz`): Left/Right Hippocampus, Left/Right
Amygdala, Left/Right Putamen, Left/Right Thalamus Proper, Brain Stem, Left/Right
Lateral Ventricle, Third Ventricle, CC_Posterior, CC_Mid_Posterior, CC_Central,
CC_Mid_Anterior, CC_Anterior, Left/Right Cerebellum Cortex.

Cortical (4, from `lh/rh.aparc.stats` / `aparc+aseg.mgz`): Left/Right Entorhinal,
Left/Right Insula.

## 8. Version Compatibility Notes

* **Stats parsing** auto-detects `# ColHeaders` (FreeSurfer 5.x+) vs. legacy
  `# TableCol N ColHeader X` directives (pre-5.x) — no version flag needed.
* **Region aliases**: `Brain-Stem`/`Brainstem`, `Left-Thalamus`/`Left-Thalamus-Proper`
  (and Right-), `3rd-Ventricle`/`Third-Ventricle`, and both `_`/`-` corpus callosum
  spellings all map to one canonical name. Extend `config.ASEG_ALIASES` if you
  encounter another variant.
* **Feature aliases**: if a stats file uses bare `Mean/StdDev/Min/Max/Range` instead
  of `normMean/normStdDev/normMin/normMax/normRange`, the bare names are used as a
  fallback automatically (`config.ASEG_FEATURE_ALIASES`).
* **Cortical masks** are read from `mri/aparc+aseg.mgz` (FreeSurfer's own volumetric
  parcellation, generated by `mri_aparc2aseg` from `lh/rh.aparc.annot` during
  `recon-all`) rather than re-deriving a surface-to-volume ribbon projection from the
  raw `.annot` files by hand — this is both the standard, reproducible source for
  cortical ROI masks and avoids reimplementing FreeSurfer's own (non-trivial)
  ribbon-mapping algorithm.

## 9. Metadata Merging

`master_features_wide.csv` always includes empty `Age, Sex, MMSE, Diagnosis, Scanner,
AcquisitionDate` columns. Pass `--metadata-csv clinical.csv` (with a `SubjectID`
column matching your FreeSurfer subject folder names) to populate them via a left
join — this is entirely optional and never blocks extraction.

## 10. Troubleshooting

* **"Voxel-level features requested but nibabel is not installed"** — run
  `pip install nibabel --break-system-packages`, or pass `--no-voxel-features`.
* **A subject shows up with all-`None` features** — check `missing_values.csv` for
  that `SubjectID`; usually means `recon-all` didn't finish (missing `stats/` or
  `mri/aseg.mgz`).
* **Unexpected extra "groups" in the output** (e.g. `fsaverage`, a scratch/staging
  folder) — pass `--groups` explicitly to whitelist only your real diagnostic
  cohorts, or ensure stray folders don't contain a `stats/` subdirectory.
* **Multiprocessing and logging** — with `--workers > 1` on Linux (fork start
  method), per-subject progress logging still comes from the main process and is
  reliable; on macOS/Windows (spawn start method) some worker-side warning/error
  logs may not reach `processing.log`. Re-run with `--workers 1` if you need to
  debug a specific subject's warnings.
* **Volume shape mismatch** — if the intensity volume's grid differs from the
  segmentation's, intensity features for the affected ROIs are skipped and recorded in
  `missing_values.csv` as `vox_intensity` (geometry features are still written). If
  `aparc+aseg.mgz` has a different grid from `aseg.mgz`, its masks are exported but flagged
  `DIM_MISMATCH` in `segmentation_summary.csv`. A corrupt or unreadable `.mgz` only affects
  the ROIs that depend on it. Nothing crashes the whole run.
* **`Status` column in `segmentation_summary.csv`** — `OK`, `EMPTY` (label not present in the
  volume), `DIM_MISMATCH`, or `ERROR: <reason>`.

## 11. Results Dashboard

`index.html` (repository root) is a single-file, static web dashboard for browsing the pipeline's
outputs. Open it directly in a browser (double-click it, or `open index.html`) —
no server, no build step, no install needed.

Load `master_features_wide.csv`, `missing_values.csv`, and `qc_report.txt` through the file
pickers at the top of the page. Everything runs client-side in your browser; nothing is
uploaded anywhere, and none of your data is stored in this repository (see `.gitignore`).

It shows: total subjects and groups, how many subjects have complete vs. missing data, a
subjects-per-group chart, a region+feature comparison chart across groups (e.g. average
hippocampus volume per diagnostic group), a searchable missing-values table, the raw QC
report text, and a per-subject table for whichever region you select.

It needs an internet connection the first time it loads, because it pulls PapaParse and
Chart.js from a CDN; your data still never leaves the browser.

To publish it as a public webpage via GitHub Pages: in the repo's Settings → Pages, set the
source to the `main` branch and `/ (root)` folder. Because the page only ever reads files the
visitor picks from their own computer, publishing it exposes no data.

## 12. Reproducibility Notes

* No subject or group name is ever hard-coded — everything is discovered from the
  filesystem at run time.
* Every ROI, alias, feature name, and output path lives in `fs_pipeline/config.py`;
  no parsing function contains inline constants.
* All segmentation ID assumptions (`config.ASEG_SEGIDS`, `config.APARC_SEGIDS`) use
  the standard FreeSurferColorLUT values, stable across FreeSurfer releases.
* Exported masks preserve the source volume's affine, voxel spacing, orientation, and
  dimensions exactly (copied from `mri/aseg.mgz` / `mri/aparc+aseg.mgz`'s header),
  so radiomics features computed later line up voxel-for-voxel with the original MRI.

## 13. Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests -q
```

The suite builds small synthetic FreeSurfer subjects on the fly (both `# ColHeaders` and
legacy `# TableCol` stats files, `.mgz` volumes with anisotropic voxels and a non-trivial
affine) and checks, among other things, that the stats-table and voxel-level feature families
agree, that exported masks keep the source geometry, that missing ROIs/files/volumes are
reported with a reason instead of crashing, and that `--workers 1` and `--workers 2` write
byte-identical CSVs. No real patient data is needed or included.
