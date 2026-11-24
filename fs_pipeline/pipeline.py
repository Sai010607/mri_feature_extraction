"""
pipeline.py
===========
CLI entry point and orchestration for the FreeSurfer AD morphometric
extraction pipeline. Wires together: subject discovery -> per-subject
extraction (stats-table + voxel-level) -> ROI validation -> segmentation
mask export -> wide/long dataset assembly -> QC / missing-value / metadata
reporting.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from . import config, qc
from .dataset import build_long_dataframe, build_wide_dataframe, discover_subjects, merge_metadata
from .extraction import extract_subject
from .models import Subject, SubjectResult
from .segmentation import NIBABEL_AVAILABLE, export_subject_segmentations
from .voxel_features import NIBABEL_MISSING_MESSAGE

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when tqdm is absent
    TQDM_AVAILABLE = False

logger = logging.getLogger("fs_pipeline")


def _process_subject(subject: Subject, cfg: config.PipelineConfig) -> SubjectResult:
    """Extract features (and optionally export segmentations) for one subject."""
    try:
        result = extract_subject(subject, compute_voxel_features=cfg.compute_voxel_features)
    except Exception as exc:  # noqa: BLE001 - one bad subject must not kill the run
        logger.error("[%s] Unhandled extraction failure: %s", subject.subject_id, exc)
        return SubjectResult(
            subject=subject, features={}, error=str(exc), n_rois_expected=len(config.ALL_ROI_NAMES),
        )

    if cfg.export_segmentations:
        try:
            result.segmentation = export_subject_segmentations(subject, cfg.output_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Segmentation export failed: %s", subject.subject_id, exc)

    return result


def _collect(future, subject: Subject) -> SubjectResult:
    """Get a worker's result; a crashed worker becomes a failed subject, not a dead run."""
    try:
        return future.result()
    except Exception as exc:  # noqa: BLE001 - e.g. BrokenProcessPool, unpicklable result
        logger.error("[%s] Worker crashed: %s", subject.subject_id, exc)
        return SubjectResult(
            subject=subject, features={}, error=f"worker crashed: {exc}",
            n_rois_expected=len(config.ALL_ROI_NAMES),
        )


def run_pipeline(cfg: config.PipelineConfig) -> None:
    """Run the full pipeline end-to-end and write every output file."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(cfg.log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    try:
        _run(cfg)
    finally:
        # Without this, calling run_pipeline twice in one process logs every line twice.
        root_logger.removeHandler(file_handler)
        file_handler.close()


def _run(cfg: config.PipelineConfig) -> None:
    subjects = discover_subjects(cfg.data_root, cfg.groups)
    groups_present = sorted({s.group for s in subjects})
    logger.info(
        "Discovered %d subject(s) across %d group(s): %s",
        len(subjects), len(groups_present), ", ".join(groups_present),
    )

    if not subjects:
        logger.error("No subjects found under %s -- nothing to do.", cfg.data_root)
        sys.exit(1)

    group_totals = {g: sum(1 for s in subjects if s.group == g) for g in groups_present}
    group_counters = {g: 0 for g in groups_present}
    results: List[SubjectResult] = []
    start = time.time()

    progress = tqdm(total=len(subjects), desc="Extracting subjects") if TQDM_AVAILABLE else None

    def record(result: SubjectResult) -> None:
        group = result.subject.group
        group_counters[group] += 1
        qc.log_subject_progress(result, group_counters[group], group_totals[group])
        results.append(result)
        if progress is not None:
            progress.update(1)

    if cfg.workers > 1:
        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {executor.submit(_process_subject, s, cfg): s for s in subjects}
            for future in as_completed(futures):
                record(_collect(future, futures[future]))
    else:
        for subject in subjects:
            record(_process_subject(subject, cfg))

    if progress is not None:
        progress.close()

    total_elapsed = time.time() - start

    # Parallel runs finish in arbitrary order; sort so every output file is
    # identical between runs (reproducibility) regardless of --workers.
    results.sort(key=lambda r: (r.subject.group, r.subject.subject_id))

    # --- Tabular outputs ---
    wide_df = build_wide_dataframe(results)
    wide_df = merge_metadata(wide_df, cfg.metadata_csv)
    wide_df.to_csv(cfg.wide_csv, index=False)
    if wide_df.empty:
        logger.warning("Wide feature table is empty; nothing usable written to %s", cfg.wide_csv)
    else:
        logger.info("Wrote %s (%d subjects x %d columns)", cfg.wide_csv, wide_df.shape[0], wide_df.shape[1])

    long_df = build_long_dataframe(results)
    long_df.to_csv(cfg.long_csv, index=False)
    logger.info("Wrote %s (%d rows)", cfg.long_csv, len(long_df))

    # --- QC / missing values ---
    missing_df = qc.write_missing_values(results, cfg.missing_values_csv)
    logger.info("Wrote %s (%d rows)", cfg.missing_values_csv, len(missing_df))

    # --- Segmentation summary ---
    if cfg.export_segmentations:
        all_seg_records = [rec for res in results for rec in res.segmentation]
        seg_df = qc.write_segmentation_summary(all_seg_records, cfg.segmentation_summary_csv)
        logger.info("Wrote %s (%d rows)", cfg.segmentation_summary_csv, len(seg_df))

    qc.generate_qc_report(results, missing_df, total_elapsed, cfg.qc_report)
    logger.info(
        "Pipeline complete in %.2f s (%.2f s/subject average).",
        total_elapsed, total_elapsed / len(subjects),
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publication-quality FreeSurfer ROI feature + segmentation extraction for AD classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Root directory containing one subfolder per diagnostic group.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("fs_pipeline_output"),
        help="Directory for all pipeline outputs (CSV/txt reports + segmented_regions/).",
    )
    parser.add_argument(
        "--groups", nargs="+", default=None,
        help="Specific group subfolder names to process (default: auto-discover all).",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes.")
    parser.add_argument(
        "--no-voxel-features", action="store_true",
        help="Skip raw voxel-level features from mri/*.mgz (requires nibabel); use stats-table features only.",
    )
    parser.add_argument(
        "--no-segmentations", action="store_true",
        help="Skip exporting individual ROI NIfTI masks.",
    )
    parser.add_argument(
        "--metadata-csv", type=Path, default=None,
        help="Optional CSV with a SubjectID column plus any of: " + ", ".join(config.METADATA_COLUMNS),
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.data_root.is_dir():
        logger.error("Data root does not exist or is not a directory: %s", args.data_root)
        sys.exit(1)
    if args.workers < 1:
        logger.error("--workers must be >= 1 (got %d)", args.workers)
        sys.exit(1)
    if not NIBABEL_AVAILABLE and not (args.no_voxel_features and args.no_segmentations):
        # Fail now instead of silently producing a run full of ERROR rows.
        if not args.no_voxel_features:
            logger.error(NIBABEL_MISSING_MESSAGE)
            sys.exit(1)
        logger.error(
            "Segmentation export requires nibabel, which is not installed. "
            "Run `pip install nibabel`, or pass --no-segmentations."
        )
        sys.exit(1)

    cfg = config.PipelineConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        groups=args.groups,
        workers=args.workers,
        export_segmentations=not args.no_segmentations,
        compute_voxel_features=not args.no_voxel_features,
        metadata_csv=args.metadata_csv,
        log_level=args.log_level,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
