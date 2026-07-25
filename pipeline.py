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
from .segmentation import export_subject_segmentations

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
        return SubjectResult(subject=subject, features={}, error=str(exc))

    if cfg.export_segmentations:
        try:
            result.segmentation = export_subject_segmentations(subject, cfg.output_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Segmentation export failed: %s", subject.subject_id, exc)

    return result


def run_pipeline(cfg: config.PipelineConfig) -> None:
    """Run the full pipeline end-to-end and write every output file."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(cfg.log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

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

    if cfg.workers > 1:
        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {executor.submit(_process_subject, s, cfg): s for s in subjects}
            for future in as_completed(futures):
                subject = futures[future]
                result = future.result()
                group_counters[subject.group] += 1
                qc.log_subject_progress(result, group_counters[subject.group], group_totals[subject.group])
                results.append(result)
                if progress is not None:
                    progress.update(1)
    else:
        for subject in subjects:
            result = _process_subject(subject, cfg)
            group_counters[subject.group] += 1
            qc.log_subject_progress(result, group_counters[subject.group], group_totals[subject.group])
            results.append(result)
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    total_elapsed = time.time() - start

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
        "--data-root", type=Path, default=Path("/home/sai/freesurfer_subjects"),
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
