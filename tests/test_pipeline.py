import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fs_pipeline import config
from fs_pipeline.dataset import (
    build_long_dataframe, build_wide_dataframe, discover_subjects, merge_metadata,
)
from fs_pipeline.extraction import extract_subject
from fs_pipeline.models import Subject
from fs_pipeline.parsing import parse_aparc, parse_aseg
from fs_pipeline.pipeline import main
from fs_pipeline.segmentation import export_subject_segmentations

from .synthetic import AFFINE, SHAPE, ZOOMS, make_subject


def _subject(root: Path, group="AD", sid="S001", **kw) -> Subject:
    path = make_subject(root, group, sid, **kw)
    return Subject(sid, group, path)


# --------------------------------------------------------------------------- parsing
def test_modern_and_legacy_stats_parse_identically(tmp_path):
    modern = make_subject(tmp_path / "a", "G", "S", legacy_stats=False)
    legacy = make_subject(tmp_path / "b", "G", "S", legacy_stats=True)
    m_aseg, m_missing = parse_aseg(modern / "stats" / "aseg.stats")
    l_aseg, l_missing = parse_aseg(legacy / "stats" / "aseg.stats")
    assert m_aseg == l_aseg and m_missing == l_missing == []
    assert set(m_aseg) == set(config.ASEG_REGIONS)  # 19 subcortical ROIs, FS7 thalamus names aliased
    m_ap, _ = parse_aparc(modern / "stats" / "lh.aparc.stats", "lh")
    l_ap, _ = parse_aparc(legacy / "stats" / "lh.aparc.stats", "lh")
    assert m_ap == l_ap and set(m_ap) == {"lh_Entorhinal", "lh_Insula"}
    assert m_ap["lh_Entorhinal"]["ThickAvg"] == pytest.approx(3.1)


# ---------------------------------------------------------------- extraction + voxel
def test_stats_and_voxel_families_agree(tmp_path):
    res = extract_subject(_subject(tmp_path))
    assert (res.n_rois_found, res.n_rois_expected) == (23, 23)
    assert res.missing == []
    for roi in config.ASEG_REGIONS:
        f = res.features
        assert f[f"{roi}_vox_NVoxels"] == f[f"{roi}_NVoxels"]
        assert f[f"{roi}_vox_Volume_mm3"] == pytest.approx(f[f"{roi}_Volume_mm3"], abs=0.06)
        assert f[f"{roi}_vox_Mean"] == pytest.approx(f[f"{roi}_normMean"], abs=1e-3)
        assert f[f"{roi}_vox_Std"] == pytest.approx(f[f"{roi}_normStdDev"], abs=1e-3)
        assert f[f"{roi}_vox_Min"] == pytest.approx(f[f"{roi}_normMin"], abs=1e-3)
    # anisotropic voxels (1x1x2 mm): volume must use all three zooms
    assert res.features["Brain_Stem_vox_Volume_mm3"] == res.features["Brain_Stem_vox_NVoxels"] * 2.0


def test_voxel_stats_geometry_and_moments(tmp_path):
    sub = _subject(tmp_path)
    res = extract_subject(sub)
    aseg = np.asanyarray(nib.load(str(sub.mri_dir / "aseg.mgz")).dataobj)
    norm = np.asanyarray(nib.load(str(sub.mri_dir / "norm.mgz")).dataobj).astype(np.float64)
    roi = "Left_Hippocampus"
    m = aseg == config.ASEG_SEGIDS[roi]
    v = norm[m]
    ijk = np.argwhere(m)
    world = (AFFINE @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3].mean(axis=0)
    f = res.features
    assert [f[f"{roi}_vox_Centroid{a}"] for a in "XYZ"] == pytest.approx(list(world))
    ext = ijk.max(0) - ijk.min(0) + 1
    assert [f[f"{roi}_vox_BBoxD{a}_mm"] for a in "XYZ"] == pytest.approx(list(ext * np.array(ZOOMS)))
    assert f[f"{roi}_vox_Median"] == pytest.approx(np.median(v))
    assert f[f"{roi}_vox_P95"] == pytest.approx(np.percentile(v, 95))
    stats = pytest.importorskip("scipy.stats")
    assert f[f"{roi}_vox_Skewness"] == pytest.approx(stats.skew(v))
    assert f[f"{roi}_vox_Kurtosis"] == pytest.approx(stats.kurtosis(v))  # excess kurtosis


def test_cortical_roi_voxel_features_come_from_aparc_aseg(tmp_path):
    res = extract_subject(_subject(tmp_path))
    for roi in ("lh_Entorhinal", "rh_Entorhinal", "lh_Insula", "rh_Insula"):
        assert res.features[f"{roi}_vox_NVoxels"] > 0
        assert res.features[f"{roi}_ThickAvg"] is not None


def test_missing_roi_and_missing_file_reasons_are_recorded(tmp_path):
    res = extract_subject(_subject(tmp_path, drop_aseg_rows={"Left_Amygdala"}))
    miss = [(m.roi, m.feature, m.reason) for m in res.missing]
    assert miss == [("Left_Amygdala", "ALL", "StructName not found in aseg.stats")]
    assert res.n_rois_found == 22 and res.features["Left_Amygdala_Volume_mm3"] is None

    sub = _subject(tmp_path, sid="S002")
    (sub.stats_dir / "aseg.stats").unlink()
    res = extract_subject(sub)
    reasons = {m.reason for m in res.missing if m.roi in config.ASEG_REGIONS and m.feature == "ALL"}
    assert reasons == {"aseg.stats file missing"}


def test_voxel_intensity_shape_mismatch_keeps_geometry(tmp_path):
    res = extract_subject(_subject(tmp_path, wrong_intensity_shape=True))
    assert res.features["Brain_Stem_vox_Mean"] is None
    assert res.features["Brain_Stem_vox_Volume_mm3"] is not None
    assert any(m.feature == "vox_intensity" and "shape mismatch" in m.reason for m in res.missing)


def test_no_volumes_gives_consistent_columns_and_vox_all_records(tmp_path):
    ok = extract_subject(_subject(tmp_path, sid="A"))
    bad = extract_subject(_subject(tmp_path, sid="B", write_volumes=False))
    assert set(ok.features) == set(bad.features)  # identical wide-table schema for every subject
    assert sum(1 for m in bad.missing if m.feature == "vox_ALL") == 23


# ------------------------------------------------------------------ segmentation
def test_exported_masks_preserve_geometry_and_counts(tmp_path):
    sub = _subject(tmp_path / "data")
    out = tmp_path / "out"
    recs = {r.region: r for r in export_subject_segmentations(sub, out)}
    assert len(recs) == 23 and {r.status for r in recs.values()} == {"OK"}
    assert "Left_Entorhinal" in recs  # friendly cortical export names
    img = nib.load(recs["Left_Hippocampus"].mask_file)
    assert img.shape == SHAPE and img.get_data_dtype() == np.uint8
    assert np.allclose(img.affine, AFFINE) and img.header.get_zooms()[:3] == pytest.approx(ZOOMS)
    data = np.asanyarray(img.dataobj)
    assert set(np.unique(data)) == {0, 1}
    assert int(data.sum()) == recs["Left_Hippocampus"].voxel_count
    assert recs["Left_Hippocampus"].volume_mm3 == recs["Left_Hippocampus"].voxel_count * 2.0


def test_segmentation_reports_missing_and_dim_mismatch(tmp_path):
    sub = _subject(tmp_path / "d1", write_volumes=False)
    recs = export_subject_segmentations(sub, tmp_path / "o1")
    assert all(r.status.startswith("ERROR: missing") for r in recs)

    sub = _subject(tmp_path / "d2")
    small = np.zeros((10, 10, 10), dtype=np.int32)
    small[2:4, 2:4, 2:4] = config.APARC_SEGIDS["lh_Insula"]
    nib.save(nib.MGHImage(small, AFFINE), str(sub.mri_dir / "aparc+aseg.mgz"))
    recs = {r.region: r for r in export_subject_segmentations(sub, tmp_path / "o2")}
    assert recs["Left_Insula"].status == "DIM_MISMATCH"
    assert recs["Left_Hippocampus"].status == "OK"

    (sub.mri_dir / "aseg.mgz").write_bytes(b"not a real volume")  # corrupt -> only its ROIs fail
    recs = {r.region: r for r in export_subject_segmentations(sub, tmp_path / "o3")}
    assert recs["Left_Hippocampus"].status.startswith("ERROR: cannot read aseg.mgz")
    assert recs["Left_Insula"].voxel_count == 8


# ----------------------------------------------------------------------- dataset
def test_discovery_skips_hidden_and_non_subject_dirs(tmp_path):
    make_subject(tmp_path, "AD", "S1")
    (tmp_path / "AD" / "fsaverage").mkdir()
    (tmp_path / "AD" / ".git").mkdir()
    (tmp_path / ".hidden" / "S9" / "stats").mkdir(parents=True)
    subs = discover_subjects(tmp_path, None)
    assert [(s.group, s.subject_id) for s in subs] == [("AD", "S1")]


def test_wide_and_long_tables(tmp_path):
    results = [extract_subject(_subject(tmp_path, sid=f"S{i}", seed=i)) for i in range(3)]
    wide = build_wide_dataframe(results)
    assert list(wide.columns[:2 + len(config.METADATA_COLUMNS)]) == ["SubjectID", "Group"] + config.METADATA_COLUMNS
    assert len(wide) == 3
    long = build_long_dataframe(results)
    assert len(long) == 3 * len([c for c in wide.columns if c not in ["SubjectID", "Group"] + config.METADATA_COLUMNS])
    row = long[(long.Region == "CC_Mid_Posterior") & (long.Feature == "vox_Volume_mm3")]
    assert len(row) == 3 and (row.Hemisphere == "").all()
    row = long[(long.Region == "lh_Entorhinal") & (long.Feature == "ThickAvg")]
    assert (row.Hemisphere == "Left").all()
    assert (long.Feature != "").all()  # every column splits into a known feature


def test_merge_metadata_keeps_schema_and_handles_messy_input(tmp_path):
    results = [extract_subject(_subject(tmp_path, sid=s, seed=i)) for i, s in enumerate(["001", "002"])]
    wide = build_wide_dataframe(results)
    cols = list(wide.columns)

    csv = tmp_path / "meta.csv"
    csv.write_text("SubjectID,Age,MMSE\n001,71,28\n001,99,1\n003,80,20\n")  # numeric-looking IDs + duplicate + extra
    merged = merge_metadata(wide, csv)
    assert list(merged.columns) == cols                                    # schema/order unchanged
    assert len(merged) == 2                                                # duplicate ID did not add rows
    assert merged.loc[merged.SubjectID == "001", "Age"].item() == 71      # first duplicate wins, str/int IDs matched
    assert merged.loc[merged.SubjectID == "002", "Age"].isna().item()
    assert merged["Sex"].isna().all()                                      # unsupplied placeholder survives

    csv.write_text("SubjectID,Unrelated\n001,x\n")
    assert list(merge_metadata(wide, csv).columns) == cols                 # placeholders NOT dropped
    csv.write_text("garbage,,\"unterminated")
    assert list(merge_metadata(wide, csv).columns) == cols                 # unreadable file never blocks


# --------------------------------------------------------------------- end to end
def _run(data, out, *extra):
    main(["--data-root", str(data), "--output-dir", str(out), *extra])


def test_end_to_end_outputs_and_worker_determinism(tmp_path):
    data = tmp_path / "data"
    for g, ids in {"AD": ["S2", "S1"], "CN": ["S3"]}.items():
        for i, sid in enumerate(ids):
            make_subject(data, g, sid, seed=i + len(g))
    make_subject(data, "CN", "S4", write_stats=False, write_volumes=False)  # incomplete subject

    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    _run(data, out1, "--workers", "1")
    _run(data, out2, "--workers", "2")

    for name in ["master_features_wide.csv", "master_features_long.csv", "missing_values.csv",
                 "segmentation_summary.csv", "qc_report.txt", "processing.log"]:
        assert (out1 / name).is_file(), name
    for name in ["master_features_wide.csv", "master_features_long.csv", "missing_values.csv"]:
        assert (out1 / name).read_text() == (out2 / name).read_text(), f"{name} differs with --workers 2"

    wide = pd.read_csv(out1 / "master_features_wide.csv")
    assert list(wide.SubjectID) == ["S1", "S2", "S3", "S4"]  # sorted by Group then SubjectID
    seg = pd.read_csv(out1 / "segmentation_summary.csv")
    assert len(seg) == 4 * 23 and (out1 / "segmented_regions" / "AD" / "S1" / "Left_Hippocampus.nii.gz").is_file()
    qc = (out1 / "qc_report.txt").read_text()
    assert "Number of Subjects           : 4" in qc and "Missing ROIs (any subject)   : 23" in qc


def test_cli_flags_and_errors(tmp_path, caplog):
    data = tmp_path / "data"
    make_subject(data, "AD", "S1")
    _run(data, tmp_path / "o", "--no-voxel-features", "--no-segmentations", "--groups", "AD")
    wide = pd.read_csv(tmp_path / "o" / "master_features_wide.csv")
    assert not any("_vox_" in c for c in wide.columns)
    assert not (tmp_path / "o" / "segmented_regions").exists()

    with pytest.raises(SystemExit):           # --data-root is required (no hard-coded personal default)
        main(["--output-dir", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        _run(tmp_path / "does_not_exist", tmp_path / "x")
    with pytest.raises(SystemExit):
        _run(data, tmp_path / "x", "--workers", "0")
    with pytest.raises(SystemExit):           # nothing discoverable
        _run(data, tmp_path / "x", "--groups", "NOPE")


def test_repeated_runs_do_not_duplicate_log_lines(tmp_path):
    data = tmp_path / "data"
    make_subject(data, "AD", "S1")
    before = len(logging.getLogger().handlers)
    _run(data, tmp_path / "o1", "--no-segmentations")
    _run(data, tmp_path / "o2", "--no-segmentations")
    assert len(logging.getLogger().handlers) == before
