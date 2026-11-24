"""
parsing.py
==========
FreeSurfer .stats text-table parsing, compatible with both the modern
"# ColHeaders" summary line and the legacy per-column "# TableCol N
ColHeader X" directives used by pre-5.x FreeSurfer releases. The format is
auto-detected per file -- no version flag needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import config

logger = logging.getLogger("fs_pipeline.parsing")


def _to_float(value: str) -> Optional[float]:
    """Best-effort string -> float conversion; returns None on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_stats_table(path: Path) -> Tuple[List[str], List[List[str]]]:
    """
    Parse the whitespace-delimited data table embedded in a FreeSurfer
    .stats file, auto-detecting the header format:

    * Modern (FreeSurfer 5.x+): a single summary line
        "# ColHeaders  Index SegId NVoxels ... StructName ..."
    * Legacy (pre-5.x): one directive line per column
        "# TableCol  1 ColHeader Index"
        "# TableCol  2 ColHeader SegId"
        ...

    If both are present (common in modern files, which write TableCol as
    verbose documentation alongside the ColHeaders summary), ColHeaders
    takes precedence since it is guaranteed to be complete and in order.

    Returns (column_headers, rows), where each row is a list of raw string
    tokens aligned with column_headers.

    Raises
    ------
    ValueError
        If neither header format is found in the file.
    """
    col_headers_line: Optional[str] = None
    table_col_entries: Dict[int, str] = {}
    rows: List[List[str]] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("# ColHeaders"):
                col_headers_line = line
                continue
            if line.startswith("# TableCol"):
                # e.g. "# TableCol  6 ColHeader normMean"
                parts = line.split()
                if len(parts) >= 5 and parts[3] == "ColHeader":
                    try:
                        col_index = int(parts[2])
                    except ValueError:
                        continue
                    table_col_entries[col_index] = parts[4]
                continue
            if line.startswith("#"):
                continue
            rows.append(line.split())

    if col_headers_line is not None:
        headers = col_headers_line.split()[2:]
    elif table_col_entries:
        headers = [table_col_entries[i] for i in sorted(table_col_entries)]
        logger.debug("%s: no '# ColHeaders' line found, using legacy '# TableCol' directives", path)
    else:
        raise ValueError(f"No '# ColHeaders' or '# TableCol' header lines found in {path}")

    return headers, rows


def _resolve_feature_indices(
    headers: List[str],
    features: List[str],
    feature_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, int], List[str]]:
    """
    Map each wanted feature name to a column index, falling back to
    feature_aliases (alias_column_name -> preferred_feature_name) when the
    preferred column is absent from this particular file.

    Returns (feature_name -> column_index, list of features not found at all).
    """
    feature_aliases = feature_aliases or {}
    index_by_header = {h: i for i, h in enumerate(headers)}

    resolved: Dict[str, int] = {}
    for feat in features:
        if feat in index_by_header:
            resolved[feat] = index_by_header[feat]

    for alias, preferred in feature_aliases.items():
        if preferred in resolved:
            continue
        if alias in index_by_header:
            resolved[preferred] = index_by_header[alias]

    missing = [feat for feat in features if feat not in resolved]
    return resolved, missing


def _extract_regions(
    path: Path,
    struct_aliases: Dict[str, str],
    features: List[str],
    feature_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Dict[str, Optional[float]]], List[str]]:
    """
    Shared logic for aseg/aparc stats-table parsing.

    Returns
    -------
    regions : {canonical_region_name: {feature_name: value}}
    missing_features : feature names that could not be found anywhere in
        the file (neither the preferred column nor any alias).
    """
    headers, rows = parse_stats_table(path)

    if "StructName" not in headers:
        raise ValueError(f"'StructName' column missing in {path}")
    struct_idx = headers.index("StructName")

    feature_idx, missing_features = _resolve_feature_indices(headers, features, feature_aliases)
    if missing_features:
        logger.warning("%s is missing expected columns: %s", path, missing_features)

    result: Dict[str, Dict[str, Optional[float]]] = {}
    for tokens in rows:
        if struct_idx >= len(tokens):
            continue
        struct_name = tokens[struct_idx]
        canonical = struct_aliases.get(struct_name)
        if canonical is None:
            continue
        result[canonical] = {
            feat: _to_float(tokens[idx]) if idx < len(tokens) else None
            for feat, idx in feature_idx.items()
        }
    return result, missing_features


def parse_aseg(path: Path) -> Tuple[Dict[str, Dict[str, Optional[float]]], List[str]]:
    """Parse aseg.stats and return features for the configured subcortical regions."""
    return _extract_regions(path, config.ASEG_ALIASES, config.ASEG_FEATURES, config.ASEG_FEATURE_ALIASES)


def parse_aparc(path: Path, hemi: str) -> Tuple[Dict[str, Dict[str, Optional[float]]], List[str]]:
    """
    Parse lh.aparc.stats or rh.aparc.stats and return features for the
    configured cortical regions, with canonical names prefixed by hemi
    ('lh' or 'rh') so left/right columns don't collide.
    """
    raw, missing_features = _extract_regions(path, config.APARC_ALIASES, config.APARC_FEATURES)
    return {f"{hemi}_{region}": feats for region, feats in raw.items()}, missing_features
