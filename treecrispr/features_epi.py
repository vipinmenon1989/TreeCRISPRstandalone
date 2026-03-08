"""
treecrispr/features_epi.py — Coordinate-aware epigenetic feature assembly.

For each guide-RNA candidate, this module:

1. Parses genomic coordinates embedded in the FASTA record ID (e.g.
   ``chr1:1,234,567-1,235,067``).
2. Locates BigWig files in :data:`~treecrispr.config.BIGWIG_DIR` by matching
   them against the expected track names in
   :data:`~treecrispr.config.EXPECTED_BIGWIGS`.
3. Calls :func:`~treecrispr.epi_seq.single_interval_features` to extract
   per-track signal at multiple window extensions.

If coordinates cannot be parsed, or BigWig files are absent, a zero-filled
feature dict is returned so that the scoring models can still run on sequence
features alone.

Public API
----------
epigenetic_features(row, logger)
    Main entry point called by the pipeline for every candidate row.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import (
    BIGWIG_DIR,
    EPIGENETIC_EXTENSIONS,
    EPIG_AGGREGATION,
    EXPECTED_BIGWIGS,
)
from .epi_seq import single_interval_features

# ---------------------------------------------------------------------------
# Coordinate parsing
# ---------------------------------------------------------------------------

_COORD_RE = re.compile(
    r"(chr[\w]+|\b[0-9XYM]+)\s*:\s*([0-9,]+)\s*-\s*([0-9,]+)",
    re.IGNORECASE,
)


def _parse_id_region(id_str: str) -> Optional[Tuple[str, int, int]]:
    """
    Extract ``(chrom, start_0based, end_0based)`` from a FASTA record ID.

    Coordinates in the ID are assumed to be 1-based closed; this function
    converts the start to 0-based half-open (BED convention).  Returns
    ``None`` if no recognisable coordinate pattern is found.
    """
    m = _COORD_RE.search(id_str)
    if not m:
        return None
    chrom = m.group(1)
    if not chrom.lower().startswith("chr"):
        chrom = "chr" + chrom
    start = int(m.group(2).replace(",", ""))
    end   = int(m.group(3).replace(",", ""))
    if start > 0:
        start -= 1  # 1-based → 0-based
    return chrom.lower(), start, end


# ---------------------------------------------------------------------------
# BigWig file discovery (cached to avoid repeated disk reads)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _map_files_to_expected_names() -> Dict[str, Path]:
    """
    Return a mapping of expected track name → Path for files found in BIGWIG_DIR.

    The lookup is case-insensitive and matches by filename prefix.  Results are
    cached after the first call so the directory is only scanned once per
    process, regardless of how many candidates are scored.
    """
    mapping: Dict[str, Path] = {}
    if not BIGWIG_DIR.exists():
        return mapping

    disk_files: Dict[str, Path] = {
        p.name.lower(): p for p in BIGWIG_DIR.iterdir() if p.is_file()
    }

    for expected in EXPECTED_BIGWIGS:
        candidates: List[Path] = [
            f for name, f in disk_files.items()
            if name.startswith(expected.lower())
        ]
        if candidates:
            # Prefer the shortest filename to avoid accidentally matching
            # a more-specific file when multiple candidates exist.
            candidates.sort(key=lambda p: len(p.name))
            mapping[expected] = candidates[0]

    return mapping


# ---------------------------------------------------------------------------
# Zero-valued feature template (stable column layout)
# ---------------------------------------------------------------------------

def _predeclare_zero_feats() -> Dict[str, float]:
    """
    Return a dict with every expected epigenetic feature column set to 0.0.

    This guarantees that the XGBoost models always receive the full feature
    matrix even when BigWig files are absent or coordinate parsing fails.
    """
    return {
        f"{base}_{int(ext)}": 0.0
        for base in EXPECTED_BIGWIGS
        for ext in EPIGENETIC_EXTENSIONS
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def epigenetic_features(row: pd.Series, logger=None) -> Dict[str, float]:
    """
    Extract epigenetic features for a single guide-RNA candidate.

    Parameters
    ----------
    row : pd.Series
        A row from the candidates DataFrame.  Must contain at least the
        columns ``"ID"``, ``"Start"``, and ``"End"``.
    logger : logging.Logger, optional
        If provided, warnings about missing coordinates or BigWig errors
        are emitted at ``WARNING`` level.

    Returns
    -------
    Dict mapping ``"{track}_{ext}"`` → float.  All expected columns are
    always present (defaulting to 0.0 on failure).
    """
    # Always return the full expected column layout.
    feats = _predeclare_zero_feats()

    # Step 1: parse genomic coordinates from the FASTA record ID.
    parsed = _parse_id_region(str(row.get("ID", "")))
    if not parsed:
        return feats

    chrom, abs_start_input, _ = parsed

    try:
        off_start = int(row.get("Start"))
        off_end   = int(row.get("End"))
    except (TypeError, ValueError):
        return feats

    abs_start = abs_start_input + off_start
    abs_end   = abs_start_input + off_end

    # Step 2: find BigWig files and overwrite the zero defaults.
    file_map = _map_files_to_expected_names()
    if not file_map:
        return feats

    try:
        found_paths = list(file_map.values())
        raw_vals = single_interval_features(
            bw_paths=found_paths,
            chrom=chrom,
            start0=abs_start,
            end0=abs_end,
            extensions=EPIGENETIC_EXTENSIONS,
            agg=EPIG_AGGREGATION,
        )

        # Map raw keys ("{file_stem}_{ext}") → model keys ("{expected_name}_{ext}").
        for expected_name, file_path in file_map.items():
            file_stem = file_path.stem
            for ext in EPIGENETIC_EXTENSIONS:
                raw_key   = f"{file_stem}_{int(ext)}"
                model_key = f"{expected_name}_{int(ext)}"
                if raw_key in raw_vals:
                    feats[model_key] = float(raw_vals[raw_key])

    except Exception as exc:
        if logger:
            logger.warning(
                "Epigenetic feature extraction failed for '%s': %s",
                row.get("ID"),
                exc,
            )

    return feats
