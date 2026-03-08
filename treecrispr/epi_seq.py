"""
treecrispr/epi_seq.py — BigWig interval extraction for epigenetic features.

This module handles all direct I/O with BigWig files via the ``pyBigWig``
library.  It is deliberately kept separate from feature assembly logic so
that it can be mocked or replaced without touching the rest of the pipeline.

Functions
---------
single_interval_features(bw_paths, chrom, start0, end0, extensions, agg)
    Compute one aggregate value per (BigWig track × window extension) pair
    for a single genomic interval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    import pyBigWig
except ImportError as exc:
    raise RuntimeError(
        "pyBigWig is required for epigenetic features.  "
        "Install it with:  conda install -c bioconda pybigwig  "
        "or:  pip install pyBigWig"
    ) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_chrom_name(bw: "pyBigWig.pyBigWig", chrom: str) -> Optional[str]:
    """
    Return the chromosome name as it appears in *bw*.

    Tries the name as provided, then with / without the ``"chr"`` prefix.
    Returns ``None`` if neither variant is present in the file.
    """
    try:
        chroms = bw.chroms()
    except Exception:
        return None
    if chrom in chroms:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
    return alt if alt in chroms else None


def _agg_values(
    bw: "pyBigWig.pyBigWig",
    chrom: str,
    start: int,
    end: int,
    agg: str = "sum",
) -> float:
    """
    Aggregate per-base BigWig values over the interval ``[start, end)``
    (0-based, half-open).

    NaN values are treated as 0.  Coordinates are clipped to valid chromosome
    bounds.  Returns ``0.0`` on any error.
    """
    name = _resolve_chrom_name(bw, chrom)
    if name is None:
        return 0.0
    try:
        clen = int(bw.chroms()[name])
    except Exception:
        return 0.0

    s = max(0, int(start))
    e = min(clen, int(end))
    if e <= s:
        return 0.0

    try:
        raw = bw.values(name, s, e, numpy=True)
        arr = np.array(raw, dtype=float)
        if arr.size == 0:
            return 0.0
        arr[np.isnan(arr)] = 0.0
        total = float(arr.sum())
        return (total / max(1, e - s)) if agg == "mean" else total
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def single_interval_features(
    bw_paths: List[Path],
    chrom: str,
    start0: int,
    end0: int,
    extensions: Iterable[int],
    agg: str = "sum",
) -> Dict[str, float]:
    """
    Compute epigenetic features for a single genomic interval.

    For each combination of BigWig track and window extension, the interval
    is symmetrically expanded by *ext* base pairs on both sides before
    aggregating the signal.

    Parameters
    ----------
    bw_paths : list of Path
        Paths to BigWig files.  Files that cannot be opened are skipped.
    chrom : str
        Chromosome name (with or without ``"chr"`` prefix — both are tried).
    start0 : int
        0-based inclusive start of the core interval.
    end0 : int
        0-based exclusive end of the core interval.
    extensions : iterable of int
        Window extensions in base pairs (e.g. ``[0, 50, 150, 250, 500, 2500]``).
    agg : str
        Aggregation method: ``"sum"`` (default) or ``"mean"``.

    Returns
    -------
    Dict mapping ``"{track_stem}_{ext}"`` → float value.  All columns are
    pre-declared to ``0.0`` so the shape is stable even when a track fails.
    """
    ext_list = list(extensions)
    out: Dict[str, float] = {}

    # Pre-declare all columns to 0.0 for shape stability.
    for p in bw_paths:
        stem = p.stem
        for ext in ext_list:
            out[f"{stem}_{int(ext)}"] = 0.0

    for p in bw_paths:
        stem = p.stem
        try:
            bw = pyBigWig.open(str(p))
        except Exception:
            continue
        try:
            for ext in ext_list:
                s = start0 - int(ext)
                e = end0 + int(ext)
                out[f"{stem}_{int(ext)}"] = _agg_values(bw, chrom, s, e, agg=agg)
        finally:
            try:
                bw.close()
            except Exception:
                pass

    return out
