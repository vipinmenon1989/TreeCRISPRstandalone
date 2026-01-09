# treecrispr/epi_seq.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import numpy as np

try:
    import pyBigWig  # pip install pybigwig
except Exception as e:
    raise RuntimeError("pyBigWig is required. pip install pybigwig") from e


def _resolve_chrom_name(bw, chrom: str) -> Optional[str]:
    try:
        chroms = bw.chroms()
    except Exception:
        return None
    if chrom in chroms:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
    return alt if alt in chroms else None


def _agg_values(bw, chrom: str, start: int, end: int, agg: str = "sum") -> float:
    """
    Sum (or mean) of per-base values from BigWig in [start, end) (0-based, half-open).
    NaNs -> 0. Out-of-bounds clipped to [0, chromLen).
    """
    name = _resolve_chrom_name(bw, chrom)
    if name is None:
        return 0.0
    try:
        clen = int(bw.chroms()[name])
    except Exception:
        return 0.0

    s = max(0, int(start)); e = min(clen, int(end))
    if e <= s:
        return 0.0

    try:
        vals = bw.values(name, s, e, numpy=True)  # 0-start, half-open
        arr = np.array(vals, dtype=float)
        if arr.size == 0:
            return 0.0
        arr[np.isnan(arr)] = 0.0
        tot = float(arr.sum())
        return (tot / max(1, e - s)) if agg == "mean" else tot
    except Exception:
        return 0.0


def single_interval_features(
    bw_paths: List[Path],
    chrom: str,
    start0: int,
    end0: int,
    extensions: Iterable[int],
    agg: str = "sum",
) -> Dict[str, float]:
    """
    Compute features for ONE interval across all (BigWig × extension).
    Returns: { "<track>_<ext>": value, ... }, with ALL columns present.
    """
    out: Dict[str, float] = {}
    # Predeclare all columns for stability (even if a track fails)
    for p in bw_paths:
        base = p.stem
        for ext in extensions:
            out[f"{base}_{int(ext)}"] = 0.0

    for p in bw_paths:
        base = p.stem
        try:
            bw = pyBigWig.open(str(p))
        except Exception:
            continue
        try:
            for ext in extensions:
                s = start0 - int(ext)
                e = end0 + int(ext)
                out[f"{base}_{int(ext)}"] = _agg_values(bw, chrom, s, e, agg=agg)
        finally:
            try:
                bw.close()
            except Exception:
                pass

    return out