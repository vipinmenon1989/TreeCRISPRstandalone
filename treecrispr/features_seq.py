"""
treecrispr/features_seq.py — Sequence-derived feature extraction.

For each 30-nt guide sequence this module computes three groups of features:

1. **Global counts** — mononucleotide counts (A/T/G/C), GC count, GC-high/low
   flags, melting temperature, Shannon entropy, and RNAfold minimum free energy.

2. **Positional one-hot (120 columns)** — ``pos{i}_{N}`` where ``i ∈ [0,29]``
   and ``N ∈ {A,T,G,C}``.  These are renamed to ``{N}{i+1}`` (e.g. ``A1``)
   before being passed to the XGBoost models (see :mod:`models`).

3. **Positional dinucleotide one-hot (464 columns)** — ``di{i}_{NN}`` where
   ``i ∈ [0,28]`` and ``NN`` is one of the 16 dinucleotides.  Renamed to
   ``{NN}{i+1}`` for the models.

Public API
----------
seq_features_for(original_seq, strand)
    Main entry point called by the pipeline for every candidate row.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from typing import Dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NUCS: str = "ATGC"
_DINUCS: list[str] = [a + b for a in _NUCS for b in _NUCS]

# Pre-compiled complement table (upper-case only; lower-case handled via upper()).
_COMP: dict[int, str] = str.maketrans("ATGCN", "TACGN")


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

def clean_seq(seq: str) -> str:
    """Upper-case *seq*, convert U→T, and strip any non-ATGCN characters."""
    s = seq.upper().replace("U", "T")
    return "".join(ch for ch in s if ch in "ATGCN")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA string (A/T/G/C/N only)."""
    return seq.upper().translate(_COMP)[::-1]


def pick_feature_sequence(seq: str, strand: str) -> str:
    """
    Return the sequence in 5′→3′ NGG orientation.

    For a ``-`` strand hit, return the reverse complement; otherwise return
    the sequence as-is.  Called before feature extraction so that all models
    receive a consistently oriented input.
    """
    s = clean_seq(seq)
    return reverse_complement(s) if str(strand).strip() in ("-", "neg", "negative") else s


# ---------------------------------------------------------------------------
# Global count features
# ---------------------------------------------------------------------------

def _mono_counts(seq: str) -> Dict[str, int]:
    return {b: seq.count(b) for b in _NUCS}


def _dinuc_counts(seq: str) -> Dict[str, int]:
    out = {d: 0 for d in _DINUCS}
    for i in range(len(seq) - 1):
        d = seq[i : i + 2]
        if d in out:
            out[d] += 1
    return out


def shannon_entropy(seq: str) -> float:
    """Compute the base-2 Shannon entropy of the nucleotide distribution."""
    total = sum(seq.count(b) for b in _NUCS)
    if total == 0:
        return 0.0
    ent = 0.0
    for b in _NUCS:
        c = seq.count(b)
        if c == 0:
            continue
        p = c / total
        ent -= p * math.log(p, 2)
    return float(ent)


def gc_count(seq: str) -> int:
    """Return the number of G and C bases in *seq*."""
    return seq.count("G") + seq.count("C")


def melting_temperature(seq: str) -> float:
    """
    Estimate melting temperature (°C) using the nearest-neighbour approximation
    for short oligonucleotides.
    """
    n = max(1, len(seq))
    gc = gc_count(seq)
    return 64.9 + 41.0 * ((gc - 16.4) / n)


def rnafold_mfe(seq: str) -> float:
    """
    Return the RNAfold minimum free energy (kcal/mol) for *seq*.

    Returns ``float('nan')`` if RNAfold is not installed or the subprocess
    fails for any reason.
    """
    if not shutil.which("RNAfold"):
        return float("nan")
    try:
        rna = seq.replace("T", "U")
        proc = subprocess.run(
            ["RNAfold", "--noPS"],
            input=(rna + "\n").encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        line = proc.stdout.decode(errors="ignore").splitlines()[1]
        m = re.search(r"\(([-+]?\d+(?:\.\d+)?)\)", line)
        return float(m.group(1)) if m else float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Positional one-hot features
# ---------------------------------------------------------------------------

def _positional_nuc_onehot(seq30: str) -> Dict[str, float]:
    """
    Generate 30 × 4 = 120 positional mononucleotide one-hot features.

    Column names follow the pattern ``pos{i}_{N}`` (0-indexed), e.g.
    ``pos0_A``, ``pos0_T``, …  These are later renamed to ``A1``, ``T1``, …
    by :func:`models.fix_column_names_for_xgboost` to match the R-trained
    XGBoost model feature names.
    """
    feats: Dict[str, float] = {}
    for i in range(30):
        ch = seq30[i] if i < len(seq30) else "N"
        for b in _NUCS:
            feats[f"pos{i}_{b}"] = 1.0 if ch == b else 0.0
    return feats


def _positional_dinuc_onehot(seq30: str) -> Dict[str, float]:
    """
    Generate 29 × 16 = 464 positional dinucleotide one-hot features.

    Column names follow the pattern ``di{i}_{NN}`` (0-indexed), e.g.
    ``di0_AA``, ``di0_AT``, …  Renamed to ``AA1``, ``AT1``, … by
    :func:`models.fix_column_names_for_xgboost`.
    """
    feats: Dict[str, float] = {
        f"di{i}_{d}": 0.0 for i in range(29) for d in _DINUCS
    }
    for i in range(min(29, len(seq30) - 1)):
        d = seq30[i : i + 2]
        if d in feats:  # only valid ATGC dinucleotides
            feats[f"di{i}_{d}"] = 1.0
    return feats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sequence_features(seq: str) -> Dict[str, float]:
    """
    Compute all sequence features for a single 30-nt guide sequence.

    Parameters
    ----------
    seq : str
        Raw 30-nt sequence (will be cleaned via :func:`clean_seq`).

    Returns
    -------
    Dict mapping feature name → float value.
    """
    s = clean_seq(seq)
    mono = _mono_counts(s)
    di   = _dinuc_counts(s)
    gc   = gc_count(s)

    base: Dict[str, float] = {
        "Entropy":           shannon_entropy(s),
        "Energy":            rnafold_mfe(s),
        "GCcount":           float(gc),
        "GChigh":            1.0 if gc > 10 else 0.0,
        "GClow":             1.0 if gc <= 10 else 0.0,
        "MeltingTemperature": float(melting_temperature(s)),
        "A":                 float(mono["A"]),
        "T":                 float(mono["T"]),
        "G":                 float(mono["G"]),
        "C":                 float(mono["C"]),
    }

    for d in _DINUCS:
        base[d] = float(di[d])

    base.update(_positional_nuc_onehot(s))
    base.update(_positional_dinuc_onehot(s))
    return base


def seq_features_for(original_seq: str, strand: str) -> Dict[str, float]:
    """
    Orient *original_seq* correctly then return all sequence features.

    This is the primary symbol imported by :mod:`pipeline`.

    Parameters
    ----------
    original_seq : str
        The 30-nt window sequence as read from the genome (forward strand).
    strand : str
        ``"+"`` or ``"-"`` (or ``"neg"`` / ``"negative"``).

    Returns
    -------
    Dict of sequence features (see :func:`sequence_features`).
    """
    feat_seq = pick_feature_sequence(original_seq, strand)
    return sequence_features(feat_seq)
