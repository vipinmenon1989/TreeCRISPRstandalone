"""
treecrispr/scanner.py — PAM-scanning logic for 30-mer guide candidates.

Exports
-------
scan_sequence(seq)
    Core scanner that returns (start, end, strand, pam) tuples from a raw
    nucleotide sequence.  This is the single source of truth for PAM detection;
    pipeline.py imports and uses it directly.

scan_targets(record_id, seq)
    Convenience wrapper that returns a list of dicts, useful for ad-hoc use
    outside the main pipeline.

Algorithm
---------
A 30-nt sliding window is moved one base at a time across the input sequence.
For each window:
  - Forward strand (NGG PAM):  positions [24:27] must match [ATGC]GG.
  - Reverse strand (CCN PAM):  positions [3:6]  must match CC[ATGC] on the
    forward strand, representing an NGG PAM on the complementary strand.

The exact PAM trinucleotide (e.g. "AGG") is returned in NGG orientation for
both strands.  Ambiguity codes are rejected — only A/T/G/C are accepted at
the N position.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

_COMP: dict[int, str] = str.maketrans("ATGCatgc", "TACGtacg")

# Valid bases for the 'N' in NGG — strict mode rejects IUPAC ambiguity codes.
_STRICT_BASES: frozenset[str] = frozenset("ATGC")


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA string (handles upper and lower case)."""
    return seq.translate(_COMP)[::-1]


def scan_sequence(seq: str) -> List[Tuple[int, int, str, str]]:
    """
    Scan *seq* for all valid NGG PAM sites and return 30-mer coordinates.

    Parameters
    ----------
    seq : str
        Raw nucleotide sequence (DNA, upper or lower case; U→T conversion applied).

    Returns
    -------
    List of (start, end, strand, pam) tuples where:
        start  — 0-based start index of the 30-mer window in *seq*
        end    — exclusive end index (start + 30)
        strand — "+" for forward NGG hit, "-" for reverse CCN hit
        pam    — exact PAM trinucleotide in NGG orientation (e.g. "AGG", "TGG")
    """
    s = seq.upper().replace("U", "T")
    n = len(s)
    out: List[Tuple[int, int, str, str]] = []

    for i in range(max(0, n - 30) + 1):
        w = s[i : i + 30]
        if len(w) < 30:
            break

        # Forward strand: guide is w[0:20], PAM is w[20:23] → but the 30-mer
        # convention here places the PAM at positions 24–26 (0-based).
        if w[25:27] == "GG" and w[24] in _STRICT_BASES:
            out.append((i, i + 30, "+", w[24:27]))

        # Reverse strand: CCN appears at positions 3–5 on the forward 30-mer.
        if w[3:5] == "CC" and w[5] in _STRICT_BASES:
            genomic_pam = w[3:6]
            out.append((i, i + 30, "-", _reverse_complement(genomic_pam)))

    return out


def scan_targets(record_id: str, seq: str) -> List[Dict]:
    """
    Scan *seq* and return results as a list of dicts (convenience wrapper).

    Each dict contains keys: ``id``, ``start``, ``end``, ``strand``, ``pam``.
    Useful for quick exploration; the main pipeline uses :func:`scan_sequence`
    directly for efficiency.
    """
    return [
        {"id": record_id, "start": s, "end": e, "strand": strand, "pam": pam}
        for s, e, strand, pam in scan_sequence(seq)
    ]
