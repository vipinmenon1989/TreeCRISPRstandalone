"""
treecrispr — Epigenetic CRISPR guide-RNA scoring package.

Submodules
----------
config        : Directory paths and environment-driven constants.
scanner       : PAM-scanning logic for 30-mer guide candidates.
io_utils      : FASTA parsing and sequence validation.
features_seq  : Sequence-derived feature extraction (one-hot, dinuc, thermodynamics).
epi_seq       : BigWig interval extraction for epigenetic features.
features_epi  : Coordinate-aware epigenetic feature assembly.
pipeline      : End-to-end orchestration (scan → feature → score).
models        : XGBoost model loading and robust prediction.
plots         : Score visualisation and pairwise statistics.
"""

__all__: list[str] = []
