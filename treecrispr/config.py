"""
treecrispr/config.py — Centralised paths and environment-driven constants.

Environment variables
---------------------
EPIG_EXTS   Comma-separated integer extensions (bp) for epigenetic window sizes.
            Default: "0,50,150,250,500,2500"
EPIG_AGG    Aggregation method for BigWig values: "sum" or "mean".
            Default: "sum"
"""

from __future__ import annotations

from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Directory layout (relative to this file's package root)
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parents[1]

UPLOAD_DIR: Path  = BASE_DIR / "uploads"
RESULTS_DIR: Path = BASE_DIR / "results_cache"
BIGWIG_DIR: Path  = BASE_DIR / "bigwig"
MODEL_DIR_I: Path = BASE_DIR / "model_crispri"
MODEL_DIR_A: Path = BASE_DIR / "model_crispra"

# ---------------------------------------------------------------------------
# Sequence constraints
# ---------------------------------------------------------------------------
MAX_SEQ_LEN: int = 500

# ---------------------------------------------------------------------------
# Epigenetic feature configuration
# ---------------------------------------------------------------------------
EPIGENETIC_EXTENSIONS: tuple[int, ...] = tuple(
    int(x) for x in os.getenv("EPIG_EXTS", "0,50,150,250,500,2500").split(",")
)

EPIG_AGGREGATION: str = os.getenv("EPIG_AGG", "sum").lower()

# ---------------------------------------------------------------------------
# Expected BigWig track names
# NOTE: names must exactly match the stem of the .bw / .bigwig files placed
#       in BIGWIG_DIR, and must match the feature column names seen by the
#       trained XGBoost models.
# ---------------------------------------------------------------------------
EXPECTED_BIGWIGS: list[str] = [
    "H2AZ",
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K79me2",
    "H3K9ac",
    "H3K9me3",
    "K562_chromatin_structure",   # fixed typo: was "strucutre"
    "K562_DNA_methylation",
    "K562_dnase",
]
