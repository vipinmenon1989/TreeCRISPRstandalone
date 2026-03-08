"""
treecrispr/pipeline.py — End-to-end orchestration of the TreeCRISPR pipeline.

Workflow
--------
1. **Scan** — :func:`build_candidates` calls :func:`~treecrispr.scanner.scan_sequence`
   to find all valid NGG PAM sites in each input sequence.
2. **Feature extraction** — :func:`compute_features_only` computes sequence
   features (one-hot, thermodynamics) and epigenetic features (BigWig signal)
   for every candidate.
3. **Scoring** — :func:`run_full_pipeline` loads XGBoost models and calls
   :func:`~treecrispr.models.score_with_models`.
4. **Output** — scored candidates are merged with their coordinates and
   returned as a single :class:`pandas.DataFrame`.

Public API
----------
build_candidates(fasta_id, seq)          → pd.DataFrame
compute_features_only(df_base, logger)   → pd.DataFrame
run_full_pipeline(records, logger, model_dir) → pd.DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import MAX_SEQ_LEN
from .features_epi import epigenetic_features
from .features_seq import reverse_complement, seq_features_for
from .models import load_models, score_with_models
from .scanner import scan_sequence

logger = logging.getLogger(__name__)

# Output columns present in every candidate row (before scores are appended).
_CANDIDATE_COLS: list[str] = [
    "ID", "Start", "End", "Strand", "Sequence", "ReverseComplement", "PAM",
]


# ---------------------------------------------------------------------------
# Candidate builder
# ---------------------------------------------------------------------------

def build_candidates(fasta_id: str, seq: str) -> pd.DataFrame:
    """
    Return a DataFrame of guide-RNA candidates for a single FASTA record.

    For each 30-mer hit returned by :func:`~treecrispr.scanner.scan_sequence`:
    - The ``Sequence`` column **always** shows the guide in NGG orientation
      (5′→3′ on the CRISPR strand).
    - The ``ReverseComplement`` column shows the opposite-strand sequence.
    - ``PAM`` is the exact 3-nt PAM in NGG form (e.g. ``"AGG"``, ``"TGG"``).

    Parameters
    ----------
    fasta_id : str
        FASTA record identifier (used to populate the ``ID`` column).
    seq : str
        Nucleotide sequence to scan (U→T conversion applied internally).

    Returns
    -------
    pd.DataFrame with columns :data:`_CANDIDATE_COLS`.
    """
    seq = seq.upper().replace("U", "T")
    rows: list[dict] = []

    for start, end, strand, pam_seq in scan_sequence(seq):
        window = seq[start:end]
        rc = reverse_complement(window)

        if strand == "+":
            # Forward hit: the 30-mer is already in NGG orientation.
            final_seq, final_rc = window, rc
        else:
            # Reverse hit (CCN on forward strand): swap so 'Sequence' is NGG-oriented.
            final_seq, final_rc = rc, window

        rows.append({
            "ID":               fasta_id,
            "Start":            start,
            "End":              end,
            "Strand":           strand,
            "Sequence":         final_seq,
            "ReverseComplement": final_rc,
            "PAM":              pam_seq,
        })

    return pd.DataFrame(rows, columns=_CANDIDATE_COLS)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_features_only(
    df_base: pd.DataFrame,
    log: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Compute sequence and epigenetic features for all rows in *df_base*.

    The ``Sequence`` column is already NGG-oriented, so features are always
    extracted with ``strand="+"`` (no additional reverse-complement step).

    Parameters
    ----------
    df_base : pd.DataFrame
        Candidate DataFrame produced by :func:`build_candidates`.
    log : logging.Logger, optional
        Logger for epigenetic feature warnings.

    Returns
    -------
    pd.DataFrame of numeric features, aligned to *df_base* by index.
    NaN values are filled with 0.0.
    """
    if df_base.empty:
        return pd.DataFrame(index=df_base.index)

    feat_rows: List[Dict[str, float]] = []
    for _, row in df_base.iterrows():
        # Sequence is always NGG-oriented; pass strand="+" directly.
        fseq = seq_features_for(row["Sequence"], "+")
        fepi = epigenetic_features(row, logger=log)
        fseq.update(fepi)
        feat_rows.append(fseq)

    return pd.DataFrame(feat_rows).fillna(0.0)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    records: List[Tuple[str, str]],
    log: Optional[logging.Logger] = None,
    model_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Execute the complete TreeCRISPR pipeline.

    Parameters
    ----------
    records : list of (id, seq) tuples
        Parsed FASTA records.  Sequences exceeding
        :data:`~treecrispr.config.MAX_SEQ_LEN` are skipped.
    log : logging.Logger, optional
        Logger for progress and warning messages.
    model_dir : Path, optional
        Directory containing ``.pkl`` / ``.joblib`` XGBoost model files.
        If ``None`` or absent, the scoring step is skipped and the returned
        DataFrame contains only candidate coordinates.

    Returns
    -------
    pd.DataFrame with candidate columns followed by one score column per
    loaded model.  Returns an empty DataFrame with :data:`_CANDIDATE_COLS`
    if no candidates are found.
    """
    _log = log or logger

    # ------------------------------------------------------------------
    # Step 1: scan sequences for guide candidates
    # ------------------------------------------------------------------
    all_frames: List[pd.DataFrame] = []
    for rid, seq in records:
        if len(seq) > MAX_SEQ_LEN:
            _log.warning("Skipping '%s': length %d exceeds MAX_SEQ_LEN=%d.", rid, len(seq), MAX_SEQ_LEN)
            continue
        cands = build_candidates(rid, seq)
        if not cands.empty:
            all_frames.append(cands)

    if not all_frames:
        return pd.DataFrame(columns=_CANDIDATE_COLS)

    df = pd.concat(all_frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Step 2: compute features
    # ------------------------------------------------------------------
    _log.info("Computing features for %d candidates …", len(df))
    F = compute_features_only(df, log=_log)
    _log.debug("Feature matrix shape: %s", F.shape)

    # ------------------------------------------------------------------
    # Step 3: score with XGBoost models
    # ------------------------------------------------------------------
    models = load_models(model_dir, logger=_log) if model_dir else {}
    if models and not F.empty:
        scores = score_with_models(F, models, model_dir=model_dir, logger=_log)
    else:
        scores = pd.DataFrame(index=F.index)

    # ------------------------------------------------------------------
    # Step 4: merge coordinates + scores
    # ------------------------------------------------------------------
    return pd.concat(
        [df.reset_index(drop=True), scores.reset_index(drop=True)],
        axis=1,
    )
