# treecrispr/pipeline.py
from __future__ import annotations
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

# Adjust imports to match your folder structure
from .config import MAX_SEQ_LEN
from .features_seq import seq_features_for, reverse_complement
from .features_epi import epigenetic_features
from .models import load_models, score_with_models

# ---- PAM scanner ----
def _scan_30mers(seq: str) -> List[Tuple[int,int,str,str]]:
    """
    Return list of (start, end, strand, pam_label) for 30-mers.
    STRICT MODE + EXACT PAM:
      - Enforces N in 'NGG' is A, T, G, or C (no ambiguity codes).
      - Returns the EXACT PAM sequence (e.g., 'AGG', 'TGG') instead of 'NGG'.
      - For Negative strand, returns the Reverse Complement PAM (e.g. 'CCT' -> 'AGG').
    """
    s = seq.upper().replace("U","T")
    out = []
    n = len(s)
    
    # Valid bases for "Strict N"
    VALID_BASES = {'A', 'T', 'G', 'C'}

    for i in range(0, max(0, n-30)+1):
        w = s[i:i+30]
        if len(w) < 30: break
        
        # --- CASE 1: Forward Strand (NGG) ---
        # Pattern: ... N(24) G(25) G(26) ...
        if w[25:27] == "GG":
            n_base = w[24]
            if n_base in VALID_BASES:
                # Capture exact PAM (e.g., AGG)
                specific_pam = w[24:27]
                out.append((i, i+30, "+", specific_pam))

        # --- CASE 2: Negative Strand (CCN) ---
        # Pattern on Fwd: ... C(3) C(4) N(5) ...
        # We want to report the PAM as it appears on the CRISPR strand (NGG style)
        if w[3:5] == "CC":
            n_base = w[5]
            if n_base in VALID_BASES:
                # Genomic is CCN (e.g. CCT). We want the RC (e.g. AGG).
                genomic_pam = w[3:6]
                pam_rc = reverse_complement(genomic_pam)
                out.append((i, i+30, "-", pam_rc))
                
    return out

def build_candidates(fasta_id: str, seq: str) -> pd.DataFrame:
    """
    Return candidates DataFrame. 
    IMPORTANT: For Negative strand (-), we SWAP Sequence and ReverseComplement
    so that the 'Sequence' column ALWAYS shows the Guide+PAM (NGG) orientation.
    """
    seq = seq.upper().replace("U","T")
    rows = []
    for start, end, strand, pam_seq in _scan_30mers(seq):
        win = seq[start:end]
        
        # Calculate RC
        rc = reverse_complement(win)
        
        if strand == "+":
            # Forward match: Sequence is already NGG-oriented
            final_seq = win
            final_rc  = rc
        else:
            # Negative match (was CCN in genome):
            # We swap them so 'Sequence' becomes the NGG-oriented guide
            final_seq = rc   
            final_rc  = win  
            
        rows.append({
            "ID": fasta_id,
            "Start": start,
            "End": end,
            "Strand": strand,
            "Sequence": final_seq,          # ALWAYS NGG oriented now
            "ReverseComplement": final_rc,  # The opposite strand
            "PAM": pam_seq                  # Specific sequence (e.g. AGG, TGG)
        })
    return pd.DataFrame(rows)

def compute_features_only(df_base: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    Returns ONLY numeric features (seq + epi) for scoring.
    """
    if df_base.empty:
        return pd.DataFrame(index=df_base.index)

    feat_rows: List[Dict[str, float]] = []
    for _, row in df_base.iterrows():
        # Since 'Sequence' is now ALWAYS the NGG-oriented version (guide),
        # we can just use row["Sequence"] directly for features.
        feat_seq = row["Sequence"]
        
        # 1. Get Sequence Features
        # We pass strand="+" because feat_seq is already oriented correctly (5'-3' NGG)
        fseq = seq_features_for(feat_seq, "+")
        
        # 2. Get Epigenetic Features 
        # (Pass the whole row; epi logic uses ID/Start/End/Strand to find genomic coords)
        fepi = epigenetic_features(row, logger=logger)
        
        # 3. Combine them
        fseq.update(fepi)
        feat_rows.append(fseq)

    # Convert to DataFrame and fill NaNs
    F = pd.DataFrame(feat_rows).fillna(0.0)
    return F

def run_full_pipeline(records: List[Tuple[str,str]], logger=None, model_dir=None) -> pd.DataFrame:
    """
    records: list of (id, seq)
    model_dir: Path to the chosen mode's model directory (i or a)
    """
    # 1. Build candidates
    all_rows = []
    for rid, seq in records:
        if len(seq) > MAX_SEQ_LEN:
            if logger: logger.warning(f"Skipping {rid}: too long ({len(seq)} > {MAX_SEQ_LEN})")
            continue
        cands = build_candidates(rid, seq)
        all_rows.append(cands)
    
    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=[
        "ID","Start","End","Strand","Sequence","ReverseComplement","PAM"
    ])

    if df.empty:
        return df

    # 2. Compute Features
    if logger: logger.info(f"Computing features for {len(df)} candidates...")
    F = compute_features_only(df, logger=logger)

    if logger:
        # Debug Log: Print shape to prove features exist
        logger.info(f"[DEBUG] Feature Matrix Shape: {F.shape}")

    # 3. Score with Models
    models = load_models(model_dir, logger=logger) if model_dir else {}
    
    if models and not F.empty:
        scores = score_with_models(F, models, model_dir=model_dir, logger=logger)
    else:
        scores = pd.DataFrame(index=F.index)

    # 4. Final Merge
    out = pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    return out