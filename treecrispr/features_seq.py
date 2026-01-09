# treecrispr/features_seq.py
from __future__ import annotations
from typing import Dict
import math, shutil, subprocess

NUCS = "ATGC"
DINUCS = [a+b for a in NUCS for b in NUCS]

def clean_seq(seq: str) -> str:
    s = seq.upper().replace("U", "T")
    return "".join(ch for ch in s if ch in "ATGCN")

def reverse_complement(seq: str) -> str:
    comp = str.maketrans({"A":"T","T":"A","G":"C","C":"G","N":"N"})
    return seq.translate(comp)[::-1]

def pick_feature_sequence(seq: str, strand: str) -> str:
    s = clean_seq(seq)
    return reverse_complement(s) if str(strand).strip() in ("-","neg","negative") else s

# ---------- global counts ----------
def _mono_counts(seq: str) -> Dict[str,int]:
    return {b: seq.count(b) for b in NUCS}

def _dinuc_counts(seq: str) -> Dict[str,int]:
    out = {d:0 for d in DINUCS}
    for i in range(len(seq)-1):
        d = seq[i:i+2]
        if d in out: out[d] += 1
    return out

def shannon_entropy(seq: str) -> float:
    total = sum(seq.count(b) for b in NUCS)
    if total == 0: return 0.0
    ent = 0.0
    for b in NUCS:
        c = seq.count(b)
        if c == 0: continue
        p = c/total
        ent -= p*math.log(p, 2)
    return float(ent)

def gc_count(seq: str) -> int:
    return seq.count("G")+seq.count("C")

def melting_temperature(seq: str) -> float:
    n = max(1, len(seq)); gc = gc_count(seq)
    return 64.9 + 41.0*((gc-16.4)/n)

def rnafold_mfe(seq: str) -> float:
    if not shutil.which("RNAfold"):
        return float("nan")
    try:
        rna = seq.replace("T","U")
        p = subprocess.run(["RNAfold","--noPS"], input=(rna+"\n").encode(),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        ln = p.stdout.decode(errors="ignore").splitlines()[1]
        import re
        m = re.search(r"\(([-+]?\d+(?:\.\d+)?)\)", ln)
        return float(m.group(1)) if m else float("nan")
    except Exception:
        return float("nan")

# ---------- positional one-hots ----------
def _positional_nuc_onehot(seq30: str) -> Dict[str, float]:
    """30 x 4 = 120 columns: pos{i}_{A|T|G|C}."""
    feats: Dict[str, float] = {}
    s = seq30
    # guard: ensure length 30; if not, still emit 0/1 across observed positions; fill rest with 0
    for i in range(30):
        ch = s[i] if i < len(s) else "N"
        for b in NUCS:
            feats[f"pos{i}_{b}"] = 1.0 if ch == b else 0.0
    return feats

def _positional_dinuc_onehot(seq30: str) -> Dict[str, float]:
    """29 x 16 = 464 columns: di{i}_{AA..TT}."""
    feats: Dict[str, float] = {f"di{i}_{d}": 0.0 for i in range(29) for d in DINUCS}
    s = seq30
    for i in range(min(29, len(s)-1)):
        d = s[i:i+2]
        if d in DINUCS:
            feats[f"di{i}_{d}"] = 1.0
    return feats

# ---------- public API ----------
def sequence_features(seq: str) -> Dict[str, float]:
    """Global (counts) + positional one-hots for a 30nt sequence."""
    s = clean_seq(seq)
    mono = _mono_counts(s); di = _dinuc_counts(s)
    gc = gc_count(s)
    base = {
        "Entropy": shannon_entropy(s),
        "Energy": rnafold_mfe(s),
        "GCcount": float(gc),
        "GChigh": 1.0 if gc > 10 else 0.0,
        "GClow" : 1.0 if gc <= 10 else 0.0,
        "MeltingTemperature": float(melting_temperature(s)),
        "A": float(mono["A"]), "T": float(mono["T"]),
        "G": float(mono["G"]), "C": float(mono["C"]),
    }
    for d in DINUCS:
        base[d] = float(di[d])

    # positional
    base.update(_positional_nuc_onehot(s))
    base.update(_positional_dinuc_onehot(s))
    return base

def seq_features_for(original_seq: str, strand: str) -> Dict[str, float]:
    """Exported symbol expected by the app/pipeline."""
    feat_seq = pick_feature_sequence(original_seq, strand)
    return sequence_features(feat_seq)