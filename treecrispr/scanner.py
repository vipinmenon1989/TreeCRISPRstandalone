from typing import Dict, List
from Bio.Seq import Seq

def revcomp(s: str) -> str:
    return str(Seq(s).reverse_complement())

def scan_targets(record_id: str, seq: str) -> List[Dict]:
    """
    Slide a 30bp window:
      + strand: w[25:27] == 'GG'  -> PAM = w[24:27] (NGG)
      - strand: w[3:5]  == 'CC'   -> PAM = w[3:6]  (CCN), RC used for features
    Start/End are 0-based, half-open within the FASTA fragment.
    """
    out = []
    n = len(seq)
    for i in range(0, n - 29):
        w = seq[i:i+30]
        # positive
        if w[25:27] == "GG":
            pam = w[24:27]
            out.append({
                "ID": record_id, "Start": i, "End": i+30, "Strand": "+",
                "Target sequence (Original)": w,
                "Reverse complement": w,
                "PAM": pam
            })
        # negative
        if w[3:5] == "CC":
            pam = w[3:6]
            out.append({
                "ID": record_id, "Start": i, "End": i+30, "Strand": "-",
                "Target sequence (Original)": w,
                "Reverse complement": revcomp(w),
                "PAM": pam
            })
    return out
