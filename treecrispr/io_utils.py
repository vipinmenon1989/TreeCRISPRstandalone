import re
from io import StringIO
from pathlib import Path
from typing import List, Tuple
from Bio import SeqIO

DNA_ALPHABET = set("ACGT")

def sanitize_seq(s: str) -> str:
    return re.sub(r"\s+", "", s.upper().replace("U", "T"))

def parse_fasta_text(text: str, max_len: int) -> List[Tuple[str, str]]:
    out = []
    handle = StringIO(text)
    for rec in SeqIO.parse(handle, "fasta"):
        seq = sanitize_seq(str(rec.seq))
        if not set(seq) <= DNA_ALPHABET:
            raise ValueError(f"{rec.id} contains non-ACGT characters.")
        if len(seq) > max_len:
            raise ValueError(f"{rec.id} exceeds {max_len} bp (len={len(seq)})")
        out.append((rec.id or "seq", seq))
    return out

def parse_fasta_file(path: Path, max_len: int) -> List[Tuple[str, str]]:
    out = []
    for rec in SeqIO.parse(str(path), "fasta"):
        seq = sanitize_seq(str(rec.seq))
        if not set(seq) <= DNA_ALPHABET:
            raise ValueError(f"{rec.id} contains non-ACGT characters.")
        if len(seq) > max_len:
            raise ValueError(f"{rec.id} exceeds {max_len} bp (len={len(seq)})")
        out.append((rec.id or "seq", seq))
    return out
