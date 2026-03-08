"""
treecrispr/io_utils.py — FASTA parsing and sequence validation.

Functions
---------
parse_fasta_text(text, max_len)
    Parse a FASTA-formatted string into a list of (id, sequence) tuples.

parse_fasta_file(path, max_len)
    Read a FASTA file from disk and delegate to :func:`parse_fasta_text`.

Notes
-----
- Only ACGT characters are retained; any other character triggers a warning
  rather than a hard failure, so ambiguous-base sequences are still processed.
- Sequences exceeding *max_len* are skipped with an informative message.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Regex to detect non-ACGT characters in a (already upper-cased) sequence.
_INVALID_RE = re.compile(r"[^ACGT]")


def parse_fasta_text(
    text: str,
    max_len: int,
) -> List[Tuple[str, str]]:
    """
    Parse a FASTA-formatted string.

    Parameters
    ----------
    text : str
        Raw FASTA content (may contain multiple records).
    max_len : int
        Maximum allowed sequence length.  Records longer than this are
        silently skipped (a warning is printed to *stderr*).

    Returns
    -------
    List of ``(record_id, sequence)`` tuples, one per valid FASTA record.

    Raises
    ------
    ValueError
        If *text* contains no recognisable FASTA records.
    """
    records: List[Tuple[str, str]] = []
    current_id: str | None = None
    seq_parts: List[str] = []

    def _commit() -> None:
        if current_id is None:
            return
        seq = "".join(seq_parts).upper()
        if len(seq) > max_len:
            print(
                f"[io_utils] WARNING: '{current_id}' skipped — "
                f"length {len(seq)} exceeds max_len={max_len}.",
                file=sys.stderr,
            )
            return
        if _INVALID_RE.search(seq):
            print(
                f"[io_utils] WARNING: '{current_id}' contains non-ACGT characters; "
                "they will be ignored during feature extraction.",
                file=sys.stderr,
            )
        records.append((current_id, seq))

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            _commit()
            current_id = line[1:].split()[0]
            seq_parts = []
        else:
            seq_parts.append(line)

    _commit()  # flush last record

    if not records:
        raise ValueError("No valid FASTA records found in the provided text.")

    return records


def parse_fasta_file(path: Path, max_len: int) -> List[Tuple[str, str]]:
    """
    Read a FASTA file from *path* and parse it.

    Parameters
    ----------
    path : Path
        Path to the input ``.fa`` / ``.fasta`` file.
    max_len : int
        Maximum allowed sequence length (passed to :func:`parse_fasta_text`).

    Returns
    -------
    List of ``(record_id, sequence)`` tuples.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        Propagated from :func:`parse_fasta_text` if no records are found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")
    return parse_fasta_text(path.read_text(encoding="utf-8", errors="replace"), max_len)
