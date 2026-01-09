# treecrispr/features_epi.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

# Import the list directly from your config
from .config import BIGWIG_DIR, EPIGENETIC_EXTENSIONS, EPIG_AGGREGATION, EXPECTED_BIGWIGS
from .epi_seq import single_interval_features

_COORD_RE = re.compile(r'(chr[\w]+|\b[0-9XYM]+)\s*:\s*([0-9,]+)\s*-\s*([0-9,]+)', re.IGNORECASE)

def _parse_id_region(id_str: str) -> Optional[Tuple[str, int, int]]:
    m = _COORD_RE.search(id_str)
    if not m:
        return None
    chrom = m.group(1)
    if not chrom.lower().startswith("chr"):
        chrom = "chr" + chrom
    start = int(m.group(2).replace(",", ""))
    end   = int(m.group(3).replace(",", ""))
    if start > 0:
        start -= 1 
    return chrom.lower(), start, end

def _map_files_to_expected_names() -> Dict[str, Path]:
    mapping = {}
    if not BIGWIG_DIR.exists():
        return mapping
    
    # Get all actual files on disk
    disk_files = {p.name.lower(): p for p in BIGWIG_DIR.iterdir() if p.is_file()}
    
    for expected in EXPECTED_BIGWIGS:
        # Tries to find "H2AZ" inside "H2AZ.bigwig" (case insensitive)
        cands = [f for name, f in disk_files.items() if name.startswith(expected.lower())]
        if cands:
            cands.sort(key=lambda p: len(p.name))
            mapping[expected] = cands[0]
    return mapping

def _predeclare_zero_feats() -> Dict[str, float]:
    """
    FORCE CREATION of all columns using the list from config.py.
    This guarantees the model sees 'H2AZ_0' etc, even if file lookup fails.
    """
    feats: Dict[str, float] = {}
    # We loop over the LIST from config, NOT the files we found
    for base in EXPECTED_BIGWIGS:
        for ext in EPIGENETIC_EXTENSIONS:
            feats[f"{base}_{int(ext)}"] = 0.0
    return feats

def epigenetic_features(row, logger=None) -> Dict[str, float]:
    # 1. First, create the dictionary with ALL expected keys (values = 0.0)
    feats = _predeclare_zero_feats()

    # 2. Check for coordinates
    parsed = _parse_id_region(str(row.get("ID", "")))
    if not parsed:
        # If no coords, return the zero-filled dict. Model runs, but result is seq-only.
        return feats 

    chrom, abs_start_input, _ = parsed
    try:
        off_start = int(row.get("Start"))
        off_end = int(row.get("End"))
    except Exception:
        return feats

    abs_start = abs_start_input + off_start
    abs_end   = abs_start_input + off_end

    # 3. If we have files, overwrite the zeros with real data
    file_map = _map_files_to_expected_names()
    
    if file_map:
        try:
            found_paths = list(file_map.values())
            
            vals = single_interval_features(
                bw_paths=found_paths,
                chrom=chrom, start0=abs_start, end0=abs_end,
                extensions=EPIGENETIC_EXTENSIONS, agg=EPIG_AGGREGATION
            )
            
            # Map the extracted values (which might use keys like "H2AZ.bigwig_0")
            # back to the expected model keys (like "H2AZ_0")
            for expected_name, file_path in file_map.items():
                file_stem = file_path.stem # e.g. "H2AZ" or "H2AZ.bigwig" -> depends on file
                
                # We check matches for the file stem
                for ext in EPIGENETIC_EXTENSIONS:
                    # The extraction engine uses the filename (stem) as key
                    raw_key = f"{file_stem}_{int(ext)}"
                    model_key = f"{expected_name}_{int(ext)}"
                    
                    if raw_key in vals:
                        feats[model_key] = float(vals[raw_key])
                        
        except Exception as e:
            if logger: logger.warning(f"EPI fail for {row.get('ID')}: {e}")
            
    return feats