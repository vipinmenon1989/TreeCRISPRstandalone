from pathlib import Path
import os

# ---- Paths (adjust BASE_DIR if needed) ----
BASE_DIR = Path(__file__).resolve().parents[1]

UPLOAD_DIR   = BASE_DIR / "uploads"
RESULTS_DIR  = BASE_DIR / "results_cache"
BIGWIG_DIR   = BASE_DIR / "bigwig"

MODEL_DIR_I  = BASE_DIR / "model_crispri"   # .pkl models for TreeCRISPRi
MODEL_DIR_A  = BASE_DIR / "model_crispra"   # .pkl models for TreeCRISPRa

for d in (UPLOAD_DIR, RESULTS_DIR, BIGWIG_DIR, MODEL_DIR_I, MODEL_DIR_A):
    d.mkdir(parents=True, exist_ok=True)

# ---- FASTA constraints ----
ALLOWED_EXT = {".fa", ".fasta", ".fna"}
MAX_SEQ_LEN = 500

# ---- Epigenetic feature settings ----
EPIGENETIC_EXTENSIONS = tuple(int(x) for x in os.getenv("EPIG_EXTS", "0,50,150,250,500,2500").split(","))
EPIG_AGGREGATION      = os.getenv("EPIG_AGG", "sum").lower()  # "sum" or "mean"
if EPIG_AGGREGATION not in ("sum", "mean"):
    EPIG_AGGREGATION = "sum"

# ---- BigWig basenames used in training (ORDER MATTERS) ----
EXPECTED_BIGWIGS = [
    "H2AZ", "H3K27ac", "H3K27me3", "H3K36me3",
    "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me2",
    "H3K9ac", "H3K9me3", "K562_chromatin_strucutre",
    "K562_DNA_methylation", "K562_dnase"
]