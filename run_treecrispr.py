#!/usr/bin/env python3
"""
run_treecrispr.py — Command-line entry point for the TreeCRISPR pipeline.

Usage
-----
python run_treecrispr.py -i <input.fa> -o <results.csv> --mode <i|a>

Arguments
---------
-i / --input   Path to the input FASTA file.
-o / --output  Path for the output CSV file (parent directories are created
               automatically if absent).
--mode         "i" for CRISPRi (Interference) or "a" for CRISPRa (Activation).

Examples
--------
# Score guides for CRISPRi:
python run_treecrispr.py -i sequences.fa -o results_crispri.csv --mode i

# Score guides for CRISPRa:
python run_treecrispr.py -i sequences.fa -o results_crispra.csv --mode a
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the package directory is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent))

try:
    from treecrispr.config import MAX_SEQ_LEN, MODEL_DIR_A, MODEL_DIR_I
    from treecrispr.io_utils import parse_fasta_file
    from treecrispr.pipeline import run_full_pipeline
except ImportError as exc:
    print(f"CRITICAL ERROR: Could not import the 'treecrispr' package.\nDetails: {exc}")
    print("\n[Expected directory structure]")
    print("  ./run_treecrispr.py")
    print("  ./treecrispr/        (pipeline package)")
    print("  ./model_crispra/     (CRISPRa .pkl models)")
    print("  ./model_crispri/     (CRISPRi .pkl models)")
    print("  ./bigwig/            (BigWig epigenetic tracks)")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def _setup_logger(name: str = "TreeCRISPR") -> logging.Logger:
    """Configure and return a console logger."""
    log = logging.getLogger(name)
    if log.handlers:
        # Avoid duplicate handlers if the function is called more than once.
        return log
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    log.addHandler(handler)
    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TreeCRISPR guide-RNA scoring pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        metavar="FASTA",
        help="Path to the input FASTA file.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=Path,
        metavar="CSV",
        help="Path for the output CSV file.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["a", "i"],
        help="'a' for CRISPRa (Activation), 'i' for CRISPRi (Interference).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log = _setup_logger()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    model_dir = MODEL_DIR_A if args.mode == "a" else MODEL_DIR_I
    mode_name = "CRISPRa" if args.mode == "a" else "CRISPRi"

    log.info("Starting TreeCRISPR — mode: %s", mode_name)
    log.info("Model directory: %s", model_dir)

    if not model_dir.exists():
        log.error("Model directory not found: %s", model_dir)
        log.error(
            "Download the model files from https://epitree.igs.umaryland.edu/epitree/ "
            "and place them in '%s'.", model_dir
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Parse FASTA
    # ------------------------------------------------------------------
    log.info("Reading sequences from %s …", args.input)
    try:
        records = parse_fasta_file(args.input, max_len=MAX_SEQ_LEN)
    except (ValueError, FileNotFoundError) as exc:
        log.error("FASTA parsing error: %s", exc)
        sys.exit(1)

    if not records:
        log.warning("No valid sequences found in the input file.")
        sys.exit(0)

    log.info("Loaded %d sequence(s).  Running pipeline …", len(records))

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    try:
        df_result = run_full_pipeline(records, log=log, model_dir=model_dir)
    except Exception as exc:
        log.error("Pipeline error: %s", exc, exc_info=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if df_result.empty:
        log.warning(
            "Pipeline finished but produced no results. "
            "Check that your sequences contain valid NGG PAM sites."
        )
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(args.output, index=False)
        log.info("Results saved to: %s", args.output)
        log.info("Total candidates scored: %d", len(df_result))


if __name__ == "__main__":
    main()
