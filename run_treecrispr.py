import argparse
import sys
import logging
import pandas as pd
from pathlib import Path

# --- Import setup ---
# This ensures we can import the 'treecrispr' package from the same directory
sys.path.append(str(Path(__file__).parent))

try:
    from treecrispr.pipeline import run_full_pipeline
    from treecrispr.io_utils import parse_fasta_file
    from treecrispr.config import MODEL_DIR_A, MODEL_DIR_I, MAX_SEQ_LEN
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import the 'treecrispr' package.\nDetails: {e}")
    print("\n[Structure Check]")
    print("Ensure your folder structure looks like this:")
    print("  ./run_treecrispr.py")
    print("  ./treecrispr/ (containing pipeline.py, config.py, etc.)")
    print("  ./model_crispra/")
    print("  ./model_crispri/")
    print("  ./bigwig/")
    sys.exit(1)

def setup_logger():
    """Sets up a simple logger to print progress to the console."""
    logger = logging.getLogger("TreeCRISPR")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def main():
    parser = argparse.ArgumentParser(description="Run TreeCRISPR pipeline on FASTA sequences.")
    
    # Required Arguments
    parser.add_argument("-i", "--input", required=True, type=Path, help="Path to input FASTA file.")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to output CSV file.")
    
    # Optional Arguments
    parser.add_argument("--mode", choices=['a', 'i'], required=True, 
                        help="Mode: 'a' for CRISPRa (Activation), 'i' for CRISPRi (Interference).")
    
    args = parser.parse_args()
    logger = setup_logger()

    # 1. Validate Paths
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # 2. Select Model Directory
    model_dir = MODEL_DIR_A if args.mode == 'a' else MODEL_DIR_I
    mode_name = "CRISPRa" if args.mode == 'a' else "CRISPRi"
    
    logger.info(f"Starting {mode_name} pipeline.")
    logger.info(f"Model Directory: {model_dir}")
    
    if not model_dir.exists():
        logger.error(f"Model directory missing: {model_dir}")
        logger.error("Please ensure your .pkl model files are in the correct folder.")
        sys.exit(1)

    # 3. Parse FASTA
    logger.info(f"Reading sequences from {args.input}...")
    try:
        # Note: config.MAX_SEQ_LEN is usually 500. 
        # Sequences longer than this are rejected by io_utils.
        records = parse_fasta_file(args.input, max_len=MAX_SEQ_LEN)
    except ValueError as e:
        logger.error(f"FASTA Parsing Error: {e}")
        sys.exit(1)
        
    if not records:
        logger.warning("No valid sequences found in input file.")
        sys.exit(0)

    logger.info(f"Loaded {len(records)} sequences. Processing...")

    # 4. Run Pipeline
    try:
        df_result = run_full_pipeline(records, logger=logger, model_dir=model_dir)
    except Exception as e:
        logger.error(f"Pipeline crashed: {e}", exc_info=True)
        sys.exit(1)

    # 5. Save Output
    if df_result.empty:
        logger.warning("Pipeline finished but generated no results (check if valid PAMs exist).")
    else:
        # Ensure output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(args.output, index=False)
        logger.info(f"Success! Results saved to: {args.output}")
        logger.info(f"Total candidates scored: {len(df_result)}")

if __name__ == "__main__":
    main()
