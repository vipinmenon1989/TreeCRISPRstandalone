# TreeCRISPR

**Epigenetic CRISPR guide-RNA design for CRISPRi and CRISPRa**

TreeCRISPR scores 20-nt Cas9 guide RNAs for CRISPR interference (CRISPRi) and CRISPR activation (CRISPRa) applications using gradient-boosted tree models trained on both sequence features and epigenetic signal from K562 cells. Given an input FASTA file of promoter or regulatory sequences (up to 500 bp each), TreeCRISPR scans for all valid NGG PAM sites, extracts a rich feature vector for each candidate, and outputs a scored table that can be used to rank and select the best guides.

Companion web server: <https://epitree.igs.umaryland.edu/epitree/>

---

## Table of Contents

1. [Algorithm overview](#algorithm-overview)
2. [Installation](#installation)
3. [Download model and BigWig files](#download-model-and-bigwig-files)
4. [Directory structure](#directory-structure)
5. [Usage](#usage)
6. [Output format](#output-format)
7. [Feature description](#feature-description)
8. [Visualisation](#visualisation)
9. [Troubleshooting](#troubleshooting)
10. [Git push commands](#git-push-commands)
11. [Citation](#citation)

---

## Algorithm overview

```
Input FASTA
     │
     ▼
PAM scanning (NGG / CCN, strict ATGC-only N)
     │  30-nt windows, exact PAM reported (e.g. AGG, TGG)
     ▼
Feature extraction
  ├── Sequence features (584+ columns)
  │     • Positional one-hot:       30 × 4   = 120 cols  (pos{i}_{N})
  │     • Positional dinucleotide:  29 × 16  = 464 cols  (di{i}_{NN})
  │     • Global counts: A, T, G, C, GC count, GC-high/low flag
  │     • Shannon entropy, melting temperature, RNAfold MFE
  └── Epigenetic features (78 columns, requires BigWig files)
        • 13 histone/chromatin tracks × 6 window extensions
        • Extensions: 0, 50, 150, 250, 500, 2500 bp
        • Tracks: H2AZ, H3K27ac, H3K27me3, H3K36me3, H3K4me1/2/3,
                  H3K79me2, H3K9ac, H3K9me3, chromatin structure,
                  DNA methylation, DNase (K562 cell line)
     │
     ▼
XGBoost scoring
  ├── CRISPRi models  (model_crispri/)
  └── CRISPRa models  (model_crispra/)
     │
     ▼
Scored CSV output
```

Guides are scored between 0 and 1 (higher = predicted stronger activity). If BigWig files are absent, the pipeline falls back to sequence-only scoring — results will still be produced but epigenetic signal will be zero-filled.

---

## Installation

### Option A — Conda (recommended)

```bash
git clone https://github.com/vipinmenon1989/TreeCRISPRstandalone.git
cd TreeCRISPRstandalone

conda env create -f TreeCRISPR.yml
conda activate TreeCRISPR
```

### Option B — pip

```bash
git clone https://github.com/vipinmenon1989/TreeCRISPRstandalone.git
cd TreeCRISPRstandalone

pip install -r requirements.txt
```

> **Note:** `pyBigWig` requires libcurl headers. On Ubuntu/Debian: `sudo apt-get install libcurl4-openssl-dev`. On macOS: `brew install curl`.

> **Note:** RNAfold (ViennaRNA) is used for minimum free energy features. If it is not installed, MFE values default to `NaN` and scoring continues without them. Install via: `conda install -c bioconda viennarna`.

---

## Download model and BigWig files

The trained XGBoost models and BigWig epigenetic tracks are not included in this repository due to file size. Download them from the companion web server:

**<https://epitree.igs.umaryland.edu/epitree/>**

Download the `model` and `bigwig` archives and extract them into the repository root. After extraction your directory should look like this:

```
TreeCRISPRstandalone/
├── run_treecrispr.py
├── treecrispr/
├── model_crispri/          ← CRISPRi XGBoost models (.pkl or .joblib)
│   ├── H3K27ac_xgb_clf.pkl
│   └── ...
├── model_crispra/          ← CRISPRa XGBoost models
│   └── ...
└── bigwig/                 ← BigWig epigenetic tracks
    ├── H2AZ.bw
    ├── H3K27ac.bw
    ├── H3K27me3.bw
    ├── H3K36me3.bw
    ├── H3K4me1.bw
    ├── H3K4me2.bw
    ├── H3K4me3.bw
    ├── H3K79me2.bw
    ├── H3K9ac.bw
    ├── H3K9me3.bw
    ├── K562_chromatin_structure.bw
    ├── K562_DNA_methylation.bw
    └── K562_dnase.bw
```

> **Important:** BigWig file names must start with the track names listed above (case-insensitive). The tool matches by prefix, so `H3K27ac.bigwig`, `H3K27ac.bw`, and `H3K27ac_K562.bw` are all valid.

---

## Directory structure

```
TreeCRISPRstandalone/
├── run_treecrispr.py          Main CLI entry point
├── TreeCRISPR.yml             Conda environment specification
├── requirements.txt           pip requirements
├── README.md
└── treecrispr/                Python package
    ├── __init__.py
    ├── config.py              Paths and environment constants
    ├── scanner.py             NGG/CCN PAM detection (30-mer windows)
    ├── io_utils.py            FASTA parsing and sequence validation
    ├── features_seq.py        Sequence feature extraction
    ├── epi_seq.py             BigWig interval extraction
    ├── features_epi.py        Epigenetic feature assembly
    ├── pipeline.py            End-to-end orchestration
    ├── models.py              Model loading and robust scoring
    └── plots.py               Score visualisation and statistics
```

---

## Usage

```bash
python run_treecrispr.py -i <input.fa> -o <output.csv> --mode <i|a>
```

| Argument | Required | Description |
|---|---|---|
| `-i` / `--input` | Yes | Path to input FASTA file (`.fa` or `.fasta`) |
| `-o` / `--output` | Yes | Path for the output CSV file |
| `--mode` | Yes | `i` = CRISPRi (interference), `a` = CRISPRa (activation) |

### Examples

Score guides for CRISPRi:
```bash
python run_treecrispr.py -i promoters.fa -o results_crispri.csv --mode i
```

Score guides for CRISPRa:
```bash
python run_treecrispr.py -i promoters.fa -o results_crispra.csv --mode a
```

### Input FASTA format

Standard FASTA format. Sequences should be ≤ 500 bp (controlled by `MAX_SEQ_LEN` in `config.py`). Sequences longer than the limit are skipped with a warning.

For epigenetic feature extraction to work, the FASTA record IDs must embed genomic coordinates in the format:
```
>gene_name chr1:1,234,567-1,235,067
```
or any pattern containing `chrom:start-end`. If coordinates are absent, the pipeline runs in sequence-only mode (all epigenetic features are set to zero).

---

## Output format

The output CSV contains one row per guide-RNA candidate, with the following columns:

| Column | Description |
|---|---|
| `ID` | FASTA record identifier |
| `Start` | 0-based start of the 30-mer window within the input sequence |
| `End` | Exclusive end of the 30-mer window (Start + 30) |
| `Strand` | `+` (forward NGG) or `-` (reverse CCN) |
| `Sequence` | 30-nt guide+PAM in NGG orientation (5′→3′) |
| `ReverseComplement` | Opposite-strand sequence |
| `PAM` | Exact 3-nt PAM (e.g. `AGG`, `TGG`, `CGG`) |
| `<ModelName>` | Score in [0, 1] from each loaded XGBoost model (one column per model) |

A higher score indicates a guide predicted to have stronger CRISPRi / CRISPRa activity. Missing values (`NaN`) in a model column indicate that the model failed to produce a prediction for that candidate (all three scoring attempts failed — see `models.py`).

---

## Feature description

### Sequence features (computed in `features_seq.py`)

| Feature | Description |
|---|---|
| `A`, `T`, `G`, `C` | Raw nucleotide counts in the 30-mer |
| `GCcount` | Total G + C count |
| `GChigh` / `GClow` | Binary flag: GC > 10 / GC ≤ 10 |
| `Entropy` | Shannon entropy of the nucleotide distribution (bits) |
| `MeltingTemperature` | Estimated melting temperature (°C) |
| `Energy` | RNAfold minimum free energy (kcal/mol); `NaN` if RNAfold not installed |
| `AA` … `TT` (16 cols) | Global dinucleotide counts |
| `pos0_A` … `pos29_C` (120 cols) | Positional mononucleotide one-hot features |
| `di0_AA` … `di28_TT` (464 cols) | Positional dinucleotide one-hot features |

> The `pos{i}_{N}` and `di{i}_{NN}` columns are automatically renamed to `{N}{i+1}` and `{NN}{i+1}` (e.g. `A1`, `AA1`) when needed to match R-trained XGBoost models.

### Epigenetic features (computed in `features_epi.py` + `epi_seq.py`)

For each of the 13 BigWig tracks, signal is aggregated (sum by default) over 6 window sizes centred on the guide locus. This yields 13 × 6 = 78 feature columns named `{track}_{extension}` (e.g. `H3K27ac_0`, `H3K27ac_50`, …, `H3K27ac_2500`).

Window extension options (configurable via the `EPIG_EXTS` environment variable):

| Extension (bp) | Biological interpretation |
|---|---|
| 0 | Signal directly overlapping the guide |
| 50 | Immediate promoter vicinity |
| 150 | Nucleosome-scale window |
| 250 | Core promoter region |
| 500 | Proximal regulatory region |
| 2500 | Broader chromatin domain |

---

## Visualisation

To generate score distribution plots and pairwise statistics from a results CSV, use the `plots` module directly:

```python
import pandas as pd
from treecrispr.plots import generate_boxplot_and_stats

df = pd.read_csv("results_crispri.csv")
stats = generate_boxplot_and_stats(df, out_dir="plots/")

print(stats["ranking_text"])   # e.g. "H3K27ac > H3K4me3 = H3K9me3 > …"
print(stats["stats"])          # pairwise Mann-Whitney U table
```

Output files written to `out_dir`:

| File | Contents |
|---|---|
| `scores_boxplot.png` | Box-and-whisker plot of each model's score distribution |
| `scores_ridgeline.png` | Ridgeline density plot (one ridge per model) |
| `scores_rank_heatmap.png` | Kendall τ-b rank-concordance matrix between models |
| `scores_dominance_heatmap.png` | P(model A > model B) per guide |
| `pairwise_stats.csv` | Mann–Whitney U p-value, Cliff's δ, Hodges–Lehmann shift, medians |

---

## Troubleshooting

**`Could not import the 'treecrispr' package`**
Run the script from the repository root: `cd TreeCRISPRstandalone && python run_treecrispr.py …`

**`Model directory missing`**
Download the model files from <https://epitree.igs.umaryland.edu/epitree/> and ensure `model_crispri/` or `model_crispra/` exists in the repository root.

**`pyBigWig` import error**
Install via conda: `conda install -c bioconda pybigwig` or pip: `pip install pyBigWig`. On Linux you may need `sudo apt-get install libcurl4-openssl-dev` first.

**All epigenetic features are zero**
Either (a) the BigWig directory is empty/missing, (b) your FASTA IDs do not contain genomic coordinates, or (c) the chromosome names in the BigWig files do not match (e.g. `1` vs `chr1` — the tool tries both automatically).

**Pipeline produces no candidates**
Check that your sequences contain valid NGG PAM sites. The tool requires strict `[ATGC]GG` PAMs — sequences with all ambiguity codes at the N position will produce no hits.

**RNAfold not found / `Energy` = NaN**
The `Energy` (MFE) feature requires ViennaRNA. Install via `conda install -c bioconda viennarna`. Scoring continues without it; the feature column will be `NaN` (filled to 0 before model prediction).

---


## Citation

If you use TreeCRISPR in your research, please cite the original publication associated with the EpiTree server:

> EpiTree: <https://epitree.igs.umaryland.edu/epitree/>

Please check the server for the most up-to-date citation information.
