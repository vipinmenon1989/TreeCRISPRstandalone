# TreeCRISPR

## Installation

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/vipinmenon1989/TreeCRISPRstandalone/tree/main
cd TreeCRISPR
```
## Setup Environment
Install the required dependencies using the provided YAML file. This creates the execution environment:

```
conda env create -f TreeCRISPR.yml
conda activate TreeCRISPR
```

## Download Model & BigWig Files

The model and BigWig files are not included in this GitHub repository due to size. You must download them separately.

Visit: https://epitree.igs.umaryland.edu/epitree/

Download the ```model``` and ```bigwig``` files.

Place them in the ```TreeCRISPR folder```. Your directory structure must look like this:

./bigwig/

./model_crispra/

./model_crispri/

## Usage
Run the tool using ```run_treecrispr.py```. You must specify the input FASTA file and the mode.
```python run_treecrispr.py <input_file> --mode <mode_type>```

## Arguments

```<input_file>```: Path to the input FASTA file (e.g., ```input.fa```).

```--mode:```

``` i ``` : CRISPRi (Interference)

``` a ``` : CRISPRa (Activation)

## Examples

Run CRISPRi (Interference):

``` python run_treecrispr.py input.fa --mode i ```

Run CRISPRa (Activation)

``` python run_treecrispr.py input.fa --mode a ```
