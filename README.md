# cueEM: Human-Like Entity Matching via Text-Centric Hybrid Attention

## Title
cueEM — Field-weighted BERT for Entity Matching on Structured ER Benchmarks

## Description
This repository contains the code for a BERT-based entity matching (record linkage) model on structured datasets.
Each record pair is converted into multiple sentence pairs (per attribute + an "overall" template). We encode each
sentence pair with BERT using the [CLS] embedding, concatenate embeddings across attributes, and classify with a
weighted linear classifier.

## Code Availability
- Code repository: https://github.com/applealice67/cueEM

## Dataset Information
Datasets are obtained from the Ditto project repository:
- Ditto datasets: https://github.com/megagonlabs/ditto

### Expected local data layout
The script expects COL/VAL formatted files placed as:
<data_dir>/<DATASET_NAME>/train.txt
<data_dir>/<DATASET_NAME>/test.txt

Default `data_dir` used in the script:
data/er_magellan/Structured

### Supported datasets in this script
- Amazon-Google
- Beer
- Fodors-Zagats
- iTunes-Amazon
- Walmart-Amazon
- DBLP-ACM
- DBLP-GoogleScholar
- Abt-Buy

### Input format (COL/VAL)
Each line contains attribute fields marked by `COL` and `VAL` tokens and ends with a binary label (0/1).
The parser converts each attribute into a tuple (left_value, right_value) and constructs an additional "overall"
sentence pair.

## Pretrained Model Information
We use the pretrained BERT model:
- Model: google-bert/bert-base-uncased
- Source URL: https://hf-mirror.com/google-bert/bert-base-uncased

You can provide either:
- a local directory (e.g., `./bert-base-uncased`), or
- a Hugging Face model id (e.g., `google-bert/bert-base-uncased`)

## Code Information (What is implemented)
Main script: `run.py`

Key components:
- `parse_line()` / `parse_file()`: parse COL/VAL formatted input into attribute pairs and label.
- Template serialization (`get_template()` + `replace_template()`): construct the "overall" sentence pair.
- Stopword removal: applied to the "overall" sentence pair (custom stopword list in code).
- Numeric preprocessing: for numeric attributes (price/ABV/year if present), replace the pair with absolute difference.
- `WeightedClassifier`: applies per-attribute weights (`attrWeight`) and outputs a binary match logit.

## Computing Infrastructure (Reproducibility)
Experiments were run on:
- GPU: NVIDIA A10
- Python: 3.13
- PyTorch: 2.7.0
- Transformers: 4.49.0


## Requirements (Dependencies)
Install dependencies:
```bash
pip install -r requirements.txt
Minimal dependencies used by the code:

torch==2.7.0
transformers==4.49.0
numpy
scikit-learn
tqdm

Usage Instructions
1) Prepare data
Download datasets from Ditto and place the processed files to match:
<data_dir>/<DATASET_NAME>/train.txt
<data_dir>/<DATASET_NAME>/test.txt
2) Train + Evaluate
Example for DBLP-ACM:

python run.py \
  --dataset DBLP-ACM \
  --data_dir data/er_magellan/Structured \
  --bert_model ./bert-base-uncased \
  --max_len 256 \
  --epochs 10 \
  --batch_size 64 \
  --lr 3e-5 \
  --lambda_l2 1e-4 \
  --seed 42
3) Outputs
The script writes:

Checkpoint: <data_dir>/<DATASET_NAME>/best_model.pth

Predictions: <data_dir>/<DATASET_NAME>/result.csv

Exported train/test (for inspection): <data_dir>/<DATASET_NAME>/train.csv, <data_dir>/<DATASET_NAME>/test.csv

Logs: ./logs/<DATASET_NAME>_<timestamp>.log

Methodology (Brief)
Parse each COL/VAL line into attribute-level pairs (left_value, right_value) and a binary label.

Construct an "overall" sentence pair by filling a dataset-specific template with attributes.

Remove stopwords from the "overall" sentence pair.

Encode each sentence pair with BERT and use the [CLS] representation.

Concatenate representations across attributes.

Apply a weighted linear classifier (attrWeight) and train with BCEWithLogitsLoss.

Address class imbalance by alternating positive and negative pairs in the training set.

Citations / References
Ditto datasets: https://github.com/megagonlabs/ditto

BERT model: https://hf-mirror.com/google-bert/bert-base-uncased

Code: https://github.com/applealice67/cueEM
