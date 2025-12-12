# config.py (English)
# Configuration file for cueEM experiments.

LANGUAGE = "en"

# Third-party sources (URLs required by the journal)
THIRD_PARTY_DATA_SOURCES = {
    "DeepMatcher (ER_Magellan benchmark)": "https://github.com/anhaidgroup/deepmatcher",
    "Ditto (ER_Magellan data directory)": "https://github.com/megagonlabs/ditto/tree/master/data/er_magellan",
    "Magellan Data Repository (landing page)": "https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository",
}

PRETRAINED_MODEL = {
    "name": "google-bert/bert-base-uncased",
    "url": "https://hf-mirror.com/google-bert/bert-base-uncased",
    # You may also set a local path, e.g., "./bert-base-uncased"
}

# Default data directory (COL/VAL formatted train.txt and test.txt)
DATA_DIR = "data/er_magellan/Structured"

# Supported datasets and their "overall" templates
TEMPLATES = {
    "Amazon-Google": "[title] by [manufacturer] now only [price]",
    "Beer": "[Beer_Name] crafted by [Brew_Factory_Name] is a [Style] beer with [ABV]",
    "Fodors-Zagats": "[name] from [class] [addr] [city] is [type] and phone is [phone]",
    "iTunes-Amazon": "[Song_Name] by [Artist_Name] from [Album_Name] in [Genre] released [Released] [CopyRight] now only [Price] with Duration [Time]",
    "Walmart-Amazon": "[title] from [brand] [category] [modelno] now only [price]",
    "DBLP-ACM": "[title] by [authors] at [venue] in [year]",
    "DBLP-GoogleScholar": "[title] by [authors] at [venue] in [year]",
    "Abt-Buy": "[name] with [description] now only [price]",
}

# Per-dataset attribute weights used by WeightedClassifier.
# NOTE: The last chunk typically corresponds to the "overall" sentence pair.
ATTR_WEIGHTS = {
    "Amazon-Google": [0.8, 0.2, 0.8, 0.5],
    "Beer": [0.8, 0.8, 0.1, 0.8, 0.5],
    "Fodors-Zagats": [0.8, 0.8, 0.2, 0.8, 0.2, 0.8, 0.5],
    "iTunes-Amazon": [0.8, 0.1, 0.8, 0.1, 0.8, 0.1, 0.8, 0.1, 0.5],
    "Walmart-Amazon": [0.01, 0.01, 0.01, 1.0, 1.0, 1.0],
    "DBLP-ACM": [1.0, 1.0, 0.0, 1.0, 1.0],
    "DBLP-GoogleScholar": [1.0, 1.0, 1.0, 0.0, 1.0],
    "Abt-Buy": [0.8, 0.1, 0.2, 0.5],
}

# Training defaults (documented for reproducibility)
DEFAULTS = {
    "max_len": 256,
    "epochs": 10,
    "batch_size": 64,
    "lr": 3e-5,
    "lambda_l2": 1e-4,
    "seed": 42,
}

# Computing environment (documented for reproducibility)
COMPUTING_ENVIRONMENT = {
    "gpu": "NVIDIA A10",
    "python": "3.13",
    "torch": "2.7.0",
    "transformers": "4.49.0",
}
