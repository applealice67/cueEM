import os
import re
import csv
import math
import random
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from tqdm import tqdm


# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------
# Stopwords (your custom list)
# -----------------------
STOP_WORDS = set([
    'such', "you'd", 'y', 't', 'down', 'i', 'by', 'whom', 'most', 'his', 'does', 'are',
    'between', 're', 'isn', 'only', 'she', 'of', 'had', 'through', 'other', 'needn',
    'be', 'below', 'should', 'when', 'on', 'for', "don't", 'until', 'can', 'to', 'a',
    'from', 'has', "you'll", 'few', 'were', "that'll", 'while', 'just', "she's",
    "didn't", 'again', 'under', 'him', 'these', 'your', 'this', 'that', 'being',
    'doing', 'all', 'with', "haven't", 'didn', 'nor', 'they', 'where', 'our', 'them',
    'couldn', 'm', "needn't", 'me', 'you', 'we', 'than', "wouldn't", "shan't", 'ma',
    'won', 'yourselves', 'wouldn', 'haven', "it's", 'against', 'ain', 'have', 's',
    'any', 'do', 'himself', 'there', 'what', 'myself', 'both', 've', 'up', 'mustn',
    'or', 'wasn', 'into', 'which', "shouldn't", 'hadn', 'as', 'own', 'o', 'mightn',
    'an', 'don', 'her', 'weren', 'itself', 'those', 'how', 'hers', "mightn't", 'is',
    'was', "wasn't", 'before', 'if', 'it', 'will', 'once', 'did', 'same', "hadn't",
    'now', 'll', 'no', 'shan', "you're", 'too', 'aren', 'he', 'some', 'my', 'over',
    "doesn't", 'shouldn', "isn't", 'ourselves', 'd', 'am', 'themselves', "aren't",
    'off', 'having', 'in', "hasn't", 'further', "mustn't", 'yourself', 'ours',
    'theirs', 'here', 'more', 'so', "won't", 'very', "should've", 'out', 'the',
    'and', 'who', 'their', 'but', "couldn't", 'hasn', 'doesn', 'not', 'above',
    'because', 'about', 'its', 'during', "weren't", 'herself', 'been', 'yours',
    "you've", 'why', 'after', 'then', 'each', 'at'
])


def no_stopwords(line: str) -> str:
    tokens = str(line).split()
    filtered = [w.lower() for w in tokens if w.lower() not in STOP_WORDS]
    return " ".join(filtered)


# -----------------------
# Numeric helpers
# -----------------------
_number_pattern = re.compile(r"\d+(\.\d+)?")

def extract_number(s: str) -> float:
    m = _number_pattern.search(str(s))
    return float(m.group()) if m else 0.0

def subtract_numbers(a: str, b: str) -> int:
    return int(abs(extract_number(a) - extract_number(b)))


# -----------------------
# Templates
# -----------------------
def get_template(dataset: str) -> str:
    if dataset == "Amazon-Google":
        return "[title] by [manufacturer] now only [price]"
    if dataset == "Beer":
        return "[Beer_Name] crafted by [Brew_Factory_Name] is a [Style] beer with [ABV]"
    if dataset == "Fodors-Zagats":
        return "[name] from [class] [addr] [city] is [type] and phone is [phone]"
    if dataset == "iTunes-Amazon":
        return "[Song_Name] by [Artist_Name] from [Album_Name] in [Genre] released [Released] [CopyRight] now only [Price] with Duration [Time]"
    if dataset == "Walmart-Amazon":
        return "[title] from [brand] [category] [modelno] now only [price]"
    if dataset == "DBLP-ACM":
        return "[title] by [authors] at [venue] in [year]"
    if dataset == "DBLP-GoogleScholar":
        return "[title] by [authors] at [venue] in [year]"
    if dataset == "Abt-Buy":
        return "[name] with [description] now only [price]"
    return ""


def replace_template(template: str, row: Dict[str, Tuple[str, str]]) -> Tuple[str, str]:
    s1, s2 = template, template
    for key, (t1, t2) in row.items():
        if key == "label":
            continue
        ph = "[" + key + "]"
        s1 = s1.replace(ph, str(t1))
        s2 = s2.replace(ph, str(t2))
    return s1, s2


def build_overall(dataset: str, row: Dict[str, Tuple[str, str]]) -> Tuple[str, str]:
    template = get_template(dataset)
    return replace_template(template, row)


# -----------------------
# Parsing COL/VAL
# -----------------------
def parse_line(dataset: str, line: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    tokens = line.strip().split()

    key = None
    value: List[str] = []

    for token in tokens:
        if token == "COL":
            if key:
                v = " ".join(value)
                if key in result:
                    if isinstance(result[key], tuple):
                        result[key] += (v,)
                    else:
                        result[key] = (result[key], v)
                else:
                    result[key] = v
            key = None
            value = []
        elif token == "VAL":
            continue
        elif key is None:
            key = token
        else:
            value.append(token)

    if key:
        v = " ".join(value)
        if key in result:
            if isinstance(result[key], tuple):
                result[key] += (v,)
            else:
                result[key] = (result[key], v)
        else:
            result[key] = v

    # label is the last token in the last field value
    if value and value[-1].isdigit():
        result["label"] = int(value[-1])

    label = int(result["label"])
    del result["label"]

    # normalize to (left,right)
    row: Dict[str, Tuple[str, str]] = {}
    for k, v in result.items():
        if isinstance(v, tuple) and len(v) >= 2:
            row[k] = (str(v[0]), str(v[1]))
        else:
            row[k] = (str(v), str(v))

    s1, s2 = build_overall(dataset, row)
    row["overall"] = (no_stopwords(s1), no_stopwords(s2))
    out: Dict[str, Any] = dict(row)
    out["label"] = label
    return out


def parse_file(dataset: str, file_path: str) -> Dict[str, List[Any]]:
    final_result: Dict[str, List[Any]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parse_line(dataset, line)
            for k, v in parsed.items():
                final_result.setdefault(k, []).append(v)

    # numeric post-processing (like your code)
    numeric_key = None
    for cand in ["price", "ABV", "Price", "year"]:
        if cand in final_result:
            numeric_key = cand
            break

    if numeric_key is not None:
        processed = []
        for (lv, rv) in final_result[numeric_key]:
            diff = subtract_numbers(lv, rv)
            processed.append(("0", str(diff)))
        final_result[numeric_key] = processed

    return final_result


def dict_to_rows(dic_data: Dict[str, List[Any]]) -> List[List[Any]]:
    keys = list(dic_data.keys())
    print("[Info] keys:", keys)
    rows: List[List[Any]] = []
    n = len(dic_data["label"])
    for i in range(n):
        rows.append([dic_data[k][i] for k in keys])
    return rows


def balance_alternate(rows: List[List[Any]]) -> List[List[Any]]:
    pos, neg = [], []
    for r in rows:
        if int(r[-1]) == 1:
            pos.append(r)
        else:
            neg.append(r)
    if not pos or not neg:
        return rows

    m = max(len(pos), len(neg))
    out = []
    for i in range(m):
        out.append(pos[i % len(pos)])
        out.append(neg[i % len(neg)])
    return out


# -----------------------
# Attribute weights (your mapping)
# -----------------------
def get_attr_weight(dataset: str) -> List[float]:
    if dataset == "Amazon-Google":
        return [0.8, 0.2, 0.8, 0.5]
    if dataset == "Beer":
        return [0.8, 0.8, 0.1, 0.8, 0.5]
    if dataset == "Fodors-Zagats":
        return [0.8, 0.8, 0.2, 0.8, 0.2, 0.8, 0.5]
    if dataset == "iTunes-Amazon":
        return [0.8, 0.1, 0.8, 0.1, 0.8, 0.1, 0.8, 0.1, 0.5]
    if dataset == "Walmart-Amazon":
        return [0.01, 0.01, 0.01, 1.0, 1.0, 1.0]
    if dataset == "DBLP-ACM":
        return [1.0, 1.0, 0.0, 1.0, 1.0]
    if dataset == "DBLP-GoogleScholar":
        return [1.0, 1.0, 1.0, 0.0, 1.0]
    if dataset == "Abt-Buy":
        return [0.8, 0.1, 0.2, 0.5]
    raise ValueError(f"Unsupported dataset: {dataset}")


# -----------------------
# Torch Dataset
# -----------------------
class SentenceSimilarityDataset(Dataset):
    def __init__(self, data: List[List[Any]], tokenizer: BertTokenizer, max_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        label = int(self.data[idx][-1])
        pairs = self.data[idx][:-1]  # list[(t1,t2), ...]
        encs = []
        for t1, t2 in pairs:
            enc = self.tokenizer(
                str(t1),
                str(t2),
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
                add_special_tokens=True,
            )
            encs.append(enc)

        input_ids = [e["input_ids"].squeeze(0) for e in encs]  # [L]
        attn = [e["attention_mask"].squeeze(0) for e in encs]
        if "token_type_ids" in encs[0]:
            tti = [e["token_type_ids"].squeeze(0) for e in encs]
        else:
            tti = [torch.zeros_like(e["input_ids"].squeeze(0)) for e in encs]

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "token_type_ids": tti,
            "labels": torch.tensor(label, dtype=torch.float32),
        }


# -----------------------
# Model
# -----------------------
class WeightedClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_chunks: int, attr_weight: List[float]):
        super().__init__()
        self.num_chunks = num_chunks
        self.linear = nn.Linear(hidden_size * num_chunks, 1)
        w = torch.tensor(attr_weight, dtype=torch.float32)
        if len(w) != num_chunks:
            raise ValueError(f"attrWeight length {len(w)} != num_chunks {num_chunks}")
        self.weights = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.chunk(self.num_chunks, dim=1)
        weighted = [c * self.weights[i] for i, c in enumerate(chunks)]
        wx = torch.cat(weighted, dim=1)
        return self.linear(wx)


# -----------------------
# Train / Eval
# -----------------------
def evaluate(
    model: BertModel,
    classifier: WeightedClassifier,
    loader: DataLoader,
    device: torch.device,
    pair_num: int,
) -> Tuple[float, float, float, float, List[int], List[int]]:
    model.eval()
    classifier.eval()

    preds: List[int] = []
    golds: List[int] = []

    with torch.no_grad():
        for batch in loader:
            # batch contains list tensors for each pair
            labels = batch["labels"].to(device).view(-1, 1)

            outputs_list = []
            for i in range(pair_num):
                input_ids = batch["input_ids"][i].to(device)
                attention_mask = batch["attention_mask"][i].to(device)
                token_type_ids = batch["token_type_ids"][i].to(device)

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                outputs_list.append(out.last_hidden_state[:, 0, :])

            combined = torch.cat(outputs_list, dim=1)
            logits = classifier(combined)
            pred = (torch.sigmoid(logits) >= 0.5).long().view(-1).tolist()

            preds.extend(pred)
            golds.extend(labels.long().view(-1).tolist())

    acc = accuracy_score(golds, preds)
    rec = recall_score(golds, preds, zero_division=0)
    prec = precision_score(golds, preds, zero_division=0)
    f1 = f1_score(golds, preds, zero_division=0)
    return acc, prec, rec, f1, golds, preds


def train_and_eval(
    dataset: str,
    data_dir: str,
    bert_model_path: str,
    max_len: int,
    epochs: int,
    lr: float,
    lambda_l2: float,
    batch_size: int,
    seed: int,
) -> None:
    set_seed(seed)

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = os.path.join("logs", f"{dataset}_{ts}.log")
    logging.basicConfig(
        filename=logfile,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Info] device:", device)
    logging.info(f"device={device}")

    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    model = BertModel.from_pretrained(bert_model_path).to(device)

    train_txt = os.path.join(data_dir, dataset, "train.txt")
    test_txt = os.path.join(data_dir, dataset, "test.txt")

    train_dic = parse_file(dataset, train_txt)
    test_dic = parse_file(dataset, test_txt)

    train_rows = balance_alternate(dict_to_rows(train_dic))
    test_rows = dict_to_rows(test_dic)

    pair_num = len(train_rows[0]) - 1
    print(f"[Info] pair_num={pair_num}, train={len(train_rows)}, test={len(test_rows)}")
    logging.info(f"pair_num={pair_num}, train={len(train_rows)}, test={len(test_rows)}")

    # also write train/test csv like your code
    os.makedirs(os.path.join(data_dir, dataset), exist_ok=True)
    with open(os.path.join(data_dir, dataset, "train.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in train_rows:
            w.writerow(r)
    with open(os.path.join(data_dir, dataset, "test.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in test_rows:
            w.writerow(r)

    attr_weight = get_attr_weight(dataset)

    classifier = WeightedClassifier(model.config.hidden_size, pair_num, attr_weight).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=lr,
        weight_decay=lambda_l2,
    )

    train_ds = SentenceSimilarityDataset(train_rows, tokenizer, max_len)
    test_ds = SentenceSimilarityDataset(test_rows, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    save_path = os.path.join(data_dir, dataset, "best_model.pth")
    result_path = os.path.join(data_dir, dataset, "result.csv")

    best_f1 = -1.0

    for epoch in range(epochs):
        model.train()
        classifier.train()

        running = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for batch in pbar:
            labels = batch["labels"].to(device).view(-1, 1)

            outputs_list = []
            for i in range(pair_num):
                input_ids = batch["input_ids"][i].to(device)
                attention_mask = batch["attention_mask"][i].to(device)
                token_type_ids = batch["token_type_ids"][i].to(device)

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                outputs_list.append(out.last_hidden_state[:, 0, :])

            combined = torch.cat(outputs_list, dim=1)
            logits = classifier(combined)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            steps += 1
            pbar.set_postfix(loss=running / max(steps, 1))

        train_loss = running / max(steps, 1)
        logging.info(f"Epoch {epoch+1} TrainLoss={train_loss:.6f}")

        acc, prec, rec, f1, _, _ = evaluate(model, classifier, test_loader, device, pair_num)
        print(f"[Eval] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
        logging.info(f"Eval acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "dataset": dataset,
                    "pair_num": pair_num,
                    "max_len": max_len,
                    "bert_model": bert_model_path,
                    "model_state_dict": model.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "attr_weight": attr_weight,
                    "timestamp": datetime.now().isoformat(),
                },
                save_path,
            )
            print(f"[Best] best_f1={best_f1:.4f} saved={save_path}")
            logging.info(f"Best f1={best_f1:.4f}")

    # load best and final eval + save predictions
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])

    learned_w = classifier.weights.data.detach().cpu().tolist()
    print("[Info] Loaded best weights:", learned_w)
    logging.info(f"Loaded best weights: {learned_w}")

    acc, prec, rec, f1, golds, preds = evaluate(model, classifier, test_loader, device, pair_num)
    print(f"[Final] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
    logging.info(f"Final acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    with open(result_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "predict"])
        for i, (y, p) in enumerate(zip(golds, preds)):
            w.writerow([i, int(y), int(p)])

    print("[Done] saved:", result_path)
    print("[Done] log:", logfile)


def parse_args():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="DBLP-ACM")
    ap.add_argument("--data_dir", type=str, default="data/er_magellan/Structured")
    ap.add_argument("--bert_model", type=str, default="./bert-base-uncased",
                    help="Local path or HF model id (e.g., ./bert-base-uncased or google-bert/bert-base-uncased)")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lambda_l2", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_eval(
        dataset=args.dataset,
        data_dir=args.data_dir,
        bert_model_path=args.bert_model,
        max_len=args.max_len,
        epochs=args.epochs,
        lr=args.lr,
        lambda_l2=args.lambda_l2,
        batch_size=args.batch_size,
        seed=args.seed,
    )
