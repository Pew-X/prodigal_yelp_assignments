"""
Task 4 — Domain Shift Evaluation

Tests the Yelp-fine-tuned DistilBERT on out-of-domain datasets.

Label alignment strategy (critical design decision):
  Yelp Full  : 5-class (0-4 -> 1-5star)         ->direct evaluation
  Amazon Polarity: binary (0=neg, 1=pos)    -> map 0->1star, 1->5star (extreme classes)
  IMDB       : binary (0=neg, 1=pos)         -> map 0->1star, 1->5star (extreme classes)

For binary datasets, we evaluate "directional accuracy":
  negative review correctly predicted as 1star or 2star
  positive review correctly predicted as 4star or 5star
This is an honest proxy for domain transfer quality.

Note: testing on binary extremes underestimates real degradation because
the model never sees ambiguous 3star examples actual degradation on
full 5-class Amazon data could be worse.

Saves:
  results/task4_domain_shift.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(ROOT, "models", "distilbert-yelp")
RESULTS_DIR = os.path.join(ROOT, "results")
SEED        = 42
MAX_LENGTH  = 256
N_PER_CLASS = 250   # 500 total for binary datasets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = MAX_LENGTH):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def predict(model, dataset, device, batch_size=64) -> np.ndarray:
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            all_logits.append(out.logits.cpu().numpy())
    logits = np.vstack(all_logits)
    return np.argmax(logits, axis=-1), torch.softmax(torch.tensor(logits), dim=-1).numpy()


def eval_yelp(model, tokenizer, device) -> dict:
    """Re-evaluate on Yelp test  baseline for comparison."""
    print("  Loading Yelp test split...")
    ds = load_dataset("yelp_review_full", split="test")
    by_label = defaultdict(list)
    for item in ds:
        by_label[item["label"]].append(item)
    random.seed(SEED)
    samples = []
    for label in range(5):
        samples.extend(random.sample(by_label[label], 200))
    texts  = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]

    dataset = InferenceDataset(texts, labels, tokenizer)
    preds, probs = predict(model, dataset, device)

    acc = accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average="macro", labels=list(range(5)), zero_division=0)
    return {
        "domain": "yelp",
        "n": len(samples),
        "label_type": "5-class (0-4)",
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(mf1), 4),
        "avg_confidence": round(float(probs.max(axis=1).mean()), 4),
    }


def eval_amazon(model, tokenizer, device) -> dict:
    """
    Amazon Polarity: binary sentiment.
    Maps: label=0 (negative) -> true_star=0 (1star), label=1 (positive) -> true_star=4 (5star)
    Directional accuracy: negative predicted as ≤1 (0 or 1), positive predicted as ≥3 (4 or 5)
    """
    print("  Loading Amazon Polarity...")
    ds = load_dataset("amazon_polarity", split="test")
    by_label = defaultdict(list)
    for item in ds:
        by_label[item["label"]].append(item)
    random.seed(SEED)
    neg = random.sample(by_label[0], N_PER_CLASS)  # negative reviews
    pos = random.sample(by_label[1], N_PER_CLASS)  # positive reviews

    texts   = [s["content"] for s in neg] + [s["content"] for s in pos]
    #map to yelp label space: negative->0 (1star), positive->4 (5star)
    true_yelp_labels = [0] * N_PER_CLASS + [4] * N_PER_CLASS
    binary_sentiment = [0] * N_PER_CLASS + [1] * N_PER_CLASS  # 0=neg, 1=pos

    dataset = InferenceDataset(texts, true_yelp_labels, tokenizer)
    preds, probs = predict(model, dataset, device)

    #directional accuracy: did we predict correct direction?
    correct_direction = [
        (sent == 0 and pred <= 1) or (sent == 1 and pred >= 3)
        for sent, pred in zip(binary_sentiment, preds)
    ]
    dir_acc = sum(correct_direction) / len(correct_direction)

    #exact accuracy (0 or 4)
    exact_acc = accuracy_score(true_yelp_labels, preds)

    #prediction distribution (what stars does the model assign?)
    pred_dist = {str(i): int((preds == i).sum()) for i in range(5)}

    return {
        "domain": "amazon_polarity",
        "n": len(texts),
        "label_type": "binary->mapped to 1star/5star",
        "directional_accuracy": round(dir_acc, 4),
        "exact_accuracy": round(float(exact_acc), 4),
        "prediction_distribution": pred_dist,
        "avg_confidence": round(float(probs.max(axis=1).mean()), 4),
    }


def eval_imdb(model, tokenizer, device) -> dict:
    """
    IMDB: binary sentiment (movie reviews — different domain and genre).
    Same mapping strategy as Amazon.
    """
    print("  Loading IMDB...")
    ds = load_dataset("imdb", split="test")
    by_label = defaultdict(list)
    for item in ds:
        by_label[item["label"]].append(item)
    random.seed(SEED)
    neg = random.sample(by_label[0], N_PER_CLASS)
    pos = random.sample(by_label[1], N_PER_CLASS)

    texts   = [s["text"][:1000] for s in neg] + [s["text"][:1000] for s in pos]
    true_yelp_labels = [0] * N_PER_CLASS + [4] * N_PER_CLASS
    binary_sentiment = [0] * N_PER_CLASS + [1] * N_PER_CLASS

    dataset = InferenceDataset(texts, true_yelp_labels, tokenizer)
    preds, probs = predict(model, dataset, device)

    correct_direction = [
        (sent == 0 and pred <= 1) or (sent == 1 and pred >= 3)
        for sent, pred in zip(binary_sentiment, preds)
    ]
    dir_acc = sum(correct_direction) / len(correct_direction)

    pred_dist = {str(i): int((preds == i).sum()) for i in range(5)}

    return {
        "domain": "imdb",
        "n": len(texts),
        "label_type": "binary->mapped to 1star/5star",
        "directional_accuracy": round(dir_acc, 4),
        "exact_accuracy": round(float(accuracy_score(true_yelp_labels, preds)), 4),
        "prediction_distribution": pred_dist,
        "avg_confidence": round(float(probs.max(axis=1).mean()), 4),
    }


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading fine-tuned model from {MODEL_DIR}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    print("\n━━━ Domain Shift Evaluation ━━━")
    results = {}

    print("\n[1/3] Yelp (in-domain baseline)")
    results["yelp"] = eval_yelp(model, tokenizer, device)

    print("\n[2/3] Amazon Polarity (out-of-domain: e-commerce)")
    results["amazon"] = eval_amazon(model, tokenizer, device)

    print("\n[3/3] IMDB (out-of-domain: movie reviews)")
    results["imdb"] = eval_imdb(model, tokenizer, device)

    #print comparison table
    print(f"\n{'='*70}")
    print(f"  Domain Shift Results")
    print(f"{'='*70}")
    print(f"  {'Domain':<20} {'Metric':<25} {'Score':>8}  {'vs Yelp':>10}")
    print(f"  {'─'*65}")

    yelp_acc = results["yelp"]["accuracy"]
    yelp_mf1 = results["yelp"]["macro_f1"]

    print(f"  {'Yelp (in-domain)':<20} {'Accuracy (5-class)':<25} {yelp_acc:>8.4f}  {'baseline':>10}")
    print(f"  {'':20} {'Macro-F1':<25} {yelp_mf1:>8.4f}  {'baseline':>10}")
    print(f"  {'':20} {'Avg Confidence':<25} {results['yelp']['avg_confidence']:>8.4f}")

    for domain in ["amazon", "imdb"]:
        r = results[domain]
        dir_drop = yelp_acc - r["directional_accuracy"]
        print(f"\n  {domain.capitalize()+' (OOD)':<20} {'Directional Acc':<25} {r['directional_accuracy']:>8.4f}  {'-'+str(round(dir_drop,4)):>10}")
        print(f"  {'':20} {'Avg Confidence':<25} {r['avg_confidence']:>8.4f}")
        print(f"  {'':20} {'Pred distribution':<25} {r['prediction_distribution']}")

    #compute performance drop
    avg_dir_acc = np.mean([results[d]["directional_accuracy"] for d in ["amazon", "imdb"]])
    print(f"\n  Performance drop (Yelp->OOD avg): {yelp_acc - avg_dir_acc:.4f}")
    print(f"  Confidence drop (Yelp->OOD avg):  {results['yelp']['avg_confidence'] - np.mean([results[d]['avg_confidence'] for d in ['amazon','imdb']]):.4f}")

    results["summary"] = {
        "yelp_accuracy":    yelp_acc,
        "avg_ood_dir_acc":  round(float(avg_dir_acc), 4),
        "performance_drop": round(float(yelp_acc - avg_dir_acc), 4),
    }

    with open(os.path.join(RESULTS_DIR, "task4_domain_shift.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> results/task4_domain_shift.json")

    return results


if __name__ == "__main__":
    main()