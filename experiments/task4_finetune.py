"""
Task 4  DistilBERT Fine-Tuning on Yelp

Design decisions:
  - DistilBERT-base-uncased: 66M params, ~250MB, fits 4GB VRAM
  - fp16=True:halves VRAM usage
  - Macro-F1 as primary metric: same as Tasks 1-2, enables direct comparison logically 
  - Best model checkpoint saved: used by domain shift + adversarial scripts

Saves:
  models/distilbert-yelp/     ← model checkpoint
  results/task4_yelp_metrics.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
import random

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(ROOT, "models", "distilbert-yelp")
RESULTS_DIR= os.path.join(ROOT, "results")
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED         = 42
N_TRAIN_PER_CLASS = 3000   
N_TEST_PER_CLASS  = 200    
MODEL_NAME   = "distilbert-base-uncased"
NUM_LABELS   = 5
MAX_LENGTH   = 256         


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_sample(dataset, n_per_class: int, seed: int = SEED) -> list:
    """Stratified sample from HuggingFace dataset. Returns list of dicts."""
    by_label = defaultdict(list)
    for item in dataset:
        by_label[item["label"]].append(item)
    random.seed(seed)
    samples = []
    for label in range(5):
        pool = by_label[label]
        chosen = random.sample(pool, min(n_per_class, len(pool)))
        samples.extend(chosen)
    random.shuffle(samples)
    return samples


def load_yelp_splits(n_train: int, n_test: int):
    """Returns train and test samples as lists of dicts."""
    print("Loading Yelp Full dataset...")
    train_ds = load_dataset("yelp_review_full", split="train")
    test_ds  = load_dataset("yelp_review_full", split="test")
    train = stratified_sample(train_ds, n_train)
    test  = stratified_sample(test_ds,  n_test)
    print(f"  Train: {len(train)} | Test: {len(test)}")
    return train, test


class YelpDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list, tokenizer, max_length: int = MAX_LENGTH):
        self.encodings = tokenizer(
            [s["text"] for s in samples],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # yelp labels are 0-indexed, keeping as-is (0->label 0 = 1 star, etc.)
        self.labels = torch.tensor([s["label"] for s in samples], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc  = accuracy_score(labels, preds)
    mf1  = f1_score(labels, preds, average="macro",
                    labels=list(range(5)), zero_division=0)
    return {"accuracy": round(acc, 4), "macro_f1": round(mf1, 4)}


def compute_full_metrics(true_labels, pred_labels) -> dict:
    """Full metrics including per-class F1 and error analysis."""
    acc = accuracy_score(true_labels, pred_labels)
    mf1 = f1_score(true_labels, pred_labels, average="macro",
                   labels=list(range(5)), zero_division=0)
    per_class = f1_score(true_labels, pred_labels, average=None,
                         labels=list(range(5)), zero_division=0)

    errors = [(t, p) for t, p in zip(true_labels, pred_labels) if t != p]
    deltas = [abs(t - p) for t, p in errors]

    return {
        "accuracy":     round(float(acc), 4),
        "macro_f1":     round(float(mf1), 4),
        "per_class_f1": {str(i+1): round(float(v), 4) for i, v in enumerate(per_class)},
        "error_analysis": {
            "total_errors": len(errors),
            "off_by_one":   sum(1 for d in deltas if d == 1),
            "off_by_two":   sum(1 for d in deltas if d == 2),
            "off_by_more":  sum(1 for d in deltas if d >= 3),
            "off_by_one_pct": round(sum(1 for d in deltas if d == 1) / len(errors), 3)
                              if errors else 0.0,
        },
    }


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data 
    train_samples, test_samples = load_yelp_splits(N_TRAIN_PER_CLASS, N_TEST_PER_CLASS)

    # Tokenizing
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(MODEL_DIR)

    train_ds = YelpDataset(train_samples, tokenizer)
    test_ds  = YelpDataset(test_samples,  tokenizer)

    # Model
    print(f"Loading model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params/1e6:.1f}M")

    #training args
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=3e-5, # explicitly lower than default 5e-5 for better convergence
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=(device == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=os.path.join(RESULTS_DIR, "logs"), 
        logging_steps=10,
        save_total_limit=1, # this automatically deletes old checkpoints
        seed=SEED,
        report_to="none",
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n━━━ Training ━━━")
    train_result = trainer.train()
    print(f"  Training complete in {train_result.metrics['train_runtime']:.0f}s")

    # evaluate on yelp test
    print("\n━━━ Evaluating on Yelp Test Set ━━━")
    preds_output = trainer.predict(test_ds)
    pred_labels  = np.argmax(preds_output.predictions, axis=-1).tolist()
    true_labels  = [s["label"] for s in test_samples]

    metrics = compute_full_metrics(true_labels, pred_labels)
    metrics["domain"]     = "yelp"
    metrics["n_samples"]  = len(test_samples)
    metrics["train_size"] = len(train_samples)
    metrics["model"]      = MODEL_NAME
    metrics["epochs"]     = int(training_args.num_train_epochs)

    print(f"\n{'='*52}")
    print(f"  DistilBERT — Yelp Test Results")
    print(f"{'='*52}")
    print(f"  Accuracy    : {metrics['accuracy']}")
    print(f"  Macro-F1    : {metrics['macro_f1']}")
    print(f"  Per-class F1:")
    for star, f1 in metrics["per_class_f1"].items():
        print(f"    {star}★ : {f1:.4f}")
    err = metrics["error_analysis"]
    print(f"  Total errors: {err['total_errors']} | Off-by-one: {err['off_by_one']} ({err['off_by_one_pct']:.1%})")

    #compare vs LLM approaches
    print(f"\n{'='*52}")
    print(f"  Cross-Task Comparison (same Yelp test domain)")
    print(f"{'='*52}")
    print(f"  {'Approach':<28} {'Accuracy':>9}  {'Macro-F1':>9}")
    print(f"  {'─'*50}")
    print(f"  {'T1: Zero-Shot LLM':<28} {'0.620':>9}  {'0.6006':>9}")
    print(f"  {'T1: Few-Shot LLM':<28} {'0.650':>9}  {'0.6390':>9}")
    print(f"  {'T2: Direct LLM':<28} {'0.665':>9}  {'0.6545':>9}")
    print(f"  {'T2: CoT LLM':<28} {'0.618':>9}  {'0.6090':>9}")
    print(f"  {'T4: DistilBERT (fine-tuned)':<28} {metrics['accuracy']:>9}  {metrics['macro_f1']:>9}")

    #save model + metrics
    trainer.save_model(MODEL_DIR)
    with open(os.path.join(RESULTS_DIR, "task4_yelp_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved → {MODEL_DIR}")
    print(f"  Saved → results/task4_yelp_metrics.json")

    return metrics


if __name__ == "__main__":
    main()