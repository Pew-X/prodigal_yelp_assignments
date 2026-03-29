"""
Task 4 — Adversarial Robustness 

Tests the fine-tuned DistilBERT on 20 handcrafted adversarial examples.
Also tests the zero-shot LLM (Task 1 Direct approach) on the same examples.

Implements a test confidence-threshold mitigation that combines both.

Adversarial taxonomy (5 types, 4 examples each):
  1. negation_trap        : double negatives, negation of positive
  2. sarcasm             : positive language, negative intent
  3. sentiment_sandwich  : positive wrapper around negative core
  4. explicit_mismatch   : review explicitly mentions stars incorrectly
  5. domain_vocabulary   : technical/formal language that confuses sentiment

Mitigation: confidence-threshold ensemble
  if classifier_confidence >= threshold → use classifier prediction
  else → fall back to zero-shot LLM (Task 2 Direct prompt)
  This directly bridges Task 4 (fine-tuning) with Tasks 1-2 (LLM prompting)

Saves:
  results/task4_adversarial.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from src.llm_client import LLMClient
from src.prompts import SYSTEM_PROMPT, build_direct_prompt

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(ROOT, "models", "distilbert-yelp")
RESULTS_DIR = os.path.join(ROOT, "results")

# adversarial dataset following same convention as yelp


ADVERSARIAL_EXAMPLES = [
    #  Type 1: Negation Traps
    {
        "text": "I cannot say enough good things about this place. Not once did I leave disappointed, and I can't imagine not coming back.",
        "true_label": 4,  # 5 star   triple negation, clearly positive
        "adversarial_type": "negation_trap",
        "explanation": "Triple negation reads as positive but keyword 'not/cannot' may trigger negative classifier weights",
    },
    {
        "text": "There was nothing wrong with the food that a good chef couldn't fix, nothing about the service that wasn't fixable, and nothing that would stop a less discerning customer from returning.",
        "true_label": 1,  # 2star  backhanded non compliments
        "adversarial_type": "negation_trap",
        "explanation": "Double negatives construct apparent positivity over fundamentally critical content",
    },
    {
        "text": "Not bad. Not disappointing. Not what I expected at all — it was genuinely excellent.",
        "true_label": 4,  # 5star
        "adversarial_type": "negation_trap",
        "explanation": "Not+negative-word sequences that resolve to positive confuse bag-of-words models",
    },
    {
        "text": "I wouldn't say the wait was short, the parking wasn't easy, and the prices aren't cheap — but the food was transcendent.",
        "true_label": 3,  # 4star  overall positive despite negative clauses
        "adversarial_type": "negation_trap",
        "explanation": "Negated negatives in subordinate clauses with genuine positive in main clause",
    },

    # Type 2: Sarcasm 
    {
        "text": "Oh wonderful, another restaurant that confuses 'artisanal' with 'undercooked'. Truly a culinary revelation for those who enjoy paying $40 for a raw chicken breast.",
        "true_label": 0,  # 1star
        "adversarial_type": "sarcasm",
        "explanation": "Positive adjectives (wonderful, culinary revelation) used sarcastically — surface positive, intent negative",
    },
    {
        "text": "Five stars! Absolutely stunning that they managed to get my order wrong three times in one visit. A true masterclass in inefficiency.",
        "true_label": 0,  # 1star  explicitly mentions five stars but is deeply negative
        "adversarial_type": "sarcasm",
        "explanation": "Explicit star mention contradicts actual sentiment; emoji-free written sarcasm",
    },
    {
        "text": "Remarkable how they've perfected the art of being aggressively mediocre. Not easy to achieve such consistent disappointment at these prices.",
        "true_label": 1,  # 2star
        "adversarial_type": "sarcasm",
        "explanation": "Words like 'remarkable' and 'perfected' are positive in isolation, negative in context",
    },
    {
        "text": "I am absolutely thrilled that they got my reservation wrong, lost my table, and then seated me next to the kitchen. Peak dining experience.",
        "true_label": 0,  # 1star
        "adversarial_type": "sarcasm",
        "explanation": "Enthusiastic framing of a cascade of failures — high positive affect vocabulary, terrible content",
    },

    # Type 3: Sentiment Sandwich 
    {
        "text": "Great location, friendly staff, and a beautiful space — but the food poisoning I got kept me bedridden for two days. Really lovely ambiance though.",
        "true_label": 0,  # 1star the middle destroys the positives
        "adversarial_type": "sentiment_sandwich",
        "explanation": "Positive opener and closer sandwich a catastrophic negative — averaging heuristics fail here",
    },
    {
        "text": "Amazing cocktails, genuinely one of the best I've had. The entree was inedible. But the dessert almost made up for it.",
        "true_label": 1,  # 2star strong positive-negative-positive, net negative
        "adversarial_type": "sentiment_sandwich",
        "explanation": "Model must weight the inedible entree as more important than the cocktail praise",
    },
    {
        "text": "Friendly service, clean tables, fast seating. The chicken parm had the texture of wet cardboard and tasted of nothing. Convenient parking though.",
        "true_label": 1,  # 2star
        "adversarial_type": "sentiment_sandwich",
        "explanation": "Peripherals praised, core product (food) failed — sentiment averaging would produce 3star",
    },
    {
        "text": "Perfect location, open late, easy to park. Rude staff, wrong order, waited 90 minutes. But hey, the napkins were nice.",
        "true_label": 0,  # 1star
        "adversarial_type": "sentiment_sandwich",
        "explanation": "Trivial positives (napkins) bookending serious service failures",
    },

    # Type 4: Explicit Star Mismatch 
    {
        "text": "I would give this place 5 stars if the food weren't cold, the service weren't glacial, the restrooms weren't filthy, and the prices weren't insulting.",
        "true_label": 0,  # 1star lists 4 critical failures despite mentioning 5 stars
        "adversarial_type": "explicit_mismatch",
        "explanation": "Conditional 5-star mention used to frame a comprehensive demolition of the restaurant",
    },
    {
        "text": "Deserves zero stars but the app won't let me go that low. So here's one, reluctantly, for managing to stay open.",
        "true_label": 0,  # 1star explicitly negative despite forced 1star
        "adversarial_type": "explicit_mismatch",
        "explanation": "Meta-commentary on rating system; content is unambiguous but framing is indirect",
    },
    {
        "text": "My friend who brought me says this is a 4-star place. I disagree: the pasta was cold, the wine was corked, and we waited an hour. I say 2 stars.",
        "true_label": 1,  # 2star
        "adversarial_type": "explicit_mismatch",
        "explanation": "Multiple star values mentioned; true rating buried in narrative disagreement",
    },
    {
        "text": "Stars: 5/5 for trying. 1/5 for execution. 0/5 for value. I averaged it out and rounded up because the owner seemed nice.",
        "true_label": 1,  # 2star
        "adversarial_type": "explicit_mismatch",
        "explanation": "Arithmetic star framing; the explicit 5/5 mention may bias toward high prediction",
    },

    # Type 5: Domain Vocabulary 
    {
        "text": "The acute dysfunction of this establishment's operational cadence creates a suboptimal consumptive experience. Recommend immediate remediation.",
        "true_label": 0,  # 1star formal/bureaucratic negative language
        "adversarial_type": "domain_vocabulary",
        "explanation": "Technical/corporate vocabulary rarely seen in Yelp training data; classifier may not recognize negativity",
    },
    {
        "text": "Tasting notes: primary flavors of disappointment with secondary notes of overpricing and a long finish of regret. Not recommended for extended palate exposure.",
        "true_label": 0,  # 1star wine review framing applied negatively
        "adversarial_type": "domain_vocabulary",
        "explanation": "Wine/sommelier vocabulary domain shift within the food review domain",
    },
    {
        "text": "Net Promoter Score for this venue: firmly in detractor territory. Would not advocate to peers or professional network. Churn probability: high.",
        "true_label": 0,  # 1star business KPI language
        "adversarial_type": "domain_vocabulary",
        "explanation": "Business metric language applied to restaurant review; OOV for Yelp-trained classifier",
    },
    {
        "text": "Phenomenologically speaking, the gestalt of this dining experience coalesced into something I can only describe as transcendent. The risotto alone justifies the Hegelian synthesis of tradition and innovation.",
        "true_label": 4,  # 5star academic/philosophical positive
        "adversarial_type": "domain_vocabulary",
        "explanation": "Academic vocabulary used positively; model may struggle with uncommon adjectives despite clear positive sentiment",
    },
]


#  Model Inference 

def predict_with_classifier(texts, model, tokenizer, device, batch_size=16):
    """Returns predicted labels (0-4) and confidence scores."""
    model.eval()
    all_preds, all_probs = [], []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
    return all_preds, all_probs


def predict_with_llm(texts, client):
    """Zero-shot LLM (Direct prompt from Task 2) on adversarial examples."""
    from tqdm import tqdm
    preds = []
    for text in tqdm(texts, desc="LLM predictions"):
        result = client.complete(SYSTEM_PROMPT, build_direct_prompt(text))
        if result["parsed"] and isinstance(result["parsed"].get("stars"), (int, float)):
            val = int(result["parsed"]["stars"])
            preds.append(val - 1 if 1 <= val <= 5 else None)  # convert to 0-indexed
        else:
            preds.append(None)
    return preds


def confidence_threshold_predict(
    texts, true_labels,
    classifier_preds, classifier_probs,
    llm_preds,
    threshold: float = 0.6,
) -> list:
    """
    Mitigation: confidence-threshold ensemble.
    If max_prob >= threshold → use classifier.
    Else → fall back to LLM prediction.
    """
    final_preds = []
    for i, (cpred, cprob, lpred) in enumerate(zip(classifier_preds, classifier_probs, llm_preds)):
        confidence = max(cprob)
        if confidence >= threshold:
            final_preds.append(cpred)
        else:
            final_preds.append(lpred if lpred is not None else cpred)
    return final_preds


# Metrics 

def adversarial_metrics(true_labels, preds, examples) -> dict:
    valid = [(t, p, ex) for t, p, ex in zip(true_labels, preds, examples) if p is not None]
    t_vals = [x[0] for x in valid]
    p_vals = [x[1] for x in valid]

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(t_vals, p_vals)

    #Per-type breakdown
    by_type = {}
    for t, p, ex in valid:
        typ = ex["adversarial_type"]
        if typ not in by_type:
            by_type[typ] = {"correct": 0, "total": 0}
        by_type[typ]["total"] += 1
        if t == p:
            by_type[typ]["correct"] += 1
    for typ in by_type:
        by_type[typ]["accuracy"] = round(by_type[typ]["correct"] / by_type[typ]["total"], 3)

    return {
        "overall_accuracy": round(float(acc), 4),
        "n": len(valid),
        "by_type": by_type,
    }


# main

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load fine-tuned model
    print(f"Loading fine-tuned model from {MODEL_DIR}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)

    texts       = [ex["text"] for ex in ADVERSARIAL_EXAMPLES]
    true_labels = [ex["true_label"] for ex in ADVERSARIAL_EXAMPLES]

    # classifier predictions 
    print("\n━━━ Classifier Predictions ━━━")
    clf_preds, clf_probs = predict_with_classifier(texts, model, tokenizer, device)
    clf_metrics = adversarial_metrics(true_labels, clf_preds, ADVERSARIAL_EXAMPLES)

    # LLM predictions 
    print("\n━━━ LLM (Direct, Zero-Shot) Predictions ━━━")
    client = LLMClient(model="llama-3.1-8b-instant")
    llm_preds = predict_with_llm(texts, client)
    llm_metrics = adversarial_metrics(true_labels, llm_preds, ADVERSARIAL_EXAMPLES)

    # mitigation: confidence thresholding
    print("\n━━━ Mitigation: Confidence Threshold Ensemble ━━━")
    THRESHOLD = 0.6
    mit_preds   = confidence_threshold_predict(
        texts, true_labels, clf_preds, clf_probs, llm_preds, THRESHOLD
    )
    mit_metrics = adversarial_metrics(true_labels, mit_preds, ADVERSARIAL_EXAMPLES)

    #How often did we fall back to LLM?
    n_fallback = sum(
        1 for cprob in clf_probs if max(cprob) < THRESHOLD
    )
    print(f"  Fallback to LLM: {n_fallback}/{len(texts)} examples (threshold={THRESHOLD})")

    #Print results 
    print(f"\n{'='*60}")
    print(f"  Adversarial Robustness Results (n={len(ADVERSARIAL_EXAMPLES)})")
    print(f"{'='*60}")
    print(f"  {'Approach':<35} {'Accuracy':>10}")
    print(f"  {'─'*48}")
    print(f"  {'DistilBERT (fine-tuned)':<35} {clf_metrics['overall_accuracy']:>10.4f}")
    print(f"  {'LLM Direct (zero-shot)':<35} {llm_metrics['overall_accuracy']:>10.4f}")
    print(f"  {'Ensemble (threshold={:.1f})':<35}".format(THRESHOLD) +
          f" {mit_metrics['overall_accuracy']:>10.4f}")

    print(f"\n  Per-type breakdown (DistilBERT):")
    print(f"  {'Type':<28} {'Accuracy':>9}  {'Correct/Total':>14}")
    for typ, stats in clf_metrics["by_type"].items():
        print(f"  {typ:<28} {stats['accuracy']:>9.3f}  {stats['correct']}/{stats['total']:>13}")

    #Per-example breakdown 
    print(f"\n{'='*60}")
    print(f"  Per-Example Results")
    print(f"{'='*60}")
    for i, ex in enumerate(ADVERSARIAL_EXAMPLES):
        clf_p   = clf_preds[i]
        llm_p   = llm_preds[i]
        true_l  = ex["true_label"]
        conf    = max(clf_probs[i])
        clf_ok  = "✓" if clf_p == true_l else "✗"
        llm_ok  = "✓" if llm_p == true_l else "✗" if llm_p is not None else "?"
        print(f"\n  [{ex['adversarial_type']}]")
        print(f"  Text     : {ex['text'][:100]}...")
        print(f"  True     : {true_l+1}star")
        print(f"  Clf      : {clf_p+1}star (conf={conf:.3f}) {clf_ok}")
        print(f"  LLM      : {llm_p+1 if llm_p is not None else '?'}star {llm_ok}")
        print(f"  Why hard : {ex['explanation']}")

    # Save
    results = {
        "n_examples": len(ADVERSARIAL_EXAMPLES),
        "threshold": THRESHOLD,
        "n_llm_fallback": n_fallback,
        "classifier":  clf_metrics,
        "llm_direct":  llm_metrics,
        "ensemble":    mit_metrics,
        "per_example": [
            {
                "text_snippet": ex["text"][:100],
                "adversarial_type": ex["adversarial_type"],
                "true_label": ex["true_label"] + 1,
                "clf_pred":   clf_preds[i] + 1,
                "llm_pred":   (llm_preds[i] + 1) if llm_preds[i] is not None else None,
                "clf_confidence": round(max(clf_probs[i]), 4),
                "clf_correct": clf_preds[i] == ex["true_label"],
                "llm_correct": llm_preds[i] == ex["true_label"],
            }
            for i, ex in enumerate(ADVERSARIAL_EXAMPLES)
        ],
    }

    with open(os.path.join(RESULTS_DIR, "task4_adversarial.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → results/task4_adversarial.json")

    return results


if __name__ == "__main__":
    main()