"""
Task 3:  Multi-Objective AI Assistant

Two-phase Test pipeline:
  Phase 1: Generation
    Input  : Yelp review (50 samples, 10 per star class)
    Output : stars + key_point + business_response (3-field json)

  Phase 2: Eval (LLM as Judge)
    Input  : review + key_point + business_response
    Output : faithfulness + actionability + tone (each 1-5, independent rubrics)

Key engineering decisions:
  1. 3-field JSON only, 4th field (e.g. sentiment) measurably raises parse failure
     rate on 8B models without proportional analytical gain as as far as observations seem to point
  2. Generation saved to CSV before judging begins, checkpoint prevents data loss
     if judge run fails in between
  3. overall_verdict computed in rules based deterministic internal scoring method in Python (faithfulness>=4 AND actionability>=4 AND tone>=4)
     NOT asked from LLM judge, removes one JSON field = one fewer failure point. 
  4. Judge does NOT receive pred_stars (predicted stars):  prevents anchoring bias in quality scores
  5. Word limits + generic-phrase ban in system prompt are the primary levers
     for faithfulness and actionability scores, not prompt length

Saves:
  results/task3_generated.csv  — checkpoint after Phase 1
  results/task3_judged.csv     — full results after Phase 2
  results/task3_metrics.json   — summary metrics
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from src.llm_client import LLMClient
from src.data_loader import load_yelp_sample
from src.prompts import (
    TASK3_SYSTEM_PROMPT,
    build_assistant_prompt,
    JUDGE_SYSTEM_TASK3,
    build_judge_prompt_task3,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

GENERATED_PATH = os.path.join(RESULTS_DIR, "task3_generated.csv")
JUDGED_PATH    = os.path.join(RESULTS_DIR, "task3_judged.csv")
METRICS_PATH   = os.path.join(RESULTS_DIR, "task3_metrics.json")


# ── Phase 1: Generation ───────────────────────────────────────────────────────

def run_generation(df: pd.DataFrame, client: LLMClient) -> pd.DataFrame:
    """
    Generates stars + key_point + business_response for each review.
    Validates stars in [1,5], key_point and business_response non-null.
    """
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        result = client.complete(
            TASK3_SYSTEM_PROMPT,
            build_assistant_prompt(row["text"]),
        )

        stars, key_point, business_response = None, None, None

        if result["parsed"]:
            p = result["parsed"]
            val = p.get("stars")
            if isinstance(val, (int, float)) and 1 <= int(val) <= 5:
                stars = int(val)
            key_point         = p.get("key_point") or None
            business_response = p.get("business_response") or None

        rows.append({
            "text":              row["text"][:800],
            "true_stars":        row["true_stars"],
            "pred_stars":        stars,
            "key_point":         key_point,
            "business_response": business_response,
            "json_valid":        result["json_valid"],
            "latency_ms":        result["latency_ms"],
            "raw_response":      result["raw_response"],
        })

    return pd.DataFrame(rows)


def print_generation_summary(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n{'='*52}")
    print(f"  Phase 1: Generation Summary  (n={n})")
    print(f"{'='*52}")
    print(f"  JSON valid           : {df['json_valid'].sum():>3} / {n}  ({df['json_valid'].mean():.1%})")
    print(f"  Stars extracted      : {df['pred_stars'].notna().sum():>3} / {n}")
    print(f"  Key points           : {df['key_point'].notna().sum():>3} / {n}")
    print(f"  Business responses   : {df['business_response'].notna().sum():>3} / {n}")

    evaluable = (
        df["key_point"].notna() &
        df["business_response"].notna() &
        df["pred_stars"].notna()
    ).sum()
    print(f"  Fully evaluable      : {evaluable:>3} / {n}")

    valid = df[df["pred_stars"].notna()].copy()
    if len(valid) > 0:
        valid["pred_stars"] = valid["pred_stars"].astype(int)
        acc = accuracy_score(valid["true_stars"], valid["pred_stars"])
        mf1 = f1_score(
            valid["true_stars"], valid["pred_stars"],
            average="macro", labels=[1,2,3,4,5], zero_division=0,
        )
        print(f"\n  Star accuracy        : {acc:.3f}")
        print(f"  Star Macro-F1        : {mf1:.3f}")
        print(f"  Avg latency (ms)     : {df['latency_ms'].mean():.0f}")


# ── Phase 2 Evaluation (LLM as Judge) 

def run_judge(df: pd.DataFrame, client: LLMClient) -> pd.DataFrame:
    """
    Runs judge on all fully-evaluable rows.
    overall_verdict computed in Python, NOT asked from LLM.
    """
    evaluable_mask = (
        df["key_point"].notna() &
        df["business_response"].notna() &
        df["pred_stars"].notna()
    )

    judge_cols = [
        "faithfulness", "faithfulness_reason",
        "actionability", "actionability_reason",
        "tone", "tone_reason",
        "overall_verdict", "judge_json_valid",
    ]
    for col in judge_cols:
        df[col] = None

    print(f"\n  Evaluable rows: {evaluable_mask.sum()} / {len(df)}")

    for idx, row in tqdm(
        df[evaluable_mask].iterrows(),
        total=evaluable_mask.sum(),
        desc="Judging",
    ):
        result = client.complete(
            JUDGE_SYSTEM_TASK3,
            build_judge_prompt_task3(
                row["text"],
                row["key_point"],
                row["business_response"],
            ),
        )

        df.at[idx, "judge_json_valid"] = result["json_valid"]

        if result["parsed"]:
            p = result["parsed"]
            scores = {}
            for dim in ["faithfulness", "actionability", "tone"]:
                val = p.get(dim)
                score = int(val) if isinstance(val, (int, float)) and 1 <= int(val) <= 5 else None
                df.at[idx, dim] = score
                df.at[idx, f"{dim}_reason"] = p.get(f"{dim}_reason")
                scores[dim] = score

            # compute verdict deterministically, not from LLM
            if all(v is not None for v in scores.values()):
                if all(v >= 4 for v in scores.values()):
                    verdict = "success"
                elif all(v <= 2 for v in scores.values()):
                    verdict = "failure"
                else:
                    verdict = "partial"
                df.at[idx, "overall_verdict"] = verdict

    return df


# metrics 

def compute_task3_metrics(df: pd.DataFrame) -> dict:
    judged = df[df["faithfulness"].notna()].copy()
    dims   = ["faithfulness", "actionability", "tone"]
    for dim in dims:
        judged[dim] = judged[dim].astype(float)

    # Per-dimension stats
    dim_metrics = {}
    for dim in dims:
        vals = judged[dim]
        dim_metrics[dim] = {
            "mean":    round(float(vals.mean()), 3),
            "std":     round(float(vals.std()),  3),
            "median":  round(float(vals.median()), 1),
            "pct_5":   round((vals == 5).mean(), 3),
            "pct_4_5": round((vals >= 4).mean(), 3),
            "pct_1_2": round((vals <= 2).mean(), 3),
        }

    # star accuracy
    valid = df[df["pred_stars"].notna()].copy()
    valid["pred_stars"] = valid["pred_stars"].astype(int)
    acc = accuracy_score(valid["true_stars"], valid["pred_stars"])
    mf1 = f1_score(
        valid["true_stars"], valid["pred_stars"],
        average="macro", labels=[1,2,3,4,5], zero_division=0,
    )

    # verdict distribution
    verdict_dist = judged["overall_verdict"].value_counts().to_dict()

    # quality by true star class
    # Tests- does response quality degrade for negative (1-2 star) reviews?
    # Hypothesis: 1-2 star reviews harder to respond to (damage control vs gratitude)
    quality_by_star = {}
    for star in range(1, 6):
        subset = judged[judged["true_stars"] == star]
        if len(subset) > 0:
            quality_by_star[str(star)] = {
                dim: round(float(subset[dim].mean()), 3)
                for dim in dims
            }

    # dimension correlations, faithfulness→actionability should be high
    # If you extract wrong, you can't respond specifically
    corr = judged[dims].corr().round(3).to_dict()

    return {
        "n_generated":           len(df),
        "n_judged":              len(judged),
        "generation_compliance": round(df["json_valid"].mean(), 4),
        "judge_compliance":      round(judged["judge_json_valid"].astype(bool).mean(), 4),
        "star_accuracy":         round(float(acc), 4),
        "star_macro_f1":         round(float(mf1), 4),
        "dimensions":            dim_metrics,
        "verdict_distribution":  verdict_dist,
        "quality_by_star_class": quality_by_star,
        "dimension_correlations": corr,
    }


def print_metrics(metrics: dict) -> None:
    print(f"\n{'='*56}")
    print(f"  Task 3 — Final Evaluation Results")
    print(f"{'='*56}")
    print(f"  Samples generated   : {metrics['n_generated']}")
    print(f"  Samples judged      : {metrics['n_judged']}")
    print(f"  Gen compliance      : {metrics['generation_compliance']:.1%}")
    print(f"  Judge compliance    : {metrics['judge_compliance']:.1%}")
    print(f"  Star accuracy       : {metrics['star_accuracy']:.3f}")
    print(f"  Star Macro-F1       : {metrics['star_macro_f1']:.3f}")

    print(f"\n  {'Dimension':<16} {'Mean':>6}  {'Std':>5}  {'Score≥4':>8}  {'Score≤2':>8}")
    print(f"  {'─'*52}")
    for dim, d in metrics["dimensions"].items():
        print(
            f"  {dim.capitalize():<16} {d['mean']:>6.3f}  {d['std']:>5.3f}  "
            f"{d['pct_4_5']:>7.1%}  {d['pct_1_2']:>7.1%}"
        )

    print(f"\n  Verdict distribution:")
    for k, v in sorted(metrics["verdict_distribution"].items()):
        total = metrics["n_judged"]
        print(f"    {k:<10}: {v:>3}  ({v/total:.1%})")

    print(f"\n  Quality by true star class:")
    print(f"  {'Star':<6}  {'Faithful':>9}  {'Actionable':>10}  {'Tone':>6}")
    print(f"  {'─'*38}")
    for star, scores in sorted(metrics["quality_by_star_class"].items()):
        print(
            f"  {star+'★':<6}  {scores['faithfulness']:>9.3f}  "
            f"{scores['actionability']:>10.3f}  {scores['tone']:>6.3f}"
        )

    print(f"\n  Dimension correlations (faithfulness→actionability expected high):")
    corr = metrics["dimension_correlations"]
    print(f"    Faithfulness ↔ Actionability : {corr['faithfulness']['actionability']}")
    print(f"    Actionability ↔ Tone         : {corr['actionability']['tone']}")
    print(f"    Faithfulness ↔ Tone          : {corr['faithfulness']['tone']}")


def print_examples(df: pd.DataFrame, n: int = 3) -> None:
    judged = df[df["faithfulness"].notna()].copy()
    for dim in ["faithfulness", "actionability", "tone"]:
        judged[dim] = judged[dim].astype(float)
    judged["avg_score"] = judged[["faithfulness","actionability","tone"]].mean(axis=1)

    for label, subset in [
        ("SUCCESS (highest avg score)", judged.nlargest(n, "avg_score")),
        ("FAILURE (lowest avg score)",  judged.nsmallest(n, "avg_score")),
    ]:
        print(f"\n{'='*64}")
        print(f"  {label}")
        print(f"{'='*64}")
        for _, row in subset.iterrows():
            print(f"\n  True: {int(row['true_stars'])}★  |  Pred: {int(row['pred_stars'])}★  |  Verdict: {row['overall_verdict']}")
            print(f"  Review    : {row['text'][:160]}...")
            print(f"  KeyPoint  : {row['key_point']}")
            print(f"  Response  : {row['business_response']}")
            print(f"  Scores    : F={row['faithfulness']:.0f}  A={row['actionability']:.0f}  T={row['tone']:.0f}  (avg={row['avg_score']:.2f})")
            print(f"  F-reason  : {row['faithfulness_reason']}")
            print(f"  A-reason  : {row['actionability_reason']}")


# Main 

def main():
    print("Loading dataset...")
    df = load_yelp_sample(n_per_class=10, seed=42)
    client = LLMClient(model="llama-3.1-8b-instant")

    #  Phase 1 
    print("\n━━━ Phase 1: Multi-Objective Generation ━━━")
    generated = run_generation(df, client)

    # Checkpoint  to save before judging
    generated.to_csv(GENERATED_PATH, index=False)
    print(f"  Checkpoint saved → {GENERATED_PATH}")
    print_generation_summary(generated)

    #  Phase 2
    print("\n━━━ Phase 2: LLM-as-Judge Evaluation ━━━")
    judged = run_judge(generated.copy(), client)
    judged.to_csv(JUDGED_PATH, index=False)
    print(f"  Saved → {JUDGED_PATH}")

    # Metrics 
    metrics = compute_task3_metrics(judged)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved → {METRICS_PATH}")

    print_metrics(metrics)
    print_examples(judged)

    return judged, metrics


if __name__ == "__main__":
    main()