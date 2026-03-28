"""
Task 2  Chain-of-Thought vs Direct Prompting

Compares:
  direct (minimal prompt, no reasoning encouraged)
  vs 
  cot (explicit step by step reasoning before star output)

Uses the SAME 200 samples as Task 1 (same seed) for fair comparison.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from tqdm import tqdm

from src.llm_client import LLMClient
from src.prompts import SYSTEM_PROMPT, build_direct_prompt, build_cot_prompt
from src.data_loader import load_yelp_sample
from src.evaluator import compute_cot_metrics, detect_reasoning_mismatch, print_metrics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# System prompt for CoT
COT_SYSTEM_PROMPT = """You are an expert Yelp review analyst.

STRICT OUTPUT FORMAT — respond with ONLY this JSON, nothing else:
{"reasoning": "<step-by-step analysis>", "stars": <integer 1-5>, "explanation": "<one sentence>"}

Rules:
- No markdown, no code fences, no extra text before or after the JSON
- Reason through positives, negatives, and overall tone before deciding stars
- "stars" must be an integer between 1 and 5"""


def run_direct(df, client):
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Direct"):
        result = client.complete(SYSTEM_PROMPT, build_direct_prompt(row["text"]))
        pred_stars = None
        if result["parsed"] and isinstance(result["parsed"].get("stars"), (int, float)):
            val = int(result["parsed"]["stars"])
            pred_stars = val if 1 <= val <= 5 else None

        rows.append({
            "text_snippet": row["text"][:150],
            "true_stars": row["true_stars"],
            "pred_stars": pred_stars,
            "reasoning": None,
            "explanation": (result["parsed"] or {}).get("explanation"),
            "json_valid": result["json_valid"],
            "raw_response": result["raw_response"],
            "latency_ms": result["latency_ms"],
            "prompt_type": "direct",
        })
    return pd.DataFrame(rows)


def run_cot(df, client):
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="CoT"):
        result = client.complete(COT_SYSTEM_PROMPT, build_cot_prompt(row["text"]))
        pred_stars = None
        reasoning = None

        if result["parsed"]:
            p = result["parsed"]
            if isinstance(p.get("stars"), (int, float)):
                val = int(p["stars"])
                pred_stars = val if 1 <= val <= 5 else None
            reasoning = p.get("reasoning")

        rows.append({
            "text_snippet": row["text"][:150],
            "true_stars": row["true_stars"],
            "pred_stars": pred_stars,
            "reasoning": reasoning,
            "explanation": (result["parsed"] or {}).get("explanation"),
            "json_valid": result["json_valid"],
            "raw_response": result["raw_response"],
            "latency_ms": result["latency_ms"],
            "prompt_type": "cot",
        })
    return pd.DataFrame(rows)


def main():
    # Same seed = same 200 samples as Task 1 
    df = load_yelp_sample(n_per_class=40, seed=42)
    client = LLMClient(model="llama-3.1-8b-instant")

    print("\n━━━ Strategy 1: Direct ━━━")
    direct_df = run_direct(df, client)
    direct_metrics = compute_cot_metrics(direct_df)
    print_metrics(direct_metrics, "Direct Results")

    print("\n━━━ Strategy 2: Chain-of-Thought ━━━")
    cot_df = run_cot(df, client)
    cot_df = detect_reasoning_mismatch(cot_df)
    cot_metrics = compute_cot_metrics(cot_df)
    print_metrics(cot_metrics, "CoT Results")

    # Save
    combined = pd.concat([direct_df, cot_df], ignore_index=True)
    combined.to_csv(os.path.join(RESULTS_DIR, "task2_raw_results.csv"), index=False)

    summary = {"direct": direct_metrics, "cot": cot_metrics}
    with open(os.path.join(RESULTS_DIR, "task2_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print reasoning mismatch examples
    mismatches = cot_df[cot_df.get("reasoning_mismatch", False) == True]
    if len(mismatches) > 0:
        print(f"\n━━━ Reasoning Mismatches Found: {len(mismatches)} ━━━")
        for _, row in mismatches.head(3).iterrows():
            print(f"\nTRUE: {int(row['true_stars'])}★ | PRED: {int(row['pred_stars'])}★")
            print(f"REASONING: {str(row['reasoning'])[:200]}...")
            print(f"TEXT: {row['text_snippet'][:100]}...")

    print(f"\nSaved → results/task2_raw_results.csv")
    print(f"Saved → results/task2_metrics.json")
    return combined, summary


if __name__ == "__main__":
    main()