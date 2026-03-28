"""
Task 1 :  Zero-Shot & Few-Shot Prompting

Runs both strategies on 200 stratified Yelp samples.
Saves:
  results/task1_raw_results.csv  --  row per prediction with relevat details
  results/task1_metrics.json     -- summary metrics for both strategies
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from tqdm import tqdm

from src.llm_client import LLMClient
from src.prompts import SYSTEM_PROMPT, build_zero_shot_prompt, build_few_shot_prompt
from src.data_loader import load_yelp_sample
from src.evaluator import compute_metrics, print_metrics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_strategy(df: pd.DataFrame, prompt_fn, client: LLMClient, label: str) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=label):
        result = client.complete(SYSTEM_PROMPT, prompt_fn(row["text"]))

        pred_stars = None
        if result["parsed"] and isinstance(result["parsed"].get("stars"), (int, float)):
            val = int(result["parsed"]["stars"])
            pred_stars = val if 1 <= val <= 5 else None
            if pred_stars is None:
                result["json_valid"] = False  # valid JSON but invalid star value

        rows.append({
            "text_snippet": row["text"],
            "true_stars": row["true_stars"],
            "pred_stars": pred_stars,
            "explanation": (result["parsed"] or {}).get("explanation"),
            "json_valid": result["json_valid"],
            "raw_response": result["raw_response"],
            "latency_ms": result["latency_ms"],
            "error": result["error"],
            "prompt_type": label,
        })

    return pd.DataFrame(rows)


def main():
    df = load_yelp_sample(n_per_class=40)  # 200 samples, 40 per star class
    client = LLMClient(model="llama-3.1-8b-instant") # using default model for now, can experiment with other models as well 

    print("\n━━━ Strategy 1: Zero-Shot ━━━")
    zs = run_strategy(df, build_zero_shot_prompt, client, "zero_shot")
    zs_metrics = compute_metrics(zs)
    print_metrics(zs_metrics, "Zero-Shot Results")

    print("\n━━━ Strategy 2: Few-Shot (5-shot, 1 per class) ━━━")
    fs = run_strategy(df, build_few_shot_prompt, client, "few_shot")
    fs_metrics = compute_metrics(fs)
    print_metrics(fs_metrics, "Few-Shot Results")

    # saving everything
    combined = pd.concat([zs, fs], ignore_index=True)
    combined.to_csv(os.path.join(RESULTS_DIR, "task1_raw_results.csv"), index=False)

    summary = {"zero_shot": zs_metrics, "few_shot": fs_metrics}
    with open(os.path.join(RESULTS_DIR, "task1_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved → {RESULTS_DIR}/task1_raw_results.csv")
    print(f"Saved → {RESULTS_DIR}/task1_metrics.json")

    return combined, summary


if __name__ == "__main__":
    main()