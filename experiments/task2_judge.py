"""
Task 2 : Reasoning Mismatch: Approach 2 (LLM-as-Judge)

Loads existing CoT results (can easily be integrated with live pipeline later) and runs a judge pass.
No new star predictions evaluation only.

"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from src.llm_client import LLMClient
from src.evaluator import detect_reasoning_mismatch_llm_judge

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "task2_raw_results.csv"))
    client = LLMClient(model="llama-3.1-8b-instant")

    print(f"Loaded {len(df)} rows | CoT rows: {(df['prompt_type']=='cot').sum()}")
    print("\nRunning LLM-as-judge consistency check on CoT responses...")

    judged = detect_reasoning_mismatch_llm_judge(df, client)
    cot_judged = judged[judged["prompt_type"] == "cot"]

    # summary
    n_inconsistent = (cot_judged["judge_consistent"] == False).sum()
    n_total = cot_judged["judge_consistent"].notna().sum()

    print(f"\n{'='*52}")
    print(f"  LLM Judge Results — CoT Consistency")
    print(f"{'='*52}")
    print(f"  Total CoT predictions evaluated : {n_total}")
    print(f"  Consistent (reasoning ↔ star)   : {(cot_judged['judge_consistent']==True).sum()}")
    print(f"  Inconsistent (mismatch)          : {n_inconsistent}")
    print(f"  Mismatch rate                    : {n_inconsistent/n_total:.3f}")
    print(f"\n  Compare — Keyword heuristic      : 0 (0.000)  ← brittle, missed all")

    # breakdown by confidence
    print(f"\n  Confidence breakdown:")
    print(cot_judged.groupby("judge_confidence")["judge_consistent"].value_counts().to_string())

    # inconsistencies
    inconsistent = cot_judged[cot_judged["judge_consistent"] == False]
    print(f"\n{'='*52}")
    print(f"  Mismatch Examples")
    print(f"{'='*52}")
    for _, row in inconsistent.head(5).iterrows():
        print(f"\n  TRUE: {int(row['true_stars'])}★ | PRED: {int(row['pred_stars'])}★")
        print(f"  REASONING: {str(row['reasoning'])[:250]}...")
        print(f"  JUDGE: {row['judge_reasoning']}")
        print(f"  CONFIDENCE: {row['judge_confidence']}")

    # save
    judged.to_csv(os.path.join(RESULTS_DIR, "task2_judged_results.csv"), index=False)

    summary = {
        "approach_1_keyword": {
            "mismatches_detected": 0,
            "mismatch_rate": 0.0,
            "verdict": "brittle  vocabulary mismatch between review language and analytical reasoning"
        },
        "approach_2_llm_judge": {
            "mismatches_detected": int(n_inconsistent),
            "mismatch_rate": round(n_inconsistent / n_total, 4),
            "total_evaluated": int(n_total),
        }
    }

    with open(os.path.join(RESULTS_DIR, "task2_judge_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved → results/task2_judged_results.csv")
    print(f"Saved → results/task2_judge_metrics.json")

    return judged, summary


if __name__ == "__main__":
    main()