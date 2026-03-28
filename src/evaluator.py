"""
Evaluation metrics for Tasks 1 (as of now)

required metrics for task 1:
- format_compliance_rate : % of responses that are valid, parseable JSON
- accuracy               : exact star match on valid predictions
- macro_f1               : unweighted mean F1 across all 5 classes
- per_class_f1           : per-star F1 for error analysis
- error_analysis         : off-by-one / off-by-two / off-by-more buckets
"""

import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Configure logging for this module
logger = logging.getLogger(__name__)


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute evaluation metrics from predictions.
    """
    logger.info(f"Computing metrics for {len(df)} predictions")
    
    total = len(df)
    n_valid = df["json_valid"].sum()
    format_compliance = n_valid / total
    logger.info(f"Format compliance: {n_valid}/{total} valid json ({format_compliance:.1%})")

    valid = df[df["pred_stars"].notna()].copy()
    logger.debug(f"Predictions with non-null stars: {len(valid)}/{total}")
    
    if len(valid) == 0:
        logger.warning("No valid predictions found")
        return {
            "total": total,
            "valid_predictions": 0,
            "format_compliance_rate": round(format_compliance, 4),
            "accuracy": None,
            "macro_f1": None,
        }

    valid["pred_stars"] = valid["pred_stars"].astype(int)
    valid["true_stars"] = valid["true_stars"].astype(int)

    accuracy = accuracy_score(valid["true_stars"], valid["pred_stars"])
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    macro_f1 = f1_score(
        valid["true_stars"], valid["pred_stars"],
        average="macro", labels=[1, 2, 3, 4, 5], zero_division=0,
    )
    logger.info(f"Macro F1: {macro_f1:.4f}")
    
    per_class = f1_score(
        valid["true_stars"], valid["pred_stars"],
        average=None, labels=[1, 2, 3, 4, 5], zero_division=0,
    )
    for i, f1_val in enumerate(per_class):
        logger.debug(f"Class {i+1} F1: {f1_val:.4f}")

    error_analysis = _error_analysis(valid)
    logger.info(f"Error analysis: {error_analysis['total_errors']} errors found")

    return {
        "total": total,
        "valid_predictions": len(valid),
        "format_compliance_rate": round(format_compliance, 4),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class_f1": {str(i + 1): round(float(v), 4) for i, v in enumerate(per_class)},
        "error_analysis": error_analysis,
    }


def _error_analysis(df: pd.DataFrame) -> dict:

    errors = df[df["pred_stars"] != df["true_stars"]]
    n_err = len(errors)
    
    logger.debug(f"Error analysis: {n_err} errors out of {len(df)} predictions ({n_err/len(df)*100:.1f}%)")
    
    if n_err == 0:
        logger.debug("No errors found - perfect predictions")
        return {"total_errors": 0, "off_by_one": 0, "off_by_two": 0, "off_by_more": 0}

    delta = (errors["pred_stars"] - errors["true_stars"]).abs()
    off_by_one = int((delta == 1).sum())
    off_by_two = int((delta == 2).sum())
    off_by_more = int((delta >= 3).sum())
    
    logger.debug(f"Error breakdown - Off by 1: {off_by_one}, Off by 2: {off_by_two}, Off by 3+: {off_by_more}")
    
    return {
        "total_errors": n_err,
        "off_by_one": off_by_one,
        "off_by_one_pct": round(off_by_one / n_err, 3) if n_err > 0 else 0,
        "off_by_two": off_by_two,
        "off_by_more": off_by_more,
    }


def print_metrics(metrics: dict, title: str = "") -> None:
    """
    Pretty-print metrics dictionary.
    
    Args:
        metrics: Metrics dictionary from compute_metrics
        title: Optional section title
    """
    if title:
        logger.info(f"Printing metrics section: {title}")
        print(f"\n{'=' * 52}")
        print(f"  {title}")
        print(f"{'=' * 52}")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")
        logger.debug(f"{k}: {v}")


# TASK 2 Eval (simple heurestic , seems very brittle with key word matches ):
def detect_reasoning_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags cot responses where the reasoning sentiment contradicts the predicted star.
    
    heuristic with simple keyword matches
    """
    POSITIVE_SIGNALS = ["excellent", "great", "love", "perfect", "outstanding",
                        "amazing", "wonderful", "best", "fantastic", "enjoyed"]
    NEGATIVE_SIGNALS = ["terrible", "awful", "horrible", "worst", "disgusting",
                        "poor", "bad", "never", "disappointed", "rude", "cold"]

    def _check(row):
        if pd.isna(row.get("reasoning")) or pd.isna(row.get("pred_stars")):
            return False
        reasoning = str(row["reasoning"]).lower()
        star = int(row["pred_stars"])

        pos_count = sum(1 for w in POSITIVE_SIGNALS if w in reasoning)
        neg_count = sum(1 for w in NEGATIVE_SIGNALS if w in reasoning)

        # Mismatch: reasoning strongly positive but star is 1-2,
        # or reasoning strongly negative but star is 4-5
        if pos_count > neg_count + 1 and star <= 2:
            return True
        if neg_count > pos_count + 1 and star >= 4:
            return True
        return False

    df = df.copy()
    df["reasoning_mismatch"] = df.apply(_check, axis=1)
    return df


def compute_cot_metrics(df: pd.DataFrame) -> dict:
    base = compute_metrics(df)
    
    if "reasoning_mismatch" in df.columns:
        valid = df[df["pred_stars"].notna()]
        n_mismatch = df["reasoning_mismatch"].sum()
        base["reasoning_mismatch_count"] = int(n_mismatch)
        base["reasoning_mismatch_rate"] = round(n_mismatch / len(valid), 4) if len(valid) > 0 else None

    return base


def detect_reasoning_mismatch_llm_judge(
    df: pd.DataFrame,
    client,  
) -> pd.DataFrame:
    """
    LLM-as-judge consistency checker instead of keyword matching.
    
     asks the judge LLM: does this reasoning actually
    support this star rating for each cot
    
    This is Approach 2  contrasted with the brittle keyword heuristic
    (Approach 1) showcasing a new evaluation for llms being meta evaluators.
    (using the task 2 cot outputs from csv as inputs to a new LLM judge prompt)
    """
    import json

    JUDGE_SYSTEM = """You are an expert evaluator of AI-generated review analysis.

Given a reasoning trace and a star rating (1-5), determine if the reasoning 
logically supports and is consistent with the star rating.

Respond ONLY with this JSON:
{"consistent": true/false, "confidence": "high/medium/low", "reason": "<one sentence>"}

Rules:
- consistent=true if the reasoning's overall sentiment and intensity matches the star rating
- consistent=false if the reasoning is mostly positive but star is 1-2, or mostly negative but star is 4-5
- No markdown, no extra text"""

    rows = []
    cot_rows = df[df["prompt_type"] == "cot"].copy()

    for _, row in tqdm(cot_rows.iterrows(), total=len(cot_rows), desc="LLM Judge"):
        if pd.isna(row.get("reasoning")) or pd.isna(row.get("pred_stars")):
            rows.append({**row, "judge_consistent": None,
                         "judge_reasoning": None, "judge_confidence": None})
            continue

        user_msg = (
            f"Reasoning: {row['reasoning']}\n"
            f"Star rating given: {int(row['pred_stars'])}"
        )
        result = client.complete(JUDGE_SYSTEM, user_msg)

        consistent = None
        judge_reason = None
        confidence = None

        if result["parsed"]:
            p = result["parsed"]
            consistent = p.get("consistent")
            judge_reason = p.get("reason")
            confidence = p.get("confidence")

        rows.append({**row,
                     "judge_consistent": consistent,
                     "judge_reasoning": judge_reason,
                     "judge_confidence": confidence})

    return pd.DataFrame(rows)