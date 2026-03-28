"""
Evaluation metrics for Tasks 1 (as of now)

required metrics for task 1:
- format_compliance_rate : % of responses that are valid, parseable JSON
- accuracy               : exact star match on valid predictions
- macro_f1               : unweighted mean F1 across all 5 classes
- per_class_f1           : per-star F1 for error analysis
- error_analysis         : off-by-one / off-by-two / off-by-more buckets
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(df: pd.DataFrame) -> dict:
    total = len(df)
    n_valid = df["json_valid"].sum()
    format_compliance = n_valid / total

    valid = df[df["pred_stars"].notna()].copy()
    valid["pred_stars"] = valid["pred_stars"].astype(int)
    valid["true_stars"] = valid["true_stars"].astype(int)

    if len(valid) == 0:
        return {
            "total": total,
            "valid_predictions": 0,
            "format_compliance_rate": round(format_compliance, 4),
            "accuracy": None,
            "macro_f1": None,
        }

    accuracy = accuracy_score(valid["true_stars"], valid["pred_stars"])
    macro_f1 = f1_score(
        valid["true_stars"], valid["pred_stars"],
        average="macro", labels=[1, 2, 3, 4, 5], zero_division=0,
    )
    per_class = f1_score(
        valid["true_stars"], valid["pred_stars"],
        average=None, labels=[1, 2, 3, 4, 5], zero_division=0,
    )

    return {
        "total": total,
        "valid_predictions": len(valid),
        "format_compliance_rate": round(format_compliance, 4),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class_f1": {str(i + 1): round(float(v), 4) for i, v in enumerate(per_class)},
        "error_analysis": _error_analysis(valid),
    }


def _error_analysis(df: pd.DataFrame) -> dict:
    errors = df[df["pred_stars"] != df["true_stars"]]
    n_err = len(errors)
    if n_err == 0:
        return {"total_errors": 0, "off_by_one": 0, "off_by_two": 0, "off_by_more": 0}

    delta = (errors["pred_stars"] - errors["true_stars"]).abs()
    return {
        "total_errors": n_err,
        "off_by_one": int((delta == 1).sum()),
        "off_by_one_pct": round((delta == 1).sum() / n_err, 3),
        "off_by_two": int((delta == 2).sum()),
        "off_by_more": int((delta >= 3).sum()),
        # "off_by_more_pct": round((delta >= 3).sum() / n_err, 3),
    }


def print_metrics(metrics: dict, title: str = "") -> None:
    if title:
        print(f"\n{'=' * 52}")
        print(f"  {title}")
        print(f"{'=' * 52}")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")