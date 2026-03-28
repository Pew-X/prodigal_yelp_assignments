"""
Task 2 Tests: Chain-of-Thought vs Direct Prompting + LLM Judge Evaluation

Covers:
- Direct and CoT prompt building
- Reasoning mismatch detection (keyword heuristic)
- CoT metrics computation
- LLM-as-judge consistency evaluation
- End-to-end task2 workflows
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.prompts import build_direct_prompt, build_cot_prompt
from src.evaluator import (
    detect_reasoning_mismatch,
    compute_cot_metrics,
    detect_reasoning_mismatch_llm_judge,
)


# ──── Test Fixtures ────────────────────────────────────────────

@pytest.fixture
def sample_cot_df():
    """Sample CoT results DataFrame with reasoning and star predictions."""
    return pd.DataFrame({
        "text_snippet": [
            "Great food and service",
            "Terrible experience",
            "Average place",
            "Best ramen ever",
            "Worst place ever",
        ],
        "true_stars": [4, 2, 3, 5, 1],
        "pred_stars": [4, 2, 3, 5, 1],
        "reasoning": [
            "Excellent service and amazing food quality made this a great visit.",
            "Poor service and cold food led to this terrible experience.",
            "Nothing remarkable but nothing bad either, just average.",
            "The best ramen I've ever had with rich broth and perfect noodles.",
            "Absolutely terrible, never coming back to this awful place.",
        ],
        "explanation": [
            "Positive experience",
            "Negative experience",
            "Neutral experience",
            "Exceptional quality",
            "Poor quality",
        ],
        "json_valid": [True, True, True, True, True],
        "raw_response": ["{}", "{}", "{}", "{}", "{}"],
        "latency_ms": [45.2, 52.1, 48.5, 61.3, 50.8],
        "prompt_type": ["cot"] * 5,
    })


@pytest.fixture
def cot_df_with_mismatches():
    """CoT DataFrame with reasoning-star mismatches."""
    return pd.DataFrame({
        "text_snippet": ["Review 1", "Review 2", "Review 3"],
        "true_stars": [4, 3, 2],
        "pred_stars": [1, 2, 5],  # Mismatched predictions
        "reasoning": [
            "This place is excellent, amazing food, wonderful service.",  # Positive but star=1
            "Decent place with good vibes.",  # Positive but star=2
            "Terrible, awful, horrible experience, never coming back.",  # Negative but star=5
        ],
        "explanation": ["Good", "Good", "Bad"],
        "json_valid": [True, True, True],
        "raw_response": ["{}", "{}", "{}"],
        "latency_ms": [45.0, 48.0, 50.0],
        "prompt_type": ["cot"] * 3,
    })


@pytest.fixture
def mixed_df():
    """DataFrame with both direct and CoT prompts."""
    return pd.DataFrame({
        "text_snippet": ["Review 1", "Review 2", "Review 3", "Review 4"],
        "true_stars": [4, 3, 2, 5],
        "pred_stars": [4, 3, 2, 5],
        "reasoning": [None, None, "Good place", "Excellent place"],
        "explanation": ["Good", "Okay", "Meh", "Great"],
        "json_valid": [True, True, True, True],
        "raw_response": ["{}", "{}", "{}", "{}"],
        "latency_ms": [45.0, 48.0, 50.0, 52.0],
        "prompt_type": ["direct", "direct", "cot", "cot"],
    })


# ──── Direct Prompt Tests ────────────────────────────────────

class TestBuildDirectPrompt:
    """Test direct prompting strategy."""

    def test_build_direct_prompt_basic(self):
        """Test basic direct prompt generation."""
        review = "Great food and service"
        prompt = build_direct_prompt(review)
        assert "Rate this Yelp review" in prompt
        assert review in prompt
        assert "1-5 star scale" in prompt

    def test_build_direct_prompt_long_review(self):
        """Test direct prompt with long review."""
        review = "A" * 500
        prompt = build_direct_prompt(review)
        assert review in prompt
        assert len(prompt) > len(review)

    def test_build_direct_prompt_special_characters(self):
        """Test direct prompt with special characters."""
        review = 'Great food! Prices "affordable" & service was excellent (really).'
        prompt = build_direct_prompt(review)
        assert review in prompt

    def test_build_direct_prompt_unicode(self):
        """Test direct prompt with unicode characters."""
        review = "Great! 很好 😊 Excellent!"
        prompt = build_direct_prompt(review)
        assert review in prompt

    def test_build_direct_prompt_empty_string_error(self):
        """Test that empty review raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            build_direct_prompt("")

    def test_build_direct_prompt_none_error(self):
        """Test that None input raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            build_direct_prompt(None)

    def test_build_direct_prompt_non_string_error(self):
        """Test that non-string input raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            build_direct_prompt(12345)

    def test_build_direct_prompt_is_string(self):
        """Test that output is a string."""
        prompt = build_direct_prompt("Test review")
        assert isinstance(prompt, str)


# ──── CoT Prompt Tests ────────────────────────────────────────

class TestBuildCotPrompt:
    """Test chain-of-thought prompting strategy."""

    def test_build_cot_prompt_basic(self):
        """Test basic CoT prompt generation."""
        review = "Great food and service"
        prompt = build_cot_prompt(review)
        assert "step by step" in prompt
        assert review in prompt
        assert "positive signals" in prompt
        assert "negative signals" in prompt
        assert "reasoning" in prompt
        assert "stars" in prompt

    def test_build_cot_prompt_contains_json_format(self):
        """Test that CoT prompt specifies JSON format."""
        review = "Test review"
        prompt = build_cot_prompt(review)
        assert '"reasoning"' in prompt
        assert '"stars"' in prompt
        assert '"explanation"' in prompt

    def test_build_cot_prompt_vs_direct_length(self):
        """Test that CoT prompt is longer than direct."""
        review = "Great food and service"
        direct = build_direct_prompt(review)
        cot = build_cot_prompt(review)
        assert len(cot) > len(direct)

    def test_build_cot_prompt_long_review(self):
        """Test CoT prompt with long review."""
        review = "A" * 500
        prompt = build_cot_prompt(review)
        assert review in prompt

    def test_build_cot_prompt_empty_string_error(self):
        """Test that empty review raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            build_cot_prompt("")

    def test_build_cot_prompt_none_error(self):
        """Test that None input raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            build_cot_prompt(None)

    def test_build_cot_prompt_non_string_error(self):
        """Test that non-string input raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            build_cot_prompt(123)

    def test_build_cot_prompt_is_string(self):
        """Test that output is a string."""
        prompt = build_cot_prompt("Test review")
        assert isinstance(prompt, str)


# ──── Reasoning Mismatch Detection Tests ────────────────────

class TestDetectReasoningMismatch:
    """Test keyword-based reasoning mismatch detection."""

    def test_detect_mismatch_positive_reasoning_low_star(self, cot_df_with_mismatches):
        """Test detection of positive reasoning with low star rating."""
        result = detect_reasoning_mismatch(cot_df_with_mismatches)
        # Row 0: positive reasoning ("excellent", "amazing") but star=1 → mismatch
        assert result.loc[0, "reasoning_mismatch"] == True

    def test_detect_mismatch_negative_reasoning_high_star(self, cot_df_with_mismatches):
        """Test detection of negative reasoning with high star rating."""
        result = detect_reasoning_mismatch(cot_df_with_mismatches)
        # Row 2: negative reasoning ("terrible", "awful") but star=5 → mismatch
        assert result.loc[2, "reasoning_mismatch"] == True

    def test_detect_mismatch_no_false_positives(self, sample_cot_df):
        """Test consistent reasoning-star pairs don't trigger mismatch."""
        result = detect_reasoning_mismatch(sample_cot_df)
        # All rows in sample_cot_df have consistent reasoning
        assert result["reasoning_mismatch"].sum() == 0

    def test_detect_mismatch_returns_dataframe(self, sample_cot_df):
        """Test that output is a DataFrame with mismatch column."""
        result = detect_reasoning_mismatch(sample_cot_df)
        assert isinstance(result, pd.DataFrame)
        assert "reasoning_mismatch" in result.columns
        assert result["reasoning_mismatch"].dtype == bool

    def test_detect_mismatch_missing_reasoning(self):
        """Test handling of missing reasoning."""
        df = pd.DataFrame({
            "reasoning": [None, "Good"],
            "pred_stars": [3, 4],
        })
        result = detect_reasoning_mismatch(df)
        assert result.loc[0, "reasoning_mismatch"] == False

    def test_detect_mismatch_missing_pred_stars(self):
        """Test handling of missing pred_stars."""
        df = pd.DataFrame({
            "reasoning": ["Good", "Good"],
            "pred_stars": [np.nan, 4],
        })
        result = detect_reasoning_mismatch(df)
        assert result.loc[0, "reasoning_mismatch"] == False

    def test_detect_mismatch_boundary_signals(self):
        """Test boundary case with equal positive/negative signals."""
        df = pd.DataFrame({
            "reasoning": ["good and bad"],  # 1 positive, 1 negative signal
            "pred_stars": [1],
        })
        result = detect_reasoning_mismatch(df)
        # Equal signals don't trigger mismatch
        assert result.loc[0, "reasoning_mismatch"] == False

    def test_detect_mismatch_does_not_modify_original(self, sample_cot_df):
        """Test that original DataFrame is not modified."""
        original_cols = set(sample_cot_df.columns)
        detect_reasoning_mismatch(sample_cot_df)
        assert set(sample_cot_df.columns) == original_cols
        assert "reasoning_mismatch" not in sample_cot_df.columns


# ──── CoT Metrics Tests ────────────────────────────────────────

class TestComputeCotMetrics:
    """Test CoT metrics computation."""

    def test_compute_cot_metrics_basic(self, sample_cot_df):
        """Test basic CoT metrics computation."""
        metrics = compute_cot_metrics(sample_cot_df)
        assert "total" in metrics
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert metrics["total"] == 5

    def test_compute_cot_metrics_with_mismatch_column(self, cot_df_with_mismatches):
        """Test CoT metrics with reasoning_mismatch column."""
        df = detect_reasoning_mismatch(cot_df_with_mismatches)
        metrics = compute_cot_metrics(df)
        assert "reasoning_mismatch_count" in metrics
        assert "reasoning_mismatch_rate" in metrics
        assert metrics["reasoning_mismatch_count"] >= 0

    def test_compute_cot_metrics_no_mismatch_column(self, sample_cot_df):
        """Test CoT metrics without reasoning_mismatch column."""
        metrics = compute_cot_metrics(sample_cot_df)
        assert "reasoning_mismatch_count" not in metrics
        assert "reasoning_mismatch_rate" not in metrics

    def test_compute_cot_metrics_is_dict(self, sample_cot_df):
        """Test that output is a dictionary."""
        metrics = compute_cot_metrics(sample_cot_df)
        assert isinstance(metrics, dict)

    def test_compute_cot_metrics_all_perfect(self):
        """Test metrics with all perfect predictions."""
        df = pd.DataFrame({
            "true_stars": [1, 2, 3, 4, 5],
            "pred_stars": [1, 2, 3, 4, 5],
            "json_valid": [True] * 5,
        })
        metrics = compute_cot_metrics(df)
        assert metrics["accuracy"] == 1.0
        assert metrics["format_compliance_rate"] == 1.0

    def test_compute_cot_metrics_all_invalid_json(self):
        """Test metrics with all invalid JSON."""
        df = pd.DataFrame({
            "true_stars": [1, 2, 3],
            "pred_stars": [np.nan, np.nan, np.nan],
            "json_valid": [False] * 3,
        })
        metrics = compute_cot_metrics(df)
        assert metrics["format_compliance_rate"] == 0.0
        assert metrics["accuracy"] is None


# ──── LLM Judge Tests ──────────────────────────────────────────

class TestDetectReasoningMismatchLLMJudge:
    """Test LLM-as-judge consistency evaluation."""

    def test_judge_basic_consistent(self):
        """Test judge evaluation of consistent reasoning."""
        df = pd.DataFrame({
            "text_snippet": ["Review 1"],
            "true_stars": [4],
            "pred_stars": [4],
            "reasoning": ["Excellent service and great food!"],
            "prompt_type": ["cot"],
        })

        mock_client = MagicMock()
        mock_client.complete.return_value = {
            "raw_response": '{"consistent": true, "confidence": "high", "reason": "Reasoning supports 4-star rating"}',
            "parsed": {"consistent": True, "confidence": "high", "reason": "Reasoning supports 4-star rating"},
            "json_valid": True,
            "latency_ms": 45.2,
            "error": None,
        }

        result = detect_reasoning_mismatch_llm_judge(df, mock_client)
        cot_rows = result[result["prompt_type"] == "cot"]
        assert cot_rows.loc[0, "judge_consistent"] == True
        assert cot_rows.loc[0, "judge_confidence"] == "high"

    def test_judge_basic_inconsistent(self):
        """Test judge evaluation of inconsistent reasoning."""
        df = pd.DataFrame({
            "text_snippet": ["Review 1"],
            "true_stars": [1],
            "pred_stars": [1],
            "reasoning": ["Excellent service and great food!"],  # Positive but star=1
            "prompt_type": ["cot"],
        })

        mock_client = MagicMock()
        mock_client.complete.return_value = {
            "raw_response": '{"consistent": false, "confidence": "high", "reason": "Positive reasoning contradicts 1-star"}',
            "parsed": {"consistent": False, "confidence": "high", "reason": "Positive reasoning contradicts 1-star"},
            "json_valid": True,
            "latency_ms": 48.5,
            "error": None,
        }

        result = detect_reasoning_mismatch_llm_judge(df, mock_client)
        cot_rows = result[result["prompt_type"] == "cot"]
        assert cot_rows.loc[0, "judge_consistent"] == False

    def test_judge_missing_reasoning(self):
        """Test judge with missing reasoning."""
        df = pd.DataFrame({
            "text_snippet": ["Review 1"],
            "true_stars": [3],
            "pred_stars": [3],
            "reasoning": [None],
            "prompt_type": ["cot"],
        })

        mock_client = MagicMock()
        result = detect_reasoning_mismatch_llm_judge(df, mock_client)
        cot_rows = result[result["prompt_type"] == "cot"]
        assert cot_rows.loc[0, "judge_consistent"] is None
        # Should not call the client for missing reasoning
        assert mock_client.complete.call_count == 0

    def test_judge_missing_pred_stars(self):
        """Test judge with missing pred_stars."""
        df = pd.DataFrame({
            "text_snippet": ["Review 1"],
            "true_stars": [3],
            "pred_stars": [np.nan],
            "reasoning": ["Good place"],
            "prompt_type": ["cot"],
        })

        mock_client = MagicMock()
        result = detect_reasoning_mismatch_llm_judge(df, mock_client)
        cot_rows = result[result["prompt_type"] == "cot"]
        assert cot_rows.loc[0, "judge_consistent"] is None
        assert mock_client.complete.call_count == 0

    def test_judge_invalid_json_response(self):
        """Test judge with invalid JSON in response."""
        df = pd.DataFrame({
            "text_snippet": ["Review 1"],
            "true_stars": [3],
            "pred_stars": [3],
            "reasoning": ["Good place"],
            "prompt_type": ["cot"],
        })

        mock_client = MagicMock()
        mock_client.complete.return_value = {
            "raw_response": "Invalid JSON",
            "parsed": None,
            "json_valid": False,
            "latency_ms": 50.0,
            "error": None,
        }

        result = detect_reasoning_mismatch_llm_judge(df, mock_client)
        cot_rows = result[result["prompt_type"] == "cot"]
        assert cot_rows.loc[0, "judge_consistent"] is None
        assert cot_rows.loc[0, "judge_confidence"] is None

    def test_judge_preserves_direct_prompts(self):
        """Test that direct prompts are preserved without judge evaluation."""
        df = pd.DataFrame({
            "text_snippet": ["Review 1", "Review 2"],
            "true_stars": [3, 4],
            "pred_stars": [3, 4],
            "reasoning": [None, "Good place"],
            "prompt_type": ["direct", "cot"],
        })

        mock_client = MagicMock()
        mock_client.complete.return_value = {
            "raw_response": '{"consistent": true, "confidence": "high", "reason": "OK"}',
            "parsed": {"consistent": True, "confidence": "high", "reason": "OK"},
            "json_valid": True,
            "latency_ms": 45.0,
            "error": None,
        }

        result = detect_reasoning_mismatch_llm_judge(df, mock_client)
        # Check that result contains both direct and cot rows
        assert len(result) == 2
        # Check the direct row is preserved
        direct_row = result[result["prompt_type"] == "direct"].iloc[0]
        assert direct_row["judge_consistent"] is None
        # Only 1 call to judge (for the CoT row only)
        assert mock_client.complete.call_count == 1

    def test_judge_returns_dataframe(self):
        """Test that judge returns a DataFrame."""
        df = pd.DataFrame({
            "text_snippet": ["Review 1"],
            "true_stars": [3],
            "pred_stars": [3],
            "reasoning": ["Good place"],
            "prompt_type": ["cot"],
        })

        mock_client = MagicMock()
        mock_client.complete.return_value = {
            "raw_response": '{"consistent": true, "confidence": "high", "reason": "OK"}',
            "parsed": {"consistent": True, "confidence": "high", "reason": "OK"},
            "json_valid": True,
            "latency_ms": 45.0,
            "error": None,
        }

        result = detect_reasoning_mismatch_llm_judge(df, mock_client)
        assert isinstance(result, pd.DataFrame)
        assert "judge_consistent" in result.columns
        assert "judge_confidence" in result.columns
        assert "judge_reasoning" in result.columns

    def test_judge_confidence_levels(self):
        """Test different confidence levels from judge."""
        df = pd.DataFrame({
            "text_snippet": ["Rev1", "Rev2", "Rev3"],
            "true_stars": [3, 4, 5],
            "pred_stars": [3, 4, 5],
            "reasoning": ["Good", "Great", "Excellent"],
            "prompt_type": ["cot", "cot", "cot"],
        })

        mock_client = MagicMock()
        responses = [
            {"consistent": True, "confidence": "high", "reason": "Very clear"},
            {"consistent": True, "confidence": "medium", "reason": "Generally OK"},
            {"consistent": False, "confidence": "low", "reason": "Unclear"},
        ]

        # Create a proper side_effect that returns different responses per call
        mock_client.complete.side_effect = [
            {
                "raw_response": "{}",
                "parsed": responses[0],
                "json_valid": True,
                "latency_ms": 45.0,
                "error": None,
            },
            {
                "raw_response": "{}",
                "parsed": responses[1],
                "json_valid": True,
                "latency_ms": 45.0,
                "error": None,
            },
            {
                "raw_response": "{}",
                "parsed": responses[2],
                "json_valid": True,
                "latency_ms": 45.0,
                "error": None,
            },
        ]

        result = detect_reasoning_mismatch_llm_judge(df, mock_client)
        cot_rows = result[result["prompt_type"] == "cot"].reset_index(drop=True)
        assert cot_rows.loc[0, "judge_confidence"] == "high"
        assert cot_rows.loc[1, "judge_confidence"] == "medium"
        assert cot_rows.loc[2, "judge_confidence"] == "low"


# ──── Integration Tests ────────────────────────────────────────

class TestTask2Integration:
    """Integration tests for Task 2 workflows."""

    def test_direct_workflow_produces_metrics(self):
        """Test complete direct prompting workflow."""
        df = pd.DataFrame({
            "text": ["Great food!"] * 10,
            "true_stars": list(range(1, 6)) * 2,
        })

        # Simulate LLM responses
        results = []
        for _, row in df.iterrows():
            results.append({
                "text_snippet": row["text"][:150],
                "true_stars": row["true_stars"],
                "pred_stars": row["true_stars"],  # Perfect predictions
                "reasoning": None,
                "explanation": "Good",
                "json_valid": True,
                "raw_response": "{}",
                "latency_ms": 45.0,
                "prompt_type": "direct",
            })

        from src.evaluator import compute_cot_metrics
        results_df = pd.DataFrame(results)
        metrics = compute_cot_metrics(results_df)

        assert metrics["accuracy"] == 1.0
        assert metrics["total"] == 10

    def test_cot_workflow_with_mismatch_detection(self):
        """Test complete CoT workflow with mismatch detection."""
        df = pd.DataFrame({
            "text_snippet": ["Review 1", "Review 2"],
            "true_stars": [4, 1],
            "pred_stars": [4, 1],
            "reasoning": [
                "Great food and service",  # Consistent: positive, star=4
                "Excellent amazing wonderful loved it",  # Mismatch: 3 positive signals but star=1
            ],
            "explanation": ["Good", "Bad"],
            "json_valid": [True, True],
            "raw_response": ["{}", "{}"],
            "latency_ms": [45.0, 48.0],
            "prompt_type": ["cot", "cot"],
        })

        df = detect_reasoning_mismatch(df)
        metrics = compute_cot_metrics(df)

        assert metrics["reasoning_mismatch_count"] == 1
        assert metrics["reasoning_mismatch_rate"] == 0.5  # 1 out of 2

    def test_mixed_direct_and_cot_workflow(self, mixed_df):
        """Test workflow with both direct and CoT prompts."""
        from src.evaluator import compute_cot_metrics
        metrics = compute_cot_metrics(mixed_df)
        assert metrics["total"] == 4
        assert metrics["accuracy"] == 1.0

    def test_judge_workflow_on_cot_results(self):
        """Test judge evaluation on CoT results."""
        df = pd.DataFrame({
            "text_snippet": ["Review 1"],
            "true_stars": [4],
            "pred_stars": [4],
            "reasoning": ["Great service and excellent food"],
            "prompt_type": ["cot"],
        })

        mock_client = MagicMock()
        mock_client.complete.return_value = {
            "raw_response": '{"consistent": true, "confidence": "high", "reason": "Reasoning matches 4-star"}',
            "parsed": {"consistent": True, "confidence": "high", "reason": "Reasoning matches 4-star"},
            "json_valid": True,
            "latency_ms": 52.1,
            "error": None,
        }

        result = detect_reasoning_mismatch_llm_judge(df, mock_client)
        assert result.iloc[0]["judge_consistent"] == True
        assert len(result) == 1
