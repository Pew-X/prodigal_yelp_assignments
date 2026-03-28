"""
Test suite for src.evaluator module.

Tests cover:
- Metric computation (accuracy, F1, compliance rate)
- Error analysis and categorization
- Edge cases (zero errors, all invalid, etc.)
- Logging verification
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluator import compute_metrics, _error_analysis, print_metrics


@pytest.mark.unit
class TestComputeMetrics:
    """Test suite for compute_metrics function."""
    
    def test_compute_metrics_perfect_predictions(self, perfect_predictions_dataframe):
        """Test metrics with perfect predictions."""
        df = perfect_predictions_dataframe
        metrics = compute_metrics(df)
        
        assert metrics["total"] == 10
        assert metrics["valid_predictions"] == 10
        assert metrics["format_compliance_rate"] == 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        
        # All classes should have F1 = 1.0
        for star in ["1", "2", "3", "4", "5"]:
            assert metrics["per_class_f1"][star] == 1.0


    def test_compute_metrics_with_errors(self, predictions_dataframe):
        """Test metrics with some prediction errors."""
        df = predictions_dataframe
        metrics = compute_metrics(df)
        
        assert metrics["total"] == 10
        assert metrics["valid_predictions"] == 10
        assert metrics["format_compliance_rate"] == 1.0
        assert 0 < metrics["accuracy"] < 1  # Some errors (6 correct out of 10)
        assert 0 < metrics["macro_f1"] < 1


    def test_compute_metrics_all_invalid(self, invalid_predictions_dataframe):
        """Test metrics with all invalid predictions."""
        df = invalid_predictions_dataframe
        metrics = compute_metrics(df)
        
        assert metrics["total"] == 3
        assert metrics["valid_predictions"] == 0
        assert metrics["format_compliance_rate"] == 0.0
        assert metrics["accuracy"] is None
        assert metrics["macro_f1"] is None


    def test_compute_metrics_returns_correct_types(self, perfect_predictions_dataframe):
        """Test that returned metrics have correct types."""
        df = perfect_predictions_dataframe
        metrics = compute_metrics(df)
        
        assert isinstance(metrics, dict)
        assert isinstance(metrics["total"], int)
        assert isinstance(metrics["valid_predictions"], int)
        assert isinstance(metrics["format_compliance_rate"], float)
        assert isinstance(metrics["accuracy"], float)
        assert isinstance(metrics["macro_f1"], float)
        assert isinstance(metrics["per_class_f1"], dict)
        assert isinstance(metrics["error_analysis"], dict)


    def test_compute_metrics_per_class_f1(self, predictions_dataframe):
        """Test that per-class F1 scores are computed."""
        df = predictions_dataframe
        metrics = compute_metrics(df)
        
        # Should have F1 for each class
        assert len(metrics["per_class_f1"]) == 5
        for star in ["1", "2", "3", "4", "5"]:
            assert star in metrics["per_class_f1"]
            assert 0 <= metrics["per_class_f1"][star] <= 1


    def test_compute_metrics_large_dataset(self):
        """Test metrics with larger dataset."""
        n_samples = 100
        df = pd.DataFrame({
            "text_snippet": [f"Review {i}" for i in range(n_samples)],
            "true_stars": np.random.randint(1, 6, n_samples),
            "pred_stars": np.random.randint(1, 6, n_samples),
            "json_valid": np.random.choice([True, False], n_samples),
            "raw_response": [f'{{}}' for _ in range(n_samples)],
            "explanation": ["Test" for _ in range(n_samples)],
            "latency_ms": [100 for _ in range(n_samples)],
            "error": [None for _ in range(n_samples)],
            "prompt_type": ["zero_shot" for _ in range(n_samples)],
        })
        
        metrics = compute_metrics(df)
        
        assert metrics["total"] == n_samples
        assert 0 <= metrics["format_compliance_rate"] <= 1
        assert metrics["accuracy"] is not None or metrics["valid_predictions"] == 0


    def test_compute_metrics_logging(self, perfect_predictions_dataframe, caplog):
        """Test that metrics computation logs appropriate messages."""
        import logging
        caplog.set_level(logging.INFO)
        
        df = perfect_predictions_dataframe
        metrics = compute_metrics(df)
        
        assert "Computing metrics" in caplog.text
        assert "Format compliance" in caplog.text


@pytest.mark.unit
class TestErrorAnalysis:
    """Test suite for _error_analysis function."""
    
    def test_error_analysis_no_errors(self, perfect_predictions_dataframe):
        """Test error analysis with no errors."""
        df = perfect_predictions_dataframe
        df_valid = df[df["pred_stars"].notna()].copy()
        
        error_analysis = _error_analysis(df_valid)
        
        assert error_analysis["total_errors"] == 0
        assert error_analysis["off_by_one"] == 0
        assert error_analysis["off_by_two"] == 0
        assert error_analysis["off_by_more"] == 0


    def test_error_analysis_off_by_one(self):
        """Test error analysis with off-by-one errors."""
        df = pd.DataFrame({
            "pred_stars": [1, 2, 4, 5],
            "true_stars": [2, 3, 5, 4],  # All off by 1
        })
        
        error_analysis = _error_analysis(df)
        
        assert error_analysis["total_errors"] == 4
        assert error_analysis["off_by_one"] == 4
        assert error_analysis["off_by_two"] == 0
        assert error_analysis["off_by_one_pct"] == 1.0


    def test_error_analysis_off_by_two(self):
        """Test error analysis with off-by-two errors."""
        df = pd.DataFrame({
            "pred_stars": [1, 2, 3],
            "true_stars": [3, 4, 5],  # All off by 2
        })
        
        error_analysis = _error_analysis(df)
        
        assert error_analysis["total_errors"] == 3
        assert error_analysis["off_by_two"] == 3
        assert error_analysis["off_by_one"] == 0


    def test_error_analysis_off_by_more_than_two(self):
        """Test error analysis with errors > 2."""
        df = pd.DataFrame({
            "pred_stars": [1, 1, 1],
            "true_stars": [4, 5, 5],  # All off by 3+
        })
        
        error_analysis = _error_analysis(df)
        
        assert error_analysis["total_errors"] == 3
        assert error_analysis["off_by_more"] == 3
        assert error_analysis["off_by_one"] == 0
        assert error_analysis["off_by_two"] == 0


    def test_error_analysis_mixed_errors(self):
        """Test error analysis with mixed error types."""
        df = pd.DataFrame({
            "pred_stars": [1, 2, 3, 5],
            "true_stars": [2, 4, 3, 1],  # off-by-1: idx0, off-by-2: idx1, no error: idx2, off-by-4: idx3
        })
        
        error_analysis = _error_analysis(df)
        
        assert error_analysis["total_errors"] == 3
        assert error_analysis["off_by_one"] == 1
        assert error_analysis["off_by_two"] == 1
        assert error_analysis["off_by_more"] == 1


    def test_error_analysis_percentages(self):
        """Test that error percentages are correctly computed."""
        df = pd.DataFrame({
            "pred_stars": [1, 2, 2, 3],
            "true_stars": [2, 4, 2, 3],  # off-by-1: idx0, off-by-2: idx1, no error: idx2, no error: idx3
        })
        
        error_analysis = _error_analysis(df)
        
        assert error_analysis["total_errors"] == 2
        assert error_analysis["off_by_one"] == 1
        assert error_analysis["off_by_one_pct"] == pytest.approx(0.5)
        assert error_analysis["off_by_two"] == 1


    def test_error_analysis_logging(self, predictions_dataframe, caplog):
        """Test that error analysis logs appropriate messages."""
        import logging
        caplog.set_level(logging.DEBUG)
        
        df = predictions_dataframe[predictions_dataframe["pred_stars"].notna()].copy()
        _error_analysis(df)
        
        assert "Error analysis" in caplog.text


@pytest.mark.unit
class TestPrintMetrics:
    """Test suite for print_metrics function."""
    
    def test_print_metrics_with_title(self, capsys, perfect_predictions_dataframe):
        """Test print_metrics with a title."""
        metrics = compute_metrics(perfect_predictions_dataframe)
        
        print_metrics(metrics, title="Test Results")
        captured = capsys.readouterr()
        
        assert "Test Results" in captured.out
        assert "=" * 52 in captured.out
        assert "total" in captured.out
        assert "accuracy" in captured.out


    def test_print_metrics_without_title(self, capsys, perfect_predictions_dataframe):
        """Test print_metrics without a title."""
        metrics = compute_metrics(perfect_predictions_dataframe)
        
        print_metrics(metrics)
        captured = capsys.readouterr()
        
        assert "total" in captured.out
        assert "accuracy" in captured.out


    def test_print_metrics_all_keys_printed(self, capsys, perfect_predictions_dataframe):
        """Test that all metric keys are printed."""
        metrics = compute_metrics(perfect_predictions_dataframe)
        
        print_metrics(metrics)
        captured = capsys.readouterr()
        
        for key in metrics.keys():
            assert key in captured.out


    def test_print_metrics_logging(self, perfect_predictions_dataframe, caplog):
        """Test that print_metrics logs appropriate messages."""
        import logging
        caplog.set_level(logging.INFO)
        
        metrics = compute_metrics(perfect_predictions_dataframe)
        print_metrics(metrics, title="Test Results")
        
        assert "Printing metrics section" in caplog.text


@pytest.mark.unit
class TestMetricsEdgeCases:
    """Test edge cases for metrics computation."""
    
    def test_compute_metrics_single_sample(self):
        """Test metrics with single sample."""
        df = pd.DataFrame({
            "text_snippet": ["Review"],
            "true_stars": [5],
            "pred_stars": [5],
            "json_valid": [True],
            "raw_response": ['{}'],
            "explanation": ["Good"],
            "latency_ms": [100],
            "error": [None],
            "prompt_type": ["zero_shot"],
        })
        
        metrics = compute_metrics(df)
        
        assert metrics["total"] == 1
        assert metrics["valid_predictions"] == 1
        assert metrics["accuracy"] == 1.0


    def test_compute_metrics_nan_handling(self):
        """Test metrics with NaN values in predictions."""
        df = pd.DataFrame({
            "text_snippet": ["Review 1", "Review 2", "Review 3"],
            "true_stars": [5, 3, 1],
            "pred_stars": [5.0, np.nan, 1.0],
            "json_valid": [True, False, True],
            "raw_response": ['{}', "invalid", '{}'],
            "explanation": ["Good", None, "Bad"],
            "latency_ms": [100, 0, 100],
            "error": [None, "error", None],
            "prompt_type": ["zero_shot", "zero_shot", "few_shot"],
        })
        
        metrics = compute_metrics(df)
        
        assert metrics["valid_predictions"] == 2
        assert metrics["format_compliance_rate"] < 1.0


    def test_compute_metrics_all_same_prediction(self):
        """Test metrics when all predictions are the same."""
        df = pd.DataFrame({
            "text_snippet": [f"Review {i}" for i in range(5)],
            "true_stars": [1, 2, 3, 4, 5],
            "pred_stars": [3, 3, 3, 3, 3],  # All predict 3
            "json_valid": [True, True, True, True, True],
            "raw_response": ['{}'] * 5,
            "explanation": ["3"] * 5,
            "latency_ms": [100] * 5,
            "error": [None] * 5,
            "prompt_type": ["zero_shot"] * 5,
        })
        
        metrics = compute_metrics(df)
        
        assert metrics["accuracy"] < 1.0
        # Class 3 should have higher F1
        assert metrics["per_class_f1"]["3"] > 0


    def test_compute_metrics_boundary_values(self):
        """Test metrics with boundary star values (1 and 5)."""
        df = pd.DataFrame({
            "text_snippet": ["Bad", "Good"],
            "true_stars": [1, 5],
            "pred_stars": [1, 5],
            "json_valid": [True, True],
            "raw_response": ['{"stars": 1}', '{"stars": 5}'],
            "explanation": ["Bad", "Good"],
            "latency_ms": [100, 100],
            "error": [None, None],
            "prompt_type": ["zero_shot", "zero_shot"],
        })
        
        metrics = compute_metrics(df)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["per_class_f1"]["1"] == 1.0
        assert metrics["per_class_f1"]["5"] == 1.0
