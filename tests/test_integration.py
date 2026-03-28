"""
Integration tests for the complete pipeline.

Tests cover:
- End-to-end workflow (data loading -> prompting -> LLM -> evaluation)
- Module interactions and data flow
- Error recovery and resilience
"""

import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import load_yelp_sample
from src.llm_client import LLMClient
from src.prompts import SYSTEM_PROMPT, build_zero_shot_prompt, build_few_shot_prompt
from src.evaluator import compute_metrics


@pytest.mark.integration
class TestEndToEndPipeline:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_zero_shot(self, mock_dataset_loader, mock_groq_client):
        """Test complete zero-shot pipeline: data -> prompts -> LLM -> metrics."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            # Step 1: Load data
            df = load_yelp_sample(n_per_class=5)
            assert len(df) == 25
            
            # Step 2: Create client
            client = LLMClient()
            
            # Step 3: Build and execute prompts
            results = []
            for _, row in df.iterrows():
                review = row["text"]
                prompt = build_zero_shot_prompt(review)
                result = client.complete(SYSTEM_PROMPT, prompt)
                
                pred_stars = None
                if result["parsed"] and "stars" in result["parsed"]:
                    pred_stars = result["parsed"]["stars"]
                
                results.append({
                    "text": review,
                    "true_stars": row["true_stars"],
                    "pred_stars": pred_stars,
                    "json_valid": result["json_valid"],
                })
            
            results_df = pd.DataFrame(results)
            
            # Step 4: Compute metrics
            metrics = compute_metrics(results_df)
            
            # Verify metrics are computed
            assert metrics["total"] == 25
            assert metrics["valid_predictions"] > 0
            assert 0 <= metrics["format_compliance_rate"] <= 1


    def test_end_to_end_few_shot(self, mock_dataset_loader, mock_groq_client):
        """Test complete few-shot pipeline."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            # Load data
            df = load_yelp_sample(n_per_class=3)
            
            # Create client
            client = LLMClient()
            
            # Process with few-shot
            results = []
            for _, row in df.iterrows():
                prompt = build_few_shot_prompt(row["text"])
                result = client.complete(SYSTEM_PROMPT, prompt)
                
                pred_stars = None
                if result["parsed"] and "stars" in result["parsed"]:
                    val = result["parsed"]["stars"]
                    if isinstance(val, (int, float)) and 1 <= val <= 5:
                        pred_stars = int(val)
                
                results.append({
                    "text": row["text"],
                    "true_stars": row["true_stars"],
                    "pred_stars": pred_stars,
                    "json_valid": result["json_valid"],
                })
            
            results_df = pd.DataFrame(results)
            metrics = compute_metrics(results_df)
            
            assert metrics["total"] == 15


    def test_pipeline_with_mixed_valid_invalid_responses(self, mock_dataset_loader):
        """Test pipeline when LLM returns both valid and invalid responses."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            with patch("src.llm_client.Groq") as mock_groq:
                mock_instance = MagicMock()
                
                # Alternate between valid and invalid responses
                responses = [
                    '{"stars": 5, "explanation": "Great"}',
                    'Not valid JSON',
                    '{"stars": 3, "explanation": "Average"}',
                    'Another invalid response',
                    '{"stars": 1, "explanation": "Bad"}',
                ]
                
                response_objects = []
                for resp in responses:
                    mock_resp = MagicMock()
                    mock_resp.choices = [MagicMock()]
                    mock_resp.choices[0].message.content = resp
                    response_objects.append(mock_resp)
                
                mock_instance.chat.completions.create.side_effect = response_objects
                mock_groq.return_value = mock_instance
                
                # Process with mixed responses
                df = load_yelp_sample(n_per_class=1)
                client = LLMClient()
                
                results = []
                for i, (_, row) in enumerate(df.iterrows()):
                    if i >= len(responses):
                        break
                    
                    result = client.complete(SYSTEM_PROMPT, build_zero_shot_prompt(row["text"]))
                    
                    pred_stars = None
                    if result["parsed"] and "stars" in result["parsed"]:
                        pred_stars = result["parsed"]["stars"]
                    
                    results.append({
                        "true_stars": row["true_stars"],
                        "pred_stars": pred_stars,
                        "json_valid": result["json_valid"],
                    })
                
                results_df = pd.DataFrame(results)
                metrics = compute_metrics(results_df)
                
                # Should handle both valid and invalid responses
                assert metrics["valid_predictions"] < metrics["total"]


    def test_pipeline_retry_mechanism(self, mock_dataset_loader):
        """Test that retry mechanism works in pipeline."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            with patch("src.llm_client.Groq") as mock_groq:
                mock_instance = MagicMock()
                
                # Simulate retry: fail once, then succeed
                call_count = 0
                def side_effect_fn(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 1:
                        raise Exception("API Error")
                    mock_resp = MagicMock()
                    mock_resp.choices = [MagicMock()]
                    mock_resp.choices[0].message.content = '{"stars": 4}'
                    return mock_resp
                
                mock_instance.chat.completions.create.side_effect = side_effect_fn
                mock_groq.return_value = mock_instance
                
                # Single request that will retry
                client = LLMClient(max_retries=2, retry_delay=0.01)
                result = client.complete(SYSTEM_PROMPT, "Test")
                
                assert result["json_valid"] is True
                assert result["error"] is None


@pytest.mark.integration
class TestDataFlowValidation:
    """Test that data flows correctly through the pipeline."""
    
    def test_data_integrity_through_pipeline(self, mock_dataset_loader, mock_groq_client):
        """Test that data is preserved through pipeline."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            # Load original data
            original_df = load_yelp_sample(n_per_class=2)
            original_texts = set(original_df["text"].values)
            original_stars = set(original_df["true_stars"].values)
            
            # Process through pipeline
            client = LLMClient()
            results = []
            
            for _, row in original_df.iterrows():
                prompt = build_zero_shot_prompt(row["text"])
                result = client.complete(SYSTEM_PROMPT, prompt)
                
                results.append({
                    "text": row["text"],
                    "true_stars": row["true_stars"],
                    "json_valid": result["json_valid"],
                })
            
            results_df = pd.DataFrame(results)
            
            # Verify data integrity
            assert set(results_df["text"].values) == original_texts
            assert set(results_df["true_stars"].values) == original_stars


    def test_metrics_consistency(self, mock_groq_client, perfect_predictions_dataframe):
        """Test that metrics are consistent across multiple computations."""
        # Compute metrics twice
        metrics1 = compute_metrics(perfect_predictions_dataframe)
        metrics2 = compute_metrics(perfect_predictions_dataframe)
        
        # Should be identical
        assert metrics1 == metrics2


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in pipeline."""
    
    def test_pipeline_handles_invalid_star_values(self, mock_dataset_loader, mock_groq_client):
        """Test pipeline handles invalid star values from LLM."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            # Mock invalid star values
            invalid_responses = [
                '{"stars": 10}',  # Out of range
                '{"stars": 0}',   # Below range
                '{"stars": -1}',  # Negative
                '{"stars": 3.5}', # Float (may be valid)
            ]
            
            with patch("src.llm_client.Groq") as mock_groq:
                mock_instance = MagicMock()
                
                response_objects = []
                for resp in invalid_responses:
                    mock_resp = MagicMock()
                    mock_resp.choices = [MagicMock()]
                    mock_resp.choices[0].message.content = resp
                    response_objects.append(mock_resp)
                
                mock_instance.chat.completions.create.side_effect = response_objects
                mock_groq.return_value = mock_instance
                
                client = LLMClient()
                results = []
                
                for i, resp in enumerate(invalid_responses):
                    result = client.complete(SYSTEM_PROMPT, f"Review {i}")
                    
                    pred_stars = None
                    if result["parsed"] and "stars" in result["parsed"]:
                        val = result["parsed"]["stars"]
                        if isinstance(val, (int, float)) and 1 <= val <= 5:
                            pred_stars = int(val)
                    
                    results.append({
                        "true_stars": 3,
                        "pred_stars": pred_stars,
                        "json_valid": result["json_valid"],
                    })
                
                results_df = pd.DataFrame(results)
                metrics = compute_metrics(results_df)
                
                # Pipeline should handle gracefully
                assert metrics is not None
                assert metrics["total"] > 0


    def test_pipeline_handles_api_failures(self, mock_dataset_loader):
        """Test pipeline handles API failures gracefully."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            with patch("src.llm_client.Groq") as mock_groq:
                mock_instance = MagicMock()
                mock_instance.chat.completions.create.side_effect = Exception("API Timeout")
                mock_groq.return_value = mock_instance
                
                client = LLMClient(max_retries=1, retry_delay=0.001)
                result = client.complete(SYSTEM_PROMPT, "Test")
                
                # Should gracefully fail with error message
                assert result["error"] is not None
                assert result["json_valid"] is False


@pytest.mark.integration
class TestScalability:
    """Test pipeline with various data scales."""
    
    def test_pipeline_large_batch(self, mock_dataset_loader, mock_groq_client):
        """Test pipeline with larger batch of data."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            df = load_yelp_sample(n_per_class=20)
            assert len(df) == 100
            
            # Verify structure
            assert "text" in df.columns
            assert "true_stars" in df.columns
            
            # Compute metrics on full batch
            results_df = pd.DataFrame({
                "true_stars": df["true_stars"],
                "pred_stars": df["true_stars"],  # Synthetic perfect predictions
                "json_valid": [True] * len(df),
            })
            
            metrics = compute_metrics(results_df)
            assert metrics["total"] == 100
            assert metrics["accuracy"] == 1.0
