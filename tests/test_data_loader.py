"""
Test suite for src.data_loader module.

Tests cover:
- load_yelp_sample function with various parameters
- Stratification validation
- Edge cases and error handling
- Logging verification
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import load_yelp_sample


@pytest.mark.unit
class TestLoadYelpSample:
    """Test suite for load_yelp_sample function."""
    
    def test_load_yelp_sample_default_params(self, mock_dataset_loader):
        """Test load_yelp_sample with default parameters."""
        df = load_yelp_sample()
        
        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "text" in df.columns
        assert "true_stars" in df.columns
        
        # Verify shape and stratification
        assert len(df) == 200  # 40 per class * 5 classes
        assert set(df["true_stars"].unique()) == {1, 2, 3, 4, 5}
        
        # Verify stratification balance
        class_counts = df["true_stars"].value_counts().to_dict()
        for star_rating in range(1, 6):
            assert class_counts[star_rating] == 40, f"Class {star_rating} should have 40 samples"


    def test_load_yelp_sample_custom_n_per_class(self, mock_dataset_loader):
        """Test load_yelp_sample with custom n_per_class parameter."""
        n = 20
        df = load_yelp_sample(n_per_class=n)
        
        assert len(df) == n * 5  # 20 per class * 5 classes
        class_counts = df["true_stars"].value_counts().to_dict()
        for star_rating in range(1, 6):
            assert class_counts[star_rating] == n


    def test_load_yelp_sample_custom_split(self, mock_dataset_loader):
        """Test load_yelp_sample with different splits."""
        for split in ["train", "test"]:
            df = load_yelp_sample(split=split)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0


    def test_load_yelp_sample_custom_seed_reproducibility(self, mock_dataset_loader):
        """Test that same seed produces same results (reproducibility)."""
        df1 = load_yelp_sample(seed=42)
        df2 = load_yelp_sample(seed=42)
        
        # Same seed should produce same sample
        pd.testing.assert_frame_equal(df1, df2)


    def test_load_yelp_sample_different_seed_different_results(self, mock_dataset_loader):
        """Test that different seeds produce different results."""
        df1 = load_yelp_sample(seed=42)
        df2 = load_yelp_sample(seed=43)
        
        # Different seeds should produce different samples (at least in order)
        assert not df1.equals(df2)


    def test_load_yelp_sample_invalid_n_per_class(self, mock_dataset_loader):
        """Test load_yelp_sample with invalid n_per_class."""
        with pytest.raises(ValueError, match="n_per_class must be >= 1"):
            load_yelp_sample(n_per_class=0)
            
        with pytest.raises(ValueError, match="n_per_class must be >= 1"):
            load_yelp_sample(n_per_class=-1)


    def test_load_yelp_sample_invalid_split(self, mock_dataset_loader):
        """Test load_yelp_sample with invalid split."""
        with pytest.raises(ValueError, match="split must be 'train' or 'test'"):
            load_yelp_sample(split="invalid")


    def test_load_yelp_sample_no_api_key(self, caplog):
        """Test load_yelp_sample when dataset loading fails."""
        with patch("src.data_loader.load_dataset") as mock_load:
            mock_load.side_effect = Exception("Network error")
            
            with pytest.raises(Exception):
                load_yelp_sample()
            
            # Verify error was logged
            assert "Failed to load dataset" in caplog.text


    def test_load_yelp_sample_text_column_not_empty(self, mock_dataset_loader):
        """Test that all text entries are non-empty strings."""
        df = load_yelp_sample(n_per_class=10)
        
        # Check that it's a string-like dtype (object, string, or str)
        dtype_str = str(df["text"].dtype).lower()
        assert any(t in dtype_str for t in ["object", "string", "str"]), f"Expected string-like dtype, got {df['text'].dtype}"
        assert not df["text"].isnull().any()  # no null values
        assert all(isinstance(text, str) and len(text) > 0 for text in df["text"])


    def test_load_yelp_sample_star_ratings_valid_range(self, mock_dataset_loader):
        """Test that all star ratings are in valid range [1, 5]."""
        df = load_yelp_sample()
        
        assert df["true_stars"].dtype in [int, "int64"]
        assert df["true_stars"].min() >= 1
        assert df["true_stars"].max() <= 5


    def test_load_yelp_sample_index_reset(self, mock_dataset_loader):
        """Test that DataFrame index is properly reset."""
        df = load_yelp_sample()
        
        # Index should be 0-indexed and sequential
        assert list(df.index) == list(range(len(df)))


    def test_load_yelp_sample_logging(self, mock_dataset_loader, caplog):
        """Test that appropriate log messages are generated."""
        with caplog.at_level("INFO"):
            df = load_yelp_sample(n_per_class=30)
        
        # Verify key log messages (n_per_class=30 * 5 classes = 150 total)
        assert "Loading Yelp" in caplog.text
        assert "Stratified sample created" in caplog.text
        assert "150" in caplog.text  # 30 * 5 classes

    
    @pytest.mark.slow
    def test_load_yelp_sample_large_n_per_class(self, mock_dataset_loader):
        """Test load_yelp_sample with large n_per_class (slow test)."""
        df = load_yelp_sample(n_per_class=100)
        
        assert len(df) == 500
        class_counts = df["true_stars"].value_counts().to_dict()
        for star_rating in range(1, 6):
            assert class_counts[star_rating] == 100


@pytest.mark.unit
class TestDataLoaderIntegration:
    """Integration tests for data loader functionality."""
    
    def test_load_yelp_sample_multiple_consecutive_loads(self, mock_dataset_loader):
        """Test multiple consecutive loads work correctly."""
        df1 = load_yelp_sample(n_per_class=10)
        df2 = load_yelp_sample(n_per_class=15)
        df3 = load_yelp_sample(n_per_class=5)
        
        assert len(df1) == 50
        assert len(df2) == 75
        assert len(df3) == 25
    
    
    def test_load_yelp_sample_dataframe_preservation(self, mock_dataset_loader):
        """Test that DataFrame structure is preserved across different parameters."""
        params = [
            {"n_per_class": 5},
            {"n_per_class": 20},
            {"n_per_class": 40},
        ]
        
        for param in params:
            df = load_yelp_sample(**param)
            assert isinstance(df, pd.DataFrame)
            assert set(df.columns) == {"text", "true_stars"}
            assert set(df["true_stars"].unique()) == {1, 2, 3, 4, 5}
