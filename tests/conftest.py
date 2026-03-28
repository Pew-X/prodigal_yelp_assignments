"""
Pytest configuration and shared fixtures for the test suite.
"""

import logging
import os
import sys
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


os.makedirs("tests/logs", exist_ok=True)


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("tests/logs/pytest.log"),
        logging.StreamHandler(),
    ],
)


@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """Setup for entire test session."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("PYTEST SESSION STARTED")
    logger.info("=" * 80)
    yield
    logger.info("=" * 80)
    logger.info("PYTEST SESSION COMPLETED")
    logger.info("=" * 80)


@pytest.fixture
def sample_reviews():
    """Fixture providing sample reviews for testing."""
    return [
        "Absolutely terrible. Food was cold, waiter was rude, waited 45 minutes despite having a reservation. Never coming back.",
        "Disappointing. Dry burger and overpriced. Service was slow but at least the staff seemed friendly enough.",
        "Decent enough spot for a quick lunch. Pasta was fine, service was average. Nothing memorable but nothing bad either.",
        "Really enjoyed dinner here. Salmon was cooked perfectly, cocktails were creative, and service was attentive without being overbearing. Would definitely return.",
        "Phenomenal. Best ramen outside of Japan. Rich complex broth, perfect noodles, and staff who made us feel like regulars on our first visit.",
    ]


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame for testing."""
    return pd.DataFrame({
        "text": [
            "Great food!",
            "Mediocre experience",
            "Terrible service",
            "Excellent quality",
            "Average place",
        ],
        "true_stars": [5, 3, 1, 5, 3],
    })


@pytest.fixture
def predictions_dataframe():
    """Fixture providing predictions DataFrame for metric testing."""
    return pd.DataFrame({
        "text_snippet": [f"Review {i}" for i in range(10)],
        "true_stars": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "pred_stars": [1, 2, 2, 2, 3, 1, 4, 5, 5, 4],  # Some errors 
        "json_valid": [True] * 10,
        "raw_response": ['{"stars": 1}'] * 10,
        "explanation": ["Test"] * 10,
        "latency_ms": [150] * 10,
        "error": [None] * 10,
        "prompt_type": ["zero_shot"] * 5 + ["few_shot"] * 5,
    })


@pytest.fixture
def perfect_predictions_dataframe():
    """Fixture providing perfect predictions DataFrame."""
    return pd.DataFrame({
        "text_snippet": [f"Review {i}" for i in range(10)],
        "true_stars": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "pred_stars": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],  # Perfect predictions (2 per class)
        "json_valid": [True] * 10,
        "raw_response": ['{"stars": 1}'] * 10,
        "explanation": ["Perfect"] * 10,
        "latency_ms": [150] * 10,
        "error": [None] * 10,
        "prompt_type": ["zero_shot"] * 5 + ["few_shot"] * 5,
    })


@pytest.fixture
def invalid_predictions_dataframe():
    """Fixture providing DataFrame with all invalid predictions."""
    return pd.DataFrame({
        "text_snippet": ["Review 1", "Review 2", "Review 3"],
        "true_stars": [5, 3, 1],
        "pred_stars": [None, None, None],
        "json_valid": [False, False, False],
        "raw_response": ["invalid json", "more invalid", "still invalid"],
        "explanation": [None, None, None],
        "latency_ms": [0, 0, 0],
        "error": ["Parse error", "Parse error", "Parse error"],
        "prompt_type": ["zero_shot", "few_shot", "zero_shot"],
    })


@pytest.fixture
def mock_groq_client():
    """Fixture providing a mocked Groq client."""
    with patch("src.llm_client.Groq") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        
        def create_response(*args, **kwargs):
            import time
            time.sleep(0.01)  # Add 10ms latency
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"stars": 4, "explanation": "Great!"}'
            return mock_response
        
        mock_instance.chat.completions.create.side_effect = create_response
        
        yield mock_instance


@pytest.fixture
def mock_groq_client_failure():
    """Fixture providing a mocked Groq client that fails."""
    with patch("src.llm_client.Groq") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.chat.completions.create.side_effect = Exception("API Error")
        
        yield mock_instance


@pytest.fixture
def mock_dataset_loader():
    """Fixture providing a mocked dataset loader."""
    with patch("src.data_loader.load_dataset") as mock_load:
        # Create mock dataset with 200 samples per class (1000 total) to support various test scenarios
        def create_mock_dataset():
            mock_dataset = []
            for label in range(5):
                for i in range(200):
                    mock_dataset.append({
                        "label": label,
                        "text": f"Sample review {i} for class {label}. This is a detailed review with enough content to be realistic."
                    })
            return mock_dataset
        
        # Create a mock object that behaves like an iterable
        # Important: Make __iter__ return a NEW iterator each time (not reuse same iterator)
        def mock_load_side_effect(*args, **kwargs):
            mock_dataset_obj = MagicMock()
            mock_dataset = create_mock_dataset()
            mock_dataset_obj.__iter__ = MagicMock(return_value=iter(mock_dataset))
            mock_dataset_obj.__len__ = MagicMock(return_value=len(mock_dataset))
            return mock_dataset_obj
        
        mock_load.side_effect = mock_load_side_effect
        yield mock_load


def pytest_configure(config):
    """Pytest hook to configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture
def caplog_handler(caplog):
    """Fixture to capture and verify log messages."""
    return caplog
