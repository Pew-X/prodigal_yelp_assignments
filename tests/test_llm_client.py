"""
Test suite for src.llm_client module.

Tests cover:
- LLMClient initialization and configuration
- JSON response parsing with multiple strategies
- Error handling and retry logic
- Logging verification
"""

import pytest
import json
from unittest.mock import patch, MagicMock, Mock
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm_client import LLMClient


@pytest.mark.unit
class TestLLMClientInitialization:
    """Test suite for LLMClient initialization."""
    
    def test_llm_client_init_default_params(self, mock_groq_client):
        """Test LLMClient initialization with default parameters."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            client = LLMClient()
            
            assert client.model == "llama-3.1-8b-instant"
            assert client.max_retries == 3
            assert client.retry_delay == 2.0


    def test_llm_client_init_custom_params(self, mock_groq_client):
        """Test LLMClient initialization with custom parameters."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            client = LLMClient(
                model="custom-model",
                max_retries=5,
                retry_delay=3.0
            )
            
            assert client.model == "custom-model"
            assert client.max_retries == 5
            assert client.retry_delay == 3.0


    def test_llm_client_init_no_api_key(self):
        """Test LLMClient initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            # Make sure GROQ_API_KEY is not set
            os.environ.pop("GROQ_API_KEY", None)
            
            with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
                LLMClient()


    def test_llm_client_init_groq_client_failure(self):
        """Test LLMClient initialization handles Groq client failure."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            with patch("src.llm_client.Groq") as mock_groq:
                mock_groq.side_effect = Exception("Connection error")
                
                with pytest.raises(Exception, match="Connection error"):
                    LLMClient()


    def test_llm_client_init_logging(self, mock_groq_client, caplog):
        """Test that initialization logs appropriate messages."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            import logging
            caplog.set_level(logging.INFO)
            
            client = LLMClient()
            
            assert "LLMClient initialized successfully" in caplog.text


@pytest.mark.unit
class TestLLMClientCompletion:
    """Test suite for LLMClient.complete method."""
    
    def test_complete_successful_response(self, mock_groq_client):
        """Test successful completion response."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            client = LLMClient()
            
            result = client.complete(
                system_prompt="You are helpful",
                user_prompt="Hello"
            )
            
            assert result["raw_response"] is not None
            assert result["json_valid"] is True
            assert result["parsed"] is not None
            assert result["latency_ms"] > 0
            assert result["error"] is None


    def test_complete_invalid_json_response(self, mock_groq_client):
        """Test completion with invalid JSON response."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            # Clear side_effect and set return_value instead
            mock_groq_client.chat.completions.create.side_effect = None
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Not valid JSON"
            mock_groq_client.chat.completions.create.return_value = mock_response
            
            client = LLMClient()
            result = client.complete(
                system_prompt="You are helpful",
                user_prompt="Hello"
            )
            
            assert result["json_valid"] is False
            assert result["parsed"] is None


    def test_complete_json_with_markdown_fences(self, mock_groq_client):
        """Test completion with JSON wrapped in markdown fences."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            # Mock response with markdown fences
            mock_response = '```json\n{"stars": 4, "explanation": "Good"}\n```'
            mock_groq_client.chat.completions.create.return_value.choices[0].message.content = mock_response
            
            client = LLMClient()
            result = client.complete(
                system_prompt="You are helpful",
                user_prompt="Hello"
            )
            
            assert result["json_valid"] is True
            assert result["parsed"]["stars"] == 4


    def test_complete_custom_temperature(self, mock_groq_client):
        """Test completion with custom temperature."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            client = LLMClient()
            
            result = client.complete(
                system_prompt="You are helpful",
                user_prompt="Hello",
                temperature=0.5
            )
            
            assert result["error"] is None


    def test_complete_latency_tracking(self, mock_groq_client):
        """Test that latency is properly tracked."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            # Add delay to mock response
            def delayed_create(*args, **kwargs):
                time.sleep(0.01)  # 10ms delay
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = '{"stars": 4}'
                return mock_response
            
            mock_groq_client.chat.completions.create.side_effect = delayed_create
            
            client = LLMClient()
            result = client.complete(
                system_prompt="You are helpful",
                user_prompt="Hello"
            )
            
            assert result["latency_ms"] >= 10  # At least 10ms (our delay)


    def test_complete_retry_on_failure(self, mock_groq_client):
        """Test retry logic on API failure."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            call_count = 0
            
            def failing_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise Exception("API Error")
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = '{"stars": 4}'
                return mock_response
            
            mock_groq_client.chat.completions.create.side_effect = failing_create
            
            client = LLMClient(max_retries=3, retry_delay=0.1)
            result = client.complete(
                system_prompt="You are helpful",
                user_prompt="Hello"
            )
            
            assert call_count == 2  # First attempt failed, second succeeded
            assert result["json_valid"] is True


    def test_complete_exhausted_retries(self, mock_groq_client):
        """Test completion fails after exhausting retries."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            mock_groq_client.chat.completions.create.side_effect = Exception("Persistent error")
            
            client = LLMClient(max_retries=2, retry_delay=0.01)
            result = client.complete(
                system_prompt="You are helpful",
                user_prompt="Hello"
            )
            
            assert result["json_valid"] is False
            assert result["raw_response"] is None
            assert "Persistent error" in result["error"]


    def test_complete_logging(self, mock_groq_client, caplog):
        """Test that completion logs appropriate messages."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            import logging
            caplog.set_level(logging.DEBUG)
            
            client = LLMClient()
            result = client.complete(
                system_prompt="You are helpful",
                user_prompt="Hello"
            )
            
            assert "Starting completion request" in caplog.text
            assert "received in" in caplog.text


@pytest.mark.unit
class TestLLMClientJSONParsing:
    """Test suite for LLMClient._parse_json method."""
    
    def test_parse_json_valid_json(self, mock_groq_client):
        """Test parsing valid JSON."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            client = LLMClient()
            
            text = '{"stars": 5, "explanation": "Excellent"}'
            parsed, is_valid = client._parse_json(text)
            
            assert is_valid is True
            assert parsed["stars"] == 5
            assert parsed["explanation"] == "Excellent"


    def test_parse_json_with_markdown_fences(self, mock_groq_client):
        """Test parsing JSON with markdown fences."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            client = LLMClient()
            
            text = '```json\n{"stars": 4}\n```'
            parsed, is_valid = client._parse_json(text)
            
            assert is_valid is True
            assert parsed["stars"] == 4


    def test_parse_json_with_extra_text(self, mock_groq_client):
        """Test parsing JSON with extra surrounding text."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            client = LLMClient()
            
            text = 'Here is the JSON: ```{"stars": 3}```'
            parsed, is_valid = client._parse_json(text)
            
            assert is_valid is True
            assert parsed["stars"] == 3


    def test_parse_json_invalid_json(self, mock_groq_client):
        """Test parsing invalid JSON."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            client = LLMClient()
            
            text = "This is not JSON at all"
            parsed, is_valid = client._parse_json(text)
            
            assert is_valid is False
            assert parsed is None


    def test_parse_json_malformed_json_in_fences(self, mock_groq_client):
        """Test parsing valid JSON with non-integer star value in fences."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            client = LLMClient()
            
            text = '```json\n{"stars": "not_an_int"}\n```'
            parsed, is_valid = client._parse_json(text)
            
            # Valid JSON even if stars is not an integer
            assert is_valid is True
            assert parsed["stars"] == "not_an_int"


    def test_parse_json_empty_string(self, mock_groq_client):
        """Test parsing empty string."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            client = LLMClient()
            
            text = ""
            parsed, is_valid = client._parse_json(text)
            
            assert is_valid is False
            assert parsed is None


    def test_parse_json_logging(self, mock_groq_client, caplog):
        """Test that JSON parsing logs appropriate messages."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            import logging
            caplog.set_level(logging.DEBUG)
            
            client = LLMClient()
            text = '{"stars": 5}'
            parsed, is_valid = client._parse_json(text)
            
            assert "Attempting to parse JSON" in caplog.text
            assert "parsed successfully" in caplog.text


@pytest.mark.unit
class TestLLMClientEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_complete_empty_prompts(self, mock_groq_client):
        """Test completion with empty prompts."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            client = LLMClient()
            
            result = client.complete(
                system_prompt="",
                user_prompt=""
            )
            
            # Should not raise error, but may not be meaningful
            assert result is not None


    def test_complete_very_long_response(self, mock_groq_client):
        """Test completion with very long response."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            long_json = '{"stars": 5, "explanation": "' + "a" * 10000 + '"}'
            mock_groq_client.chat.completions.create.return_value.choices[0].message.content = long_json
            
            client = LLMClient()
            result = client.complete(
                system_prompt="You are helpful",
                user_prompt="Hello"
            )
            
            assert result["json_valid"] is True


    def test_complete_special_characters_in_response(self, mock_groq_client):
        """Test completion with special characters in JSON."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            json_with_special = '{"stars": 5, "explanation": "Great! Special chars: !@#$%^&*()"}'
            mock_groq_client.chat.completions.create.side_effect = None
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json_with_special
            mock_groq_client.chat.completions.create.return_value = mock_response
            
            client = LLMClient()
            result = client.complete(
                system_prompt="You are helpful",
                user_prompt="Hello"
            )
            
            assert result["json_valid"] is True
            assert "Special chars" in result["parsed"]["explanation"]


    def test_complete_unicode_in_response(self, mock_groq_client):
        """Test completion with Unicode characters in response."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
            json_with_unicode = '{"stars": 5, "explanation": "Amazing! ✨🌟"}'
            mock_groq_client.chat.completions.create.side_effect = None
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json_with_unicode
            mock_groq_client.chat.completions.create.return_value = mock_response
            
            client = LLMClient()
            result = client.complete(
                system_prompt="You are helpful",
                user_prompt="Hello"
            )
            
            assert result["json_valid"] is True
            assert "✨" in result["parsed"]["explanation"]
