"""
Test suite for src.prompts module.

Tests cover:
- Prompt builder functions (zero-shot, few-shot)
- System prompt and example consistency
- Error handling and validation
- Logging verification
"""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.prompts import (
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    build_zero_shot_prompt,
    build_few_shot_prompt,
)


@pytest.mark.unit
class TestSystemPrompt:
    """Test suite for SYSTEM_PROMPT constant."""
    
    def test_system_prompt_exists(self):
        """Test that SYSTEM_PROMPT is defined."""
        assert SYSTEM_PROMPT is not None
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0


    def test_system_prompt_contains_instructions(self):
        """Test that system prompt contains key instructions."""
        assert "JSON" in SYSTEM_PROMPT or "json" in SYSTEM_PROMPT
        assert "stars" in SYSTEM_PROMPT
        assert "1" in SYSTEM_PROMPT and "5" in SYSTEM_PROMPT


    def test_system_prompt_format_guidelines(self):
        """Test that system prompt specifies format requirements."""
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "format" in prompt_lower or "output" in prompt_lower


@pytest.mark.unit
class TestFewShotExamples:
    """Test suite for FEW_SHOT_EXAMPLES constant."""
    
    def test_few_shot_examples_count(self):
        """Test that there are 5 examples (one per star rating)."""
        assert len(FEW_SHOT_EXAMPLES) == 5


    def test_few_shot_examples_structure(self):
        """Test that each example has correct structure."""
        for i, example in enumerate(FEW_SHOT_EXAMPLES):
            assert "review" in example
            assert "label" in example
            assert isinstance(example["review"], str)
            assert isinstance(example["label"], dict)
            assert "stars" in example["label"]
            assert "explanation" in example["label"]


    def test_few_shot_examples_star_ratings(self):
        """Test that examples cover all star ratings 1-5."""
        ratings = [ex["label"]["stars"] for ex in FEW_SHOT_EXAMPLES]
        assert set(ratings) == {1, 2, 3, 4, 5}
        assert len(ratings) == 5


    def test_few_shot_examples_review_non_empty(self):
        """Test that all reviews are non-empty strings."""
        for example in FEW_SHOT_EXAMPLES:
            assert isinstance(example["review"], str)
            assert len(example["review"]) > 0


    def test_few_shot_examples_explanation_non_empty(self):
        """Test that all explanations are non-empty strings."""
        for example in FEW_SHOT_EXAMPLES:
            explanation = example["label"]["explanation"]
            assert isinstance(explanation, str)
            assert len(explanation) > 0


    def test_few_shot_examples_ratings_are_integers(self):
        """Test that all star ratings are integers."""
        for example in FEW_SHOT_EXAMPLES:
            rating = example["label"]["stars"]
            assert isinstance(rating, int)
            assert 1 <= rating <= 5


@pytest.mark.unit
class TestBuildZeroShotPrompt:
    """Test suite for build_zero_shot_prompt function."""
    
    def test_build_zero_shot_basic(self):
        """Test basic zero-shot prompt building."""
        review = "Great food!"
        prompt = build_zero_shot_prompt(review)
        
        assert isinstance(prompt, str)
        assert review in prompt
        assert "Classify" in prompt or "classify" in prompt


    def test_build_zero_shot_contains_review(self, sample_reviews):
        """Test that zero-shot prompt contains the review."""
        for review in sample_reviews:
            prompt = build_zero_shot_prompt(review)
            assert review in prompt


    def test_build_zero_shot_prompt_format(self):
        """Test that zero-shot prompt has expected format."""
        review = "Test review"
        prompt = build_zero_shot_prompt(review)
        
        # Should be relatively short (no examples)
        assert len(prompt) < 1000
        # Should not contain multiple examples
        assert prompt.count("Review:") <= 1


    def test_build_zero_shot_empty_review_error(self):
        """Test that empty review raises error."""
        with pytest.raises(ValueError, match="non-empty string"):
            build_zero_shot_prompt("")


    def test_build_zero_shot_non_string_input_error(self):
        """Test that non-string input raises error."""
        with pytest.raises(ValueError, match="non-empty string"):
            build_zero_shot_prompt(None)
        
        with pytest.raises(ValueError, match="non-empty string"):
            build_zero_shot_prompt(123)


    def test_build_zero_shot_special_characters(self):
        """Test zero-shot prompt with special characters in review."""
        review = 'Great food! Special chars: !@#$%^&*() "quoted"'
        prompt = build_zero_shot_prompt(review)
        
        assert review in prompt


    def test_build_zero_shot_long_review(self):
        """Test zero-shot prompt with very long review."""
        review = "A" * 5000  # Very long review
        prompt = build_zero_shot_prompt(review)
        
        assert review in prompt


    def test_build_zero_shot_logging(self, caplog):
        """Test that zero-shot prompt building logs messages."""
        import logging
        caplog.set_level(logging.DEBUG)
        
        review = "Test review"
        prompt = build_zero_shot_prompt(review)
        
        assert "Building zero-shot prompt" in caplog.text
        assert "built successfully" in caplog.text


@pytest.mark.unit
class TestBuildFewShotPrompt:
    """Test suite for build_few_shot_prompt function."""
    
    def test_build_few_shot_basic(self):
        """Test basic few-shot prompt building."""
        review = "Great food!"
        prompt = build_few_shot_prompt(review)
        
        assert isinstance(prompt, str)
        assert review in prompt
        # Should contain examples
        assert "Review:" in prompt


    def test_build_few_shot_contains_examples(self):
        """Test that few-shot prompt contains all examples."""
        review = "Test review"
        prompt = build_few_shot_prompt(review)
        
        # Should contain multiple examples
        assert prompt.count("Review:") >= 5  # At least examples + target
        
        # Should contain all examples
        for example in FEW_SHOT_EXAMPLES:
            assert example["review"][:20] in prompt or example["review"] in prompt


    def test_build_few_shot_contains_labels(self):
        """Test that few-shot prompt contains all labels."""
        review = "Test review"
        prompt = build_few_shot_prompt(review)
        
        # Should contain JSON labels for each example
        assert '"stars":' in prompt or '"stars" :' in prompt
        assert '"explanation":' in prompt or '"explanation" :' in prompt


    def test_build_few_shot_all_star_ratings_present(self):
        """Test that all star ratings are present in examples."""
        review = "Test review"
        prompt = build_few_shot_prompt(review)
        
        # Should contain all ratings 1-5 in the examples
        for rating in [1, 2, 3, 4, 5]:
            assert str(rating) in prompt


    def test_build_few_shot_target_review_at_end(self):
        """Test that target review appears at the end."""
        review = "Final test review"
        prompt = build_few_shot_prompt(review)
        
        # Target review should appear near the end
        idx = prompt.rfind(review)
        assert idx > len(prompt) // 2  # Should be in second half


    def test_build_few_shot_empty_review_error(self):
        """Test that empty review raises error."""
        with pytest.raises(ValueError, match="non-empty string"):
            build_few_shot_prompt("")


    def test_build_few_shot_non_string_input_error(self):
        """Test that non-string input raises error."""
        with pytest.raises(ValueError, match="non-empty string"):
            build_few_shot_prompt(None)
        
        with pytest.raises(ValueError, match="non-empty string"):
            build_few_shot_prompt(123)


    def test_build_few_shot_longer_than_zero_shot(self):
        """Test that few-shot prompt is longer than zero-shot."""
        review = "Test review"
        zero_shot = build_zero_shot_prompt(review)
        few_shot = build_few_shot_prompt(review)
        
        assert len(few_shot) > len(zero_shot)


    def test_build_few_shot_special_characters(self):
        """Test few-shot prompt with special characters."""
        review = 'Special chars: !@#$%^&*() "quoted"'
        prompt = build_few_shot_prompt(review)
        
        assert review in prompt


    def test_build_few_shot_logging(self, caplog):
        """Test that few-shot prompt building logs messages."""
        import logging
        caplog.set_level(logging.DEBUG)
        
        review = "Test review"
        prompt = build_few_shot_prompt(review)
        
        assert "Building few-shot prompt" in caplog.text
        assert "built successfully" in caplog.text


@pytest.mark.unit
class TestPromptComparison:
    """Test comparison between zero-shot and few-shot prompts."""
    
    def test_zero_vs_few_shot_same_review(self):
        """Test that both methods can handle the same review."""
        review = "Excellent service!"
        
        zero_shot = build_zero_shot_prompt(review)
        few_shot = build_few_shot_prompt(review)
        
        assert review in zero_shot
        assert review in few_shot
        assert zero_shot != few_shot


    def test_zero_vs_few_shot_length_difference(self):
        """Test that few-shot is significantly longer."""
        review = "Test review"
        
        zero_shot = build_zero_shot_prompt(review)
        few_shot = build_few_shot_prompt(review)
        
        # Few-shot should be 3-10x longer due to examples
        assert len(few_shot) > 3 * len(zero_shot)


    def test_prompt_consistency_with_examples(self):
        """Test that prompt builders are consistent with examples."""
        for example in FEW_SHOT_EXAMPLES:
            review = example["review"]
            zero_shot = build_zero_shot_prompt(review)
            few_shot = build_few_shot_prompt(review)
            
            assert review in zero_shot
            assert review in few_shot
            assert len(few_shot) > len(zero_shot)


@pytest.mark.unit
class TestPromptEdgeCases:
    """Test edge cases for prompt functions."""
    
    def test_prompt_with_newlines(self):
        """Test prompts with newlines in review."""
        review = "Line 1\nLine 2\nLine 3"
        
        zero_shot = build_zero_shot_prompt(review)
        few_shot = build_few_shot_prompt(review)
        
        assert review in zero_shot
        assert review in few_shot


    def test_prompt_with_quotes(self):
        """Test prompts with various quote styles."""
        review = 'He said "Great!" and she replied \'Excellent!\''
        
        zero_shot = build_zero_shot_prompt(review)
        few_shot = build_few_shot_prompt(review)
        
        assert zero_shot is not None
        assert few_shot is not None


    def test_prompt_with_unicode(self):
        """Test prompts with Unicode characters."""
        review = "Amazing! 🌟✨ Best experience ever!"
        
        zero_shot = build_zero_shot_prompt(review)
        few_shot = build_few_shot_prompt(review)
        
        assert review in zero_shot
        assert review in few_shot


    def test_prompt_with_html_entities(self):
        """Test prompts with HTML-like entities."""
        review = "Great & Good < Excellent > OK"
        
        zero_shot = build_zero_shot_prompt(review)
        few_shot = build_few_shot_prompt(review)
        
        assert review in zero_shot
        assert review in few_shot


    def test_prompt_with_very_long_review(self):
        """Test prompts with extremely long reviews."""
        review = "Word " * 2000  # Very long review
        
        zero_shot = build_zero_shot_prompt(review)
        few_shot = build_few_shot_prompt(review)
        
        assert review in zero_shot
        assert review in few_shot
        assert len(few_shot) > len(zero_shot)
