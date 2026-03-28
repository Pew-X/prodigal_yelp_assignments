"""
Task 3 Tests: Multi-objective Assistant Prompts

Covers:
- Assistant prompt generation with review analysis
- Judge prompt generation for evaluating extraction quality
- Task 3 system prompts validation
- End-to-end Task 3 workflows
"""

import pytest
import pandas as pd
from src.prompts import (
    build_assistant_prompt,
    build_judge_prompt_task3,
    TASK3_SYSTEM_PROMPT,
    JUDGE_SYSTEM_TASK3,
)


# ──── Test Fixtures ────────────────────────────────────────────

@pytest.fixture
def sample_review():
    """Sample Yelp review for testing."""
    return "Food was terrible, waiter was rude, waited forever. Never coming back."


@pytest.fixture
def sample_key_point():
    """Sample extracted key point."""
    return "Terrible food quality and rude staff with excessive wait times."


@pytest.fixture
def sample_business_response():
    """Sample business response to a review."""
    return "We sincerely apologize for this disappointing experience. Your feedback about service speed and staff behavior is important. We'd like the opportunity to make this right with a complimentary meal."


@pytest.fixture
def sample_task3_results():
    """Sample Task 3 results DataFrame."""
    return pd.DataFrame({
        "review": [
            "Amazing food, perfect service, will definitely return!",
            "Terrible experience, never coming back.",
            "Average place, nothing special.",
        ],
        "stars": [5, 1, 3],
        "key_point": [
            "Exceptional service and high-quality cuisine.",
            "Poor food quality and inattentive staff.",
            "Mediocre food and average service.",
        ],
        "business_response": [
            "Thank you for the wonderful feedback! We take pride in our team and kitchen excellence.",
            "We deeply apologize and would welcome the opportunity to restore your confidence.",
            "We appreciate the feedback and welcome you back for another visit.",
        ],
    })


# ──── Task 3 System Prompt Tests ────────────────────────────

class TestTask3SystemPrompts:
    """Test Task 3 system prompts are properly defined."""

    def test_task3_system_prompt_exists(self):
        """Test that Task 3 system prompt is defined."""
        assert TASK3_SYSTEM_PROMPT is not None
        assert isinstance(TASK3_SYSTEM_PROMPT, str)
        assert len(TASK3_SYSTEM_PROMPT) > 0

    def test_task3_system_prompt_contains_instructions(self):
        """Test system prompt has all required instructions."""
        assert "stars" in TASK3_SYSTEM_PROMPT
        assert "key_point" in TASK3_SYSTEM_PROMPT
        assert "business_response" in TASK3_SYSTEM_PROMPT
        assert "JSON" in TASK3_SYSTEM_PROMPT

    def test_task3_system_prompt_has_specific_requirements(self):
        """Test system prompt includes specific requirements."""
        assert "concrete details" in TASK3_SYSTEM_PROMPT
        assert "item names" in TASK3_SYSTEM_PROMPT or "specific" in TASK3_SYSTEM_PROMPT
        assert "empathetic" in TASK3_SYSTEM_PROMPT

    def test_judge_system_prompt_task3_exists(self):
        """Test that judge system prompt for Task 3 is defined."""
        assert JUDGE_SYSTEM_TASK3 is not None
        assert isinstance(JUDGE_SYSTEM_TASK3, str)
        assert len(JUDGE_SYSTEM_TASK3) > 0

    def test_judge_system_prompt_contains_dimensions(self):
        """Test judge prompt defines all three evaluation dimensions."""
        assert "faithfulness" in JUDGE_SYSTEM_TASK3
        assert "actionability" in JUDGE_SYSTEM_TASK3
        assert "tone" in JUDGE_SYSTEM_TASK3

    def test_judge_system_prompt_has_rubrics(self):
        """Test judge prompt includes detailed scoring rubrics."""
        assert "1 =" in JUDGE_SYSTEM_TASK3
        assert "3 =" in JUDGE_SYSTEM_TASK3
        assert "5 =" in JUDGE_SYSTEM_TASK3


# ──── Build Assistant Prompt Tests ────────────────────────────

class TestBuildAssistantPrompt:
    """Test Task 3 assistant prompt generation."""

    def test_build_assistant_prompt_basic(self, sample_review):
        """Test basic assistant prompt generation."""
        prompt = build_assistant_prompt(sample_review)
        assert "Analyze this Yelp review" in prompt
        assert "structured business response" in prompt
        assert sample_review in prompt

    def test_build_assistant_prompt_long_review(self):
        """Test assistant prompt with long review."""
        review = "A" * 500
        prompt = build_assistant_prompt(review)
        assert review in prompt
        assert len(prompt) > len(review)

    def test_build_assistant_prompt_special_characters(self):
        """Test assistant prompt with special characters."""
        review = 'Great food! Prices "reasonable" & service (excellent).'
        prompt = build_assistant_prompt(review)
        assert review in prompt

    def test_build_assistant_prompt_unicode(self):
        """Test assistant prompt with unicode characters."""
        review = "Excellent! 很好 😊 Amazing!"
        prompt = build_assistant_prompt(review)
        assert review in prompt

    def test_build_assistant_prompt_empty_string_error(self):
        """Test that empty review raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            build_assistant_prompt("")

    def test_build_assistant_prompt_none_error(self):
        """Test that None input raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            build_assistant_prompt(None)

    def test_build_assistant_prompt_non_string_error(self):
        """Test that non-string input raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            build_assistant_prompt(12345)

    def test_build_assistant_prompt_is_string(self, sample_review):
        """Test that output is a string."""
        prompt = build_assistant_prompt(sample_review)
        assert isinstance(prompt, str)

    def test_build_assistant_prompt_multiline_review(self):
        """Test assistant prompt with multiline review."""
        review = """This place is amazing!
Excellent service and great food.
Highly recommend."""
        prompt = build_assistant_prompt(review)
        assert review in prompt
        assert "\n" in prompt


# ──── Build Judge Prompt Task 3 Tests ────────────────────────

class TestBuildJudgePromptTask3:
    """Test Task 3 judge prompt generation."""

    def test_build_judge_prompt_basic(
        self, sample_review, sample_key_point, sample_business_response
    ):
        """Test basic judge prompt generation."""
        prompt = build_judge_prompt_task3(
            sample_review, sample_key_point, sample_business_response
        )
        assert 'Review: "' in prompt
        assert 'Extracted key_point: "' in prompt
        assert 'Business response: "' in prompt
        assert sample_review in prompt
        assert sample_key_point in prompt
        assert sample_business_response in prompt

    def test_build_judge_prompt_structure(
        self, sample_review, sample_key_point, sample_business_response
    ):
        """Test judge prompt has correct structure."""
        prompt = build_judge_prompt_task3(
            sample_review, sample_key_point, sample_business_response
        )
        # Check that it's structured with newlines
        assert prompt.count("\n\n") >= 2  # At least 2 double newlines for separation

    def test_build_judge_prompt_long_inputs(self):
        """Test judge prompt with long inputs."""
        review = "A" * 300
        key_point = "B" * 100
        response = "C" * 200
        prompt = build_judge_prompt_task3(review, key_point, response)
        assert review in prompt
        assert key_point in prompt
        assert response in prompt

    def test_build_judge_prompt_special_characters(self):
        """Test judge prompt with special characters."""
        review = 'Review with "quotes" and \'apostrophes\'.'
        key_point = "Key point: excellent (quality) & service"
        response = "Response: Thank you! We're grateful."
        prompt = build_judge_prompt_task3(review, key_point, response)
        assert review in prompt
        assert key_point in prompt
        assert response in prompt

    def test_build_judge_prompt_unicode(self):
        """Test judge prompt with unicode characters."""
        review = "Great! 很好 😊"
        key_point = "Key point: excellent service"
        response = "Response: 感谢你的好评! 🙏"
        prompt = build_judge_prompt_task3(review, key_point, response)
        assert review in prompt
        assert key_point in prompt
        assert response in prompt

    def test_build_judge_prompt_empty_review_error(self, sample_key_point, sample_business_response):
        """Test that empty review raises ValueError."""
        with pytest.raises(ValueError, match="Review must be a non-empty string"):
            build_judge_prompt_task3("", sample_key_point, sample_business_response)

    def test_build_judge_prompt_empty_key_point_error(self, sample_review, sample_business_response):
        """Test that empty key_point raises ValueError."""
        with pytest.raises(ValueError, match="Key point must be a non-empty string"):
            build_judge_prompt_task3(sample_review, "", sample_business_response)

    def test_build_judge_prompt_empty_response_error(self, sample_review, sample_key_point):
        """Test that empty business_response raises ValueError."""
        with pytest.raises(ValueError, match="Business response must be a non-empty string"):
            build_judge_prompt_task3(sample_review, sample_key_point, "")

    def test_build_judge_prompt_none_review_error(self, sample_key_point, sample_business_response):
        """Test that None review raises ValueError."""
        with pytest.raises(ValueError):
            build_judge_prompt_task3(None, sample_key_point, sample_business_response)

    def test_build_judge_prompt_none_key_point_error(self, sample_review, sample_business_response):
        """Test that None key_point raises ValueError."""
        with pytest.raises(ValueError):
            build_judge_prompt_task3(sample_review, None, sample_business_response)

    def test_build_judge_prompt_none_response_error(self, sample_review, sample_key_point):
        """Test that None business_response raises ValueError."""
        with pytest.raises(ValueError):
            build_judge_prompt_task3(sample_review, sample_key_point, None)

    def test_build_judge_prompt_non_string_review_error(self, sample_key_point, sample_business_response):
        """Test that non-string review raises ValueError."""
        with pytest.raises(ValueError):
            build_judge_prompt_task3(12345, sample_key_point, sample_business_response)

    def test_build_judge_prompt_non_string_key_point_error(self, sample_review, sample_business_response):
        """Test that non-string key_point raises ValueError."""
        with pytest.raises(ValueError):
            build_judge_prompt_task3(sample_review, 12345, sample_business_response)

    def test_build_judge_prompt_non_string_response_error(self, sample_review, sample_key_point):
        """Test that non-string business_response raises ValueError."""
        with pytest.raises(ValueError):
            build_judge_prompt_task3(sample_review, sample_key_point, 12345)

    def test_build_judge_prompt_is_string(
        self, sample_review, sample_key_point, sample_business_response
    ):
        """Test that output is a string."""
        prompt = build_judge_prompt_task3(
            sample_review, sample_key_point, sample_business_response
        )
        assert isinstance(prompt, str)


# ──── Integration Tests ────────────────────────────────────────

class TestTask3Integration:
    """Integration tests for Task 3 workflows."""

    def test_assistant_and_judge_workflow(
        self, sample_review, sample_key_point, sample_business_response
    ):
        """Test complete Task 3 workflow: assistant prompt + judge prompt."""
        # Step 1: Build assistant prompt (would be sent to LLM)
        assistant_prompt = build_assistant_prompt(sample_review)
        assert sample_review in assistant_prompt

        # Step 2: Build judge prompt (with extracted key_point and response)
        judge_prompt = build_judge_prompt_task3(
            sample_review, sample_key_point, sample_business_response
        )
        assert sample_review in judge_prompt
        assert sample_key_point in judge_prompt
        assert sample_business_response in judge_prompt

    def test_task3_prompts_contain_required_fields(self, sample_task3_results):
        """Test that Task 3 prompts are consistent with expected fields."""
        for _, row in sample_task3_results.iterrows():
            # Build assistant prompt
            assistant_prompt = build_assistant_prompt(row["review"])
            assert row["review"] in assistant_prompt

            # Build judge prompt
            judge_prompt = build_judge_prompt_task3(
                row["review"], row["key_point"], row["business_response"]
            )
            assert row["review"] in judge_prompt
            assert row["key_point"] in judge_prompt
            assert row["business_response"] in judge_prompt

    def test_assistant_prompt_length_reasonable(self, sample_review):
        """Test that assistant prompt length is reasonable."""
        prompt = build_assistant_prompt(sample_review)
        # Prompt should be longer than review (adds instruction)
        assert len(prompt) > len(sample_review)
        # But not excessively long (no redundancy)
        assert len(prompt) < len(sample_review) * 3

    def test_judge_prompt_length_scales_with_inputs(self):
        """Test that judge prompt length scales with input lengths."""
        review_short = "Good"
        key_point_short = "Good service"
        response_short = "Thank you!"

        prompt_short = build_judge_prompt_task3(
            review_short, key_point_short, response_short
        )

        review_long = "A" * 200
        key_point_long = "B" * 100
        response_long = "C" * 150

        prompt_long = build_judge_prompt_task3(
            review_long, key_point_long, response_long
        )

        # Longer inputs should produce longer prompts
        assert len(prompt_long) > len(prompt_short)

    def test_batch_task3_prompts(self, sample_task3_results):
        """Test building Task 3 prompts for a batch of reviews."""
        results = []
        for _, row in sample_task3_results.iterrows():
            assistant_prompt = build_assistant_prompt(row["review"])
            judge_prompt = build_judge_prompt_task3(
                row["review"], row["key_point"], row["business_response"]
            )
            results.append({
                "review": row["review"],
                "assistant_prompt": assistant_prompt,
                "judge_prompt": judge_prompt,
            })

        assert len(results) == len(sample_task3_results)
        # All prompts should be valid strings
        for result in results:
            assert isinstance(result["assistant_prompt"], str)
            assert isinstance(result["judge_prompt"], str)
            assert len(result["assistant_prompt"]) > 0
            assert len(result["judge_prompt"]) > 0

    def test_judge_prompt_preserves_field_order(
        self, sample_review, sample_key_point, sample_business_response
    ):
        """Test that judge prompt maintains correct field order."""
        prompt = build_judge_prompt_task3(
            sample_review, sample_key_point, sample_business_response
        )
        # Review should appear before key_point
        review_pos = prompt.find(sample_review)
        key_point_pos = prompt.find(sample_key_point)
        response_pos = prompt.find(sample_business_response)

        assert review_pos < key_point_pos
        assert key_point_pos < response_pos

    def test_multiline_inputs_preserved(self):
        """Test that multiline inputs are properly preserved."""
        review = """This is a multiline review.
It has multiple lines.
Each line is important."""
        key_point = """Multiple issues: slow service and cold food.
Both mentioned specifically."""
        response = """We apologize sincerely.
We will improve service speed.
We invite you to return."""

        prompt = build_judge_prompt_task3(review, key_point, response)

        # All lines should be preserved
        assert review in prompt
        assert key_point in prompt
        assert response in prompt
