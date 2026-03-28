import json
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

# System Prompt
SYSTEM_PROMPT = """You are an expert Yelp review analyst. Classify reviews into star ratings from 1 to 5.

STRICT OUTPUT FORMAT — respond with ONLY this JSON, nothing else:
{"stars": <integer 1-5>, "explanation": "<one sentence explaining the rating>"}

Rules:
- No markdown, no code fences, no extra text before or after the JSON
- "stars" must be an strict integer between 1 and 5
- "explanation" must be a single sentence under 30 words"""


# Few-Shot Examples : 
# Manually curated for clarity
#   Selection rationale:
# - Each example has unambiguous sentiment for its star level
# - Language cues taken care of (e.g., "never coming back" = 1 star)
# - Diverse domain mixtures (food, service, ambiance) to prevent domain overfitting

FEW_SHOT_EXAMPLES = [
    {
        "review": "Absolutely terrible. Food was cold, waiter was rude, waited 45 minutes despite having a reservation. Never coming back.",
        "label": {"stars": 1, "explanation": "Multiple critical failures in food quality, service, and wait time with no redeeming factors."},
    },
    {
        "review": "Disappointing. Dry burger and overpriced. Service was slow but at least the staff seemed friendly enough.",
        "label": {"stars": 2, "explanation": "Below-average food quality and value with only minor positives in staff attitude."},
    },
    {
        "review": "Decent  enough spot for a quick lunch. Pasta was fine, service was average. Nothing memorable but nothing bad either.",
        "label": {"stars": 3, "explanation": "Neutral experience with no standout positives or negatives indicating average quality."},
    },
    {
        "review": "Really enjoyed dinner here. Salmon was cooked perfectly, cocktails were creative, and service was attentive without being overbearing. Would definitely return.",
        "label": {"stars": 4, "explanation": "Strong positive experience with specific praise for food and service quality, with intent to return."},
    },
    {
        "review": "Phenomenal. Best ramen outside  of Japan. Rich complex broth, perfect noodles, and staff who made us feel like regulars on our first visit.",
        "label": {"stars": 5, "explanation": "Exceptional experience with enthusiastic praise for both food craft and hospitality."},
    },
]


# ── TASK 1 Prompts ────────────────────────────────────────────────────────────

def build_zero_shot_prompt(review: str) -> str:

    if not review or not isinstance(review, str):
        logger.error(f"Invalid review input: {type(review)}")
        raise ValueError("Review must be a non-empty string")
    
    logger.debug(f"Building zero-shot prompt (review length: {len(review)})")
    prompt = f'Classify this Yelp review:\n\n"{review}"'
    logger.debug("Zero-shot prompt built successfully")
    return prompt


def build_few_shot_prompt(review: str) -> str:

    if not review or not isinstance(review, str):
        logger.error(f"Invalid review input: {type(review)}")
        raise ValueError("Review must be a non-empty string")
    
    logger.debug(f"Building few-shot prompt with {len(FEW_SHOT_EXAMPLES)} examples (review length: {len(review)})")
    
    examples_block = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES):
        examples_block += (
            f'Review: "{ex["review"]}"\n'
            f'Output: {json.dumps(ex["label"])}\n\n'
        )
        logger.debug(f"Added example {i+1}/{len(FEW_SHOT_EXAMPLES)} (class {ex['label']['stars']})")

    prompt = (
        f"Here are examples of correct classifications:\n\n"
        f"{examples_block.strip()}\n\n"
        f"Now classify this review:\n\n"
        f'Review: "{review}"\n'
        f"Output:"
    )
    logger.debug("Few-shot prompt built successfully")
    return prompt


# ── Task 2 Prompts ─────────────────────────────────────────────────────

def build_direct_prompt(review: str) -> str:
    """
    Direct strategy: minimal instruction, no reasoning encouraged.
    """
    return f'Rate this Yelp review on a 1-5 star scale:\n\n"{review}"'


def build_cot_prompt(review: str) -> str:
    """
    Chain-of-Thought strategy

    Output format:
    {
      "reasoning": "step-by-step analysis...",
      "stars": 3,
      "explanation": "one-sentence summary"
    }
    """
    return (
        f'Analyze this Yelp review step by step before rating it.\n\n'
        f'Review: "{review}"\n\n'
        f'Think through: (1) specific positive signals, (2) specific negative signals, '
        f'(3) overall tone and intensity, then decide the star rating.\n\n'
        f'Output ONLY this JSON:\n'
        f'{{"reasoning": "<your step-by-step analysis>", "stars": <1-5>, '
        f'"explanation": "<one sentence summary>"}}'
    )