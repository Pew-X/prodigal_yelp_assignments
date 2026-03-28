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
    if not review or not isinstance(review, str):
        logger.error(f"Invalid review input for direct prompt: {type(review)}")
        raise ValueError("Review must be a non-empty string")
    
    logger.debug(f"Building direct prompt (review length: {len(review)})")
    prompt = f'Rate this Yelp review on a 1-5 star scale:\n\n"{review}"'
    logger.debug("Direct prompt built successfully")
    return prompt


def build_cot_prompt(review: str) -> str:
    """
    Chain-of-Thought strategy: encourages step-by-step reasoning.

    Output format:
    {
      "reasoning": "step-by-step analysis...",
      "stars": 3,
      "explanation": "one-sentence summary"
    }
    """
    if not review or not isinstance(review, str):
        logger.error(f"Invalid review input for CoT prompt: {type(review)}")
        raise ValueError("Review must be a non-empty string")
    
    logger.debug(f"Building CoT prompt (review length: {len(review)})")
    prompt = (
        f'Analyze this Yelp review step by step before rating it.\n\n'
        f'Review: "{review}"\n\n'
        f'Think through: (1) specific positive signals, (2) specific negative signals, '
        f'(3) overall tone and intensity, then decide the star rating.\n\n'
        f'Output ONLY this JSON:\n'
        f'{{"reasoning": "<your step-by-step analysis>", "stars": <1-5>, '
        f'"explanation": "<one sentence summary>"}}'
    )
    logger.debug("CoT prompt built successfully")
    return prompt


# ── Task 3 Prompts (Multi objective assistant) ─────────────────────────────────────────
#
# Design decisions:
# - 3-field JSON only (stars, key_point, business_response)
#   Adding a 4th field (sentiment) raises parse failure rate on 8B models as seen previously
# - Explicit word limits + ban on generic phrases enforced stritcly 
#   directly raising faithfulness and actionability judge scores
# - business_response prompt chain: acknowledge → address specifically → invite return
#   This structure  I believe is what separates a usable response from a normal apology

TASK3_SYSTEM_PROMPT = """You are an expert business analyst for Yelp reviews.
For each review, provide a structured analysis to help the business owner respond.

STRICT OUTPUT FORMAT — respond with ONLY this JSON, nothing else:
{
  "stars": <integer 1-5>,
  "key_point": "<the single most specific complaint OR compliment from this exact review — include concrete details like item names, wait times, staff names, specific dishes — one sentence, max 25 words>",
  "business_response": "<professional, empathetic response the owner could post publicly — must directly reference the specific issue in key_point — 2-3 sentences, max 60 words>"
}

Rules:
- No markdown, no code fences, no extra text before or after the JSON
- key_point must name something SPECIFIC: never generic phrases like 'poor service' or 'good food'
- business_response must use a specific detail from key_point, never a generic apology template
- business_response tone: acknowledge → address specifically → invite return"""


def build_assistant_prompt(review: str) -> str:
    if not review or not isinstance(review, str):
        logger.error(f"Invalid review input for assistant prompt: {type(review)}")
        raise ValueError("Review must be a non-empty string")
    
    logger.debug(f"Building Task 3 assistant prompt (review length: {len(review)})")
    prompt = (
        f"Analyze this Yelp review and generate a structured business response:\n\n"
        f'"{review}"'
    )
    logger.debug("Task 3 assistant prompt built successfully")
    return prompt


# ── Task 3's  LLM as Judge ──────────────────────────────────────────────────────
#
# Design decisions:
# - Three independent dimensions, not a single holistic score
#   A holistic score hides WHICH dimension fails will be useless for analysis
# - Rubric anchors defined at 1, 3, 5 explicitly   hoping to reduce judge score variance
# - overall_verdict is NOT asked from the judge, computed deterministically in Python code in a rules based manner.
#   (faithfulness>=4 AND actionability>=4 AND tone>=4 -> success)
#   Removing it from the JSON cuts one more failure point for a small model (hypothesis)
# - Judge sees: review + key_point + business_response
#   NOT the true star, trying to avoid anchoring bias in judge scores

JUDGE_SYSTEM_TASK3 = """You are a strict expert evaluator of AI-generated Yelp review analysis.

Score the key_point extraction and business_response on THREE independent dimensions.

STRICT OUTPUT FORMAT — respond with ONLY this JSON:
{
  "faithfulness": <1-5>,
  "faithfulness_reason": "<one sentence>",
  "actionability": <1-5>,
  "actionability_reason": "<one sentence>",
  "tone": <1-5>,
  "tone_reason": "<one sentence>"
}

SCORING RUBRICS:

faithfulness (Does key_point accurately reflect the review?):
  1 = fabricated or completely misrepresents the review
  3 = correct but generic, could apply to many similar reviews
  5 = precise, captures the exact core issue with specific details from the review

actionability (Does business_response specifically address the key_point?):
  1 = generic template with no connection to the actual review content
  3 = somewhat specific but could apply to similar reviews
  5 = directly addresses the specific issue, proposes concrete next step

tone (Is business_response professional, empathetic, non-defensive?):
  1 = rude, dismissive, or defensive in any way
  3 = professional but flat, generic warmth only
  5 = empathetic, specific, warm, naturally invites return visit

No markdown, no extra text, strictly only the JSON."""


def build_judge_prompt_task3(
    review: str,
    key_point: str,
    business_response: str,
) -> str:
    # Validate inputs
    if not review or not isinstance(review, str):
        logger.error(f"Invalid review input: {type(review)}")
        raise ValueError("Review must be a non-empty string")
    if not key_point or not isinstance(key_point, str):
        logger.error(f"Invalid key_point input: {type(key_point)}")
        raise ValueError("Key point must be a non-empty string")
    if not business_response or not isinstance(business_response, str):
        logger.error(f"Invalid business_response input: {type(business_response)}")
        raise ValueError("Business response must be a non-empty string")
    
    logger.debug(f"Building Task 3 judge prompt (review: {len(review)}c, key_point: {len(key_point)}c, response: {len(business_response)}c)")
    prompt = (
        f'Review: "{review}"\n\n'
        f'Extracted key_point: "{key_point}"\n\n'
        f'Business response: "{business_response}"'
    )
    logger.debug("Task 3 judge prompt built successfully")
    return prompt