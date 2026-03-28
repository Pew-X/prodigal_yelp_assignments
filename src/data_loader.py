"""
Stratified sampler for Yelp Full dataset.
"""

import random
import logging
from collections import defaultdict
from datasets import load_dataset
import pandas as pd

# logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_yelp_sample(
    n_per_class: int = 40,
    split: str = "test",
    seed: int = 33,
) -> pd.DataFrame:
    """
    Returns a stratified DataFrame with columns: text, true_stars (1-5).
    Total rows = n_per_class * 5 = 200 by default.
    
    """
    logger.info(f"Loading Yelp ({split} split) with n_per_class={n_per_class}, seed={seed}")
    
    if n_per_class < 1:
        logger.error(f"Invalid n_per_class: {n_per_class}. Must be >= 1")
        raise ValueError("n_per_class must be >= 1")
    
    if split not in ["train", "test"]:
        logger.error(f"Invalid split: {split}. Must be 'train' or 'test'")
        raise ValueError(f"split must be 'train' or 'test', got {split}")
    
    try:
        dataset = load_dataset("yelp_review_full", split=split)
        logger.debug(f"Dataset loaded successfully. Total samples: {len(dataset)}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

    by_label = defaultdict(list)
    for item in dataset:
        by_label[item["label"]].append(item)
    
    logger.debug(f"Grouped samples by label: {dict(sorted({k: len(v) for k, v in by_label.items()}.items()))}")

    random.seed(seed)
    samples = []
    for label in range(5):
        pool = by_label[label]
        num_to_sample = min(n_per_class, len(pool))
        logger.debug(f"Label {label}: sampling {num_to_sample} from {len(pool)} available")
        chosen = random.sample(pool, num_to_sample)
        for item in chosen:
            samples.append({
                "text": item["text"],
                "true_stars": label + 1,  # dataset is 0-indexed, convert to 1-5
            })

    random.shuffle(samples)
    df = pd.DataFrame(samples).reset_index(drop=True)
    logger.info(f"Stratified sample created: {len(df)} total | {n_per_class} per class | classes: {sorted(df['true_stars'].unique().tolist())}")
    
    return df

# df=load_yelp_sample()
# print(df.head())