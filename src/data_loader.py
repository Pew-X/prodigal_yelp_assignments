"""
Stratified sampler for Yelp Full dataset.
"""

import random
from collections import defaultdict
from datasets import load_dataset
import pandas as pd


def load_yelp_sample(
    n_per_class: int = 40,
    split: str = "test",
    seed: int = 33,
) -> pd.DataFrame:
    """
    Returns a stratified DataFrame with columns: text, true_stars (1-5).
    Total rows = n_per_class * 5 = 200 by default.
    """
    print(f"Loading Yelp ({split} split)")
    dataset = load_dataset("yelp_review_full", split=split)

    by_label = defaultdict(list)
    for item in dataset:
        by_label[item["label"]].append(item)

    random.seed(seed)
    samples = []
    for label in range(5):
        pool = by_label[label]
        chosen = random.sample(pool, min(n_per_class, len(pool))) # randomly choosiing per class
        for item in chosen:
            samples.append({
                "text": item["text"],
                "true_stars": label + 1,  # dataset is 0-indexed, convert to 1-5
            })

    random.shuffle(samples)
    df = pd.DataFrame(samples).reset_index(drop=True)
    print(f"Loaded {len(df)} samples | {n_per_class} per class | classes: {sorted(df['true_stars'].unique())}")
    return df

# df=load_yelp_sample()
# print(df.head())