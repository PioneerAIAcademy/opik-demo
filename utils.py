"""
Reusable utility functions for prompt optimization.

These functions are generic and can be used in other projects for:
- Extracting scores from LLM responses
- Loading and splitting datasets with stratification
- Loading text templates
- Creating timestamped output directories

Students: Feel free to copy these functions into your own projects!
"""

import re
import os
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def extract_score_from_text(text: str, min_score: float = 1.0, max_score: float = 5.0) -> Optional[float]:
    """
    Extract a numerical score from text using common formatting patterns.

    This is a reusable pattern for extracting scores from LLM responses.
    Works with various formats like "**Score:** 4", "Score: 3.5", etc.

    Args:
        text: The text to search for a score
        min_score: Minimum valid score (default: 1.0)
        max_score: Maximum valid score (default: 5.0)

    Returns:
        The extracted score, or None if not found or invalid

    Examples:
        >>> extract_score_from_text("**Score:** 4")
        4.0
        >>> extract_score_from_text("The answer gets a score of 3.5")
        3.5
        >>> extract_score_from_text("No score here")
        None

    Reuse this in your projects:
        score = extract_score_from_text(llm_response)
        if score is not None:
            print(f"Extracted score: {score}")
    """
    # Common patterns for score formatting
    patterns = [
        r'\*\*Score:\*\*\s*(\d+\.?\d*)',      # **Score:** 4 or 4.5
        r'\*\*Score\*\*:\s*(\d+\.?\d*)',      # **Score**: 4 or 4.5
        r'Score:\s*(\d+\.?\d*)',               # Score: 4 or 4.5
        r'score:\s*(\d+\.?\d*)',               # score: 4 or 4.5 (case-insensitive)
        r'\bscore\s+(?:of\s+)?(\d+\.?\d*)',   # "score of 3" or "score 3"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            # Validate score is in valid range
            if min_score <= score <= max_score:
                return score

    return None


def load_csv_with_stratified_split(
    csv_path: str,
    sample_size: Optional[int] = None,
    stratify_column: str = "score",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a CSV file and split it into train/test sets with stratification.

    Uses dynamic 2/3 train, 1/3 test split based on sample size.
    Stratification ensures that the distribution of values in the stratify_column
    is maintained in both train and test sets. This is important for evaluation!

    Works with datasets of any size - handles small and large datasets robustly.

    Args:
        csv_path: Path to the CSV file
        sample_size: Total number of samples to use (None = use all data)
        stratify_column: Column to stratify on (maintains distribution)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)

    Raises:
        ValueError: If dataset doesn't have enough rows

    Example:
        >>> train, test = load_csv_with_stratified_split(
        ...     "evaluations.csv",
        ...     sample_size=15,  # 10 train, 5 test
        ...     stratify_column="score"
        ... )
        >>> print(f"Train: {len(train)}, Test: {len(test)}")

    Reuse this in your projects:
        # For any CSV with categorical or continuous targets
        train_data, test_data = load_csv_with_stratified_split(
            "my_data.csv",
            sample_size=None,  # Use all data
            stratify_column="label"  # or "score", "rating", etc.
        )
    """
    print(f"📊 Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"   Total rows available: {len(df)}")
    print(f"   Columns: {list(df.columns)}")

    # Round scores to integers for stratification grouping
    score_groups = df[stratify_column].round().astype(int)

    # Show distribution of stratify column
    print(f"\n   Distribution of '{stratify_column}' (rounded to integers):")
    print(score_groups.value_counts().sort_index())

    # Determine sample size
    if sample_size is None:
        sample_size = len(df)
        print(f"\n   Using full dataset: {sample_size} rows")
    else:
        print(f"\n   Using sample size: {sample_size} rows")

    if len(df) < sample_size:
        raise ValueError(
            f"Dataset has only {len(df)} rows, but requested {sample_size} samples"
        )

    # Calculate train/test sizes (2/3 train, 1/3 test)
    train_size = int(sample_size * 2 / 3)
    test_size = sample_size - train_size

    print(f"\n   Split: {train_size} train ({train_size/sample_size*100:.0f}%), "
          f"{test_size} test ({test_size/sample_size*100:.0f}%)")

    # First, sample the data if needed (with stratification)
    if sample_size < len(df):
        # Use train_test_split to get stratified sample, then take only one part
        df_sample, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=score_groups,
            random_state=random_state
        )
    else:
        df_sample = df

    # Now split the sampled data into train/test (2/3:1/3)
    # Recalculate score groups for the sample
    sample_score_groups = df_sample[stratify_column].round().astype(int)

    # Try stratified split, fall back to random split if too few samples per class
    try:
        train_df, test_df = train_test_split(
            df_sample,
            train_size=train_size,
            test_size=test_size,
            stratify=sample_score_groups,
            random_state=random_state
        )
        print(f"   ✓ Using stratified split")
    except ValueError as e:
        # Stratification requires at least 2 members per class
        # Fall back to random split for small samples
        print(f"   ⚠ Stratification not possible (sample too small), using random split")
        train_df, test_df = train_test_split(
            df_sample,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state
        )

    print(f"\n   ✅ Split complete:")
    print(f"      Training: {len(train_df)} samples")
    print(f"      Test: {len(test_df)} samples")

    # Show score distribution in each split
    print(f"\n   Training score distribution:")
    print(train_df[stratify_column].round().value_counts().sort_index())
    print(f"\n   Test score distribution:")
    print(test_df[stratify_column].round().value_counts().sort_index())

    return train_df, test_df


def load_text_template(file_path: str) -> str:
    """
    Load a text template from a file.

    Simple utility for loading prompt templates, configuration files, etc.

    Args:
        file_path: Path to the text file

    Returns:
        File contents as a string

    Example:
        >>> template = load_text_template("my_prompt.txt")
        >>> formatted = template.format(question="What is AI?")

    Reuse this in your projects:
        prompt_template = load_text_template("prompts/system_message.txt")
        config_text = load_text_template("config/settings.txt")
    """
    print(f"📝 Loading template from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"   ✅ Loaded {len(content)} characters")
    return content


def create_timestamped_run_dir(base_dir: str = "runs") -> str:
    """
    Create a timestamped directory for storing optimization run outputs.

    Prevents file overwrites by creating a new directory for each run.
    Format: runs/YYYY-MM-DD_HH-MM-SS/

    Args:
        base_dir: Base directory name (default: "runs")

    Returns:
        Path to the created directory

    Example:
        >>> run_dir = create_timestamped_run_dir()
        >>> print(run_dir)
        runs/2025-12-12_14-30-22/

    Reuse this in your projects:
        output_dir = create_timestamped_run_dir("experiments")
        with open(f"{output_dir}/results.txt", "w") as f:
            f.write("Results...")
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"📁 Created output directory: {run_dir}/")
    return run_dir
