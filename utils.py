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
    random_state: int = 42,
    split_type: str = "train_test",
    train_dev_test_ratio: str = "40/40/20"
) -> Tuple[pd.DataFrame, ...]:
    """
    Load a CSV file and split it into train/dev/test sets with stratification.

    Supports two split types:
    - "train_test": 2/3 train, 1/3 test (legacy mode)
    - "train_dev_test": Custom ratio (default 40/40/20)

    Stratification ensures that the distribution of values in the stratify_column
    is maintained in all sets. This is critical for reliable evaluation!

    For prompt optimization with ~150 samples, 40/40/20 is recommended:
    - Larger dev set (40%) provides reliable optimization signal
    - Smaller train set (40%) is sufficient (optimizers only need 30-40 samples)
    - Test set (20%) adequate for final evaluation

    Works with datasets of any size - handles small and large datasets robustly.

    Args:
        csv_path: Path to the CSV file
        sample_size: Total number of samples to use (None = use all data)
        stratify_column: Column to stratify on (maintains distribution)
        random_state: Random seed for reproducibility
        split_type: "train_test" or "train_dev_test"
        train_dev_test_ratio: Ratio as "train/dev/test" (e.g., "40/40/20", "30/50/20")
                             Only used when split_type="train_dev_test"

    Returns:
        If split_type="train_test": Tuple of (train_df, test_df)
        If split_type="train_dev_test": Tuple of (train_df, dev_df, test_df)

    Raises:
        ValueError: If dataset doesn't have enough rows

    Example:
        >>> # Two-way split
        >>> train, test = load_csv_with_stratified_split(
        ...     "evaluations.csv",
        ...     sample_size=15,
        ...     split_type="train_test"
        ... )
        >>> print(f"Train: {len(train)}, Test: {len(test)}")

        >>> # Three-way split (recommended for optimization)
        >>> train, dev, test = load_csv_with_stratified_split(
        ...     "evaluations.csv",
        ...     sample_size=None,
        ...     split_type="train_dev_test"
        ... )
        >>> print(f"Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")

    Reuse this in your projects:
        # For optimization tasks with validation
        train_data, dev_data, test_data = load_csv_with_stratified_split(
            "my_data.csv",
            sample_size=None,
            stratify_column="label",
            split_type="train_dev_test"
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

    # Recalculate score groups for the sample
    sample_score_groups = df_sample[stratify_column].round().astype(int)

    # Perform split based on type
    if split_type == "train_dev_test":
        # Parse the ratio string (e.g., "40/40/20")
        try:
            train_pct, dev_pct, test_pct = map(int, train_dev_test_ratio.split('/'))
            if train_pct + dev_pct + test_pct != 100:
                raise ValueError(f"Split ratios must sum to 100, got {train_pct + dev_pct + test_pct}")
        except Exception as e:
            raise ValueError(f"Invalid train_dev_test_ratio '{train_dev_test_ratio}': {e}")

        # Calculate sizes based on ratio
        train_size = int(sample_size * train_pct / 100)
        dev_size = int(sample_size * dev_pct / 100)
        test_size = sample_size - train_size - dev_size  # Remainder to handle rounding

        print(f"\n   Split ratio: {train_dev_test_ratio}")
        print(f"   Split: {train_size} train ({train_size/sample_size*100:.0f}%), "
              f"{dev_size} dev ({dev_size/sample_size*100:.0f}%), "
              f"{test_size} test ({test_size/sample_size*100:.0f}%)")

        # Try stratified split, fall back to random split if too few samples per class
        try:
            # First split: train vs (dev+test)
            train_df, dev_test_df = train_test_split(
                df_sample,
                train_size=train_size,
                stratify=sample_score_groups,
                random_state=random_state
            )

            # Second split: dev vs test
            dev_test_score_groups = dev_test_df[stratify_column].round().astype(int)
            dev_df, test_df = train_test_split(
                dev_test_df,
                train_size=dev_size,
                test_size=test_size,
                stratify=dev_test_score_groups,
                random_state=random_state
            )
            print("   ✓ Using stratified split")
        except ValueError:
            # Stratification requires at least 2 members per class
            # Fall back to random split for small samples
            print("   ⚠ Stratification not possible (sample too small), using random split")
            train_df, dev_test_df = train_test_split(
                df_sample,
                train_size=train_size,
                random_state=random_state
            )
            dev_df, test_df = train_test_split(
                dev_test_df,
                train_size=dev_size,
                test_size=test_size,
                random_state=random_state
            )

        print("\n   ✅ Split complete:")
        print(f"      Training: {len(train_df)} samples")
        print(f"      Dev/Validation: {len(dev_df)} samples")
        print(f"      Test: {len(test_df)} samples")

        # Show score distribution in each split
        print("\n   Training score distribution:")
        print(train_df[stratify_column].round().value_counts().sort_index())
        print("\n   Dev score distribution:")
        print(dev_df[stratify_column].round().value_counts().sort_index())
        print("\n   Test score distribution:")
        print(test_df[stratify_column].round().value_counts().sort_index())

        return train_df, dev_df, test_df

    else:  # train_test split (legacy)
        # Two-way split: 2/3 train, 1/3 test
        train_size = int(sample_size * 2 / 3)
        test_size = sample_size - train_size

        print(f"\n   Split: {train_size} train ({train_size/sample_size*100:.0f}%), "
              f"{test_size} test ({test_size/sample_size*100:.0f}%)")

        # Try stratified split, fall back to random split if too few samples per class
        try:
            train_df, test_df = train_test_split(
                df_sample,
                train_size=train_size,
                test_size=test_size,
                stratify=sample_score_groups,
                random_state=random_state
            )
            print("   ✓ Using stratified split")
        except ValueError:
            # Stratification requires at least 2 members per class
            # Fall back to random split for small samples
            print("   ⚠ Stratification not possible (sample too small), using random split")
            train_df, test_df = train_test_split(
                df_sample,
                train_size=train_size,
                test_size=test_size,
                random_state=random_state
            )

        print("\n   ✅ Split complete:")
        print(f"      Training: {len(train_df)} samples")
        print(f"      Test: {len(test_df)} samples")

        # Show score distribution in each split
        print("\n   Training score distribution:")
        print(train_df[stratify_column].round().value_counts().sort_index())
        print("\n   Test score distribution:")
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
