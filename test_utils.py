"""
Unit tests for utility functions.

Run with: pytest test_utils.py -v
Run with coverage: pytest test_utils.py -v --cov=utils
"""

import pytest
import pandas as pd
import os
from utils import (
    extract_score_from_text,
    load_csv_with_stratified_split,
    load_text_template,
    create_timestamped_run_dir
)


class TestScoreExtraction:
    """Test score extraction from various text formats."""

    def test_standard_format(self):
        """Test **Score:** 4 format."""
        assert extract_score_from_text("**Score:** 4") == 4.0

    def test_standard_format_decimal(self):
        """Test **Score:** 4.5 format."""
        assert extract_score_from_text("**Score:** 4.5") == 4.5

    def test_alternate_format(self):
        """Test Score: 3 format."""
        assert extract_score_from_text("Score: 3") == 3.0

    def test_alternate_format_decimal(self):
        """Test Score: 3.2 format."""
        assert extract_score_from_text("Score: 3.2") == 3.2

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        assert extract_score_from_text("score: 5") == 5.0

    def test_score_of_format(self):
        """Test 'score of X' format."""
        assert extract_score_from_text("The answer gets a score of 3") == 3.0

    def test_score_of_format_decimal(self):
        """Test 'score of X.X' format."""
        assert extract_score_from_text("The answer gets a score of 3.8") == 3.8

    def test_no_score_found(self):
        """Test when no score is present."""
        assert extract_score_from_text("No score here") is None

    def test_invalid_range_high(self):
        """Test score above max range."""
        assert extract_score_from_text("Score: 10") is None

    def test_invalid_range_low(self):
        """Test score below min range."""
        assert extract_score_from_text("Score: 0") is None

    def test_invalid_range_negative(self):
        """Test negative score."""
        assert extract_score_from_text("Score: -1") is None

    def test_multiple_scores_takes_first(self):
        """Test that first valid score is returned."""
        text = "Score: 3... and later Score: 4"
        assert extract_score_from_text(text) == 3.0

    def test_score_in_context(self):
        """Test score extraction from realistic LLM output."""
        text = """**Analysis:** The AI answer is well-grounded.
**Score:** 4.8
**Justification:** The answer accurately reflects the information."""
        assert extract_score_from_text(text) == 4.8


class TestDatasetLoading:
    """Test dataset loading and splitting."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV for testing."""
        csv_path = tmp_path / "test.csv"
        # Create 30 rows with balanced score distribution
        df = pd.DataFrame({
            "question": [f"Q{i}" for i in range(30)],
            "human_answer": [f"A{i}" for i in range(30)],
            "ai_answer": [f"AI{i}" for i in range(30)],
            "retrieved_content": [f"C{i}" for i in range(30)],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0] * 6  # Balanced distribution
        })
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_basic_split_full_dataset(self, sample_csv):
        """Test loading full dataset with 2/3:1/3 split."""
        train, test = load_csv_with_stratified_split(
            sample_csv,
            sample_size=None  # Use all data
        )
        # 30 rows → 20 train, 10 test (2/3:1/3)
        assert len(train) == 20
        assert len(test) == 10
        assert set(train.columns) == {"question", "human_answer", "ai_answer", "retrieved_content", "score"}

    def test_basic_split_custom_sample(self, sample_csv):
        """Test loading with custom sample size."""
        train, test = load_csv_with_stratified_split(
            sample_csv,
            sample_size=15
        )
        # 15 samples → 10 train, 5 test (2/3:1/3)
        assert len(train) == 10
        assert len(test) == 5

    def test_stratification_maintains_distribution(self, sample_csv):
        """Test that stratification works."""
        train, test = load_csv_with_stratified_split(
            sample_csv,
            sample_size=None
        )
        # Both should have variety of scores (after rounding to integers)
        train_scores = train["score"].round().unique()
        test_scores = test["score"].round().unique()
        assert len(train_scores) >= 3
        assert len(test_scores) >= 3

    def test_insufficient_data_raises_error(self, sample_csv):
        """Test that insufficient data raises ValueError."""
        with pytest.raises(ValueError):
            load_csv_with_stratified_split(
                sample_csv,
                sample_size=100  # More than available (30)
            )

    def test_split_ratio(self, sample_csv):
        """Test that split maintains 2/3:1/3 ratio."""
        train, test = load_csv_with_stratified_split(
            sample_csv,
            sample_size=21
        )
        # 21 samples → 14 train (2/3), 7 test (1/3)
        assert len(train) == 14
        assert len(test) == 7


class TestTemplateLoading:
    """Test template loading."""

    @pytest.fixture
    def sample_template(self, tmp_path):
        """Create a sample template file."""
        template_path = tmp_path / "test_template.txt"
        content = "Question: {question}\nAnswer: {ai_answer}"
        with open(template_path, "w") as f:
            f.write(content)
        return str(template_path)

    def test_load_template(self, sample_template):
        """Test template loading."""
        template = load_text_template(sample_template)
        assert "{question}" in template
        assert "{ai_answer}" in template
        assert len(template) > 0

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_text_template("nonexistent.txt")


class TestTimestampedRunDir:
    """Test timestamped run directory creation."""

    def test_creates_directory(self, tmp_path):
        """Test that directory is created."""
        os.chdir(tmp_path)
        run_dir = create_timestamped_run_dir("test_runs")
        assert os.path.exists(run_dir)
        assert os.path.isdir(run_dir)
        assert run_dir.startswith("test_runs/")

    def test_unique_timestamps(self, tmp_path):
        """Test that multiple calls create unique directories."""
        os.chdir(tmp_path)
        import time
        run_dir1 = create_timestamped_run_dir("test_runs")
        time.sleep(1.1)  # Ensure different timestamp (seconds precision)
        run_dir2 = create_timestamped_run_dir("test_runs")
        assert run_dir1 != run_dir2
        assert os.path.exists(run_dir1)
        assert os.path.exists(run_dir2)

    def test_default_base_dir(self, tmp_path):
        """Test default base directory name."""
        os.chdir(tmp_path)
        run_dir = create_timestamped_run_dir()
        assert run_dir.startswith("runs/")
