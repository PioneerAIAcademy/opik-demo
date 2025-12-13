# %% [markdown]
"""
# Grading Rubric Prompt Optimization Workshop

Welcome! This notebook teaches you how to optimize prompts using Opik's agent optimizer.

**What you'll learn:**
1. How to set up Opik for optimization
2. How to load and prepare evaluation datasets with stratification
3. How to create custom metrics for scoring
4. How to run baseline evaluations
5. How to use three different optimizers
6. How to compare and analyze results

**Prerequisites:**
- Python 3.8+
- OpenAI API key
- Comet API key (free at comet.com)
- Basic understanding of prompt engineering

**Estimated time:** 45-90 minutes (most time is spent waiting for optimizations to run)
"""

# %% Configuration
"""
CONFIGURATION - Default values (can be overridden via CLI arguments)
"""

# Default configuration values
DEFAULT_SAMPLE_SIZE = None  # None = use full dataset (~153 rows → 102 train, 51 test)
DEFAULT_N_TRIALS = 5        # Number of optimization rounds per optimizer
DEFAULT_N_THREADS = 8       # Parallel threads for evaluation

# Model in LiteLLM format: "provider/endpoint/model"
# Examples: "openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022"
# "responses" endpoint enables reasoning models (GPT-5, etc.)
# See: https://docs.litellm.ai/docs/providers
DEFAULT_MODEL = "openai/responses/gpt-5-mini"

# ⚠️  COST & TIME WARNING:
# - Full dataset (153 samples): ~4,500 API calls, 45-90 minutes, ~$20-50
# - Small test (15 samples): ~150 API calls, 5-10 minutes, ~$1-2
# 💡 Recommendation: Start with --sample-size 15 to verify everything works!

# Default Responses API parameters (LiteLLM flat format)
# See: https://docs.litellm.ai/docs/providers/openai/responses_api
DEFAULT_REASONING_EFFORT = "low"         # "low" | "medium" | "high" | "xhigh"
DEFAULT_VERBOSITY = "low"                # "low" | "medium" | "high"
DEFAULT_MAX_OUTPUT_TOKENS = 65536        # Max tokens (includes reasoning + output)

# %% Setup and imports
import os
from dotenv import load_dotenv
import pandas as pd
from typing import Any
import time
import argparse
import sys
import json

# Import Opik and optimizer
import opik
from opik import Opik
from opik.evaluation.metrics import BaseMetric, score_result
from opik_optimizer import (
    MetaPromptOptimizer,
    HierarchicalReflectiveOptimizer,
    FewShotBayesianOptimizer,
    ChatPrompt
)

# Import our custom utilities
from utils import (
    extract_score_from_text,
    load_csv_with_stratified_split,
    load_text_template,
    create_timestamped_run_dir
)

# Load environment variables from .env file
load_dotenv()

# %% Parse command-line arguments
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Grading Rubric Prompt Optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE,
                       help='Number of samples (None = full dataset ~153)')
    parser.add_argument('--n-trials', type=int, default=DEFAULT_N_TRIALS,
                       help='Optimization trials per optimizer')
    parser.add_argument('--n-threads', type=int, default=DEFAULT_N_THREADS,
                       help='Parallel threads for evaluation')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       help='LLM model (LiteLLM format)')
    parser.add_argument('--optimizers', type=str, default='all',
                       help='Comma-separated: metaprompt,hierarchical,fewshot (default: all)')
    parser.add_argument('--output-dir', type=str, default='runs',
                       help='Base directory for outputs')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    parser.add_argument('--reasoning-effort', type=str, default=DEFAULT_REASONING_EFFORT,
                       choices=['low', 'medium', 'high', 'xhigh'],
                       help='Reasoning effort for Responses API')
    parser.add_argument('--verbosity', type=str, default=DEFAULT_VERBOSITY,
                       choices=['low', 'medium', 'high'],
                       help='Output verbosity for Responses API')
    parser.add_argument('--max-output-tokens', type=int, default=DEFAULT_MAX_OUTPUT_TOKENS,
                       help='Max output tokens for Responses API')
    parser.add_argument('--split-ratio', type=str, default='40/40/20',
                       choices=['60/20/20', '50/25/25', '40/40/20', '30/50/20', '33/33/33'],
                       help='Train/Dev/Test split ratio (default: 40/40/20 recommended for small datasets)')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Build model parameters from CLI arguments
model_params = {
    "reasoning_effort": args.reasoning_effort,
    "verbosity": args.verbosity,
    "max_output_tokens": args.max_output_tokens
}

# Validate configuration values
if args.sample_size is not None and args.sample_size < 5:
    print(f"❌ ERROR: sample_size must be at least 5 (got {args.sample_size})")
    sys.exit(1)

if args.n_trials < 1:
    print(f"❌ ERROR: n_trials must be at least 1 (got {args.n_trials})")
    sys.exit(1)

if args.n_threads < 1:
    print(f"❌ ERROR: n_threads must be at least 1 (got {args.n_threads})")
    sys.exit(1)

# Print configuration
if not args.quiet:
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Sample size: {args.sample_size if args.sample_size else 'Full dataset (~153 rows)'}")
    print(f"Optimization trials: {args.n_trials}")
    print(f"Parallel threads: {args.n_threads}")
    print(f"Model: {args.model}")
    print(f"Model parameters: {model_params}")
    print(f"Optimizers: {args.optimizers}")
    print("=" * 80)

    if args.sample_size is None:
        print("\n⚠️  WARNING: Using full dataset!")
        print("   This will make ~4,500 API calls and may cost $20-50.")
        print("   Consider using --sample-size 15 for initial testing.")
        print("=" * 80)

# Verify API keys are present
if not os.getenv("COMET_API_KEY"):
    print("❌ ERROR: COMET_API_KEY not found")
    print("\nHow to fix:")
    print("1. Create a .env file in this directory")
    print("2. Add: COMET_API_KEY=your_key_here")
    print("3. Get a key at: https://www.comet.com/signup")
    sys.exit(1)

# Check for appropriate API key based on model provider
if args.model.startswith("openai/"):
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found")
        print("\nHow to fix:")
        print("1. Add to .env file: OPENAI_API_KEY=your_key_here")
        print("2. Get a key at: https://platform.openai.com/api-keys")
        sys.exit(1)
elif args.model.startswith("gemini/"):
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ ERROR: GEMINI_API_KEY not found")
        print("\nHow to fix:")
        print("1. Add to .env file: GEMINI_API_KEY=your_key_here")
        print("2. Get a key at: https://aistudio.google.com/app/apikey")
        sys.exit(1)
else:
    print(f"⚠️  Warning: Unknown model provider '{args.model.split('/')[0]}' - make sure appropriate API key is set")

if not args.quiet:
    print("\n✅ API keys loaded successfully")

# %% [markdown]
"""
## Configure Opik

Opik handles tracking and visualization of your optimization runs.
If this is your first time, you'll be prompted to enter your Comet API key.
"""

# %% Configure Opik
opik.configure()
print("✅ Opik configured")

# %% [markdown]
"""
## Create Output Directory

We create a timestamped directory for this run's outputs.
This prevents overwriting previous results and keeps everything organized.
"""

# %% Create output directory
RUN_DIR = create_timestamped_run_dir(base_dir=args.output_dir)
if not args.quiet:
    print(f"   All outputs will be saved to: {RUN_DIR}/")

# %% [markdown]
"""
## Load the Dataset

We use **stratified splitting** to maintain score distribution across ALL splits.

**Three-way split (train/dev/test):**
- Train (60%): Used for failure analysis and understanding patterns
- Dev (20%): Used during optimization to score and select best candidates
- Test (20%): Held-out for final evaluation (NEVER touched during optimization)

Why this matters:
- Without dev set: Optimizer overfits to training data
- With dev set: Optimizer can't overfit - must generalize to unseen dev data
- Test set: Measures true performance on completely unseen data

Example: ~153 samples → 92 train, 31 dev, 30 test

**Why stratification matters:**
- Maintains score distribution across all three sets
- Prevents train having all 5s while test has all 1s
- Ensures reliable, comparable evaluation
"""

# %% Load and split dataset
train_df, dev_df, test_df = load_csv_with_stratified_split(
    csv_path="answer-evaluation.csv",
    sample_size=args.sample_size,
    stratify_column="score",
    random_state=42,
    split_type="train_dev_test",              # Three-way split for proper optimization
    train_dev_test_ratio=args.split_ratio     # Use command-line specified ratio
)

# %% [markdown]
"""
## Create Opik Datasets

Opik datasets let us version and track our evaluation data.
We create THREE separate datasets:
- **Train**: For understanding failure patterns and initial context
- **Dev**: For scoring candidates during optimization (prevents overfitting)
- **Test**: For final evaluation (never touched during optimization)
"""

# %% Create Opik datasets
# Initialize Opik client with a project name
opik_client = Opik(project_name="grading-rubric-optimization")
print("✅ Connected to Opik project: grading-rubric-optimization")


def create_opik_dataset(name: str, df: pd.DataFrame):
    """
    Create an Opik dataset from a pandas DataFrame.

    This maps our CSV columns to Opik dataset fields.
    The 'score' column becomes 'expected_output' for our metric to use.

    ⚠️ IMPORTANT: This deletes any existing dataset with the same name to prevent
    data duplication across multiple runs. Each run should have fresh datasets.
    """
    print(f"\n📦 Getting or creating Opik dataset '{name}'...")

    # Delete existing dataset to prevent duplication across runs
    # This ensures each run starts with fresh data
    try:
        existing = opik_client.get_dataset(name=name)
        if existing:
            print(f"   ⚠️  Deleting existing dataset '{name}' to prevent duplication...")
            opik_client.delete_dataset(name=name)
    except Exception:
        # Dataset doesn't exist, which is fine
        pass

    # Create fresh dataset
    dataset = opik_client.get_or_create_dataset(
        name=name,
        description=f"Grading evaluation dataset - {name}"
    )

    # Insert data with column mapping
    # Opik will map dataset fields to template variables by name!
    dataset.insert_from_pandas(
        dataframe=df,
        keys_mapping={
            "question": "question",           # Maps to {question} in template
            "human_answer": "human_answer",   # Maps to {human_answer} in template
            "ai_answer": "ai_answer",         # Maps to {ai_answer} in template
            "retrieved_content": "retrieved_content",  # Maps to {retrieved_content}
            "score": "expected_output"        # Used by our metric for comparison
        }
    )

    print(f"   ✅ Dataset created with {len(df)} items")
    return dataset


# Create all three datasets
train_dataset = create_opik_dataset("grading-train-dataset", train_df)
dev_dataset = create_opik_dataset("grading-dev-dataset", dev_df)
test_dataset = create_opik_dataset("grading-test-dataset", test_df)

# %% [markdown]
"""
## Load the Prompt Template

Our grading rubric prompt is stored in a separate text file for easy editing.
It uses Python format strings like {question}, {ai_answer}, etc.

**How Opik fills the template:**
Opik automatically maps dataset fields to template variables by name:
- dataset_item["question"] → {question}
- dataset_item["ai_answer"] → {ai_answer}
- And so on...
"""

# %% Load prompt template
prompt_template = load_text_template("grading-rubric-prompt.txt")

print("\n📄 Template preview (first 300 chars):")
print("-" * 80)
print(prompt_template[:300] + "...")
print("-" * 80)

# %% [markdown]
"""
## Create the Custom Metric

Our metric compares LLM-generated scores (1-5) to human scores.

**How it works:**
1. The LLM evaluates an answer using our prompt
2. The LLM's response contains "**Score:** X"
3. We extract that score
4. We compare it to the human score from the dataset
5. We return an accuracy value (0.0 to 1.0)

**Accuracy formula:**
- Perfect match (LLM=4, Human=4) → 1.0
- Off by 0.2 (LLM=4, Human=4.2) → 0.96
- Off by 1 (LLM=4, Human=5) → 0.8
- Off by 2 (LLM=3, Human=5) → 0.6
- Failed to extract score → 0.0

**Important:** This metric does NOT call the LLM - it just evaluates the LLM's output!
"""

# %% Define custom metric class


class ScoreAccuracyMetric(BaseMetric):
    """
    Custom metric that measures score prediction accuracy.

    Key insight: This metric does NOT call the LLM itself.
    It evaluates the LLM's output that was already generated.

    Flow:
    1. Optimizer generates LLM response using the prompt
    2. LLM response contains "**Score:** 4.5"
    3. This metric extracts that score
    4. Compares to expected human score
    5. Returns accuracy (0.0 to 1.0)

    Inherits from Opik's BaseMetric to integrate with the optimizer.
    """

    def __init__(self, name: str = "score_accuracy"):
        """Initialize the metric with a name."""
        super().__init__(name=name)

    def score(
        self,
        output: str,
        expected_output: str,
        **ignored_kwargs: Any
    ) -> score_result.ScoreResult:
        """
        Score the LLM output by comparing extracted score to expected score.

        Args:
            output: The LLM's full response (contains the score)
            expected_output: The expected score from the dataset
            **ignored_kwargs: Other dataset fields (we don't need them)

        Returns:
            ScoreResult with accuracy value between 0.0 and 1.0
        """
        # Extract the score from LLM output
        llm_score = extract_score_from_text(output, min_score=1.0, max_score=5.0)

        if llm_score is None:
            # Failed to extract score - return 0
            return score_result.ScoreResult(
                value=0.0,
                name=self.name,
                reason="Could not extract score from LLM response"
            )

        # Parse expected score
        try:
            human_score = float(expected_output)
            if not (1.0 <= human_score <= 5.0):
                return score_result.ScoreResult(
                    value=0.0,
                    name=self.name,
                    reason=f"Invalid expected score: {expected_output}"
                )
        except (ValueError, TypeError):
            return score_result.ScoreResult(
                value=0.0,
                name=self.name,
                reason=f"Could not parse expected score: {expected_output}"
            )

        # Calculate accuracy: 1 - (difference / 4)
        # This gives us a value between 0.0 and 1.0
        # Max difference is 4 (1.0 to 5.0), so we normalize by 4
        difference = abs(llm_score - human_score)
        accuracy = max(0.0, 1.0 - (difference / 4.0))

        return score_result.ScoreResult(
            value=accuracy,
            name=self.name,
            reason=f"LLM score: {llm_score}, Human score: {human_score}, Diff: {difference:.2f}"
        )


# Create metric instance
metric = ScoreAccuracyMetric(name="score_accuracy")
print("✅ Created custom metric: score_accuracy")

# %% [markdown]
"""
## Test the Metric

Let's test our metric with an example to make sure it works!
"""

# %% Test metric
test_output = """**Analysis:** The AI answer is well-grounded in the retrieved content.
**Score:** 4.5
**Justification:** The answer accurately reflects the information."""

test_expected = "4.5"

test_result = metric.score(output=test_output, expected_output=test_expected)
print("\n🧪 Metric test:")
print("   Input: LLM score=4.5, Expected score=4.5")
print(f"   Result: {test_result.value:.2f} (should be 1.00)")
print(f"   Reason: {test_result.reason}")

if test_result.value == 1.0:
    print("   ✅ Metric working correctly!")
else:
    print("   ❌ Metric test failed - check implementation")

# %% [markdown]
"""
## Create the Metric Function

Following the pattern from OpikOptimizerIntro.ipynb, we create a function
that takes `dataset_item` and `llm_output` as parameters.

**This is the signature Opik expects!**
"""

# %% Define metric function


def score_accuracy_metric_func(dataset_item: dict, llm_output: str) -> score_result.ScoreResult:
    """
    Metric function for optimizer.

    This follows the exact pattern from OpikOptimizerIntro.ipynb.

    Args:
        dataset_item: Full dataset item with all fields
        llm_output: The LLM's response text

    Returns:
        ScoreResult from our custom metric
    """
    return metric.score(
        output=llm_output,
        expected_output=dataset_item["expected_output"]
    )


print("✅ Metric function ready")

# %% [markdown]
"""
## Create the Initial ChatPrompt

The ChatPrompt wraps our template in the format Opik expects.
We put the template in the "user" role message.

**Note:** We only specify the template here, NOT the model!
The model configuration goes in the optimizer.
"""

# %% Create initial ChatPrompt
initial_prompt = ChatPrompt(
    messages=[
        {
            "role": "user",
            "content": prompt_template  # The template with {question}, {ai_answer}, etc.
        }
    ],
    model=args.model,               # Use user's model for prompt evaluation
    model_parameters=model_params   # Pass Responses API parameters to prompt
)

print("✅ Created initial ChatPrompt")

# %% Optimizer selection
# Parse which optimizers to run
enabled_optimizers = set()
if args.optimizers.lower() == 'all':
    enabled_optimizers = {'metaprompt', 'hierarchical', 'fewshot'}
else:
    enabled_optimizers = set(opt.strip().lower() for opt in args.optimizers.split(','))

# Validate optimizer names
valid_optimizers = {'metaprompt', 'hierarchical', 'fewshot'}
invalid = enabled_optimizers - valid_optimizers
if invalid:
    print(f"❌ Invalid optimizer names: {invalid}")
    print(f"   Valid options: {valid_optimizers}")
    sys.exit(1)

if not args.quiet:
    print(f"\n✅ Will run optimizers: {', '.join(sorted(enabled_optimizers))}")

# Initialize optimizer results (will remain None if optimizer not selected)
metaprompt_result = None
hierarchical_result = None
fewshot_result = None

# %% [markdown]
"""
---
# BASELINE EVALUATION

Before optimizing, let's see how well our initial prompt performs.
This gives us a baseline to compare against!
"""

# %% Baseline evaluation
print("\n" + "=" * 80)
print("📊 BASELINE EVALUATION")
print("=" * 80)
print("⏳ Evaluating on train and dev sets...\n")

# We'll use MetaPromptOptimizer for baseline evaluation
baseline_optimizer = MetaPromptOptimizer(
    prompts_per_round=3,
    enable_context=True,
    num_task_examples=3,
    n_threads=args.n_threads,
    verbose=1,
    seed=42
)

start_time = time.time()

# Evaluate baseline on TRAINING data
baseline_train_score = baseline_optimizer.evaluate_prompt(
    prompt=initial_prompt,
    dataset=train_dataset,
    metric=score_accuracy_metric_func,
    n_samples=len(train_df),
    n_threads=args.n_threads
)

# Evaluate baseline on DEV data (what optimizer will use for scoring)
baseline_dev_score = baseline_optimizer.evaluate_prompt(
    prompt=initial_prompt,
    dataset=dev_dataset,
    metric=score_accuracy_metric_func,
    n_samples=len(dev_df),
    n_threads=args.n_threads
)

elapsed_time = time.time() - start_time

print(f"\n📈 Baseline Scores:")
print(f"   Train: {baseline_train_score:.4f} ({baseline_train_score*100:.1f}% accuracy)")
print(f"   Dev:   {baseline_dev_score:.4f} ({baseline_dev_score*100:.1f}% accuracy)")
print(f"   Evaluation time: {elapsed_time/60:.1f} minutes")

# Validate baseline scores are reasonable
if baseline_dev_score < 0.0 or baseline_dev_score > 1.0:
    print(f"\n❌ ERROR: Baseline dev score {baseline_dev_score:.4f} is outside valid range [0, 1]")
    print("   This indicates a problem with the metric or dataset.")
    print("   Please check your metric implementation and dataset format.")
    sys.exit(1)

if baseline_dev_score == 0.0:
    print("\n❌ ERROR: Baseline dev score is 0.0 - model is not working")
    print("   This means the model failed to produce any valid scores.")
    print("   Please check:")
    print("   1. Model API key is valid")
    print("   2. Model name is correct")
    print("   3. Metric is extracting scores correctly")
    sys.exit(1)

# Save baseline scores
with open(f"{RUN_DIR}/baseline_score.txt", "w") as f:
    f.write(f"Baseline Train Score: {baseline_train_score:.4f} ({baseline_train_score*100:.1f}% accuracy)\n")
    f.write(f"Baseline Dev Score: {baseline_dev_score:.4f} ({baseline_dev_score*100:.1f}% accuracy)\n")
    f.write(f"Evaluation time: {elapsed_time/60:.1f} minutes\n")

# %% [markdown]
"""
---
# OPTIMIZATION EXPERIMENTS

We'll run three different optimizers and compare their results:

1. **MetaPrompt** - Uses an LLM to critique and improve the prompt iteratively
2. **Hierarchical Reflective** - Analyzes failure patterns and addresses root causes
3. **Few-Shot Bayesian** - Optimizes few-shot examples using Bayesian search

Each optimizer will run on the SAME training data, so we can fairly compare them!
"""

# %% [markdown]
"""
## Experiment 1: MetaPrompt Optimizer

**How it works:**
1. Analyzes the current prompt
2. Identifies weaknesses based on failures
3. Generates improved candidate prompts
4. Tests each candidate
5. Repeats for multiple rounds

**Parameters:**
- `prompts_per_round=3`: Generate 3 candidates each round
- `enable_context=True`: Use task examples for context
- `num_task_examples=3`: Include 3 examples
- `n_threads=8`: Parallel evaluation for speed
- `max_trials=N_TRIALS`: Run N_TRIALS optimization rounds
"""

# %% Run MetaPrompt optimization
if 'metaprompt' in enabled_optimizers:
    print("\n" + "=" * 80)
    print("🚀 OPTIMIZATION 1: MetaPrompt Optimizer")
    print("=" * 80)
    print("⏳ This may take several minutes...\n")

    metaprompt_optimizer = MetaPromptOptimizer(
        prompts_per_round=3,
        enable_context=True,
        num_task_examples=3,
        n_threads=args.n_threads,
        verbose=1,
        seed=42
    )

    start_time = time.time()

    metaprompt_result = metaprompt_optimizer.optimize_prompt(
        prompt=initial_prompt,
        dataset=train_dataset,              # For failure analysis
        validation_dataset=dev_dataset,     # For scoring candidates ⭐
        metric=score_accuracy_metric_func,
        n_samples=len(train_df),
        n_trials=args.n_trials
    )

    elapsed_time = time.time() - start_time

    # %% Display MetaPrompt results
    print("\n" + "=" * 80)
    print("✅ MetaPrompt Optimization Complete!")
    print("=" * 80)

    metaprompt_result.display()

    print("\n📊 Summary:")
    print(f"   Baseline Dev Score:  {baseline_dev_score:.4f}")
    print(f"   Optimized Dev Score: {metaprompt_result.score:.4f}")
    print(f"   Optimization time: {elapsed_time/60:.1f} minutes")

    if metaprompt_result.initial_score is not None:
        improvement = metaprompt_result.score - metaprompt_result.initial_score
        improvement_pct = (improvement / metaprompt_result.initial_score) * 100
        print(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")

    # %% Save MetaPrompt result
    output_file = f"{RUN_DIR}/optimized-metaprompt-messages.txt"
    with open(output_file, "w") as f:
        f.write("MetaPrompt Optimization Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Baseline Dev Score: {baseline_dev_score:.4f}\n")
        f.write(f"Optimized Dev Score: {metaprompt_result.score:.4f}\n")
        f.write(f"Optimization Time: {elapsed_time/60:.1f} minutes\n\n")
        f.write("=" * 80 + "\n")
        f.write("OPTIMIZED PROMPT MESSAGES:\n")
        f.write("=" * 80 + "\n\n")
        for msg in metaprompt_result.prompt:
            f.write(f"Role: {msg.get('role')}\n")
            f.write(f"Content:\n{msg.get('content')}\n")
            f.write("-" * 80 + "\n\n")

    print(f"\n💾 Saved optimized prompt to: {output_file}")
else:
    print("\n⏭️  Skipping MetaPrompt optimizer")

# %% [markdown]
"""
---
## Experiment 2: Hierarchical Reflective Optimizer

**How it works:**
1. Groups failures into batches
2. Analyzes root causes hierarchically
3. Generates fixes targeting those causes
4. Tests and iterates

**Parameters:**
- `max_parallel_batches=3`: Process up to 3 batches in parallel
- `batch_size=20`: Analyze 20 failures per batch
- `convergence_threshold=0.01`: Stop if improvement < 1%
"""

# %% Run Hierarchical optimization
if 'hierarchical' in enabled_optimizers:
    print("\n" + "=" * 80)
    print("🚀 OPTIMIZATION 2: Hierarchical Reflective Optimizer")
    print("=" * 80)
    print("⏳ This may take several minutes...\n")

    hierarchical_optimizer = HierarchicalReflectiveOptimizer(
        max_parallel_batches=3,
        batch_size=20,
        convergence_threshold=0.01,
        n_threads=args.n_threads,
        verbose=1,
        seed=42
    )

    start_time = time.time()

    hierarchical_result = hierarchical_optimizer.optimize_prompt(
        prompt=initial_prompt,
        dataset=train_dataset,              # For failure analysis
        validation_dataset=dev_dataset,     # For scoring candidates ⭐
        metric=score_accuracy_metric_func,
        n_samples=len(train_df),
        n_trials=args.n_trials
    )

    elapsed_time = time.time() - start_time

    # %% Display Hierarchical results
    print("\n" + "=" * 80)
    print("✅ Hierarchical Reflective Optimization Complete!")
    print("=" * 80)

    hierarchical_result.display()

    print("\n📊 Summary:")
    print(f"   Baseline Dev Score:  {baseline_dev_score:.4f}")
    print(f"   Optimized Dev Score: {hierarchical_result.score:.4f}")
    print(f"   Optimization time: {elapsed_time/60:.1f} minutes")

    if hierarchical_result.initial_score is not None:
        improvement = hierarchical_result.score - hierarchical_result.initial_score
        improvement_pct = (improvement / hierarchical_result.initial_score) * 100
        print(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")

    # %% Save Hierarchical result
    output_file = f"{RUN_DIR}/optimized-hierarchical-messages.txt"
    with open(output_file, "w") as f:
        f.write("Hierarchical Reflective Optimization Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Baseline Dev Score: {baseline_dev_score:.4f}\n")
        f.write(f"Optimized Dev Score: {hierarchical_result.score:.4f}\n")
        f.write(f"Optimization Time: {elapsed_time/60:.1f} minutes\n\n")
        f.write("=" * 80 + "\n")
        f.write("OPTIMIZED PROMPT MESSAGES:\n")
        f.write("=" * 80 + "\n\n")
        for msg in hierarchical_result.prompt:
            f.write(f"Role: {msg.get('role')}\n")
            f.write(f"Content:\n{msg.get('content')}\n")
            f.write("-" * 80 + "\n\n")

    print(f"\n💾 Saved optimized prompt to: {output_file}")
else:
    print("\n⏭️  Skipping Hierarchical optimizer")

# %% [markdown]
"""
---
## Experiment 3: Few-Shot Bayesian Optimizer

**How it works:**
1. Searches for optimal few-shot examples
2. Uses Bayesian optimization to explore efficiently
3. Finds the best combination of demonstration examples

**Parameters:**
- `n_trials=10`: Number of Bayesian optimization trials
"""

# %% Run Few-Shot optimization
if 'fewshot' in enabled_optimizers:
    print("\n" + "=" * 80)
    print("🚀 OPTIMIZATION 3: Few-Shot Bayesian Optimizer")
    print("=" * 80)
    print("⏳ This may take several minutes...\n")

    fewshot_optimizer = FewShotBayesianOptimizer(
        min_examples=2,
        max_examples=8,
        n_threads=args.n_threads,
        verbose=1,
        seed=42
    )

    start_time = time.time()

    fewshot_result = fewshot_optimizer.optimize_prompt(
        prompt=initial_prompt,
        dataset=train_dataset,              # For failure analysis
        validation_dataset=dev_dataset,     # For scoring candidates ⭐
        metric=score_accuracy_metric_func,
        n_samples=len(train_df),
        max_trials=args.n_trials
    )

    elapsed_time = time.time() - start_time

    # %% Display Few-Shot results
    print("\n" + "=" * 80)
    print("✅ Few-Shot Bayesian Optimization Complete!")
    print("=" * 80)

    fewshot_result.display()

    print("\n📊 Summary:")
    print(f"   Baseline Dev Score:  {baseline_dev_score:.4f}")
    print(f"   Optimized Dev Score: {fewshot_result.score:.4f}")
    print(f"   Optimization time: {elapsed_time/60:.1f} minutes")

    if fewshot_result.initial_score is not None:
        improvement = fewshot_result.score - fewshot_result.initial_score
        improvement_pct = (improvement / fewshot_result.initial_score) * 100
        print(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")

    # %% Save Few-Shot result
    output_file = f"{RUN_DIR}/optimized-fewshot-messages.txt"
    with open(output_file, "w") as f:
        f.write("Few-Shot Bayesian Optimization Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Baseline Dev Score: {baseline_dev_score:.4f}\n")
        f.write(f"Optimized Dev Score: {fewshot_result.score:.4f}\n")
        f.write(f"Optimization Time: {elapsed_time/60:.1f} minutes\n\n")
        f.write("=" * 80 + "\n")
        f.write("OPTIMIZED PROMPT MESSAGES:\n")
        f.write("=" * 80 + "\n\n")
        for msg in fewshot_result.prompt:
            f.write(f"Role: {msg.get('role')}\n")
            f.write(f"Content:\n{msg.get('content')}\n")
            f.write("-" * 80 + "\n\n")

    print(f"\n💾 Saved optimized prompt to: {output_file}")
else:
    print("\n⏭️  Skipping Few-Shot optimizer")

# %% [markdown]
"""
---
# TEST SET EVALUATION

Now evaluate the best prompts on the held-out test set to measure generalization!
"""

# %% Evaluate on test set
print("\n" + "=" * 80)
print("🧪 TEST SET EVALUATION")
print("=" * 80)
print("Evaluating optimized prompts on held-out test data...")
print()

# Evaluate baseline on test set
print("Evaluating baseline on test set...")
baseline_test_score = baseline_optimizer.evaluate_prompt(
    prompt=initial_prompt,
    dataset=test_dataset,  # ⚠️ Use test set!
    metric=score_accuracy_metric_func,
    n_samples=len(test_df),
    n_threads=args.n_threads
)
print(f"✅ Baseline test score: {baseline_test_score:.4f}")

# Evaluate each optimizer's result on test set
metaprompt_test_score = None
hierarchical_test_score = None
fewshot_test_score = None

if metaprompt_result is not None:
    print("\nEvaluating MetaPrompt on test set...")
    metaprompt_test_prompt = ChatPrompt(
        messages=metaprompt_result.prompt,
        model=args.model,
        model_parameters=model_params
    )
    metaprompt_test_score = baseline_optimizer.evaluate_prompt(
        prompt=metaprompt_test_prompt,
        dataset=test_dataset,
        metric=score_accuracy_metric_func,
        n_samples=len(test_df),
        n_threads=args.n_threads
    )
    print(f"✅ MetaPrompt test score: {metaprompt_test_score:.4f}")

if hierarchical_result is not None:
    print("\nEvaluating Hierarchical on test set...")
    hierarchical_test_prompt = ChatPrompt(
        messages=hierarchical_result.prompt,
        model=args.model,
        model_parameters=model_params
    )
    hierarchical_test_score = baseline_optimizer.evaluate_prompt(
        prompt=hierarchical_test_prompt,
        dataset=test_dataset,
        metric=score_accuracy_metric_func,
        n_samples=len(test_df),
        n_threads=args.n_threads
    )
    print(f"✅ Hierarchical test score: {hierarchical_test_score:.4f}")

if fewshot_result is not None:
    print("\nEvaluating FewShot on test set...")
    fewshot_test_prompt = ChatPrompt(
        messages=fewshot_result.prompt,
        model=args.model,
        model_parameters=model_params
    )
    fewshot_test_score = baseline_optimizer.evaluate_prompt(
        prompt=fewshot_test_prompt,
        dataset=test_dataset,
        metric=score_accuracy_metric_func,
        n_samples=len(test_df),
        n_threads=args.n_threads
    )
    print(f"✅ FewShot test score: {fewshot_test_score:.4f}")

# %% [markdown]
"""
---
# FINAL COMPARISON

Let's compare all three optimizers side-by-side!
"""

# %% Compare all results
# Calculate improvements
def calc_improvement(result, baseline):
    if result.initial_score is not None and result.initial_score != 0:
        return ((result.score - result.initial_score) / result.initial_score) * 100
    elif baseline != 0:
        return ((result.score - baseline) / baseline) * 100
    else:
        return 0.0

# Build results list dynamically based on what ran
results_data = [{
    "Optimizer": "Baseline",
    "Train Score": baseline_dev_score,
    "Test Score": baseline_test_score,
    "Train Improvement": "—",
    "Test Improvement": "—",
    "Output File": "—"
}]

if metaprompt_result is not None:
    train_imp = calc_improvement(metaprompt_result, baseline_dev_score)
    test_imp = ((metaprompt_test_score - baseline_test_score) / baseline_test_score) * 100 if baseline_test_score != 0 else 0.0
    results_data.append({
        "Optimizer": "MetaPrompt",
        "Train Score": metaprompt_result.score,
        "Test Score": metaprompt_test_score,
        "Train Improvement": f"{train_imp:+.1f}%",
        "Test Improvement": f"{test_imp:+.1f}%",
        "Output File": "optimized-metaprompt-messages.txt"
    })

if hierarchical_result is not None:
    train_imp = calc_improvement(hierarchical_result, baseline_dev_score)
    test_imp = ((hierarchical_test_score - baseline_test_score) / baseline_test_score) * 100 if baseline_test_score != 0 else 0.0
    results_data.append({
        "Optimizer": "Hierarchical",
        "Train Score": hierarchical_result.score,
        "Test Score": hierarchical_test_score,
        "Train Improvement": f"{train_imp:+.1f}%",
        "Test Improvement": f"{test_imp:+.1f}%",
        "Output File": "optimized-hierarchical-messages.txt"
    })

if fewshot_result is not None:
    train_imp = calc_improvement(fewshot_result, baseline_dev_score)
    test_imp = ((fewshot_test_score - baseline_test_score) / baseline_test_score) * 100 if baseline_test_score != 0 else 0.0
    results_data.append({
        "Optimizer": "Few-Shot",
        "Train Score": fewshot_result.score,
        "Test Score": fewshot_test_score,
        "Train Improvement": f"{train_imp:+.1f}%",
        "Test Improvement": f"{test_imp:+.1f}%",
        "Output File": "optimized-fewshot-messages.txt"
    })

results_df = pd.DataFrame(results_data)

print("\n" + "=" * 80)
print("📊 FINAL RESULTS COMPARISON (TRAIN vs TEST)")
print("=" * 80)
print()
print(results_df.to_string(index=False))

# Find winner based on TEST score (excluding baseline)
test_scores = results_df[results_df["Optimizer"] != "Baseline"]["Test Score"]
if len(test_scores) > 0:
    best_idx = test_scores.idxmax()
    winner = results_df.loc[best_idx, "Optimizer"]
    winner_test_score = results_df.loc[best_idx, "Test Score"]
    winner_train_score = results_df.loc[best_idx, "Train Score"]

    print(f"\n🏆 WINNER (by test score): {winner}")
    print(f"   Test Score: {winner_test_score:.4f}")
    print(f"   Train Score: {winner_train_score:.4f}")

    # Check for overfitting
    if winner_train_score > winner_test_score + 0.02:
        print(f"   ⚠️  WARNING: Possible overfitting detected!")
        print(f"   Train score is {winner_train_score - winner_test_score:.4f} higher than test")
else:
    winner = "N/A"
    winner_test_score = 0.0
print(f"\n📁 All results saved to: {RUN_DIR}/")

# Save comparison table
with open(f"{RUN_DIR}/comparison_table.txt", "w") as f:
    f.write("FINAL RESULTS COMPARISON (TRAIN vs TEST)\n")
    f.write("=" * 80 + "\n\n")
    f.write(results_df.to_string(index=False))
    if len(test_scores) > 0:
        f.write(f"\n\n🏆 WINNER (by test score): {winner}")
        f.write(f"\n   Test Score: {winner_test_score:.4f}")
        f.write(f"\n   Train Score: {winner_train_score:.4f}\n")

# %% Save JSON summary (always created for easy parsing/testing)
summary = {
    "test_config": {
        "sample_size": args.sample_size,
        "n_trials": args.n_trials,
        "n_threads": args.n_threads,
        "model": args.model,
        "timestamp": os.path.basename(RUN_DIR),
        "enabled_optimizers": list(enabled_optimizers)
    },
    "baseline": {
        "train_score": float(baseline_dev_score),
        "test_score": float(baseline_test_score)
    },
    "optimizers": {},
    "status": "success"
}

# Add optimizer results
optimizer_map = [
    ('MetaPrompt', metaprompt_result, metaprompt_test_score),
    ('Hierarchical', hierarchical_result, hierarchical_test_score),
    ('FewShot', fewshot_result, fewshot_test_score)
]

for name, result_obj, test_score in optimizer_map:
    if result_obj is not None and test_score is not None:
        train_imp = ((result_obj.score - baseline_dev_score) / baseline_dev_score) * 100 if baseline_dev_score != 0 else 0.0
        test_imp = ((test_score - baseline_test_score) / baseline_test_score) * 100 if baseline_test_score != 0 else 0.0
        summary["optimizers"][name] = {
            "train_score": float(result_obj.score),
            "test_score": float(test_score),
            "train_improvement_pct": float(train_imp),
            "test_improvement_pct": float(test_imp)
        }

# Find winner based on TEST score
if summary["optimizers"]:
    winner_name = max(summary["optimizers"].keys(),
                     key=lambda k: summary["optimizers"][k]["test_score"])
    summary["winner"] = {
        "name": winner_name,
        "train_score": summary["optimizers"][winner_name]["train_score"],
        "test_score": summary["optimizers"][winner_name]["test_score"]
    }

# Save JSON
json_path = f"{RUN_DIR}/results_summary.json"
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

if not args.quiet:
    print(f"\n💾 JSON summary saved to: {json_path}")

# %% [markdown]
"""
## View Results in Opik Dashboard

You can see detailed visualizations, trace logs, and experiment tracking
in the Opik web interface.
"""

# %% Link to Opik dashboard
print("\n🔗 View detailed results in Opik:")
print("   https://www.comet.com/opik")
print("\n✨ Workshop complete!")
print("\n📚 Key Takeaways:")
print("   1. Always measure baseline before optimizing")
print("   2. Different optimizers work better for different tasks")
print("   3. Stratification matters for reliable evaluation")
print("   4. Custom metrics let you optimize for your specific goals")
print("   5. The patterns in utils.py can be reused in your projects")
print(f"\n📁 All outputs saved to: {RUN_DIR}/")
