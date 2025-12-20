# %% [markdown]
"""
# Emotion Classification Prompt Optimization Workshop

Welcome! This notebook teaches you how to optimize prompts using Opik's agent optimizer.

**What you'll learn:**
1. How to set up Opik for optimization
2. How to load and prepare evaluation datasets with stratification
3. How to create custom metrics for scoring
4. How to run baseline evaluations
5. How to use five different optimizers
6. How to compare and analyze results

**Task:** Classify social media text into one of four emotions:
- joy, anger, sadness, surprise

**Prerequisites:**
- Python 3.12+
- OpenAI API key
- Comet API key (free at comet.com)
- Basic understanding of prompt engineering
"""

# %% Configuration
"""
CONFIGURATION - Default values (can be overridden via CLI arguments)
"""

# Default configuration values
DEFAULT_SAMPLE_SIZE = None  # None = use full dataset (~300 rows)
DEFAULT_N_TRIALS = 10       # Number of optimization rounds per optimizer
DEFAULT_N_THREADS = 4       # Parallel threads for evaluation

# Two models are used during optimization:
# 1. REASONING_MODEL: Used by the optimizer for generating candidates, mutations, analysis
# 2. TASK_MODEL: Used to evaluate prompts on the dataset
#
# Model format is LiteLLM: "provider/model"
# Examples: "openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-20241022"
# See: https://docs.litellm.ai/docs/providers
DEFAULT_REASONING_MODEL = "openai/gpt-4o"       # Smarter model for optimization logic
DEFAULT_TASK_MODEL = "openai/gpt-4o-mini"       # Faster/cheaper model for evaluation

# 💡 Recommendation: Start with --sample-size 30 to verify everything works!

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
    GepaOptimizer,
    EvolutionaryOptimizer,
    ChatPrompt
)

# Import our custom utilities
from utils import (
    extract_emotion_from_text,
    load_csv_with_stratified_split,
    load_text_template,
    create_timestamped_run_dir,
    save_optimizer_result_to_file
)

# Load environment variables from .env file
load_dotenv()

# %% Parse command-line arguments
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Emotion Classification Prompt Optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE,
                       help='Number of samples (None = full dataset ~300)')
    parser.add_argument('--n-trials', type=int, default=DEFAULT_N_TRIALS,
                       help='Optimization trials per optimizer')
    parser.add_argument('--n-threads', type=int, default=DEFAULT_N_THREADS,
                       help='Parallel threads for evaluation')
    parser.add_argument('--reasoning-model', type=str, default=DEFAULT_REASONING_MODEL,
                       help='Model for optimizer reasoning/generation (LiteLLM format)')
    parser.add_argument('--task-model', type=str, default=DEFAULT_TASK_MODEL,
                       help='Model for evaluating prompts on dataset (LiteLLM format)')
    parser.add_argument('--optimizers', type=str, default='all',
                       help='Comma-separated: metaprompt,hierarchical,fewshot,gepa,evolutionary (default: all)')
    parser.add_argument('--output-dir', type=str, default='runs',
                       help='Base directory for outputs')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    parser.add_argument('--split-ratio', type=str, default='40/40/20',
                       choices=['60/20/20', '50/25/25', '40/40/20', '30/50/20', '33/33/33'],
                       help='Train/Dev/Test split ratio (default: 40/40/20 recommended for small datasets)')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Build model parameters from CLI arguments
# These parameters apply to both reasoning and task model calls
model_params = {
    "num_retries": 5  # Retry on transient API errors (connection drops, rate limits)
}

# Convert hyphenated CLI args to underscore attributes
reasoning_model = args.reasoning_model
task_model = args.task_model

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
    print(f"Sample size: {args.sample_size if args.sample_size else 'Full dataset (~300 rows)'}")
    print(f"Optimization trials: {args.n_trials}")
    print(f"Parallel threads: {args.n_threads}")
    print(f"Reasoning model: {reasoning_model} (for optimizer logic)")
    print(f"Task model: {task_model} (for prompt evaluation)")
    print(f"Model parameters: {model_params}")
    print(f"Optimizers: {args.optimizers}")
    print("=" * 80)

# Verify API keys are present
if not os.getenv("OPIK_API_KEY"):
    print("❌ ERROR: OPIK_API_KEY not found")
    print("\nHow to fix:")
    print("1. Create a .env file in this directory")
    print("2. Add: OPIK_API_KEY=your_key_here")
    print("3. Get a key at: https://www.comet.com/signup")
    sys.exit(1)


def check_model_api_key(model_name: str, model_purpose: str) -> None:
    """Check that the appropriate API key is set for a model."""
    if model_name.startswith("openai/"):
        if not os.getenv("OPENAI_API_KEY"):
            print(f"❌ ERROR: OPENAI_API_KEY not found (required for {model_purpose}: {model_name})")
            print("\nHow to fix:")
            print("1. Add to .env file: OPENAI_API_KEY=your_key_here")
            print("2. Get a key at: https://platform.openai.com/api-keys")
            sys.exit(1)
    elif model_name.startswith("gemini/"):
        if not os.getenv("GEMINI_API_KEY"):
            print(f"❌ ERROR: GEMINI_API_KEY not found (required for {model_purpose}: {model_name})")
            print("\nHow to fix:")
            print("1. Add to .env file: GEMINI_API_KEY=your_key_here")
            print("2. Get a key at: https://aistudio.google.com/app/apikey")
            sys.exit(1)
    else:
        provider = model_name.split('/')[0] if '/' in model_name else model_name
        print(f"⚠️  Warning: Unknown provider '{provider}' for {model_purpose} - ensure API key is set")


# Check API keys for both models
check_model_api_key(reasoning_model, "reasoning model")
check_model_api_key(task_model, "task model")

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

We use **stratified splitting** to maintain emotion distribution across ALL splits.

**Three-way split (train/dev/test):**
- Train (40%): Used for failure analysis and understanding patterns
- Dev (40%): Used during optimization to score and select best candidates
- Test (20%): Held-out for final evaluation (NEVER touched during optimization)

Why this matters:
- Without dev set: Optimizer overfits to training data
- With dev set: Optimizer can't overfit - must generalize to unseen dev data
- Test set: Measures true performance on completely unseen data

Example: ~300 samples → 120 train, 120 dev, 60 test

**Why stratification matters:**
- Maintains emotion distribution across all three sets
- Prevents train having all joy while test has all anger
- Ensures reliable, comparable evaluation
"""

# %% Load and split dataset
train_df, dev_df, test_df = load_csv_with_stratified_split(
    csv_path="sampled_emotions.csv",
    sample_size=args.sample_size,
    stratify_column="emotion",
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
opik_client = Opik(project_name="emotion-classification")
print("✅ Connected to Opik project: emotion-classification")


def create_opik_dataset(name: str, df: pd.DataFrame):
    """
    Create an Opik dataset from a pandas DataFrame.

    This maps our CSV columns to Opik dataset fields.
    The 'emotion' column becomes 'expected_output' for our metric to use.

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
        description=f"Emotion classification dataset - {name}"
    )

    # Insert data with column mapping
    # Opik will map dataset fields to template variables by name!
    dataset.insert_from_pandas(
        dataframe=df,
        keys_mapping={
            "text": "text",                   # Maps to {text} in template
            "emotion": "expected_output"      # Used by our metric for comparison
        }
    )

    print(f"   ✅ Dataset created with {len(df)} items")
    return dataset


# Create all three datasets
train_dataset = create_opik_dataset("emotion-train-dataset", train_df)
dev_dataset = create_opik_dataset("emotion-dev-dataset", dev_df)
test_dataset = create_opik_dataset("emotion-test-dataset", test_df)

# Allow Opik server to sync (eventual consistency)
# This prevents dataset_item_id mismatches in GEPA optimizer
if not args.quiet:
    print("\n⏳ Waiting for Opik server sync...")
time.sleep(3)

# Verify dataset items have IDs (debug for GEPA issues)
train_items = train_dataset.get_items()
if train_items:
    first_id = train_items[0].get("id")
    if not args.quiet:
        print(f"   ✅ Train dataset verified: {len(train_items)} items, first ID: {first_id[:20]}...")
else:
    print("   ⚠️  WARNING: Train dataset has no items!")

# %% [markdown]
"""
## Load the Prompt Templates

Our prompt is split into TWO files for optimizer compatibility:
- **emotion-prompt-system.txt**: The classification instruction (system message)
- **emotion-prompt-user.txt**: Template variable for text (user message)

**Why this split?**
MetaPrompt optimizer only modifies system messages by default. By putting the
classification instruction in the system message, the optimizer can improve it.
The user message with the template variable is preserved during optimization.

**How Opik fills the template:**
Opik automatically maps dataset fields to template variables by name:
- dataset_item["text"] → {text}
"""

# %% Load prompt templates (system + user for optimizer compatibility)
# MetaPrompt optimizer only modifies system messages by default, so we split:
# - System message: Contains the classification instruction (will be optimized)
# - User message: Contains only template variable (preserved during optimization)
system_template = load_text_template("emotion-prompt-system.txt")
user_template = load_text_template("emotion-prompt-user.txt")

print("\n📄 System template preview (first 300 chars):")
print("-" * 80)
print(system_template[:300] + "...")
print("-" * 80)
print("\n📄 User template (full):")
print("-" * 80)
print(user_template)
print("-" * 80)

# %% [markdown]
"""
## Create the Custom Metric

Our metric compares LLM-predicted emotions to expected emotions.

**How it works:**
1. The LLM classifies text using our prompt
2. The LLM's response contains the emotion (e.g., "joy")
3. We extract that emotion
4. We compare it to the expected emotion from the dataset
5. We return 1.0 for correct, 0.0 for incorrect

**Important:** This metric does NOT call the LLM - it just evaluates the LLM's output!
"""

# %% Define custom metric class


class EmotionAccuracyMetric(BaseMetric):
    """
    Custom metric that measures emotion classification accuracy.

    Key insight: This metric does NOT call the LLM itself.
    It evaluates the LLM's output that was already generated.

    Flow:
    1. Optimizer generates LLM response using the prompt
    2. LLM response contains "joy" (or other emotion)
    3. This metric extracts that emotion
    4. Compares to expected emotion
    5. Returns 1.0 if correct, 0.0 if incorrect

    Inherits from Opik's BaseMetric to integrate with the optimizer.
    """

    def __init__(self, name: str = "emotion_accuracy"):
        """Initialize the metric with a name."""
        super().__init__(name=name)

    def score(
        self,
        output: str,
        expected_output: str,
        **ignored_kwargs: Any
    ) -> score_result.ScoreResult:
        """
        Score the LLM output by comparing extracted emotion to expected emotion.

        Args:
            output: The LLM's full response (contains the emotion)
            expected_output: The expected emotion from the dataset
            **ignored_kwargs: Other dataset fields (we don't need them)

        Returns:
            ScoreResult with value 1.0 (correct) or 0.0 (incorrect)
        """
        # Extract the emotion from LLM output
        predicted_emotion = extract_emotion_from_text(output)
        expected_emotion = expected_output.strip().lower()

        if predicted_emotion is None:
            # Failed to extract emotion - return 0
            return score_result.ScoreResult(
                value=0.0,
                name=self.name,
                reason=f"Could not extract emotion from: {output[:50]}..."
            )

        # Compare emotions (case-insensitive)
        is_correct = (predicted_emotion == expected_emotion)

        return score_result.ScoreResult(
            value=1.0 if is_correct else 0.0,
            name=self.name,
            reason=f"Predicted: {predicted_emotion}, Expected: {expected_emotion}"
        )


# Create metric instance
metric = EmotionAccuracyMetric(name="emotion_accuracy")
print("✅ Created custom metric: emotion_accuracy")

# %% [markdown]
"""
## Test the Metric

Let's test our metric with an example to make sure it works!
"""

# %% Test metric
test_output = "joy"
test_expected = "joy"

test_result = metric.score(output=test_output, expected_output=test_expected)
print("\n🧪 Metric test:")
print("   Input: LLM output='joy', Expected='joy'")
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


def emotion_accuracy_metric_func(dataset_item: dict, llm_output: str) -> score_result.ScoreResult:
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

The ChatPrompt wraps our templates in the format Opik expects.
We use a two-message structure:
- **System message**: Contains the classification instruction (will be optimized by MetaPrompt)
- **User message**: Contains template variable (preserved during optimization)

**Note:** We only specify the template here, NOT the model!
The model configuration goes in the optimizer.
"""

# %% Create initial ChatPrompt
# Using system + user message structure so MetaPrompt can optimize the instruction
initial_prompt = ChatPrompt(
    messages=[
        {
            "role": "system",
            "content": system_template  # The classification instruction (will be optimized)
        },
        {
            "role": "user",
            "content": user_template    # Template variable only (preserved)
        }
    ],
    model=task_model,               # Task model evaluates prompts on dataset
    model_parameters=model_params   # Pass model parameters to prompt
)

print("✅ Created initial ChatPrompt (system + user messages)")

# %% Optimizer selection
# Parse which optimizers to run
enabled_optimizers = set()
if args.optimizers.lower() == 'all':
    enabled_optimizers = {'metaprompt', 'hierarchical', 'fewshot', 'gepa', 'evolutionary'}
else:
    enabled_optimizers = set(opt.strip().lower() for opt in args.optimizers.split(','))

# Validate optimizer names
valid_optimizers = {'metaprompt', 'hierarchical', 'fewshot', 'gepa', 'evolutionary'}
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
gepa_result = None
evolutionary_result = None

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
    model=reasoning_model,          # Reasoning model for optimizer logic
    model_parameters=model_params,
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
    metric=emotion_accuracy_metric_func,
    n_samples=len(train_df),
    n_threads=args.n_threads
)

# Evaluate baseline on DEV data (what optimizer will use for scoring)
baseline_dev_score = baseline_optimizer.evaluate_prompt(
    prompt=initial_prompt,
    dataset=dev_dataset,
    metric=emotion_accuracy_metric_func,
    n_samples=len(dev_df),
    n_threads=args.n_threads
)

baseline_elapsed_time = time.time() - start_time

print("\n📈 Baseline Scores:")
print(f"   Train: {baseline_train_score:.4f} ({baseline_train_score*100:.1f}% accuracy)")
print(f"   Dev:   {baseline_dev_score:.4f} ({baseline_dev_score*100:.1f}% accuracy)")
print(f"   Evaluation time: {baseline_elapsed_time/60:.1f} minutes")

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
    f.write(f"Evaluation time: {baseline_elapsed_time/60:.1f} minutes\n")

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
        model=reasoning_model,          # Reasoning model for optimizer logic
        model_parameters=model_params,
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
        metric=emotion_accuracy_metric_func,
        n_samples=len(train_df),
        n_trials=args.n_trials
    )

    metaprompt_elapsed_time = time.time() - start_time

    # %% Display MetaPrompt results
    print("\n" + "=" * 80)
    print("✅ MetaPrompt Optimization Complete!")
    print("=" * 80)

    metaprompt_result.display()

    print("\n📊 Summary:")
    print(f"   Baseline Dev Score:  {baseline_dev_score:.4f}")
    print(f"   Optimized Dev Score: {metaprompt_result.score:.4f}")
    print(f"   Optimization time: {metaprompt_elapsed_time/60:.1f} minutes")

    if metaprompt_result.initial_score is not None:
        improvement = metaprompt_result.score - metaprompt_result.initial_score
        improvement_pct = (improvement / metaprompt_result.initial_score) * 100
        print(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")

    # %% Save MetaPrompt result
    output_file = save_optimizer_result_to_file(
        run_dir=RUN_DIR,
        optimizer_name="MetaPrompt",
        result=metaprompt_result,
        baseline_score=baseline_dev_score,
        elapsed_time=metaprompt_elapsed_time
    )
    print(f"\n💾 Saved optimized prompt to: {output_file}")
else:
    metaprompt_elapsed_time = None
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
        model=reasoning_model,          # Reasoning model for optimizer logic
        model_parameters=model_params,
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
        metric=emotion_accuracy_metric_func,
        n_samples=len(train_df),
        n_trials=args.n_trials
    )

    hierarchical_elapsed_time = time.time() - start_time

    # %% Display Hierarchical results
    print("\n" + "=" * 80)
    print("✅ Hierarchical Reflective Optimization Complete!")
    print("=" * 80)

    hierarchical_result.display()

    print("\n📊 Summary:")
    print(f"   Baseline Dev Score:  {baseline_dev_score:.4f}")
    print(f"   Optimized Dev Score: {hierarchical_result.score:.4f}")
    print(f"   Optimization time: {hierarchical_elapsed_time/60:.1f} minutes")

    if hierarchical_result.initial_score is not None:
        improvement = hierarchical_result.score - hierarchical_result.initial_score
        improvement_pct = (improvement / hierarchical_result.initial_score) * 100
        print(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")

    # %% Save Hierarchical result
    output_file = save_optimizer_result_to_file(
        run_dir=RUN_DIR,
        optimizer_name="Hierarchical Reflective",
        result=hierarchical_result,
        baseline_score=baseline_dev_score,
        elapsed_time=hierarchical_elapsed_time
    )
    print(f"\n💾 Saved optimized prompt to: {output_file}")
else:
    hierarchical_elapsed_time = None
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
        model=reasoning_model,          # Reasoning model for optimizer logic
        model_parameters=model_params,
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
        metric=emotion_accuracy_metric_func,
        n_samples=len(train_df),
        max_trials=args.n_trials
    )

    fewshot_elapsed_time = time.time() - start_time

    # %% Display Few-Shot results
    print("\n" + "=" * 80)
    print("✅ Few-Shot Bayesian Optimization Complete!")
    print("=" * 80)

    fewshot_result.display()

    print("\n📊 Summary:")
    print(f"   Baseline Dev Score:  {baseline_dev_score:.4f}")
    print(f"   Optimized Dev Score: {fewshot_result.score:.4f}")
    print(f"   Optimization time: {fewshot_elapsed_time/60:.1f} minutes")

    if fewshot_result.initial_score is not None:
        improvement = fewshot_result.score - fewshot_result.initial_score
        improvement_pct = (improvement / fewshot_result.initial_score) * 100
        print(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")

    # %% Save Few-Shot result
    output_file = save_optimizer_result_to_file(
        run_dir=RUN_DIR,
        optimizer_name="Few-Shot Bayesian",
        result=fewshot_result,
        baseline_score=baseline_dev_score,
        elapsed_time=fewshot_elapsed_time
    )
    print(f"\n💾 Saved optimized prompt to: {output_file}")
else:
    fewshot_elapsed_time = None
    print("\n⏭️  Skipping Few-Shot optimizer")

# %% [markdown]
"""
---
## Experiment 4: GEPA Optimizer

**How it works:**
1. Uses reflection to analyze evaluation results
2. Employs evolutionary algorithms to explore the prompt space
3. Combines both approaches for effective optimization

**Parameters:**
- Uses default GEPA parameters for initial testing
"""

# %% Run GEPA optimization
if 'gepa' in enabled_optimizers:
    print("\n" + "=" * 80)
    print("🧬 OPTIMIZATION 4: GEPA Optimizer")
    print("=" * 80)
    print("⏳ This may take several minutes...\n")

    gepa_optimizer = GepaOptimizer(
        model=reasoning_model,          # Reasoning model for optimizer logic
        model_parameters=model_params,
        n_threads=args.n_threads,
        verbose=1 if not args.quiet else 0,
        seed=42
    )

    start_time = time.time()

    gepa_result = gepa_optimizer.optimize_prompt(
        prompt=initial_prompt,
        dataset=train_dataset,              # For failure analysis
        validation_dataset=dev_dataset,     # For scoring candidates
        metric=emotion_accuracy_metric_func,
        n_samples=len(train_df),
        max_trials=args.n_trials
    )

    gepa_elapsed_time = time.time() - start_time

    # %% Display GEPA results
    print("\n" + "=" * 80)
    print("✅ GEPA Optimization Complete!")
    print("=" * 80)

    gepa_result.display()

    print("\n📊 Summary:")
    print(f"   Baseline Dev Score:  {baseline_dev_score:.4f}")
    print(f"   Optimized Dev Score: {gepa_result.score:.4f}")
    print(f"   Optimization time: {gepa_elapsed_time/60:.1f} minutes")

    if gepa_result.initial_score is not None:
        improvement = gepa_result.score - gepa_result.initial_score
        improvement_pct = (improvement / gepa_result.initial_score) * 100
        print(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")

    # %% Save GEPA result
    output_file = save_optimizer_result_to_file(
        run_dir=RUN_DIR,
        optimizer_name="GEPA",
        result=gepa_result,
        baseline_score=baseline_dev_score,
        elapsed_time=gepa_elapsed_time
    )
    print(f"\n💾 Saved optimized prompt to: {output_file}")
else:
    gepa_elapsed_time = None
    print("\n⏭️  Skipping GEPA optimizer")

# %% [markdown]
"""
---
## Experiment 5: Evolutionary Optimizer

**How it works:**
1. Uses genetic algorithms to evolve prompt populations
2. Enables discovery of novel prompt structures
3. Supports multi-objective optimization (e.g., score vs. length)

**Parameters:**
- Uses default Evolutionary parameters for initial testing
"""

# %% Run Evolutionary optimization
if 'evolutionary' in enabled_optimizers:
    print("\n" + "=" * 80)
    print("🧬 OPTIMIZATION 5: Evolutionary Optimizer")
    print("=" * 80)
    print("⏳ This may take several minutes...\n")

    evolutionary_optimizer = EvolutionaryOptimizer(
        model=reasoning_model,          # Reasoning model for optimizer logic
        model_parameters=model_params,
        n_threads=args.n_threads,
        verbose=1 if not args.quiet else 0,
        seed=42
    )

    start_time = time.time()

    evolutionary_result = evolutionary_optimizer.optimize_prompt(
        prompt=initial_prompt,
        dataset=train_dataset,              # For failure analysis
        validation_dataset=dev_dataset,     # For scoring candidates ⭐
        metric=emotion_accuracy_metric_func,
        n_samples=len(train_df),
        max_trials=args.n_trials
    )

    evolutionary_elapsed_time = time.time() - start_time

    # %% Display Evolutionary results
    print("\n" + "=" * 80)
    print("✅ Evolutionary Optimization Complete!")
    print("=" * 80)

    evolutionary_result.display()

    print("\n📊 Summary:")
    print(f"   Baseline Dev Score:  {baseline_dev_score:.4f}")
    print(f"   Optimized Dev Score: {evolutionary_result.score:.4f}")
    print(f"   Optimization time: {evolutionary_elapsed_time/60:.1f} minutes")

    if evolutionary_result.initial_score is not None:
        improvement = evolutionary_result.score - evolutionary_result.initial_score
        improvement_pct = (improvement / evolutionary_result.initial_score) * 100
        print(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")

    # %% Save Evolutionary result
    output_file = save_optimizer_result_to_file(
        run_dir=RUN_DIR,
        optimizer_name="Evolutionary",
        result=evolutionary_result,
        baseline_score=baseline_dev_score,
        elapsed_time=evolutionary_elapsed_time
    )
    print(f"\n💾 Saved optimized prompt to: {output_file}")
else:
    evolutionary_elapsed_time = None
    print("\n⏭️  Skipping Evolutionary optimizer")

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

test_eval_start_time = time.time()

# Evaluate baseline on test set
print("Evaluating baseline on test set...")
baseline_test_score = baseline_optimizer.evaluate_prompt(
    prompt=initial_prompt,
    dataset=test_dataset,  # ⚠️ Use test set!
    metric=emotion_accuracy_metric_func,
    n_samples=len(test_df),
    n_threads=args.n_threads
)
print(f"✅ Baseline test score: {baseline_test_score:.4f}")

# Evaluate each optimizer's result on test set
metaprompt_test_score = None
hierarchical_test_score = None
fewshot_test_score = None
gepa_test_score = None
evolutionary_test_score = None

if metaprompt_result is not None:
    print("\nEvaluating MetaPrompt on test set...")
    metaprompt_test_prompt = ChatPrompt(
        messages=metaprompt_result.prompt,
        model=task_model,
        model_parameters=model_params
    )
    metaprompt_test_score = baseline_optimizer.evaluate_prompt(
        prompt=metaprompt_test_prompt,
        dataset=test_dataset,
        metric=emotion_accuracy_metric_func,
        n_samples=len(test_df),
        n_threads=args.n_threads
    )
    print(f"✅ MetaPrompt test score: {metaprompt_test_score:.4f}")

if hierarchical_result is not None:
    print("\nEvaluating Hierarchical on test set...")
    hierarchical_test_prompt = ChatPrompt(
        messages=hierarchical_result.prompt,
        model=task_model,
        model_parameters=model_params
    )
    hierarchical_test_score = baseline_optimizer.evaluate_prompt(
        prompt=hierarchical_test_prompt,
        dataset=test_dataset,
        metric=emotion_accuracy_metric_func,
        n_samples=len(test_df),
        n_threads=args.n_threads
    )
    print(f"✅ Hierarchical test score: {hierarchical_test_score:.4f}")

if fewshot_result is not None:
    print("\nEvaluating FewShot on test set...")
    fewshot_test_prompt = ChatPrompt(
        messages=fewshot_result.prompt,
        model=task_model,
        model_parameters=model_params
    )
    fewshot_test_score = baseline_optimizer.evaluate_prompt(
        prompt=fewshot_test_prompt,
        dataset=test_dataset,
        metric=emotion_accuracy_metric_func,
        n_samples=len(test_df),
        n_threads=args.n_threads
    )
    print(f"✅ FewShot test score: {fewshot_test_score:.4f}")

if gepa_result is not None:
    print("\nEvaluating GEPA on test set...")
    gepa_test_prompt = ChatPrompt(
        messages=gepa_result.prompt,
        model=task_model,
        model_parameters=model_params
    )
    gepa_test_score = baseline_optimizer.evaluate_prompt(
        prompt=gepa_test_prompt,
        dataset=test_dataset,
        metric=emotion_accuracy_metric_func,
        n_samples=len(test_df),
        n_threads=args.n_threads
    )
    print(f"✅ GEPA test score: {gepa_test_score:.4f}")

if evolutionary_result is not None:
    print("\nEvaluating Evolutionary on test set...")
    evolutionary_test_prompt = ChatPrompt(
        messages=evolutionary_result.prompt,
        model=task_model,
        model_parameters=model_params
    )
    evolutionary_test_score = baseline_optimizer.evaluate_prompt(
        prompt=evolutionary_test_prompt,
        dataset=test_dataset,
        metric=emotion_accuracy_metric_func,
        n_samples=len(test_df),
        n_threads=args.n_threads
    )
    print(f"✅ Evolutionary test score: {evolutionary_test_score:.4f}")

test_eval_elapsed_time = time.time() - test_eval_start_time
print(f"\n⏱️  Test evaluation time: {test_eval_elapsed_time/60:.1f} minutes")

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
    "Dev Score": baseline_dev_score,
    "Test Score": baseline_test_score,
    "Dev Improvement": "—",
    "Test Improvement": "—",
    "Output File": "—"
}]

if metaprompt_result is not None:
    dev_imp = calc_improvement(metaprompt_result, baseline_dev_score)
    test_imp = ((metaprompt_test_score - baseline_test_score) / baseline_test_score) * 100 if baseline_test_score != 0 else 0.0
    results_data.append({
        "Optimizer": "MetaPrompt",
        "Dev Score": metaprompt_result.score,
        "Test Score": metaprompt_test_score,
        "Dev Improvement": f"{dev_imp:+.1f}%",
        "Test Improvement": f"{test_imp:+.1f}%",
        "Output File": "optimized-metaprompt-messages.txt"
    })

if hierarchical_result is not None:
    dev_imp = calc_improvement(hierarchical_result, baseline_dev_score)
    test_imp = ((hierarchical_test_score - baseline_test_score) / baseline_test_score) * 100 if baseline_test_score != 0 else 0.0
    results_data.append({
        "Optimizer": "Hierarchical",
        "Dev Score": hierarchical_result.score,
        "Test Score": hierarchical_test_score,
        "Dev Improvement": f"{dev_imp:+.1f}%",
        "Test Improvement": f"{test_imp:+.1f}%",
        "Output File": "optimized-hierarchical-messages.txt"
    })

if fewshot_result is not None:
    dev_imp = calc_improvement(fewshot_result, baseline_dev_score)
    test_imp = ((fewshot_test_score - baseline_test_score) / baseline_test_score) * 100 if baseline_test_score != 0 else 0.0
    results_data.append({
        "Optimizer": "Few-Shot",
        "Dev Score": fewshot_result.score,
        "Test Score": fewshot_test_score,
        "Dev Improvement": f"{dev_imp:+.1f}%",
        "Test Improvement": f"{test_imp:+.1f}%",
        "Output File": "optimized-fewshot-messages.txt"
    })

if gepa_result is not None:
    dev_imp = calc_improvement(gepa_result, baseline_dev_score)
    test_imp = ((gepa_test_score - baseline_test_score) / baseline_test_score) * 100 if baseline_test_score != 0 else 0.0
    results_data.append({
        "Optimizer": "GEPA",
        "Dev Score": gepa_result.score,
        "Test Score": gepa_test_score,
        "Dev Improvement": f"{dev_imp:+.1f}%",
        "Test Improvement": f"{test_imp:+.1f}%",
        "Output File": "optimized-gepa-messages.txt"
    })

if evolutionary_result is not None:
    dev_imp = calc_improvement(evolutionary_result, baseline_dev_score)
    test_imp = ((evolutionary_test_score - baseline_test_score) / baseline_test_score) * 100 if baseline_test_score != 0 else 0.0
    results_data.append({
        "Optimizer": "Evolutionary",
        "Dev Score": evolutionary_result.score,
        "Test Score": evolutionary_test_score,
        "Dev Improvement": f"{dev_imp:+.1f}%",
        "Test Improvement": f"{test_imp:+.1f}%",
        "Output File": "optimized-evolutionary-messages.txt"
    })

results_df = pd.DataFrame(results_data)

print("\n" + "=" * 80)
print("📊 FINAL RESULTS COMPARISON (DEV vs TEST)")
print("=" * 80)
print()
print(results_df.to_string(index=False))

# Find winner based on TEST score (excluding baseline)
test_scores = results_df[results_df["Optimizer"] != "Baseline"]["Test Score"]
if len(test_scores) > 0:
    best_idx = test_scores.idxmax()
    winner = results_df.loc[best_idx, "Optimizer"]
    winner_test_score = results_df.loc[best_idx, "Test Score"]
    winner_dev_score = results_df.loc[best_idx, "Dev Score"]

    print(f"\n🏆 WINNER (by test score): {winner}")
    print(f"   Test Score: {winner_test_score:.4f}")
    print(f"   Dev Score: {winner_dev_score:.4f}")

    # Check for overfitting
    if winner_dev_score > winner_test_score + 0.02:
        print("   ⚠️  WARNING: Possible overfitting detected!")
        print(f"   Dev score is {winner_dev_score - winner_test_score:.4f} higher than test")
else:
    winner = "N/A"
    winner_test_score = 0.0
print(f"\n📁 All results saved to: {RUN_DIR}/")

# Save comparison table
with open(f"{RUN_DIR}/comparison_table.txt", "w") as f:
    f.write("FINAL RESULTS COMPARISON (DEV vs TEST)\n")
    f.write("=" * 80 + "\n\n")
    f.write(results_df.to_string(index=False))
    if len(test_scores) > 0:
        f.write(f"\n\n🏆 WINNER (by test score): {winner}")
        f.write(f"\n   Test Score: {winner_test_score:.4f}")
        f.write(f"\n   Dev Score: {winner_dev_score:.4f}\n")

# %% Save JSON summary (always created for easy parsing/testing)
summary = {
    "test_config": {
        "sample_size": args.sample_size,
        "n_trials": args.n_trials,
        "n_threads": args.n_threads,
        "reasoning_model": reasoning_model,
        "task_model": task_model,
        "model_parameters": model_params,
        "timestamp": os.path.basename(RUN_DIR),
        "enabled_optimizers": list(enabled_optimizers)
    },
    "baseline": {
        "dev_score": float(baseline_dev_score),
        "test_score": float(baseline_test_score)
    },
    "optimizers": {},
    "status": "success"
}

# Add optimizer results
optimizer_map = [
    ('MetaPrompt', metaprompt_result, metaprompt_test_score),
    ('Hierarchical', hierarchical_result, hierarchical_test_score),
    ('FewShot', fewshot_result, fewshot_test_score),
    ('GEPA', gepa_result, gepa_test_score),
    ('Evolutionary', evolutionary_result, evolutionary_test_score)
]

for name, result_obj, test_score in optimizer_map:
    if result_obj is not None and test_score is not None:
        dev_imp = ((result_obj.score - baseline_dev_score) / baseline_dev_score) * 100 if baseline_dev_score != 0 else 0.0
        test_imp = ((test_score - baseline_test_score) / baseline_test_score) * 100 if baseline_test_score != 0 else 0.0
        summary["optimizers"][name] = {
            "dev_score": float(result_obj.score),
            "test_score": float(test_score),
            "dev_improvement_pct": float(dev_imp),
            "test_improvement_pct": float(test_imp)
        }

# Find winner based on TEST score
if summary["optimizers"]:
    winner_name = max(summary["optimizers"].keys(),
                     key=lambda k: summary["optimizers"][k]["test_score"])
    summary["winner"] = {
        "name": winner_name,
        "dev_score": summary["optimizers"][winner_name]["dev_score"],
        "test_score": summary["optimizers"][winner_name]["test_score"]
    }

# Save JSON
json_path = f"{RUN_DIR}/results_summary.json"
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

if not args.quiet:
    print(f"\n💾 JSON summary saved to: {json_path}")

# %% Timings table
# Build timings data
timings_data = [{"Stage": "Baseline Evaluation", "Time (min)": f"{baseline_elapsed_time/60:.1f}"}]

if metaprompt_elapsed_time is not None:
    timings_data.append({"Stage": "MetaPrompt Optimizer", "Time (min)": f"{metaprompt_elapsed_time/60:.1f}"})
if hierarchical_elapsed_time is not None:
    timings_data.append({"Stage": "Hierarchical Optimizer", "Time (min)": f"{hierarchical_elapsed_time/60:.1f}"})
if fewshot_elapsed_time is not None:
    timings_data.append({"Stage": "Few-Shot Optimizer", "Time (min)": f"{fewshot_elapsed_time/60:.1f}"})
if gepa_elapsed_time is not None:
    timings_data.append({"Stage": "GEPA Optimizer", "Time (min)": f"{gepa_elapsed_time/60:.1f}"})
if evolutionary_elapsed_time is not None:
    timings_data.append({"Stage": "Evolutionary Optimizer", "Time (min)": f"{evolutionary_elapsed_time/60:.1f}"})

timings_data.append({"Stage": "Test Set Evaluation", "Time (min)": f"{test_eval_elapsed_time/60:.1f}"})

# Calculate total time
total_time = baseline_elapsed_time + test_eval_elapsed_time
if metaprompt_elapsed_time is not None:
    total_time += metaprompt_elapsed_time
if hierarchical_elapsed_time is not None:
    total_time += hierarchical_elapsed_time
if fewshot_elapsed_time is not None:
    total_time += fewshot_elapsed_time
if gepa_elapsed_time is not None:
    total_time += gepa_elapsed_time
if evolutionary_elapsed_time is not None:
    total_time += evolutionary_elapsed_time

timings_data.append({"Stage": "TOTAL", "Time (min)": f"{total_time/60:.1f}"})

timings_df = pd.DataFrame(timings_data)

print("\n" + "=" * 80)
print("⏱️  TIMINGS")
print("=" * 80)
print()
print(timings_df.to_string(index=False))

# Save timings to JSON summary
summary["timings"] = {
    "baseline_minutes": round(baseline_elapsed_time/60, 1),
    "metaprompt_minutes": round(metaprompt_elapsed_time/60, 1) if metaprompt_elapsed_time else None,
    "hierarchical_minutes": round(hierarchical_elapsed_time/60, 1) if hierarchical_elapsed_time else None,
    "fewshot_minutes": round(fewshot_elapsed_time/60, 1) if fewshot_elapsed_time else None,
    "gepa_minutes": round(gepa_elapsed_time/60, 1) if gepa_elapsed_time else None,
    "evolutionary_minutes": round(evolutionary_elapsed_time/60, 1) if evolutionary_elapsed_time else None,
    "test_eval_minutes": round(test_eval_elapsed_time/60, 1),
    "total_minutes": round(total_time/60, 1)
}

# Re-save JSON with timings
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

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
