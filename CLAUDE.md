# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a prompt optimization workshop that teaches how to optimize AI prompts using Opik's agent optimizer. The project classifies social media text into four emotions (joy, anger, sadness, surprise) using five different optimization algorithms.

## Commands

### Run the optimizer
```bash
# Quick test (recommended for first run)
python optimize.py --sample-size 30 --n-trials 1

# Run specific optimizer only
python optimize.py --sample-size 30 --optimizers metaprompt

# Full run with all optimizers
python optimize.py
```

### Run tests
```bash
# Unit tests
pytest test_utils.py -v

# Unit tests with coverage
pytest test_utils.py -v --cov=utils

# End-to-end test (uses real APIs, ~5 min)
pytest test_optimize_e2e.py -v -s

# Skip expensive E2E tests
pytest -m "not e2e" -v
```

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Architecture

### Core Files
- `optimize.py` - Main optimization script with CLI. Uses VS Code cell markers (`# %%`) for interactive execution. Configures five optimizers (MetaPrompt, Hierarchical Reflective, Few-Shot Bayesian, GEPA, Evolutionary), runs baseline evaluation, then optimization, and compares results.
- `utils.py` - Reusable utility functions for emotion extraction, dataset loading with stratification, template loading, and output directory management.

### Data Flow
1. Load emotion classification CSV with stratified train/dev/test split (40/40/20 by default)
2. Create Opik datasets from pandas DataFrames
3. Run baseline evaluation on train and dev sets
4. Run selected optimizers (each uses train for failure analysis, dev for scoring)
5. Evaluate final prompts on held-out test set
6. Output comparison table and JSON summary to timestamped `runs/` directory

### Key Integration Points
- **Opik**: Connects to Comet for experiment tracking. Uses `opik.configure()` and `Opik(project_name=...)`.
- **LiteLLM**: Model specification format is `provider/model` (e.g., `openai/gpt-4o-mini`).
- **Custom Metric**: `EmotionAccuracyMetric` extracts emotions from LLM output and compares to expected emotion (exact match: 1.0 if correct, 0.0 if wrong).

### Optimizer Pattern
All five optimizers follow the same pattern:
```python
optimizer = SomeOptimizer(params...)
result = optimizer.optimize_prompt(
    prompt=initial_prompt,
    dataset=train_dataset,           # For failure analysis
    validation_dataset=dev_dataset,  # For scoring candidates
    metric=emotion_accuracy_metric_func,
    n_samples=len(train_df),
    n_trials=N
)
```

## Model Notes

Two models are used during optimization:
- **Reasoning model** (`--reasoning-model`): Used by the optimizer for generating candidates, mutations, analysis. Default: `openai/gpt-4o`
- **Task model** (`--task-model`): Used to evaluate prompts on the dataset. Default: `openai/gpt-4o-mini`

Model format is LiteLLM: `provider/model`
- OpenAI: `openai/gpt-4o`, `openai/gpt-4o-mini`
- Anthropic: `anthropic/claude-3-5-sonnet-20241022`

## Environment Variables

Required in `.env`:
- `OPIK_API_KEY` - From comet.com (formerly COMET_API_KEY)
- `OPENAI_API_KEY` or `GEMINI_API_KEY` depending on model
