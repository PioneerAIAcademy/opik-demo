# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a prompt optimization workshop that teaches how to optimize AI prompts using Opik's agent optimizer. The project evaluates chatbot-generated answers against human scores using three different optimization algorithms.

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

# End-to-end test (uses real APIs, ~5-10 min, ~$1-2)
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
- `optimize.py` - Main optimization script with CLI. Uses VS Code cell markers (`# %%`) for interactive execution. Configures three optimizers (MetaPrompt, Hierarchical Reflective, Few-Shot Bayesian), runs baseline evaluation, then optimization, and compares results.
- `utils.py` - Reusable utility functions for score extraction, dataset loading with stratification, template loading, and output directory management.

### Data Flow
1. Load evaluation CSV with stratified train/dev/test split (40/40/20 by default)
2. Create Opik datasets from pandas DataFrames
3. Run baseline evaluation on train and dev sets
4. Run selected optimizers (each uses train for failure analysis, dev for scoring)
5. Evaluate final prompts on held-out test set
6. Output comparison table and JSON summary to timestamped `runs/` directory

### Key Integration Points
- **Opik**: Connects to Comet for experiment tracking. Uses `opik.configure()` and `Opik(project_name=...)`.
- **LiteLLM**: Model specification format is `provider/endpoint/model` (e.g., `openai/responses/gpt-5-mini`).
- **Custom Metric**: `ScoreAccuracyMetric` extracts scores from LLM output via regex patterns and compares to human scores using formula `1 - abs(diff) / 4.0`.

### Optimizer Pattern
All three optimizers follow the same pattern:
```python
optimizer = SomeOptimizer(params...)
result = optimizer.optimize_prompt(
    prompt=initial_prompt,
    dataset=train_dataset,           # For failure analysis
    validation_dataset=dev_dataset,  # For scoring candidates
    metric=score_accuracy_metric_func,
    n_samples=len(train_df),
    n_trials=N
)
```

## Model Notes

- GPT-5 models are reasoning models requiring OpenAI's Responses API
- Use `openai/responses/gpt-5-mini` format in LiteLLM for reasoning models
- This requires `fastapi` and `orjson` packages (don't use `litellm[proxy]` - it has version conflicts with opik)
- Standard models (GPT-4o, etc.) use `openai/gpt-4o` format without the `responses` endpoint

## Environment Variables

Required in `.env`:
- `OPIK_API_KEY` - From comet.com (formerly COMET_API_KEY)
- `OPENAI_API_KEY` or `GEMINI_API_KEY` depending on model
