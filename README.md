# Grading Rubric Prompt Optimization Workshop

Welcome to the prompt optimization workshop! Learn how to use Opik's agent optimizer to improve AI prompts through hands-on experimentation.

## 🎯 What You'll Learn

- How to optimize prompts using three different algorithms (MetaPrompt, Hierarchical Reflective, Few-Shot Bayesian)
- How to create custom evaluation metrics
- How to work with evaluation datasets and stratified splits
- How to measure baseline performance before optimization
- Reusable patterns for your own projects

## 📚 Prerequisites

- Python 3.8 or higher
- Basic Python knowledge
- OpenAI API key or Gemini API key 
- Comet API key ([Free signup](https://www.comet.com/signup))

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Navigate to project directory
cd opik-demo

# Create virtual environment and activate it
python -m venv .venv
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project directory:

```env
OPIK_API_KEY=your_comet_key_here
OPENAI_API_KEY=your_openai_key_here
# or
GEMINI_API_KEY=your_gemini_key_here
```

### 3. Configure Your Run

Edit the configuration section at the top of `optimize.py`:

```python
# Configuration
SAMPLE_SIZE = 30  # Start wit 30 for quick testing (12 train, 12 dev, 6 test)
                   # Set to None for full dataset
N_TRIALS = 1      # Number of optimization rounds - set to 5 for a real run
N_THREADS = 4     # Parallel threads for speed
```

⚠️  **Important:** Start with `SAMPLE_SIZE=30` and `N_TRIALS=1`
for your first run to verify everything works! Full dataset optimization with 5 trials can take over and hour and cost several dollars in API calls if using OpenAI.

### 4. Run the Workshop

**Option A: Interactive (Recommended)**
Open `optimize.py` in VS Code and run cells interactively (Shift+Enter):
- Run cells one at a time to understand each step
- Perfect for learning and experimentation

**Option B: Full Script**
Run the entire script from command line:
```bash
python optimize.py
```

## 🎛️ CLI Usage

`optimize.py` supports command-line arguments for flexible configuration:

### Quick Test (Recommended for first run)
```bash
python optimize.py --sample-size 30 --n-trials 1
```

### Run Specific Optimizers (Fast!)
```bash
# Run only MetaPrompt (~5-10 min instead of 15-30 min)
python optimize.py --sample-size 30 --optimizers metaprompt

# Run two optimizers
python optimize.py --optimizers metaprompt,hierarchical
```

### Full Test
```bash
# Run all three optimizers with full dataset (over an hour)
python optimize.py
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample-size N` | None | Number of samples (None = full) |
| `--n-trials N` | 5 | Optimization trials per optimizer |
| `--n-threads N` | 4 | Parallel threads for evaluation |
| `--model MODEL` | openai/responses/gpt-5-mini | LLM model (LiteLLM format) |
| `--optimizers LIST` | all | Comma-separated optimizer list |
| `--output-dir DIR` | runs | Output directory |
| `--split-ratio RATIO` | 40/40/20 | Train/Dev/Test split ratio |
| `--reasoning-effort LEVEL` | low | Reasoning effort (low/medium/high/xhigh) |
| `--verbosity LEVEL` | low | Output verbosity (low/medium/high) |
| `--max-output-tokens N` | 65536 | Max output tokens for Responses API |
| `--quiet` | False | Suppress verbose output |

**Note**: JSON summary (`results_summary.json`) is automatically created in every run for easy verification.

## 📁 Project Structure

```
opik-demo/
├── optimize.py                    # Main workshop script (interactive cells)
├── utils.py                       # Reusable utility functions
├── grading-rubric-prompt.txt      # Prompt template (Python format strings)
├── grading-rubric.md              # Grading rubric documentation
├── answer-evaluation.csv          # Evaluation dataset (153 examples)
├── test_utils.py                  # Unit tests
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── .env                           # API keys (you create this)
└── runs/                          # Output directory (auto-created)
    └── YYYY-MM-DD_HH-MM-SS/       # Timestamped run folder
        ├── baseline_score.txt
        ├── optimized-metaprompt-messages.txt
        ├── optimized-hierarchical-messages.txt
        ├── optimized-fewshot-messages.txt
        ├── comparison_table.txt
        └── results_summary.json
```

## 🧪 Run Tests

### Unit Tests
Verify utilities work correctly:

```bash
# Run unit tests with coverage
pytest test_utils.py -v --cov=utils

# Run specific test class
pytest test_utils.py::TestScoreExtraction -v
```

### End-to-End Test
Test the full optimize.py script with real API calls:

```bash
# Run the E2E test (uses real APIs, ~5-10 min, ~$1-2)
pytest test_optimize_e2e.py -v -s

# Run only unit tests (skip expensive E2E)
pytest -m "not e2e" -v
```

**Note**: The E2E test requires API keys and makes real API calls. It validates the entire optimization workflow with a small dataset (10 samples, 1 trial, all 3 optimizers).

## 📖 Understanding the Code

- **Dataset**: 153 examples with human scores (1-5), split 40% train, 40% dev, 20% test with stratification
- **Metric**: Custom `ScoreAccuracyMetric` compares LLM scores to human scores (formula: `1 - abs(diff) / 4.0`)
- **Optimizers**: MetaPrompt (iterative critique), Hierarchical (failure analysis), Few-Shot (Bayesian search)
- **Pattern**: All optimizers follow: create → evaluate baseline → optimize → display results

## 🔧 Reusable Functions

All functions in `utils.py` can be copied to your own projects:

- `extract_score_from_text()` - Extract scores from LLM responses
- `load_csv_with_stratified_split()` - Load and split datasets
- `load_text_template()` - Load text templates
- `create_timestamped_run_dir()` - Create timestamped output directories
- `save_optimizer_result_to_file()` - Save optimization results to formatted text files

See the docstrings for usage examples!

## 📊 Understanding Results

After running, you'll get:

### Output Files

All outputs are saved in `runs/YYYY-MM-DD_HH-MM-SS/`:
- `baseline_score.txt` - Initial prompt performance
- `optimized-metaprompt-messages.txt` - MetaPrompt results
- `optimized-hierarchical-messages.txt` - Hierarchical results
- `optimized-fewshot-messages.txt` - Few-Shot results
- `comparison_table.txt` - Side-by-side comparison

### Comparison Table

```
Optimizer      Score  Improvement  Output File
Baseline       0.65   —            —
MetaPrompt     0.78   +20.0%       optimized-metaprompt-messages.txt
Hierarchical   0.82   +26.2%       optimized-hierarchical-messages.txt
Few-Shot       0.75   +15.4%       optimized-fewshot-messages.txt

🏆 WINNER: Hierarchical with score 0.82
```

### Opik Dashboard

View detailed analysis at [comet.com/opik](https://www.comet.com/opik):
- Trace logs for each evaluation
- Score trends over optimization rounds
- Prompt evolution history
- Individual example predictions

## 💡 Best Practices

### For Learning
1. **Start small**: Use `SAMPLE_SIZE=15` for quick iterations
2. **Run cells interactively**: Understand each step before proceeding
3. **Examine outputs**: Look at optimized prompts to see what changed
4. **Compare metrics**: Try different scoring approaches

### For Production
1. **Use full dataset**: Set `SAMPLE_SIZE=None` for final optimization
2. **Increase trials**: Set `N_TRIALS=5` or higher for better results
3. **Test on holdout set**: Evaluate final prompt on `test_dataset`
4. **Monitor costs**: Track API usage in OpenAI dashboard

### Cost Management

| Configuration | Optimizers | Samples | API Calls | Time | Est. Cost |
|--------------|------------|---------|-----------|------|-----------|
| Quick test | 1 | 30 | ~50 | 2-5 min | $0.50-$1 |
| Medium | 1 | 45 | ~300 | 8-15 min | $2-$5 |
| Full single | 1 | None | ~1,500 | 15-30 min | $8-$20 |
| Full all | 3 | None | ~4,500 | 45-90 min | $20-$50 |

## 🆘 Troubleshooting

Having issues? See **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** for detailed solutions to common problems.

## 📚 Learn More

- [Opik Documentation](https://www.comet.com/docs/opik)
- [Agent Optimization Guide](https://www.comet.com/docs/opik/agent_optimization/overview)
- [Custom Metrics Guide](https://www.comet.com/docs/opik/evaluation/metrics/custom_metric)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [DSPy Optimization Guide](https://dspy-docs.vercel.app/)

## 🎓 Next Steps

- **Experiment**: Try different sample sizes, models, and optimizer combinations
- **Adapt**: Use the patterns in `utils.py` to optimize your own prompts
- **Production**: Deploy the best prompt and monitor performance over time

## 🤝 Contributing

This is an educational project for Pioneer AI Academy students. Feel free to:
- Ask questions about the code
- Suggest improvements
- Share your results
- Adapt for your own projects

---

Happy optimizing! 🚀

**Questions?** Reach out to your instructor or check the Opik documentation.
