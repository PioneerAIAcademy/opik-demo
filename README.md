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
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Comet API key ([Free signup](https://www.comet.com/signup))

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Navigate to project directory
cd /home/dallan/pioneeraiacademy/opik-demo

# Create virtual environment
python -m venv .venv

# Install packages
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project directory:

```env
COMET_API_KEY=your_comet_key_here
OPENAI_API_KEY=your_openai_key_here
```

### 3. Configure Your Run

Edit the configuration section at the top of `optimize.py`:

```python
# Configuration
SAMPLE_SIZE = 15  # Start with 15 for quick testing (10 train, 5 test)
                   # Set to None for full dataset (102 train, 51 test)
N_TRIALS = 3      # Number of optimization rounds
N_THREADS = 8     # Parallel threads for speed
```

⚠️  **Important:** Start with `SAMPLE_SIZE=15` for your first run to verify everything works! Full dataset optimization can take 45-90 minutes and cost $20-50 in API calls.

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

`optimize.py` now supports command-line arguments for flexible configuration:

### Quick Test (Recommended for first run)
```bash
python optimize.py --sample-size 10 --n-trials 1
```

### Specific Optimizers
```bash
python optimize.py --optimizers metaprompt,hierarchical
```

### Full Dataset
```bash
python optimize.py  # Uses all ~153 samples
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample-size N` | None | Number of samples (None = full ~153) |
| `--n-trials N` | 3 | Optimization trials per optimizer |
| `--n-threads N` | 8 | Parallel threads for evaluation |
| `--model MODEL` | openai/gpt-5-mini | LLM model (LiteLLM format) |
| `--optimizers LIST` | all | Comma-separated optimizer list |
| `--output-dir DIR` | runs | Output directory |
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
        └── comparison_table.txt
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

### The Dataset

- **153 examples** of AI-generated answers with human scores (1-5)
- **Columns**: question, human_answer, ai_answer, retrieved_content, score
- **Dynamic Split**: 2/3 train, 1/3 test based on sample size
  - Full dataset (153): 102 train, 51 test
  - Sample (15): 10 train, 5 test
- **Stratified**: Maintains score distribution in both sets

### The Metric

Our custom `ScoreAccuracyMetric` compares LLM scores to human scores:

```python
# Perfect match (LLM=4.0, Human=4.0) → accuracy = 1.00
# Off by 0.5 (LLM=4.0, Human=4.5) → accuracy = 0.875
# Off by 1.0 (LLM=4.0, Human=5.0) → accuracy = 0.75
# Formula: 1 - (abs(difference) / 4.0)
```

### The Optimizers

1. **MetaPrompt** - Uses LLM to iteratively critique and improve the prompt
2. **Hierarchical Reflective** - Analyzes failure patterns systematically
3. **Few-Shot Bayesian** - Optimizes demonstration examples using Bayesian search

### The Pattern

All optimizers follow the same pattern from OpikOptimizerIntro.ipynb:

```python
# 1. Create optimizer
optimizer = MetaPromptOptimizer(model="openai/gpt-5-mini", ...)

# 2. Evaluate baseline
baseline_score = optimizer.evaluate_prompt(prompt, dataset, metric, ...)

# 3. Optimize
result = optimizer.optimize_prompt(prompt, dataset, metric, ...)

# 4. Display results
result.display()
```

## 🔧 Reusable Functions

All functions in `utils.py` can be copied to your own projects:

- `extract_score_from_text()` - Extract scores from LLM responses
- `load_csv_with_stratified_split()` - Load and split datasets
- `load_text_template()` - Load text templates
- `create_timestamped_run_dir()` - Create timestamped output directories

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

| Configuration | Samples | API Calls | Time | Est. Cost |
|--------------|---------|-----------|------|-----------|
| Quick test | 15 | ~150 | 5-10 min | $1-2 |
| Medium | 45 | ~1,000 | 20-30 min | $5-10 |
| Full dataset | 153 | ~4,500 | 45-90 min | $20-50 |

## 🆘 Troubleshooting

### Error: "COMET_API_KEY not found"
- Make sure `.env` file exists with your API keys
- Check that `python-dotenv` is installed
- Verify file is in the same directory as `optimize.py`

### Error: "Dataset has only X rows"
- The CSV should have 153 rows
- Check that `answer-evaluation.csv` exists and is not corrupted
- Try reading it manually: `pd.read_csv("answer-evaluation.csv")`

### Optimization is slow
- This is normal - each optimizer tests many variations
- Can take 5-15 minutes per optimizer with small sample
- Reduce `SAMPLE_SIZE` for faster testing (but less accurate results)
- Increase `N_THREADS` if you have more CPU cores

### Errors about model parameters
- Ensure you're using a reasoning model (gpt-5-mini)
- Check that your OpenAI API key has access to the model
- Try removing model_parameters if model doesn't support reasoning

### High API costs
- Start with `SAMPLE_SIZE=15` to test
- Monitor usage in OpenAI dashboard
- Consider using cheaper models for initial experiments

### Score extraction fails
- Check that LLM output includes "**Score:** X" format
- Update patterns in `extract_score_from_text()` if needed
- Add debug prints to see raw LLM outputs

## 📚 Learn More

- [Opik Documentation](https://www.comet.com/docs/opik)
- [Agent Optimization Guide](https://www.comet.com/docs/opik/agent_optimization/overview)
- [Custom Metrics Guide](https://www.comet.com/docs/opik/evaluation/metrics/custom_metric)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [DSPy Optimization Guide](https://dspy-docs.vercel.app/)

## 🎓 Next Steps

1. **Experiment with configuration**
   - Try different sample sizes
   - Adjust number of trials
   - Test different models

2. **Optimize your own prompts**
   - Replace the grading rubric with your prompt
   - Create a dataset for your use case
   - Define custom metrics

3. **Compare optimizers**
   - Which works best for your task?
   - How do results change with more data?
   - What's the cost/benefit tradeoff?

4. **Use in production**
   - Deploy the best optimized prompt
   - Monitor performance over time
   - Re-optimize periodically with new data

## 🤝 Contributing

This is an educational project for Pioneer AI Academy students. Feel free to:
- Ask questions about the code
- Suggest improvements
- Share your results
- Adapt for your own projects

## 📄 License

This workshop material is for educational purposes. The reusable utility functions in `utils.py` are free to use in your own projects.

---

Happy optimizing! 🚀

**Questions?** Reach out to your instructor or check the Opik documentation.
