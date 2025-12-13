# Troubleshooting Guide

Common issues and solutions for the Grading Rubric Prompt Optimization Workshop.

## Error: "COMET_API_KEY not found"

**Cause:** Missing or incorrectly configured API key

**Solution:**
- Make sure `.env` file exists with your API keys
- Check that `python-dotenv` is installed
- Verify file is in the same directory as `optimize.py`

## Error: "Dataset has only X rows"

**Cause:** Dataset file is missing or corrupted

**Solution:**
- The CSV should have 153 rows
- Check that `answer-evaluation.csv` exists and is not corrupted
- Try reading it manually: `pd.read_csv("answer-evaluation.csv")`

## Optimization is slow

**Cause:** This is normal behavior - each optimizer tests many variations

**What to expect:**
- Can take 5-15 minutes per optimizer with small sample
- Full dataset takes 45-90 minutes for all three optimizers

**Solutions:**
- Reduce `--sample-size` for faster testing (but less accurate results)
- Increase `--n-threads` if you have more CPU cores
- Run only one optimizer with `--optimizers metaprompt`

## Errors about model parameters

**Cause:** Model doesn't support the reasoning parameters

**Solutions:**
- Ensure you're using a reasoning model (gpt-5-mini)
- Check that your OpenAI API key has access to the model
- Try removing model_parameters if model doesn't support reasoning
- Use a different model: `--model openai/gpt-4o`

## High API costs

**Cause:** Running expensive configurations

**Solutions:**
- Start with `--sample-size 15` to test (~$1-2)
- Monitor usage in OpenAI dashboard
- Consider using cheaper models for initial experiments
- Run only one optimizer instead of all three

**Cost estimates:**
| Configuration | API Calls | Est. Cost |
|--------------|-----------|-----------|
| Quick test (15 samples, 1 optimizer) | ~50 | $0.50-$1 |
| Medium (45 samples, 1 optimizer) | ~300 | $2-$5 |
| Full (153 samples, 1 optimizer) | ~1,500 | $8-$20 |
| Full (153 samples, 3 optimizers) | ~4,500 | $20-$50 |

## Score extraction fails

**Cause:** LLM output doesn't match expected format

**Solutions:**
- Check that LLM output includes "**Score:** X" format
- Update patterns in `extract_score_from_text()` if needed
- Add debug prints to see raw LLM outputs
- Verify the prompt template produces the expected format

## Import errors or module not found

**Cause:** Missing dependencies

**Solutions:**
- Activate virtual environment: `source .venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`
- Check Python version is 3.8 or higher: `python --version`

## Tests are failing

**Cause:** Various potential issues

**Solutions:**
- Check API keys are set: `cat .env`
- Run only unit tests (skip E2E): `pytest -m "not e2e" -v`
- Check that all files exist: `ls *.py *.csv *.txt`
- Review test output for specific error messages

## Out of memory errors

**Cause:** Dataset or model outputs are too large

**Solutions:**
- Reduce `--sample-size`
- Reduce `--n-threads`
- Reduce `--max-output-tokens`
- Close other applications

## Still having issues?

1. Check the [Opik Documentation](https://www.comet.com/docs/opik)
2. Review your configuration with `--quiet` flag removed
3. Try the minimal test: `python optimize.py --sample-size 10 --n-trials 1 --optimizers metaprompt`
4. Check git status for uncommitted changes that might affect behavior
5. Reach out to your instructor with the error message and configuration used
