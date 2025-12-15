"""
Evaluate an optimized prompt from a runs directory.

This script:
1. Reads an optimized prompt file (e.g., optimized-metaprompt-messages.txt)
2. Loads model config from results_summary.json in the same directory
3. Runs the prompt against all rows in answer-evaluation.csv
4. Generates an HTML report with scores and color coding

Usage:
    # Basic usage
    python evaluate_prompt.py --prompt-file runs/2025-12-14_18-22-33/optimized-metaprompt-messages.txt

    # With sampling for quick test
    python evaluate_prompt.py --prompt-file runs/2025-12-14_18-22-33/optimized-metaprompt-messages.txt --sample-size 10

    # Override model
    python evaluate_prompt.py --prompt-file runs/... --model openai/gpt-4o
"""

import os
import re
import json
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
from dotenv import load_dotenv
import pandas as pd
from litellm import completion

from utils import extract_score_from_text

# Load environment variables
load_dotenv()

# Suppress LiteLLM verbose output
import litellm
litellm.suppress_debug_info = True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate an optimized prompt on the full dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--prompt-file', type=str, required=True,
                       help='Path to optimized prompt file (e.g., runs/.../optimized-metaprompt-messages.txt)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of samples to evaluate (None = full dataset)')
    parser.add_argument('--n-threads', type=int, default=4,
                       help='Parallel threads for LLM calls')
    parser.add_argument('--model', type=str, default=None,
                       help='Override model (default: read from results_summary.json)')
    parser.add_argument('--csv-path', type=str, default='answer-evaluation.csv',
                       help='Path to evaluation CSV')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    return parser.parse_args()


def parse_prompt_file(file_path: str) -> Tuple[str, str]:
    """
    Parse an optimized prompt file to extract system and user prompts.

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find system message
    system_match = re.search(
        r'Role:\s*system\s*\nContent:\s*\n(.*?)(?=-{20,})',
        content, re.DOTALL | re.IGNORECASE
    )
    if not system_match:
        raise ValueError("Could not find system message in prompt file")
    system_prompt = system_match.group(1).strip()

    # Find user message
    user_match = re.search(
        r'Role:\s*user\s*\nContent:\s*\n(.*?)(?=-{20,}|$)',
        content, re.DOTALL | re.IGNORECASE
    )
    if not user_match:
        raise ValueError("Could not find user message in prompt file")
    user_prompt = user_match.group(1).strip()

    return system_prompt, user_prompt


def load_model_config(run_dir: str) -> Optional[dict]:
    """Load model configuration from results_summary.json."""
    json_path = os.path.join(run_dir, 'results_summary.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return None


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    row: dict,
    model_params: dict
) -> Tuple[str, Optional[float], float, str]:
    """
    Call LLM with the prompt and extract score.

    Returns:
        Tuple of (llm_output, extracted_score, accuracy, reason)
    """
    # Fill in template variables
    filled_user = user_prompt.format(
        question=row['question'],
        human_answer=row['human_answer'],
        ai_answer=row['ai_answer'],
        retrieved_content=row['retrieved_content']
    )

    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": filled_user}
            ],
            **model_params
        )
        llm_output = response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}", None, 0.0, f"LLM call failed: {str(e)}"

    # Extract score
    extracted_score = extract_score_from_text(llm_output, min_score=1.0, max_score=5.0)

    if extracted_score is None:
        return llm_output, None, 0.0, "Could not extract score from LLM response"

    # Calculate accuracy
    human_score = float(row['score'])
    difference = abs(extracted_score - human_score)
    accuracy = max(0.0, 1.0 - (difference / 4.0))
    reason = f"LLM: {extracted_score}, Human: {human_score}, Diff: {difference:.2f}"

    return llm_output, extracted_score, accuracy, reason


def get_color_class(accuracy: float) -> str:
    """Get CSS color class based on accuracy."""
    if accuracy >= 0.9:
        return "score-green"
    elif accuracy >= 0.75:
        return "score-yellow"
    else:
        return "score-red"


def generate_html_report(
    results: list,
    average_accuracy: float,
    prompt_name: str,
    model: str,
    system_prompt: str,
    user_prompt: str
) -> str:
    """Generate HTML report with results."""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt Evaluation: {prompt_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary h2 {{
            margin-top: 0;
        }}
        .avg-score {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .prompt-section {{
            background: #f0f0f0;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 12px;
            max-height: 200px;
            overflow-y: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
            vertical-align: top;
        }}
        th {{
            background: #333;
            color: white;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .score-green {{
            background-color: #d4edda;
        }}
        .score-yellow {{
            background-color: #fff3cd;
        }}
        .score-red {{
            background-color: #f8d7da;
        }}
        .cell-text {{
            max-width: 300px;
            max-height: 150px;
            overflow: auto;
            white-space: pre-wrap;
            font-size: 12px;
        }}
        .llm-output {{
            max-width: 400px;
            max-height: 200px;
            overflow: auto;
            white-space: pre-wrap;
            font-size: 12px;
            background: #f9f9f9;
            padding: 8px;
            border-radius: 4px;
        }}
        .score-cell {{
            font-weight: bold;
            text-align: center;
        }}
    </style>
</head>
<body>
    <h1>Prompt Evaluation Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Prompt:</strong> {prompt_name}</p>
        <p><strong>Model:</strong> {model}</p>
        <p><strong>Samples Evaluated:</strong> {len(results)}</p>
        <p><strong>Average Accuracy:</strong> <span class="avg-score">{average_accuracy:.1%}</span></p>

        <h3>System Prompt</h3>
        <div class="prompt-section">{system_prompt}</div>

        <h3>User Prompt Template</h3>
        <div class="prompt-section">{user_prompt}</div>
    </div>

    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Question</th>
                <th>Human Answer</th>
                <th>AI Answer</th>
                <th>LLM Output</th>
                <th>LLM Score</th>
                <th>Human Score</th>
                <th>Accuracy</th>
            </tr>
        </thead>
        <tbody>
"""

    for i, r in enumerate(results, 1):
        color_class = get_color_class(r['accuracy'])
        llm_score_display = f"{r['llm_score']:.1f}" if r['llm_score'] is not None else "N/A"

        html += f"""            <tr class="{color_class}">
                <td>{i}</td>
                <td><div class="cell-text">{escape_html(r['question'])}</div></td>
                <td><div class="cell-text">{escape_html(r['human_answer'])}</div></td>
                <td><div class="cell-text">{escape_html(r['ai_answer'])}</div></td>
                <td><div class="llm-output">{escape_html(r['llm_output'])}</div></td>
                <td class="score-cell">{llm_score_display}</td>
                <td class="score-cell">{r['human_score']:.1f}</td>
                <td class="score-cell">{r['accuracy']:.1%}</td>
            </tr>
"""

    html += """        </tbody>
    </table>
</body>
</html>
"""
    return html


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


def main():
    args = parse_args()

    # Validate prompt file exists
    if not os.path.exists(args.prompt_file):
        print(f"ERROR: Prompt file not found: {args.prompt_file}")
        sys.exit(1)

    # Get run directory and prompt name from file path
    run_dir = os.path.dirname(args.prompt_file)
    prompt_filename = os.path.basename(args.prompt_file)

    # Extract prompt name for output file (e.g., "metaprompt" from "optimized-metaprompt-messages.txt")
    prompt_name_match = re.search(r'optimized-(.+?)-messages\.txt', prompt_filename)
    if prompt_name_match:
        prompt_name = prompt_name_match.group(1)
    else:
        prompt_name = prompt_filename.replace('.txt', '')

    if not args.quiet:
        print("=" * 80)
        print("PROMPT EVALUATION")
        print("=" * 80)
        print(f"Prompt file: {args.prompt_file}")
        print(f"Prompt name: {prompt_name}")

    # Parse prompt file
    if not args.quiet:
        print("\nParsing prompt file...")
    system_prompt, user_prompt = parse_prompt_file(args.prompt_file)
    if not args.quiet:
        print(f"  System prompt: {len(system_prompt)} chars")
        print(f"  User prompt: {len(user_prompt)} chars")

    # Load model config
    model = args.model
    config = load_model_config(run_dir)
    if model is None:
        if config and 'test_config' in config:
            model = config['test_config'].get('model')
        if model is None:
            print("ERROR: No model specified and couldn't find results_summary.json")
            print("Use --model to specify a model")
            sys.exit(1)

    # Load model parameters (with defaults for backward compatibility)
    default_model_params = {
        "reasoning_effort": "low",
        "verbosity": "low",
        "max_output_tokens": 65536,
        "num_retries": 5
    }
    model_params = default_model_params.copy()
    if config and 'test_config' in config:
        saved_params = config['test_config'].get('model_parameters', {})
        if saved_params:
            model_params.update(saved_params)

    if not args.quiet:
        print(f"Model: {model}")
        print(f"Model params: {model_params}")

    # Load CSV
    if not args.quiet:
        print(f"\nLoading CSV from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    # Sample if requested
    if args.sample_size is not None and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=42)
        if not args.quiet:
            print(f"  Sampled {args.sample_size} rows")

    if not args.quiet:
        print(f"  Total rows to evaluate: {len(df)}")

    # Evaluate each row in parallel
    if not args.quiet:
        print(f"\nEvaluating with {args.n_threads} threads...")

    results = []
    total = len(df)
    completed = 0

    with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        # Submit all tasks
        future_to_idx = {}
        for idx, row in df.iterrows():
            future = executor.submit(
                call_llm,
                system_prompt,
                user_prompt,
                model,
                row.to_dict(),
                model_params
            )
            future_to_idx[future] = (idx, row.to_dict())

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx, row = future_to_idx[future]
            llm_output, llm_score, accuracy, reason = future.result()

            results.append({
                'question': row['question'],
                'human_answer': row['human_answer'],
                'ai_answer': row['ai_answer'],
                'llm_output': llm_output,
                'llm_score': llm_score,
                'human_score': float(row['score']),
                'accuracy': accuracy,
                'reason': reason
            })

            completed += 1
            if not args.quiet:
                print(f"  Progress: {completed}/{total} ({completed/total*100:.0f}%)", end='\r')

    if not args.quiet:
        print()  # Newline after progress

    # Calculate average accuracy
    average_accuracy = sum(r['accuracy'] for r in results) / len(results) if results else 0.0

    # Generate and save HTML report
    html = generate_html_report(
        results=results,
        average_accuracy=average_accuracy,
        prompt_name=prompt_name,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    output_file = os.path.join(run_dir, f"evaluation-{prompt_name}.html")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Average Accuracy: {average_accuracy:.4f} ({average_accuracy*100:.1f}%)")
    print(f"Samples Evaluated: {len(results)}")

    # Score distribution
    green_count = sum(1 for r in results if r['accuracy'] >= 0.9)
    yellow_count = sum(1 for r in results if 0.75 <= r['accuracy'] < 0.9)
    red_count = sum(1 for r in results if r['accuracy'] < 0.75)

    print(f"\nScore Distribution:")
    print(f"  Green (>=90%):  {green_count} ({green_count/len(results)*100:.1f}%)")
    print(f"  Yellow (75-90%): {yellow_count} ({yellow_count/len(results)*100:.1f}%)")
    print(f"  Red (<75%):     {red_count} ({red_count/len(results)*100:.1f}%)")

    print(f"\nHTML report saved to: {output_file}")


if __name__ == "__main__":
    main()
