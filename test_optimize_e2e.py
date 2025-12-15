"""
End-to-end tests for optimize.py with real API calls.

Run with: pytest test_optimize_e2e.py -v -s
"""

import pytest
import subprocess
import os
import json
import time
from pathlib import Path


@pytest.fixture(scope="session")
def test_env():
    """Provide environment with API keys."""
    env = os.environ.copy()

    required_keys = ['OPIK_API_KEY', 'OPENAI_API_KEY']
    missing = [k for k in required_keys if not env.get(k)]

    if missing:
        pytest.skip(f"Missing API keys: {missing}")

    return env


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory."""
    test_output = tmp_path / "test_runs"
    test_output.mkdir()
    return str(test_output)


class TestOptimizeE2E:
    """End-to-end tests for optimize.py."""

    def test_minimal_run_all_optimizers(self, test_env, output_dir):
        """
        Test complete optimization run with minimal dataset.

        Validates:
        - CLI argument parsing
        - Dataset loading and splitting
        - Baseline evaluation
        - All three optimizers
        - Output file generation
        - Results comparison
        """
        print("\n" + "=" * 80)
        print("STARTING E2E TEST: Minimal run with all optimizers")
        print("Expected duration: 15-30 minutes")
        print("=" * 80)

        start_time = time.time()

        # Run optimize.py
        result = subprocess.run(
            [
                '.venv/bin/python', 'optimize.py',
                '--sample-size', '30',  # Updated from 10 (need 12/12/6 with 40/40/20 split)
                '--n-trials', '1',
                '--n-threads', '4',
                '--output-dir', output_dir,
            ],
            capture_output=True,
            text=True,
            env=test_env,
            cwd=os.getcwd()
        )

        elapsed = time.time() - start_time

        # Print output for debugging
        print("\n" + "=" * 80)
        print("STDOUT:")
        print("=" * 80)
        print(result.stdout)

        if result.stderr:
            print("\n" + "=" * 80)
            print("STDERR:")
            print("=" * 80)
            print(result.stderr)

        print(f"\nTest completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

        # ASSERTIONS
        assert result.returncode == 0, f"Failed with code {result.returncode}"

        # Find run directory
        run_dirs = sorted(Path(output_dir).glob("2025-*"))
        assert len(run_dirs) > 0, "No run directory created"
        run_dir = run_dirs[-1]
        print(f"\n✓ Run directory: {run_dir}")

        # Verify JSON summary
        json_path = run_dir / "results_summary.json"
        assert json_path.exists(), f"JSON summary not found at {json_path}"

        with open(json_path) as f:
            summary = json.load(f)

        print("\n✓ JSON summary:")
        print(json.dumps(summary, indent=2))

        # Validate structure
        assert "test_config" in summary
        assert "baseline" in summary
        assert "optimizers" in summary
        assert summary["test_config"]["sample_size"] == 30  # Updated from 10
        assert summary["test_config"]["n_trials"] == 1

        # Validate baseline (now has dev_score and test_score)
        assert "dev_score" in summary["baseline"]
        assert "test_score" in summary["baseline"]
        baseline_dev = summary["baseline"]["dev_score"]
        baseline_test = summary["baseline"]["test_score"]
        assert 0.0 <= baseline_dev <= 1.0
        assert 0.0 <= baseline_test <= 1.0
        print(f"\n✓ Baseline: dev={baseline_dev:.4f}, test={baseline_test:.4f}")

        # Validate optimizers
        expected_optimizers = {"MetaPrompt", "Hierarchical", "FewShot", "GEPA", "Evolutionary"}
        actual_optimizers = set(summary["optimizers"].keys())
        assert actual_optimizers == expected_optimizers, \
            f"Expected {expected_optimizers}, got {actual_optimizers}"

        for opt_name, opt_data in summary["optimizers"].items():
            dev_score = opt_data["dev_score"]
            test_score = opt_data["test_score"]
            assert 0.0 <= dev_score <= 1.0, f"{opt_name} dev_score {dev_score} out of range"
            assert 0.0 <= test_score <= 1.0, f"{opt_name} test_score {test_score} out of range"
            print(f"✓ {opt_name}: dev={dev_score:.4f}, test={test_score:.4f} "
                  f"(test improvement: {opt_data['test_improvement_pct']:+.1f}%)")

        # Verify output files
        expected_files = [
            "baseline_score.txt",
            "optimized-metaprompt-messages.txt",
            "optimized-hierarchical-reflective-messages.txt",
            "optimized-few-shot-bayesian-messages.txt",
            "optimized-gepa-messages.txt",
            "optimized-evolutionary-messages.txt",
            "comparison_table.txt",
            "results_summary.json"
        ]

        for filename in expected_files:
            filepath = run_dir / filename
            assert filepath.exists(), f"Missing: {filename}"
            assert filepath.stat().st_size > 0, f"Empty: {filename}"

        print(f"\n✓ All {len(expected_files)} files created")

        # Verify winner (now has dev_score and test_score)
        assert "winner" in summary
        assert summary["winner"]["name"] in expected_optimizers
        assert "dev_score" in summary["winner"]
        assert "test_score" in summary["winner"]
        print(f"\n✓ Winner: {summary['winner']['name']} "
              f"(dev={summary['winner']['dev_score']:.4f}, "
              f"test={summary['winner']['test_score']:.4f})")

        # Verify stdout markers
        stdout = result.stdout
        assert "CONFIGURATION" in stdout
        assert "BASELINE EVALUATION" in stdout
        assert "FINAL RESULTS COMPARISON" in stdout

        print("\n" + "=" * 80)
        print("✓ ALL ASSERTIONS PASSED")
        print("=" * 80)


# Pytest markers
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
]
