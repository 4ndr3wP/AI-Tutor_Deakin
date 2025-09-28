#!/usr/bin/env python3
"""
AI Tutor Test Runner
===================

Quick wrapper to run tests from anywhere in the project.

NOTE: You must `conda activate test_app` before running any test. `python AI-Tutor/run_sh.sh` will do this for you.

Examples:
  python run_tests.py                         # Run all tests
  python run_tests.py --test off-topic        # Test off-topic detection only
  python run_tests.py --test memory           # Test memory functionality only
  python run_tests.py --test performance      # Test performance only
  python run_tests.py --test isolation        # Test session isolation only
  python run_tests.py --test cold-start       # Test cold start vs warm start
  python run_tests.py --test hallucination    # Test hallucination detection only
  python run_tests.py --test socratic         # Test Socratic response only
  python run_tests.py --test gpu_stress       # Test GPU stress only
  python run_tests.py --test reference_accuracy # Test Reference Accuracy
  python run_tests.py --test ontrack_tasks    # Test OnTrack task retrieval only

NOTE: All tests are found in `_my_tests/test_cases/`.
"""


import subprocess
import sys
import os

def main():
    # Change to tests directory
    os.chdir('_my_tests')
    
    # Run the actual test runner with all arguments
    subprocess.run([sys.executable, 'test_runner.py'] + sys.argv[1:])

if __name__ == "__main__":
    main()