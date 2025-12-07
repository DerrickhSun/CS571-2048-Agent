#!/usr/bin/env python3
"""Convenience script to run MCTS agent without GUI."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Run the headless MCTS test script with any arguments passed
    script_dir = Path(__file__).parent
    result = subprocess.run(
        [sys.executable, "run_mcts_headless.py"] + sys.argv[1:],
        cwd=str(script_dir)
    )
    sys.exit(result.returncode)
