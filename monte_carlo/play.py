#!/usr/bin/env python3
"""Quick launcher for MCTS headless runner."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    monte_carlo_dir = Path(__file__).parent
    
    # Run the headless MCTS script with any arguments passed
    result = subprocess.run(
        [sys.executable, "run_mcts_headless.py"] + sys.argv[1:],
        cwd=str(monte_carlo_dir)
    )
    sys.exit(result.returncode)
