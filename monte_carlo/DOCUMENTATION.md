# Monte Carlo (MCTS) — Documentation

This single consolidated document replaces the previous multiple guides (quick start, Tkinter issue, fix summary).

## Overview

This folder contains Monte Carlo Tree Search implementations and runners for the 2048 game puzzle.

Implemented agents:
- `ImprovedMCTSAgent`: recommended — player/chance nodes + UCB1 + heuristic rollouts.
- `RandomPlayoutAgent`: fast playout-only variant.

## Quick Start

From project root (GUI):

```bash
python3 main.py mcts
```

Or run GUI with the faster agent:

```bash
python3 main.py mcts_playout
```

Or run headless (if Tkinter is not available):

*Headless runner is not included in this version. Use the GUI mode above.*

## Parameters

Parameters for agent behavior are set in `improved_mcts.py` as module-level defaults.

## Running Examples

Quick start (GUI):

```bash
python3 main.py mcts
```

## GUI

`test_improved_mcts.py` requires `tkinter` and a display; install system Tk or run on a desktop environment.

## Implementation Notes

- Uses UCB1: value/visits + c*sqrt(ln(parent_visits)/visits).
- Improved agent models player and chance nodes separately and handles 2/4 spawn probabilities correctly.
- Rollouts use epsilon-greedy policy combining heuristic and random moves.

## Troubleshooting

1. If you see `ModuleNotFoundError: No module named '_tkinter'`, run headless mode or install Tkinter (platform-specific instructions).
2. If performance is slow, reduce `--sims` and `--depth`.

## Files in this folder

- `improved_mcts.py` — improved agent implementation
- `test_improved_mcts.py` — GUI test script
- `__init__.py` — package exports

---

