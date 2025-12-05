#!/usr/bin/env python3
"""
Test the improved MCTS agent with GUI.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game_files.puzzle import GameGrid
from game_files import logic
from improved_mcts import ImprovedMCTSAgent
from tkinter import Tk, messagebox


def run_mcts_game(num_simulations=200, rollout_depth=40, rollout_epsilon=0.15):
    """Run 2048 with ImprovedMCTSAgent."""
    root = Tk()
    game_grid = GameGrid(auto_start=False, root=root)
    
    agent = ImprovedMCTSAgent(
        num_simulations=num_simulations,
        exploration_constant=1.4,
        rollout_depth=rollout_depth,
        rollout_epsilon=rollout_epsilon
    )
    
    state = {
        'moves': 0,
        'start_time': time.time(),
        'running': True
    }
    
    def make_move():
        if not state['running']:
            return
        
        # Check game state
        gs = logic.game_state(game_grid.matrix)
        if gs == 'win':
            elapsed = time.time() - state['start_time']
            messagebox.showinfo(
                "Game Won!", 
                f"Reached 2048!\n\nScore: {game_grid.score}\nMoves: {state['moves']}\nTime: {elapsed:.1f}s"
            )
            state['running'] = False
            return
        elif gs == 'lose':
            elapsed = time.time() - state['start_time']
            messagebox.showinfo(
                "Game Over",
                f"No more moves!\n\nScore: {game_grid.score}\nMoves: {state['moves']}\nTime: {elapsed:.1f}s"
            )
            state['running'] = False
            return
        
        # Get move
        direction = agent.next_move(game_grid)
        if direction is None:
            elapsed = time.time() - state['start_time']
            messagebox.showinfo(
                "Game Over",
                f"No valid moves!\n\nScore: {game_grid.score}\nMoves: {state['moves']}\nTime: {elapsed:.1f}s"
            )
            state['running'] = False
            return
        
        # Execute move
        success = game_grid.make_move(direction)
        if success:
            state['moves'] += 1
        
        # Update title
        elapsed = time.time() - state['start_time']
        max_tile = max(max(row) for row in game_grid.matrix)
        root.title(f"2048 - ImprovedMCTS | Moves: {state['moves']} | Score: {game_grid.score} | Max: {max_tile} | {elapsed:.0f}s")
        
        # Schedule next move
        root.after(50, make_move)
    
    root.title("2048 - ImprovedMCTS Agent")
    print(f"Starting 2048 with ImprovedMCTSAgent")
    print(f"  Simulations: {num_simulations}")
    print(f"  Rollout depth: {rollout_depth}")
    print(f"  Rollout epsilon: {rollout_epsilon}")
    
    root.after(100, make_move)
    root.mainloop()
    
    elapsed = time.time() - state['start_time']
    max_tile = max(max(row) for row in game_grid.matrix)
    
    print(f"\n{'='*50}")
    print(f"GAME RESULTS")
    print(f"{'='*50}")
    print(f"Final Score:   {game_grid.score}")
    print(f"Moves Made:    {state['moves']}")
    print(f"Max Tile:      {max_tile}")
    print(f"Total Time:    {elapsed:.1f}s")
    print(f"Time/Move:     {elapsed/state['moves'] if state['moves'] > 0 else 0:.2f}s")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Improved MCTS GUI with parameters")
    parser.add_argument('--sims', type=int, default=150, help='Simulations per move')
    parser.add_argument('--depth', type=int, default=40, help='Rollout depth')
    parser.add_argument('--epsilon', type=float, default=0.15, help='Rollout epsilon')

    args = parser.parse_args()

    run_mcts_game(
        num_simulations=args.sims,
        rollout_depth=args.depth,
        rollout_epsilon=args.epsilon,
    )
