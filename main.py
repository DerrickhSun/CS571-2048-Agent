#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for the 2048 game.
Can run in interactive mode (user input) or with an agent method.
"""

import sys
import time
from tkinter import Tk
from game_files.puzzle import GameGrid
import game_files.logic as logic
from agents.naive import RandomAgent
from monte_carlo.improved_mcts import ImprovedMCTSAgent, RandomPlayoutAgent

# Dictionary mapping argument strings to agent classes
AGENT_CLASSES = {
    # Basic agents
    'random': RandomAgent,
    
    # Monte Carlo Tree Search agents
    'mcts': ImprovedMCTSAgent,
    'mcts_playout': RandomPlayoutAgent,
}


def run_with_agent(game_grid, agent):
    """
    Run the game with an agent that decides moves.
    
    Args:
        game_grid: The GameGrid instance
        agent: An Agent instance with a next_move method
    """
    def make_agent_move():
        """Make one move using the agent, then schedule the next one."""
        # Check game state
        state = logic.game_state(game_grid.matrix)
        if state == 'win':
            print("Game won!")
            return
        elif state == 'lose':
            print("Game lost!")
            return
        
        # Get move from agent
        direction = agent.next_move(game_grid)
        
        if direction is None:
            print("No valid moves available!")
            return
        
        # Execute the move
        success = game_grid.make_move(direction)
        
        if not success:
            print(f"Move {direction} was invalid, trying again...")
        
        # Schedule next move
        game_grid.master.after(100, make_agent_move)
    
    # Start the agent loop
    game_grid.master.after(100, make_agent_move)
    
    # Start the mainloop
    game_grid.mainloop()


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Agent mode
        agent_name = sys.argv[1]
        
        if agent_name not in AGENT_CLASSES:
            print(f"Unknown agent: {agent_name}")
            print(f"Available agents: {', '.join(AGENT_CLASSES.keys())}")
            sys.exit(1)
        
        # Create game grid without auto-starting mainloop
        root = Tk()
        game_grid = GameGrid(auto_start=False, root=root)
        
        # Instantiate the agent
        agent_class = AGENT_CLASSES[agent_name]
        agent = agent_class()
        
        print(f"Starting game with agent: {agent_name}")
        print("Close the window to exit.")
        
        # Run with agent
        run_with_agent(game_grid, agent)
        
    else:
        # Interactive mode (user input)
        print("Starting game in interactive mode...")
        print("Use arrow keys or WASD to play.")
        game_grid = GameGrid(auto_start=True)


if __name__ == "__main__":
    main()
