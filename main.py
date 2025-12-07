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
from generation_methods import Random2, Default, Scaling

# Dictionary mapping argument strings to agent classes
AGENT_CLASSES = {
    # Basic agents
    'random': RandomAgent,
    
    # Monte Carlo Tree Search agents
    'mcts': ImprovedMCTSAgent,
    'mcts_playout': RandomPlayoutAgent,
}

# Dictionary mapping argument strings to generation method classes
GENERATION_METHOD_CLASSES = {
    'random2': Random2,
    'default': Default,
    'scaling': Scaling,
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
    # Parse generation method (second argument, optional)
    generation_method = None
    if len(sys.argv) > 2:
        generation_method_name = sys.argv[2]
        
        if generation_method_name not in GENERATION_METHOD_CLASSES:
            print(f"Unknown generation method: {generation_method_name}")
            print(f"Available generation methods: {', '.join(GENERATION_METHOD_CLASSES.keys())}")
            sys.exit(1)
        
        generation_method_class = GENERATION_METHOD_CLASSES[generation_method_name]
        generation_method = generation_method_class()
        print(f"Using generation method: {generation_method_name}")
    
    if len(sys.argv) > 1:
        agent_name = sys.argv[1]
        
        # Check if it's manual mode
        if agent_name == 'manual':
            # Interactive mode (user input) with optional generation method
            print("Starting game in interactive mode...")
            print("Use arrow keys or WASD to play.")
            game_grid = GameGrid(auto_start=True, generation_method=generation_method)
        else:
            # Agent mode
            if agent_name not in AGENT_CLASSES:
                print(f"Unknown agent: {agent_name}")
                print(f"Available agents: {', '.join(AGENT_CLASSES.keys())}, 'manual'")
                sys.exit(1)
            
            # Create game grid without auto-starting mainloop
            root = Tk()
            game_grid = GameGrid(auto_start=False, root=root, generation_method=generation_method)
            
            # Instantiate the agent
            agent_class = AGENT_CLASSES[agent_name]
            agent = agent_class()
            
            print(f"Starting game with agent: {agent_name}")
            print("Close the window to exit.")
            
            # Run with agent
            run_with_agent(game_grid, agent)
        
    else:
        # Interactive mode (user input) with default generation method
        print("Starting game in interactive mode...")
        print("Use arrow keys or WASD to play.")
        game_grid = GameGrid(auto_start=True, generation_method=generation_method)


if __name__ == "__main__":
    main()
