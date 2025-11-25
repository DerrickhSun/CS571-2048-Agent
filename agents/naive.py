"""
Naive agent implementations for the 2048 game.
These are simple heuristic agents for demonstration purposes.
"""

import random
import copy
from agents.base import Agent


class RandomAgent(Agent):
    """
    Simple random agent that picks a random valid move.
    """
    
    def next_move(self, game_grid):
        """
        Pick a random valid move.
        
        Args:
            game_grid: The GameGrid instance
            
        Returns:
            A direction string ('up', 'down', 'left', 'right') or None if no valid moves
        """
        directions = ['up', 'down', 'left', 'right']
        random.shuffle(directions)
        
        for direction in directions:
            # Check if move is valid by trying it
            # We can check by seeing if the board would change
            test_matrix = copy.deepcopy(game_grid.matrix)
            move_func = game_grid.direction_map[direction]
            new_matrix, done = move_func(test_matrix)
            
            if done:
                return direction
        
        return None
