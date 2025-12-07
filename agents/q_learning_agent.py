"""
Q-Learning agent for the 2048 game.
Uses a pre-trained Q-table to make decisions.
"""

import random
import pickle
import os
import sys

# Handle imports for q_learning utilities
try:
    # Try to import from q_learning directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'q_learning'))
    from env_2048 import encode_state
except ImportError:
    # Fallback: define encode_state here if q_learning not available
    def encode_state(matrix):
        return tuple(v for row in matrix for v in row)

from agents.base import Agent


class QLearningAgent(Agent):
    """
    Q-Learning agent that uses a pre-trained Q-table to make decisions.
    """
    
    def __init__(self, q_table_path=None):
        """
        Initialize the Q-Learning agent.
        
        Args:
            q_table_path: Path to the Q-table pickle file. If None, tries to find
                         q_table_shaped.pkl in the q_learning directory.
        """
        super().__init__()
        
        # Try to load Q table
        if q_table_path is None:
            # Try common locations
            possible_paths = [
                os.path.join('q_learning', 'q_table_shaped.pkl'),
                'q_table_shaped.pkl',
                os.path.join(os.path.dirname(__file__), '..', 'q_learning', 'q_table_shaped.pkl'),
            ]
            q_table_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    q_table_path = path
                    break
        
        self.Q = {}
        if q_table_path and os.path.exists(q_table_path):
            try:
                with open(q_table_path, "rb") as f:
                    self.Q = pickle.load(f)
                print(f"Loaded Q table with {len(self.Q)} states from {q_table_path}")
            except Exception as e:
                print(f"Warning: Could not load Q table from {q_table_path}: {e}")
        else:
            print("Warning: Q table not found. Agent will use random moves.")
    
    def next_move(self, game_grid):
        """
        Determine the next move using the Q-table.
        
        Args:
            game_grid: The GameGrid instance containing the current game state
            
        Returns:
            A direction string ('up', 'down', 'left', 'right') or None if no valid moves
        """
        # Encode the current state
        state = encode_state(game_grid.matrix)
        
        # Action to direction mapping
        action_to_direction = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        
        # Check if state is in Q table
        if state not in self.Q or len(self.Q) == 0:
            # Unseen state or no Q table, fallback to random valid move
            actions = [0, 1, 2, 3]
            random.shuffle(actions)
            for action in actions:
                # Check if move is valid
                new_matrix, done, _ = game_grid.apply_move_direct(action)
                if done:
                    return action_to_direction[action]
            return None
        
        # Get Q values for this state
        q_vals = self.Q[state]
        max_q = max(q_vals)
        best_actions = [a for a, q in enumerate(q_vals) if q == max_q]
        
        # Try best actions in order, return first valid one
        for action in best_actions:
            # Check if move is valid using apply_move_direct
            new_matrix, done, _ = game_grid.apply_move_direct(action)
            if done:
                return action_to_direction[action]
        
        # If no best action is valid, try any valid move
        actions = [0, 1, 2, 3]
        random.shuffle(actions)
        for action in actions:
            new_matrix, done, _ = game_grid.apply_move_direct(action)
            if done:
                return action_to_direction[action]
        
        return None

