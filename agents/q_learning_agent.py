"""
Q-Learning agent for the 2048 game.
Uses the q_learning system from the q_learning folder.
"""

import random
import pickle
import os
import sys

# Handle imports for q_learning utilities
try:
    # Try to import from q_learning directory
    q_learning_path = os.path.join(os.path.dirname(__file__), '..', 'q_learning')
    if os.path.exists(q_learning_path):
        sys.path.insert(0, q_learning_path)
        from env_2048 import encode_state
        Q_LEARNING_AVAILABLE = True
    else:
        Q_LEARNING_AVAILABLE = False
        def encode_state(matrix):
            return tuple(v for row in matrix for v in row)
except ImportError:
    # Fallback: define encode_state here if q_learning not available
    Q_LEARNING_AVAILABLE = False
    def encode_state(matrix):
        return tuple(v for row in matrix for v in row)

from agents.base import Agent


def agent_best_action(matrix, Q):
    """
    Get the best action from Q table for the given matrix state.
    This matches the function from q_learning/puzzle.py
    """
    state = encode_state(matrix)
    
    if state not in Q:
        # Unseen state, fallback to any move
        return random.choice([0, 1, 2, 3])
    
    q_vals = Q[state]
    max_q = max(q_vals)
    best_actions = [a for a, q in enumerate(q_vals) if q == max_q]
    return random.choice(best_actions)


class QLearningAgent(Agent):
    """
    Q-Learning agent that uses a pre-trained Q-table from q_learning/q_learning.py.
    Uses the same agent_best_action logic as q_learning/puzzle.py
    """
    
    def __init__(self, q_table_path=None):
        """
        Initialize the Q-Learning agent.
        
        Args:
            q_table_path: Path to the Q-table pickle file. If None, tries to find
                         q_table_shaped.pkl in the q_learning directory.
        """
        super().__init__()
        
        if not Q_LEARNING_AVAILABLE:
            print("Warning: q_learning module not found. Agent will use random moves.")
            self.Q = {}
            return
        
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
        Determine the next move using the Q-table and agent_best_action from q_learning.
        
        Args:
            game_grid: The GameGrid instance containing the current game state
            
        Returns:
            A direction string ('up', 'down', 'left', 'right') or None if no valid moves
        """
        # Use the same agent_best_action logic as q_learning/puzzle.py
        action = agent_best_action(game_grid.matrix, self.Q)
        
        # Action to direction mapping (matching q_learning: 0=up, 1=down, 2=left, 3=right)
        action_to_direction = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        direction = action_to_direction[action]
        
        # Verify the move is valid before returning
        new_matrix, done, _ = game_grid.apply_move_direct(action)
        if done:
            return direction
        
        # If the suggested move is invalid, try other moves
        actions = [0, 1, 2, 3]
        random.shuffle(actions)
        for a in actions:
            new_matrix, done, _ = game_grid.apply_move_direct(a)
            if done:
                return action_to_direction[a]
        
        return None

