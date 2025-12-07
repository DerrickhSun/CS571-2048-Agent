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

# Standard 2048 logic functions for pure matrix simulation
from game_files import logic
from agents.base import Agent


def agent_best_action(matrix, Q):
    """
    Get the best action from Q table for the given matrix state.
    This matches the function from q_learning/puzzle.py
    """
    state = encode_state(matrix)

    if state not in Q:
        return random.choice([0, 1, 2, 3])

    q_vals = Q[state]
    max_q = max(q_vals)
    best_actions = [a for a, q in enumerate(q_vals) if q == max_q]
    return random.choice(best_actions)


def simulate_move_matrix(matrix, action):
    # Simulate movement through logic module without touching the game grid
    if action == 0:
        new_matrix, reward, done = logic.up(matrix)
    elif action == 1:
        new_matrix, reward, done = logic.down(matrix)
    elif action == 2:
        new_matrix, reward, done = logic.left(matrix)
    elif action == 3:
        new_matrix, reward, done = logic.right(matrix)
    else:
        raise ValueError("Invalid action")

    return new_matrix, reward, done


class QLearningAgent(Agent):
    """
    Q-Learning agent that uses a pre-trained Q-table from q_learning/q_learning.py.
    """

    def __init__(self, q_table_path=None):
        super().__init__()

        if not Q_LEARNING_AVAILABLE:
            print("Warning: q_learning module not found. Agent will use random moves.")
            self.Q = {}
            return

        if q_table_path is None:
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
        Uses matrix simulation to verify move validity instead of touching game_grid.
        """

        matrix = game_grid.matrix

        action = agent_best_action(matrix, self.Q)

        action_to_direction = {
            0: 'up',
            1: 'down',
            2: 'left',
            3: 'right'
        }

        # Check suggested move
        _, reward, _ = simulate_move_matrix(matrix, action)
        if reward >= 0:
            return action_to_direction[action]

        # Try other moves if Q suggests an invalid one
        actions = [0, 1, 2, 3]
        random.shuffle(actions)

        for a in actions:
            _, reward, _ = simulate_move_matrix(matrix, a)
            if reward >= 0:
                return action_to_direction[a]

        return None
