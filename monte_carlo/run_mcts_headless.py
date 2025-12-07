#!/usr/bin/env python3
"""
Headless script to run MCTS agent without GUI dependencies.
Directly uses game_files.logic for all game operations.
"""

import sys
import time
import random
import copy
from pathlib import Path

# Ensure imports work from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only non-GUI components
from game_files import logic, constants as c
from improved_mcts import ImprovedMCTSAgent, RandomPlayoutAgent


class SimpleGameState:
    """Minimal game state for headless testing."""
    
    def __init__(self):
        self.matrix = logic.new_game(c.GRID_LEN)
        self.score = 0
    
    def get_available_moves(self):
        """Get list of available moves."""
        moves = []
        direction_funcs = [logic.up, logic.down, logic.left, logic.right]
        for i, func in enumerate(direction_funcs):
            # Try the move on a copy
            test_matrix = [row[:] for row in self.matrix]
            try:
                result = func(test_matrix)
                if isinstance(result, tuple) and len(result) == 3:
                    new_matrix, moved, score = result
                    if moved:  # Move actually changed the board
                        moves.append(i)
            except:
                pass
        return moves
    
    def make_move(self, move_idx):
        """Make a move and return resulting state and points."""
        direction_funcs = [logic.up, logic.down, logic.left, logic.right]
        func = direction_funcs[move_idx]
        
        new_matrix, moved, points = func(self.matrix)
        if moved:
            self.matrix = new_matrix
            self.score += points
            # Add a random tile (2 with 90% probability, 4 with 10%)
            self.matrix = self._add_random_tile()
        
        return points
    
    def _add_random_tile(self):
        """Add a random 2 or 4 to an empty cell."""
        empty = [(i, j) for i in range(len(self.matrix)) 
                 for j in range(len(self.matrix[0])) 
                 if self.matrix[i][j] == 0]
        if not empty:
            return self.matrix
        i, j = random.choice(empty)
        new_matrix = [row[:] for row in self.matrix]
        new_matrix[i][j] = 2 if random.random() < 0.9 else 4
        return new_matrix
    
    def is_game_over(self):
        """Check if game is over using logic.game_state."""
        state = logic.game_state(self.matrix)
        return state == 'lose'
    
    def copy(self):
        """Return a deep copy of the game state."""
        new_state = SimpleGameState()
        new_state.matrix = [row[:] for row in self.matrix]
        new_state.score = self.score
        return new_state
    
    def get_board(self):
        """Get board for display."""
        return self.matrix
    
    def get_max_tile(self):
        """Get maximum tile value."""
        return max(max(row) for row in self.matrix)


def display_board(game_state):
    """Display the game board in terminal."""
    matrix = game_state.get_board()
    print("\n" + "=" * 40)
    for row in matrix:
        print(" ".join(f"{val:5d}" for val in row))
    print(f"Score: {game_state.score:8d} | Max Tile: {game_state.get_max_tile()}")
    print("=" * 40)


def run_game(agent_class, num_simulations=200, rollout_depth=40, rollout_epsilon=0.15, verbose=True):
    """Run a single game with the given agent."""
    
    game = SimpleGameState()
    agent = agent_class(
        num_simulations=num_simulations,
        rollout_depth=rollout_depth,
        rollout_epsilon=rollout_epsilon
    )
    
    move_count = 0
    
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Starting game with {agent_class.__name__}")
        print(f"Simulations: {num_simulations}, Depth: {rollout_depth}, Epsilon: {rollout_epsilon}")
        print(f"{'=' * 50}")
        display_board(game)
    
    while not game.is_game_over():
        move_count += 1
        
        # Get best move
        available_moves = game.get_available_moves()
        
        if not available_moves:
            break
        
        start_time = time.time()
        move_direction = agent.next_move(game)
        elapsed = time.time() - start_time
        
        # Convert direction to index
        direction_to_idx = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        move = direction_to_idx.get(move_direction, -1)
        
        if move == -1:
            # Invalid move returned, pick random valid move
            move = random.choice(available_moves) if available_moves else 0
        
        if move not in available_moves:
            # Fallback to random if agent returned invalid move
            move = random.choice(available_moves)
        
        # Make the move
        points = game.make_move(move)
        
        direction_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        if verbose:
            print(f"\nMove {move_count}: {direction_names[move]} (search: {elapsed:.2f}s, +{points} pts)")
            if move_count % 5 == 0:  # Display board every 5 moves
                display_board(game)
    
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"GAME OVER!")
        display_board(game)
        print(f"Total moves: {move_count}")
        print(f"{'=' * 50}\n")
    
    return game.score, game.get_max_tile(), move_count


def run_multiple_games(agent_class, num_games=3, num_simulations=200, 
                      rollout_depth=40, rollout_epsilon=0.15):
    """Run multiple games and report statistics."""
    
    scores = []
    max_tiles = []
    move_counts = []
    
    print(f"\n{'#' * 60}")
    print(f"Running {num_games} games with {agent_class.__name__}")
    print(f"Parameters: sims={num_simulations}, depth={rollout_depth}, eps={rollout_epsilon}")
    print(f"{'#' * 60}")
    
    start_total = time.time()
    
    for game_num in range(num_games):
        print(f"\n[Game {game_num + 1}/{num_games}]")
        score, max_tile, moves = run_game(
            agent_class,
            num_simulations=num_simulations,
            rollout_depth=rollout_depth,
            rollout_epsilon=rollout_epsilon,
            verbose=(num_games <= 2)  # Only verbose for 1-2 games
        )
        
        scores.append(score)
        max_tiles.append(max_tile)
        move_counts.append(moves)
        
        print(f"  Score: {score:8d} | Max Tile: {max_tile:6d} | Moves: {moves:3d}")
    
    total_time = time.time() - start_total
    
    print(f"\n{'=' * 60}")
    print(f"STATISTICS ({num_games} games, {total_time:.1f}s total):")
    print(f"  Scores:    avg={sum(scores)/len(scores):.0f}, min={min(scores)}, max={max(scores)}")
    print(f"  Max Tiles: avg={sum(max_tiles)/len(max_tiles):.0f}, min={min(max_tiles)}, max={max(max_tiles)}")
    print(f"  Moves:     avg={sum(move_counts)/len(move_counts):.1f}, min={min(move_counts)}, max={max(move_counts)}")
    print(f"{'=' * 60}\n")
    
    return scores, max_tiles, move_counts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MCTS agent headlessly")
    parser.add_argument("--agent", choices=["improved", "playout"], default="improved",
                       help="Which agent to use")
    parser.add_argument("--games", type=int, default=1,
                       help="Number of games to play")
    parser.add_argument("--sims", type=int, default=200,
                       help="Number of simulations per move")
    parser.add_argument("--depth", type=int, default=40,
                       help="Rollout depth")
    parser.add_argument("--epsilon", type=float, default=0.15,
                       help="Epsilon for epsilon-greedy rollouts")
    
    args = parser.parse_args()
    
    agent_class = ImprovedMCTSAgent if args.agent == "improved" else RandomPlayoutAgent
    
    if args.games == 1:
        score, max_tile, moves = run_game(
            agent_class,
            num_simulations=args.sims,
            rollout_depth=args.depth,
            rollout_epsilon=args.epsilon,
            verbose=True
        )
    else:
        run_multiple_games(
            agent_class,
            num_games=args.games,
            num_simulations=args.sims,
            rollout_depth=args.depth,
            rollout_epsilon=args.epsilon
        )
