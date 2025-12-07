#!/usr/bin/env python3
"""
Comprehensive evaluation script for the project proposal.
Tests different agents under:
1. Standard vs Modified (Scaling) tile distribution
2. Different grid sizes (3x3, 4x4, 5x5, 6x6)
3. Evaluates heuristic reliability across conditions

Supports: Expectimax, MCTS (Monte Carlo), Random, and other agents
"""

import sys
import time
import copy
from game_files import logic
from game_files import constants as c
from agents.expectimax import ExpectimaxAgent, ExpectimaxAgentFast, ExpectimaxAgentDeep
from agents.naive import RandomAgent
from monte_carlo.improved_mcts import ImprovedMCTSAgent, RandomPlayoutAgent
from generation_methods import Default, Scaling, Random2
from agents.q_learning_agent import QLearningAgent


# Import all agents from main.py
AGENT_CLASSES = {
    'random': RandomAgent,
    'mcts': ImprovedMCTSAgent,
    'mcts_playout': RandomPlayoutAgent,
    'expectimax': ExpectimaxAgent,
    'expectimax_fast': ExpectimaxAgentFast,
    'expectimax_deep': ExpectimaxAgentDeep,
    'qlearning': QLearningAgent,
}


def run_single_game(agent, grid_size=4, generation_method=None, max_moves=10000, verbose=False):
    """
    Run a single game with specified parameters.
    
    Args:
        agent: The agent to test
        grid_size: Size of the grid (3, 4, or 5)
        generation_method: GenerationMethod instance (Default or Scaling)
        max_moves: Maximum number of moves
        verbose: Print detailed progress
    
    Returns:
        dict with game statistics
    """
    # Temporarily change GRID_LEN for this game
    original_grid_len = c.GRID_LEN
    c.GRID_LEN = grid_size
    
    try:
        # Initialize game with specified grid size
        matrix = logic.new_game(grid_size)
        total_score = 0
        moves = 0
        
        # Use default generation if none provided
        if generation_method is None:
            generation_method = Default()
        
        # Create a mock game_grid object
        class MockGameGrid:
            def __init__(self, matrix):
                self.matrix = matrix
                self.direction_map = {
                    'up': logic.up,
                    'down': logic.down,
                    'left': logic.left,
                    'right': logic.right
                }
        
        game_grid = MockGameGrid(matrix)
        
        # Play until game over
        while moves < max_moves:
            state = logic.game_state(matrix)
            if state in ['win', 'lose']:
                break
            
            # Get agent's move
            game_grid.matrix = matrix
            direction = agent.next_move(game_grid)
            
            if direction is None:
                break
            
            # Execute move
            move_func = game_grid.direction_map[direction]
            new_matrix, done, score = move_func(matrix)
            
            if not done:
                continue
            
            # Update state - use generation method
            matrix = generation_method.add_tile(new_matrix)
            total_score += score
            moves += 1
            
            if verbose and moves % 100 == 0:
                max_tile = max(max(row) for row in matrix)
                print(f"  Move {moves}: Score={total_score}, Max tile={max_tile}")
        
        # Get final statistics
        max_tile = max(max(row) for row in matrix)
        final_state = logic.game_state(matrix)
        
        return {
            'score': total_score,
            'max_tile': max_tile,
            'moves': moves,
            'won': final_state == 'win',
            'grid_size': grid_size,
            'generation_method': generation_method.__class__.__name__
        }
    finally:
        # Restore original GRID_LEN
        c.GRID_LEN = original_grid_len


def run_experiment(agent_class, grid_size, generation_method, num_games=10, agent_name=None):
    """
    Run an experiment with specific configuration.
    """
    if agent_name is None:
        agent_name = agent_class.__name__
    
    method_name = generation_method.__class__.__name__
    
    # Determine tile_distribution for Expectimax agent
    if method_name == 'Default':
        tile_dist = 'standard'
    elif method_name == 'Scaling':
        tile_dist = 'modified'
    else:
        tile_dist = 'standard'
    
    print(f"\n{'='*70}")
    print(f"Experiment: {agent_name} | Grid: {grid_size}x{grid_size} | Tiles: {method_name}")
    print(f"{'='*70}")
    
    results = []
    start_time = time.time()
    
    for i in range(num_games):
        print(f"Game {i+1}/{num_games}...", end=' ', flush=True)
        
        # Create agent based on type
        agent_class_name = agent_class.__name__
        if 'Expectimax' in agent_class_name:
            # Expectimax variants with tile distribution
            if 'Fast' in agent_class_name:
                agent = ExpectimaxAgentFast(verbose=False, tile_distribution=tile_dist)
            elif 'Deep' in agent_class_name:
                agent = ExpectimaxAgentDeep(verbose=False, tile_distribution=tile_dist)
            else:
                agent = ExpectimaxAgent(depth=3, verbose=False, tile_distribution=tile_dist)
        elif 'MCTS' in agent_class_name or 'Monte' in agent_class_name:
            # MCTS agents - use default parameters from module
            agent = agent_class()
        else:
            # Other agents (Random, etc.)
            agent = agent_class()
        
        result = run_single_game(agent, grid_size=grid_size, generation_method=generation_method)
        results.append(result)
        
        print(f"Score: {result['score']:5d}, Max tile: {result['max_tile']:4d}, Moves: {result['moves']:3d}")
    
    elapsed_time = time.time() - start_time
    
    # Calculate statistics
    scores = [r['score'] for r in results]
    max_tiles = [r['max_tile'] for r in results]
    moves = [r['moves'] for r in results]
    wins = sum(1 for r in results if r['won'])
    
    # Tile achievements
    tile_thresholds = [128, 256, 512, 1024, 2048, 4096]
    tile_counts = {t: sum(1 for tile in max_tiles if tile >= t) for t in tile_thresholds}
    
    # Print summary
    print(f"\n{'-'*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'-'*70}")
    print(f"Configuration: {grid_size}x{grid_size} grid, {method_name} tiles")
    print(f"Games played: {num_games}")
    print(f"Total time: {elapsed_time:.2f}s (avg: {elapsed_time/num_games:.2f}s per game)")
    print(f"\nScore Statistics:")
    print(f"  Average: {sum(scores)/len(scores):.2f}")
    print(f"  Min: {min(scores)}, Max: {max(scores)}")
    print(f"\nMax Tile Statistics:")
    print(f"  Average: {sum(max_tiles)/len(max_tiles):.2f}")
    print(f"  Min: {min(max_tiles)}, Max: {max(max_tiles)}")
    print(f"\nMove Statistics:")
    print(f"  Average: {sum(moves)/len(moves):.2f}")
    print(f"\nTile Achievement Rates:")
    for tile, count in sorted(tile_counts.items()):
        if tile <= max(max_tiles):  # Only show relevant tiles
            print(f"  {tile:4d}+: {count:2d}/{num_games} ({100*count/num_games:5.1f}%)")
    print(f"\nWin Rate (2048): {wins}/{num_games} ({100*wins/num_games:.1f}%)")
    print(f"{'='*70}\n")
    
    return {
        'config': f"{grid_size}x{grid_size}_{method_name}",
        'avg_score': sum(scores)/len(scores),
        'avg_max_tile': sum(max_tiles)/len(max_tiles),
        'win_rate': wins/num_games,
        'tile_counts': tile_counts,
        'results': results
    }


def main():
    """Main experimental evaluation."""
    # Parse arguments
    if len(sys.argv) <= 1:
        num_games = 10
        test_agents = ['random', 'mcts', 'expectimax']  # Default agents to test
    else:
        num_games = int(sys.argv[1])
        # Optional: specify agents to test as additional arguments
        test_agents = sys.argv[2:] if len(sys.argv) > 2 else list(AGENT_CLASSES.keys())
    
    print("\n" + "="*70)
    print("COMPREHENSIVE AGENT EVALUATION")
    print("Testing generalization across grid sizes and tile distributions")
    print(f"Agents to test: {', '.join(test_agents)}")
    print("="*70)
    
    # Create generation methods
    default_gen = Default()
    scaling_gen = Scaling()
    
    # Store all results for comparison
    all_results = []
    
    # Test each agent
    for agent_key in test_agents:
        if agent_key not in AGENT_CLASSES:
            print(f"\nWarning: Unknown agent '{agent_key}', skipping...")
            continue
        
        agent_class = AGENT_CLASSES[agent_key]
        agent_display_name = agent_key.upper()
        
        print("\n" + "#"*70)
        print(f"# TESTING AGENT: {agent_display_name}")
        print("#"*70)
        
        # Test 6 scenarios: 4x4, 5x5, 6x6 with both Default and Scaling tiles
        for grid_size in [4, 5, 6]:
            # Default tiles
            exp_name = f"{agent_display_name}_{grid_size}x{grid_size}_Default"
            print(f"\n--- {agent_display_name} on {grid_size}x{grid_size} (Default tiles) ---")
            result = run_experiment(agent_class, grid_size=grid_size, generation_method=default_gen,
                                  num_games=num_games, agent_name=exp_name)
            all_results.append((exp_name, result))
            
            # Scaling tiles
            exp_name_scaling = f"{agent_display_name}_{grid_size}x{grid_size}_Scaling"
            print(f"\n--- {agent_display_name} on {grid_size}x{grid_size} (Scaling tiles) ---")
            result_scaling = run_experiment(agent_class, grid_size=grid_size, generation_method=scaling_gen,
                                          num_games=num_games, agent_name=exp_name_scaling)
            all_results.append((exp_name_scaling, result_scaling))
    
    # ==========================================
    # FINAL COMPARISON
    # ==========================================
    print("\n" + "="*70)
    print("FINAL COMPARISON - ALL EXPERIMENTS")
    print("="*70)
    print(f"\n{'Configuration':<25} {'Avg Score':>12} {'Avg Max Tile':>12} {'Win Rate':>10}")
    print("-"*70)
    
    for name, result in all_results:
        print(f"{name:<25} {result['avg_score']:>12.1f} {result['avg_max_tile']:>12.1f} {result['win_rate']:>9.1%}")
    
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nTested {len(test_agents)} agent(s) across multiple configurations")
    print(f"Total experiments run: {len(all_results)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()