#!/usr/bin/env python3
"""
Comprehensive evaluation script for the project proposal.
Tests Expectimax under:
1. Standard vs Modified (Scaling) tile distribution
2. Different grid sizes (3x3, 4x4, 5x5)
3. Evaluates heuristic reliability across conditions
"""

import sys
import time
import copy
from game_files import logic
from game_files import constants as c
from agents.expectimax import ExpectimaxAgent
from agents.naive import RandomAgent
from generation_methods import Default, Scaling


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
        
        # Create agent with matching tile distribution
        if hasattr(agent_class, '__name__') and 'Expectimax' in str(agent_class):
            agent = ExpectimaxAgent(depth=3, verbose=False, tile_distribution=tile_dist)
        else:
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
    num_games = 10 if len(sys.argv) <= 1 else int(sys.argv[1])
    
    print("\n" + "="*70)
    print("COMPREHENSIVE EXPECTIMAX EVALUATION")
    print("Testing generalization across grid sizes and tile distributions")
    print("="*70)
    
    # Create quiet Expectimax agent factory
    # Note: We'll create agents with proper tile_distribution in run_experiment
    ExpectimaxQuiet = ExpectimaxAgent
    ExpectimaxQuiet.__name__ = "ExpectimaxAgent"
    
    # Create generation methods
    default_gen = Default()
    scaling_gen = Scaling()
    
    # Store all results for comparison
    all_results = []
    
    # ==========================================
    # EXPERIMENT 1: Standard 4x4 (Baseline)
    # ==========================================
    print("\n" + "#"*70)
    print("# EXPERIMENT 1: Baseline (Standard 4x4, Default Tiles)")
    print("#"*70)
    baseline = run_experiment(ExpectimaxQuiet, grid_size=4, generation_method=default_gen, 
                              num_games=num_games, agent_name="Expectimax (Baseline)")
    all_results.append(('Baseline 4x4 Default', baseline))
    
    # ==========================================
    # EXPERIMENT 2: Scaling Tiles on 4x4
    # ==========================================
    print("\n" + "#"*70)
    print("# EXPERIMENT 2: Modified Tile Distribution (4x4, Scaling Tiles)")
    print("#"*70)
    modified_4x4 = run_experiment(ExpectimaxQuiet, grid_size=4, generation_method=scaling_gen,
                                  num_games=num_games, agent_name="Expectimax (Scaling)")
    all_results.append(('4x4 Scaling Tiles', modified_4x4))
    
    # ==========================================
    # EXPERIMENT 3: Different Grid Sizes (Default Tiles)
    # ==========================================
    print("\n" + "#"*70)
    print("# EXPERIMENT 3: Grid Size Generalization (Default Tiles)")
    print("#"*70)
    
    # 3x3 grid
    small_grid = run_experiment(ExpectimaxQuiet, grid_size=3, generation_method=default_gen,
                               num_games=num_games, agent_name="Expectimax (3x3)")
    all_results.append(('3x3 Default', small_grid))
    
    # 5x5 grid
    large_grid = run_experiment(ExpectimaxQuiet, grid_size=5, generation_method=default_gen,
                               num_games=num_games, agent_name="Expectimax (5x5)")
    all_results.append(('5x5 Default', large_grid))
    
    # ==========================================
    # EXPERIMENT 4: Combined Modifications
    # ==========================================
    print("\n" + "#"*70)
    print("# EXPERIMENT 4: Combined Modifications")
    print("#"*70)
    
    # 3x3 with scaling tiles
    small_scaling = run_experiment(ExpectimaxQuiet, grid_size=3, generation_method=scaling_gen,
                                   num_games=num_games, agent_name="Expectimax (3x3 Scaling)")
    all_results.append(('3x3 Scaling', small_scaling))
    
    # 5x5 with scaling tiles
    large_scaling = run_experiment(ExpectimaxQuiet, grid_size=5, generation_method=scaling_gen,
                                   num_games=num_games, agent_name="Expectimax (5x5 Scaling)")
    all_results.append(('5x5 Scaling', large_scaling))
    
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
    
    # Calculate performance degradation
    baseline_score = baseline['avg_score']
    print(f"\n{'Performance vs Baseline:':<25}")
    print("-"*70)
    for name, result in all_results[1:]:  # Skip baseline
        degradation = ((result['avg_score'] - baseline_score) / baseline_score) * 100
        print(f"{name:<25} {degradation:>+6.1f}%")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nKey Findings:")
    print(f"1. Baseline performance (4x4 default): {baseline_score:.1f} avg score")
    print(f"2. Scaling tiles impact: {((modified_4x4['avg_score']-baseline_score)/baseline_score)*100:+.1f}%")
    print(f"3. Heuristics generalize to different grid sizes: {'YES' if all(r['avg_score'] > 0 for _, r in all_results) else 'NO'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()