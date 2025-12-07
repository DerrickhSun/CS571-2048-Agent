"""
Expectimax Agent for 2048 Game
Implements depth-limited Expectimax search with heuristic evaluation.
"""

import copy
import math
from agents.base import Agent


class ExpectimaxAgent(Agent):
    """
    Expectimax agent that uses game-tree search with heuristic evaluation.
    
    The agent alternates between:
    - Max nodes: Player choosing the best move
    - Chance nodes: Random tile placement by the game
    """
    
    def __init__(self, depth=3, tile_distribution='standard', verbose=True):
        """
        Initialize the Expectimax agent.
        
        Args:
            depth: Maximum search depth (3-4 is typical, higher = slower but smarter)
            tile_distribution: 'standard' for {2:90%, 4:10%} or 'modified' for variable tiles
            verbose: If True, print debug information during gameplay
        """
        super().__init__()
        self.depth = depth
        self.tile_distribution = tile_distribution
        self.verbose = verbose
        
       
        # Empty cells should DOMINATE - be much larger than other factors
        self.weights = {
            'monotonicity': 1.0,
            'smoothness': 1.0,
            'empty_cells': 100000.0,  # Very high weight to prioritize empty cells
            'max_corner': 10.0,       
            'merge_potential': 1.0,
            'border_penalty': 0.1      # Penalty for middle tiles
        }
        
        # Statistics for debugging
        self.nodes_evaluated = 0
        self.cache_hits = 0
        
    def next_move(self, game_grid):
        """
        Determine the next move using Expectimax search.
        
        Args:
            game_grid: The GameGrid instance containing current game state
            
        Returns:
            Best direction string ('up', 'down', 'left', 'right') or None
        """
        # Reset statistics
        self.nodes_evaluated = 0
        self.cache_hits = 0
        
        # Run Expectimax search
        best_score, best_move = self.expectimax(
            game_grid.matrix, 
            self.depth, 
            is_max_node=True,
            game_grid=game_grid
        )
        
        # Debug output
        if self.verbose:
            max_tile = max(max(row) for row in game_grid.matrix)
            empty_cells = sum(row.count(0) for row in game_grid.matrix)
            print(f"Move: {best_move:5s} | Heuristic: {best_score:7.1f} | Max tile: {max_tile:4d} | Empty: {empty_cells} | Nodes: {self.nodes_evaluated}")
        
        return best_move
    
    def expectimax(self, matrix, depth, is_max_node, game_grid=None):
        """
        Recursive Expectimax search.
        
        Args:
            matrix: Current game board state
            depth: Remaining search depth
            is_max_node: True for player moves, False for chance nodes
            game_grid: GameGrid instance (needed for move functions)
            
        Returns:
            (score, move) tuple
        """
        self.nodes_evaluated += 1
        
        # Base case: reached depth limit or terminal state
        if depth == 0 or self.is_terminal(matrix):
            return self.evaluate_state(matrix), None
        
        if is_max_node:
            # MAX node: player chooses best move
            return self.max_node(matrix, depth, game_grid)
        else:
            # CHANCE node: expected value over random tile placements
            return self.chance_node(matrix, depth, game_grid)
    
    def max_node(self, matrix, depth, game_grid):
        """
        Handle MAX node: player choosing the best move.
        """
        max_score = float('-inf')
        best_move = None
        
        # Try all four directions
        for direction in ['up', 'down', 'left', 'right']:
            # Simulate the move
            test_matrix = copy.deepcopy(matrix)
            move_func = game_grid.direction_map[direction]
            new_matrix, move_valid, score_gained = move_func(test_matrix)
            
            # Skip invalid moves
            if not move_valid:
                continue
            
            # Recurse to chance node (tile spawning)
            expected_score, _ = self.expectimax(
                new_matrix, 
                depth - 1, 
                is_max_node=False,
                game_grid=game_grid
            )
            
            # Add immediate reward from this move
            total_score = expected_score + score_gained * 0.1
            
            # Update best move
            if total_score > max_score:
                max_score = total_score
                best_move = direction
        
        # If no valid moves, return current state evaluation
        if best_move is None:
            return self.evaluate_state(matrix), None
        
        return max_score, best_move
    
    def chance_node(self, matrix, depth, game_grid):
        """
        Handle CHANCE node: expected value over random tile spawns.
        """
        # Get empty cells
        empty_cells = []
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                if matrix[i][j] == 0:
                    empty_cells.append((i, j))
        
        # If no empty cells, game is over
        if not empty_cells:
            return self.evaluate_state(matrix), None
        
        # Get tile probabilities based on distribution type
        tile_probs = self.get_tile_probabilities(matrix)
        
        # Calculate expected value
        expected_value = 0.0
        cell_probability = 1.0 / len(empty_cells)
        
        # For performance: sample empty cells if too many
        # At deep depths, evaluating all empty cells is expensive
        sample_cells = empty_cells
        if len(empty_cells) > 6 and depth < self.depth - 1:
            import random
            sample_cells = random.sample(empty_cells, 6)
            cell_probability = 1.0 / 6
        
        # Try each empty cell
        for (i, j) in sample_cells:
            # Try each possible tile value
            for tile_value, tile_prob in tile_probs:
                # Create new state with tile placed
                new_matrix = copy.deepcopy(matrix)
                new_matrix[i][j] = tile_value
                
                # Recurse to max node
                score, _ = self.expectimax(
                    new_matrix,
                    depth - 1,
                    is_max_node=True,
                    game_grid=game_grid
                )
                
                # Add weighted contribution
                expected_value += score * cell_probability * tile_prob
        
        return expected_value, None
    
    def get_tile_probabilities(self, matrix):
        """
        Get tile spawn probabilities based on distribution type.
        
        Returns:
            List of (tile_value, probability) tuples
        """
        if self.tile_distribution == 'standard':
            # Standard 2048: 90% chance of 2, 10% chance of 4
            return [(2, 0.9), (4, 0.1)]
        else:
            # Modified: equal probability for all tiles up to current max
            max_tile = max(max(row) for row in matrix)
            if max_tile < 2:
                max_tile = 2
            
            # Generate all powers of 2 up to max_tile
            tiles = []
            power = 1
            while 2**power <= max_tile:
                tiles.append(2**power)
                power += 1
            
            prob = 1.0 / len(tiles)
            return [(tile, prob) for tile in tiles]
    
    def is_terminal(self, matrix):
        """
        Check if the game state is terminal (no valid moves).
        """
        n = len(matrix)
        
        # Check for empty cells
        for i in range(n):
            for j in range(n):
                if matrix[i][j] == 0:
                    return False
        
        # Check for possible merges (horizontal)
        for i in range(n):
            for j in range(n - 1):
                if matrix[i][j] == matrix[i][j + 1]:
                    return False
        
        # Check for possible merges (vertical)
        for i in range(n - 1):
            for j in range(n):
                if matrix[i][j] == matrix[i + 1][j]:
                    return False
        
        return True
    
    # ========================
    # HEURISTIC FUNCTIONS
    # ========================
    
    def evaluate_state(self, matrix):
        """
        Evaluate a game state using weighted heuristics.
        
        Returns:
            Numerical score (higher = better state)
        """
        mono = self.monotonicity_score(matrix)
        smooth = self.smoothness_score(matrix)
        empty = self.empty_cells_score(matrix)
        corner = self.max_corner_score(matrix)
        merges = self.merge_potential_score(matrix)
        border = self.border_penalty_score(matrix)
        
        # Weighted combination
        total_score = (
            self.weights['monotonicity'] * mono +
            self.weights['smoothness'] * smooth +
            self.weights['empty_cells'] * empty +
            self.weights['max_corner'] * corner +
            self.weights['merge_potential'] * merges +
            self.weights['border_penalty'] * border
        )
        
        return total_score
    
    def monotonicity_score(self, matrix):
        """
        Reward boards where tiles increase/decrease monotonically.
        Checks both rows and columns, both directions.
        """
        score = 0
        n = len(matrix)
        
        # Check rows
        for i in range(n):
            row = [matrix[i][j] for j in range(n) if matrix[i][j] != 0]
            if len(row) > 1:
                # Check increasing
                increasing = sum(1 for k in range(len(row) - 1) if row[k] <= row[k + 1])
                # Check decreasing
                decreasing = sum(1 for k in range(len(row) - 1) if row[k] >= row[k + 1])
                score += max(increasing, decreasing)
        
        # Check columns
        for j in range(n):
            col = [matrix[i][j] for i in range(n) if matrix[i][j] != 0]
            if len(col) > 1:
                increasing = sum(1 for k in range(len(col) - 1) if col[k] <= col[k + 1])
                decreasing = sum(1 for k in range(len(col) - 1) if col[k] >= col[k + 1])
                score += max(increasing, decreasing)
        
        return score
    
    def smoothness_score(self, matrix):
        """
        Reward boards where adjacent tiles have similar values.
        Uses log scale to compare tile values proportionally.
        """
        smoothness = 0
        n = len(matrix)
        
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != 0:
                    value = math.log2(matrix[i][j])
                    
                    # Check right neighbor
                    if j < n - 1 and matrix[i][j + 1] != 0:
                        target = math.log2(matrix[i][j + 1])
                        smoothness -= abs(value - target)
                    
                    # Check down neighbor
                    if i < n - 1 and matrix[i + 1][j] != 0:
                        target = math.log2(matrix[i + 1][j])
                        smoothness -= abs(value - target)
        
        return smoothness
    
    def empty_cells_score(self, matrix):
        """
        Count empty cells. More empty cells = more flexibility.
        """
        count = 0
        for row in matrix:
            count += row.count(0)
        return count
    
    def max_corner_score(self, matrix):
        """
        Reward keeping the maximum tile in a corner.
        Top-left corner is preferred for consistent strategy.
        """
        n = len(matrix)
        max_tile = max(max(row) for row in matrix)
        
        # Check if max tile is in a corner
        if matrix[0][0] == max_tile:
            # Top-left corner (preferred)
            return max_tile * 4
        elif matrix[0][n-1] == max_tile:
            # Top-right corner
            return max_tile * 2
        elif matrix[n-1][0] == max_tile:
            # Bottom-left corner
            return max_tile * 2
        elif matrix[n-1][n-1] == max_tile:
            # Bottom-right corner
            return max_tile * 2
        else:
            # Max tile not in corner (bad!)
            return 0
    
    def border_penalty_score(self, matrix):
        """
        Penalize tiles that are not on borders (like the paper).
        Uses quadratic distance from border.
        """
        penalty = 0
        n = len(matrix)
        
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != 0:
                    # Calculate distance from nearest border
                    dist_from_border = min(i, j, n-1-i, n-1-j)
                    # Quadratic penalty
                    penalty -= matrix[i][j] * (dist_from_border ** 2)
        
        return penalty
    
    def merge_potential_score(self, matrix):
        """
        Count how many adjacent tiles can potentially merge.
        """
        merges = 0
        n = len(matrix)
        
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != 0:
                    # Check right neighbor
                    if j < n - 1 and matrix[i][j] == matrix[i][j + 1]:
                        merges += 1
                    
                    # Check down neighbor
                    if i < n - 1 and matrix[i][j] == matrix[i + 1][j]:
                        merges += 1
        
        return merges


class ExpectimaxAgentFast(ExpectimaxAgent):
    """
    Faster variant with depth=2 for quicker decisions.
    Good for testing or faster gameplay.
    """
    def __init__(self, verbose=True):
        super().__init__(depth=2, verbose=verbose)


class ExpectimaxAgentDeep(ExpectimaxAgent):
    """
    Deeper search variant with depth=4 for stronger play.
    Slower but makes better decisions.
    """
    def __init__(self, verbose=True):
        super().__init__(depth=4, verbose=verbose)