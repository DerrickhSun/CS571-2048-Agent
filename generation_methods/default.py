"""
Default generation method implementation for the 2048 game.
This mimics the standard 2048 behavior - randomly places a "2" (90% chance) or "4" (10% chance) in an empty cell.
"""

import random
from generation_methods.base import GenerationMethod


class Default(GenerationMethod):
    """
    Default generation method that places a "2" (90% chance) or "4" (10% chance) in a random empty cell.
    This matches the standard 2048 game behavior.
    """
    
    def add_tile(self, matrix):
        """
        Add a "2" or "4" tile to a random empty cell in the matrix.
        Probability: 90% for "2", 10% for "4".
        
        Args:
            matrix: The current game matrix (list of lists)
            
        Returns:
            The modified matrix with a new tile added
        """
        # Find all empty cells
        empty_cells = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    empty_cells.append((i, j))
        
        # If there are empty cells, randomly select one and place a tile
        if empty_cells:
            i, j = random.choice(empty_cells)
            # 90% chance for "2", 10% chance for "4"
            tile_value = 2 if random.random() < 0.9 else 4
            matrix[i][j] = tile_value
        
        return matrix

