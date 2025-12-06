"""
Random generation method implementation for the 2048 game.
This mimics the original add_two behavior - randomly places a "2" in an empty cell.
"""

import random
from generation_methods.base import GenerationMethod


class Random2(GenerationMethod):
    """
    Random generation method that places a "2" in a random empty cell.
    This is the default behavior matching the original add_two function.
    """
    
    def add_tile(self, matrix):
        """
        Add a "2" tile to a random empty cell in the matrix.
        
        Args:
            matrix: The current game matrix (list of lists)
            
        Returns:
            The modified matrix with a new "2" tile added
        """
        # Find all empty cells
        empty_cells = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    empty_cells.append((i, j))
        
        # If there are empty cells, randomly select one and place a "2"
        if empty_cells:
            i, j = random.choice(empty_cells)
            matrix[i][j] = 2
        
        return matrix

