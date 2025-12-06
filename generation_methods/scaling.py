"""
Scaling generation method implementation for the 2048 game.
This method generates tiles based on the current maximum value on the board.
"""

import random
from generation_methods.base import GenerationMethod


class Scaling(GenerationMethod):
    """
    Scaling generation method that places a tile with value equal to any power of 2
    less than or equal to the maximum value on the board, with equal probability for each.
    """
    
    def add_tile(self, matrix):
        """
        Add a tile to a random empty cell. The tile value is a power of 2
        less than or equal to the maximum value on the board.
        
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
        
        # If there are no empty cells, return matrix unchanged
        if not empty_cells:
            return matrix
        
        # Find the maximum value on the board
        max_value = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] > max_value:
                    max_value = matrix[i][j]
        
        # If board is empty (max_value is 0), default to generating 2
        if max_value == 0:
            max_value = 2
        
        # Find all powers of 2 less than or equal to max_value
        # Start from 2 and go up to max_value
        possible_values = []
        power = 2
        while power <= max_value:
            possible_values.append(power)
            power *= 2
        
        # Randomly select one of the possible values
        tile_value = random.choice(possible_values)
        
        # Place the tile in a random empty cell
        i, j = random.choice(empty_cells)
        matrix[i][j] = tile_value
        
        return matrix

