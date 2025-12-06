"""
Base generation method class for 2048 tile generation.
All generation methods should inherit from this class and implement the add_tile method.
"""


class GenerationMethod:
    """
    Base class for all 2048 tile generation methods.
    
    Generation methods can maintain state between tile additions using instance variables.
    """
    
    def __init__(self):
        """Initialize the generation method. Override to set up any needed state."""
        pass
    
    def add_tile(self, matrix):
        """
        Add a new tile to the given game matrix.
        
        Args:
            matrix: The current game matrix (list of lists)
            
        Returns:
            The modified matrix with a new tile added
        """
        raise NotImplementedError("Subclasses must implement add_tile method")

