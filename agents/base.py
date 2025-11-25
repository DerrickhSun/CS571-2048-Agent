"""
Base agent class for 2048 game agents.
All agents should inherit from this class and implement the next_move method.
"""


class Agent:
    """
    Base class for all 2048 game agents.
    
    Agents can maintain state between moves using instance variables.
    """
    
    def __init__(self):
        """Initialize the agent. Override to set up any needed state."""
        pass
    
    def next_move(self, game_grid):
        """
        Determine the next move given the current game state.
        
        Args:
            game_grid: The GameGrid instance containing the current game state
            
        Returns:
            A direction string ('up', 'down', 'left', 'right') or None if no valid moves
        """
        raise NotImplementedError("Subclasses must implement next_move method")

