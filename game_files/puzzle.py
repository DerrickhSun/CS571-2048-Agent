from tkinter import Frame, Label, CENTER
import random

# Handle imports for both direct execution and package import
try:
    # Try relative imports first (works when imported as package)
    from . import logic
    from . import constants as c
except ImportError:
    # Try absolute package imports (works when run from parent directory)
    try:
        from game_files import logic
        from game_files import constants as c
    except ImportError:
        # Fallback to simple imports (works when run directly from game_files directory)
        import logic
        import constants as c

# Handle imports for generation methods
try:
    from generation_methods import Random2
except ImportError:
    # Fallback if generation_methods is not available
    Random2 = None

def gen():
    return random.randint(0, c.GRID_LEN - 1)


class GameGrid(Frame):
    def __init__(self, auto_start=True, root=None, generation_method=None):
        if root is None:
            Frame.__init__(self)
        else:
            Frame.__init__(self, root)

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)
        
        # Set up generation method (default to Random2 if none provided)
        if generation_method is None:
            if Random2 is not None:
                self.generation_method = Random2()
            else:
                # Fallback to None - will use logic.add_two directly
                self.generation_method = None
        else:
            self.generation_method = generation_method

        self.commands = {
            c.KEY_UP: logic.up,
            c.KEY_DOWN: logic.down,
            c.KEY_LEFT: logic.left,
            c.KEY_RIGHT: logic.right,
            c.KEY_UP_ALT1: logic.up,
            c.KEY_DOWN_ALT1: logic.down,
            c.KEY_LEFT_ALT1: logic.left,
            c.KEY_RIGHT_ALT1: logic.right,
            c.KEY_UP_ALT2: logic.up,
            c.KEY_DOWN_ALT2: logic.down,
            c.KEY_LEFT_ALT2: logic.left,
            c.KEY_RIGHT_ALT2: logic.right,
        }
        
        # Map direction strings and integers to logic functions
        self.direction_map = {
            'up': logic.up,
            'down': logic.down,
            'left': logic.left,
            'right': logic.right,
            0: logic.up,
            1: logic.down,
            2: logic.left,
            3: logic.right,
        }

        self.grid_cells = []
        self.score = 0
        self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = []
        self.update_grid_cells()
        self.update_score_display()

        if auto_start:
            self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME,width=c.SIZE, height=c.SIZE)
        background.grid(row=0, column=0)

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)
        
        # Create score label below the game grid
        score_frame = Frame(self, bg=c.BACKGROUND_COLOR_GAME)
        score_frame.grid(row=1, column=0, pady=10)
        
        score_label_text = Label(
            master=score_frame,
            text="Score:",
            bg=c.BACKGROUND_COLOR_GAME,
            fg="#f9f6f2",
            font=("Verdana", 20, "bold")
        )
        score_label_text.pack(side="left", padx=5)
        
        self.score_label = Label(
            master=score_frame,
            text="0",
            bg=c.BACKGROUND_COLOR_GAME,
            fg="#f9f6f2",
            font=("Verdana", 20, "bold")
        )
        self.score_label.pack(side="left", padx=5)

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="",bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number]
                    )
        self.update_idletasks()
    
    def update_score_display(self):
        """Update the score label with the current score."""
        if hasattr(self, 'score_label'):
            self.score_label.configure(text=str(self.score))

    def _handle_post_move(self):
        """Common logic after a successful move."""
        # Use generation method if available, otherwise fallback to logic.add_two
        if self.generation_method is not None:
            self.matrix = self.generation_method.add_tile(self.matrix)
        else:
            self.matrix = logic.add_two(self.matrix)
        # record last move
        self.history_matrixs.append(self.matrix)
        self.update_grid_cells()
        if hasattr(self, 'score_label'):
            self.update_score_display()
        
        game_state = logic.game_state(self.matrix)
        if game_state == 'win':
            self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        elif game_state == 'lose':
            self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)

    def apply_move_direct(self, action):
        """
        Apply a move directly using action integer and return raw results.
        This is useful for agents that need direct access to move results.
        
        Args:
            action: Integer (0=up, 1=down, 2=left, 3=right)
        
        Returns:
            Tuple of (new_matrix, done, score) where:
            - new_matrix: The matrix after the move (copy)
            - done: Boolean indicating if the move was valid
            - score: The score from this move
        """
        if action not in [0, 1, 2, 3]:
            return self.matrix, False, 0
        
        move_function = self.direction_map[action]
        new_matrix, done, score = move_function(self.matrix)
        return new_matrix, done, score

    def make_move(self, direction):
        """
        Execute a move in the specified direction.
        
        Args:
            direction: Can be a string ('up', 'down', 'left', 'right') or 
                      an integer (0=up, 1=down, 2=left, 3=right)
        
        Returns:
            True if the move was valid and executed, False otherwise.
        """
        if direction not in self.direction_map:
            return False
        
        move_function = self.direction_map[direction]
        self.matrix, done, score = move_function(self.matrix)
        if done:
            self.score += score
            self._handle_post_move()
            return True
        
        return False
    
    def key_down(self, event):
        key = event.keysym
        print(event)
        if key == c.KEY_QUIT: exit()
        if key == c.KEY_BACK and len(self.history_matrixs) > 1:
            self.matrix = self.history_matrixs.pop()
            self.update_grid_cells()
            print('back on step total step:', len(self.history_matrixs))
        elif key in self.commands:
            # Use the new make_move method
            direction_map_key = None
            if key in [c.KEY_UP, c.KEY_UP_ALT1, c.KEY_UP_ALT2]:
                direction_map_key = 'up'
            elif key in [c.KEY_DOWN, c.KEY_DOWN_ALT1, c.KEY_DOWN_ALT2]:
                direction_map_key = 'down'
            elif key in [c.KEY_LEFT, c.KEY_LEFT_ALT1, c.KEY_LEFT_ALT2]:
                direction_map_key = 'left'
            elif key in [c.KEY_RIGHT, c.KEY_RIGHT_ALT1, c.KEY_RIGHT_ALT2]:
                direction_map_key = 'right'
            
            if direction_map_key:
                self.make_move(direction_map_key)

    def generate_next(self):
        index = (gen(), gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (gen(), gen())
        self.matrix[index[0]][index[1]] = 2


if __name__ == "__main__":
    game_grid = GameGrid()