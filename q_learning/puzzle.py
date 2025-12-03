import sys
from tkinter import Frame, Label, CENTER
import random
import logic
import constants as c


import pickle
from env_2048 import encode_state 

# Load Q table
try:
    with open("q_table_shaped.pkl", "rb") as f:
        Q = pickle.load(f)
    print("Loaded Q table with", len(Q), "states")
except:
    print("Warning: Q table not found")
    

def agent_best_action(matrix):
    state = encode_state(matrix)

    if state not in Q:
        # Unseen state, fallback to any move
        return random.choice([0, 1, 2, 3])

    q_vals = Q[state]
    max_q = max(q_vals)
    best_actions = [a for a, q in enumerate(q_vals) if q == max_q]
    return random.choice(best_actions)

def gen():
    return random.randint(0, c.GRID_LEN - 1)

class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)
        # Info panel for agent decision display
        self.info_frame = Frame(self, bg="#faf8ef")
        self.info_frame.grid(row=0, column=0, pady=10)

        self.q_label = Label(self.info_frame, text="Q: [0,0,0,0]", font=("Helvetica", 12))
        self.q_label.grid(row=0, column=0, padx=5)

        self.action_label = Label(self.info_frame, text="Action: -", font=("Helvetica", 12))
        self.action_label.grid(row=0, column=1, padx=5)

        self.reward_label = Label(self.info_frame, text="Reward: 0", font=("Helvetica", 12))
        self.reward_label.grid(row=0, column=2, padx=5)

        self.known_label = Label(self.info_frame, text="State known: No", font=("Helvetica", 12))
        self.known_label.grid(row=0, column=3, padx=5)


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

        self.grid_cells = []
        self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = []
        self.update_grid_cells()
        if AUTO_AGENT:
            # start agent autoplay as soon as the GUI loads
            self.after(200, self.agent_autoplay)


        self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME,width=c.SIZE, height=c.SIZE)
        background.grid()

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

    def _handle_post_move(self):
        """Common logic after a successful move."""
        self.matrix = logic.add_two(self.matrix)
        self.history_matrixs.append(self.matrix)
        self.update_grid_cells()

        state = logic.game_state(self.matrix)
        if state == "win":
            self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        elif state == "lose":
            self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)

    def agent_autoplay(self):
        if logic.game_state(self.matrix) in ("win", "lose"):
            return

        state = encode_state(self.matrix)

        # Check whether state is in Q
        known = state in Q
        q_vals = Q[state] if known else None

        # Pick action
        action = agent_best_action(self.matrix)

        move_map = {
            0: logic.up,
            1: logic.down,
            2: logic.left,
            3: logic.right
        }
        move_fn = move_map[action]

        new_matrix, done, merge_reward = move_fn(self.matrix)

        # Update info panel before applying the move
        self.update_info_panel(q_vals, action, merge_reward if done else 0, known)

        if done:
            self.matrix = new_matrix
            self._handle_post_move()

        # continue loop
        self.after(50, self.agent_autoplay)




    def key_down(self, event):
        key = event.keysym
        print(event)
        if key == c.KEY_QUIT: exit()
        if key == c.KEY_BACK and len(self.history_matrixs) > 1:
            self.matrix = self.history_matrixs.pop()
            self.update_grid_cells()
            print('back on step total step:', len(self.history_matrixs))
            return

        # Single step agent move with "a"
        if key == "a":
            state = encode_state(self.matrix)
            known = state in Q
            q_vals = Q[state] if known else None

            action = agent_best_action(self.matrix)
            move_map = {0: logic.up, 1: logic.down, 2: logic.left, 3: logic.right}
            move_fn = move_map[action]

            new_matrix, done, merge_reward = move_fn(self.matrix)

            self.update_info_panel(q_vals, action, merge_reward if done else 0, known)

            if done:
                self.matrix = new_matrix
                self._handle_post_move()
            return


       
        # Human arrow key or alternate mapping
        if key in self.commands:
            self.matrix, done, _ = self.commands[key](self.matrix)

            if done:
                self._handle_post_move()

    def generate_next(self):
        index = (gen(), gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (gen(), gen())
        self.matrix[index[0]][index[1]] = 2

    def update_info_panel(self, q_vals, action, reward, known):
        # q_vals is a list of four floats
        if q_vals is None:
            self.q_label.config(text="Q: n/a")
        else:
            self.q_label.config(text=f"Q: [{q_vals[0]:.3f}, {q_vals[1]:.3f}, {q_vals[2]:.3f}, {q_vals[3]:.3f}]")

        # action indicator
        action_map = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
        if action is None:
            self.action_label.config(text="Action: -")
        else:
            self.action_label.config(text=f"Action: {action_map[action]}")

        # merge reward for that move
        self.reward_label.config(text=f"Reward: {reward}")

        # known state
        known_text = "Yes" if known else "No"
        self.known_label.config(text=f"State known: {known_text}")


AUTO_AGENT = False
if len(sys.argv) > 1 and sys.argv[1] == "agent":
    AUTO_AGENT = True
    print("Running in agent autoplay mode")

game_grid = GameGrid()