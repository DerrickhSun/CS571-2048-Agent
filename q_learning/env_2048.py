import logic
import constants as c
import random


# Action mapping:
# 0 = up, 1 = down, 2 = left, 3 = right
ACTIONS = [0, 1, 2, 3]

def encode_state(matrix):
    # Flatten the matrix into a tuple so it can be a dict key
    return tuple(v for row in matrix for v in row)

class Game2048Env:
    def __init__(self):
        self.matrix = None

    def reset(self):
        # New game matrix
        self.matrix = logic.new_game(c.GRID_LEN)

        return encode_state(self.matrix)

    def step(self, action):
        # pick the right move function
        if action == 0:
            move_fn = logic.up
        elif action == 1:
            move_fn = logic.down
        elif action == 2:
            move_fn = logic.left
        elif action == 3:
            move_fn = logic.right
        else:
            print("Cannot Step Environment! Invalid action received!")
            raise ValueError("Cannot Step Environment! Invalid action received!")

        # Apply move
        new_matrix, done_move, merge_score = move_fn(self.matrix)

        # If move is invalid: no tile moved or merged
        if not done_move:
            # small penalty, and state does not change
            return encode_state(self.matrix), -1.0, False

        # Valid move: use merge_score as base reward
        reward = merge_score

        # After a valid move, spawn a new random tile
        new_matrix = logic.add_two(new_matrix)
        self.matrix = new_matrix

        # Check terminal
        game_status = logic.game_state(self.matrix)

        if game_status == "win":
            reward += 100.0
            return encode_state(self.matrix), reward, True

        if game_status == "lose":
            reward -= 100.0
            return encode_state(self.matrix), reward, True

        # Otherwise game continues
        return encode_state(self.matrix), reward, False

    def get_valid_actions(self) -> list[int]:
        """Optional helper for smarter exploration."""
        valid = []
        for action in ACTIONS:
            if action == 0:
                move_fn = logic.up
            elif action == 1:
                move_fn = logic.down
            elif action == 2:
                move_fn = logic.left
            elif action == 3:
                move_fn = logic.right
            else:
                raise ValueError("Action found in ACTIONS list not a defined!")
            
            _, done_move, _ = move_fn([row[:] for row in self.matrix])
            if done_move:
                valid.append(action)
        return valid
