"""Improved MCTS agent for 2048 game.

Module-level defaults are defined here so callers (e.g. `main.py`) can
instantiate agents without passing parameters and still get consistent
behaviour. To change simulation depth/count/epsilon, edit the constants
below.
"""

import math
import random
from agents.base import Agent
from game_files import logic

DEFAULT_NUM_SIMULATIONS = 40  # Simulations per move
DEFAULT_ROLLOUT_DEPTH = 30    # Max rollout depth
DEFAULT_ROLLOUT_EPSILON = 0.15 # Epsilon for rollout policy

MOVE_FUNCS = {d: getattr(logic, d) for d in ['up', 'down', 'left', 'right']}


def add_random_tile(state):
    """Add a random tile (2 or 4) to an empty cell."""
    empty = [(i, j) for i, row in enumerate(state) for j, v in enumerate(row) if v == 0]
    if not empty:
        return state
    i, j = random.choice(empty)
    new = [row[:] for row in state]
    new[i][j] = 2 if random.random() < 0.9 else 4
    return new


def all_spawn_children(state):
    """Return all possible tile spawn outcomes and their probabilities."""
    empty = [(i, j) for i, row in enumerate(state) for j, v in enumerate(row) if v == 0]
    if not empty:
        return []
    ecount = len(empty)
    p2, p4 = 0.9 / ecount, 0.1 / ecount
    children = []
    for i, j in empty:
        for val, p in [(2, p2), (4, p4)]:
            s = [row[:] for row in state]
            s[i][j] = val
            children.append((s, p))
    return children


def apply_move(state, direction):
    """Apply a move to the board. Returns (new_state, moved, score)."""
    board = [row[:] for row in state]
    res = MOVE_FUNCS[direction](board)
    if isinstance(res, tuple) and len(res) == 3:
        a, b, c = res
        if isinstance(a, list):
            return a, bool(b), c
        elif isinstance(b, list):
            return b, bool(a), c
    raise RuntimeError(f"Unexpected result from move function: {res}")


def heuristic(state):
    """Heuristic: empty cells * 100 + log2(max_tile) * 20."""
    empty = sum(v == 0 for row in state for v in row)
    max_tile = max(v for row in state for v in row)
    return empty * 100.0 + math.log(max_tile + 1, 2) * 20.0


class MCTSNode:
    """Node for MCTS tree (player/chance)."""
    __slots__ = ('state', 'parent', 'children', 'visits', 'value', 'node_type', 'prob')

    def __init__(self, state, parent=None, node_type='player', prob=1.0):
        self.state = [row[:] for row in state]
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.node_type = node_type
        self.prob = prob

    def is_terminal(self):
        """True if game is over."""
        return logic.game_state(self.state) != 'not over'

    def ucb1(self, parent_visits, c):
        """UCB1 score for selection."""
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c * math.sqrt(math.log(parent_visits) / self.visits)


class ImprovedMCTSAgent(Agent):
    """Monte Carlo Tree Search agent for 2048."""
    
    def __init__(self, num_simulations=DEFAULT_NUM_SIMULATIONS, exploration_constant=1.4, 
                 rollout_depth=DEFAULT_ROLLOUT_DEPTH, rollout_epsilon=DEFAULT_ROLLOUT_EPSILON):
        super().__init__()
        self.num_simulations = num_simulations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.rollout_epsilon = rollout_epsilon
        self.root = None

    def next_move(self, game_grid):
        """Return best move using MCTS simulations."""
        root_state = [row[:] for row in game_grid.matrix]
        self.root = MCTSNode(root_state, parent=None, node_type='player')
        
        # Quick check: if terminal, no move
        if self.root.is_terminal():
            return None

        for _ in range(self.num_simulations):
            node, path = self._select(self.root)
            
            if not node.is_terminal():
                self._expand(node)
                # After expansion, pick one child to simulate from
                if node.children:
                    if node.node_type == 'chance':
                        # Sample a child proportionally to their probabilities
                        children_list = list(node.children.items())
                        total = sum(child.prob for _, child in children_list)
                        if total <= 0:
                            chosen = random.choice(list(node.children.values()))
                        else:
                            r = random.random() * total
                            acc = 0.0
                            chosen = children_list[-1][1]
                            for _, child in children_list:
                                acc += child.prob
                                if r <= acc:
                                    chosen = child
                                    break
                    else:
                        # Player node -> pick a random child
                        chosen = random.choice(list(node.children.values()))
                    node = chosen

            # Simulate from node.state
            reward = self._simulate(node.state)
            # Backpropagate reward
            self._backpropagate(node, reward)

        # After simulations, choose best root child by average value
        if not self.root.children:
            return None

        best_dir = None
        best_avg = -float('inf')
        for direction, chance_node in self.root.children.items():
            if chance_node.visits == 0:
                avg = -float('inf')
            else:
                avg = chance_node.value / chance_node.visits
            if avg > best_avg:
                best_avg = avg
                best_dir = direction

        return best_dir

    def _select(self, root):
        """Descend tree using UCB1 until expansion/simulation needed."""
        node = root
        path = [node]
        
        while True:
            if node.node_type == 'player':
                # If no children or terminal, return for expansion/simulation
                if not node.children or node.is_terminal():
                    return node, path
                
                # Select child with max UCB over chance nodes
                best_score = -float('inf')
                best_child = None
                for direction, child in node.children.items():
                    score = child.ucb1(node.visits or 1, self.c)
                    if score > best_score:
                        best_score = score
                        best_child = child
                
                node = best_child
                path.append(node)
            else:  # chance node
                # If no children or terminal, return for expansion/simulation
                if not node.children or node.is_terminal():
                    return node, path
                
                # For chance nodes, sample according to spawn probabilities
                children_list = list(node.children.values())
                probs = [c.prob for c in children_list]
                total = sum(probs)
                
                if total <= 0:
                    node = random.choice(children_list)
                else:
                    r = random.random() * total
                    acc = 0.0
                    chosen = children_list[-1]
                    for c in children_list:
                        acc += c.prob
                        if r <= acc:
                            chosen = c
                            break
                    node = chosen
                
                path.append(node)

    def _expand(self, node):
        """Expand node: add children for valid moves or spawns."""
        if node.node_type == 'player':
            # For each valid direction, create a chance node
            for d in MOVE_FUNCS.keys():
                try:
                    new_board, moved, gained = apply_move(node.state, d)
                except Exception:
                    continue
                
                if not moved:
                    continue
                
                # Create chance node (spawn not yet applied)
                chance_node = MCTSNode(new_board, parent=node, 
                                      node_type='chance', prob=1.0)
                node.children[d] = chance_node
        else:
            # Chance node: enumerate all spawn outcomes
            spawn_children = all_spawn_children(node.state)
            for idx, (s, p) in enumerate(spawn_children):
                child = MCTSNode(s, parent=node, node_type='player', prob=p)
                node.children[idx] = child

    def _simulate(self, state):
        """Run rollout with epsilon-greedy policy."""
        current = [row[:] for row in state]
        total_score = 0
        
        for depth in range(self.rollout_depth):
            if logic.game_state(current) != 'not over':
                break
            
            # Choose next move
            if random.random() < self.rollout_epsilon:
                # Random move
                dirs = list(MOVE_FUNCS.keys())
                random.shuffle(dirs)
                chosen = None
                for d in dirs:
                    newb, moved, gained = apply_move(current, d)
                    if moved:
                        chosen = (newb, gained)
                        break
                if chosen is None:  # no valid moves
                    break
                new_state, gained = chosen
            else:
                # Greedy by heuristic: pick move that maximizes
                # heuristic(new_state) + gained
                best_val = -float('inf')
                best_state = None
                best_gain = 0
                for d in MOVE_FUNCS.keys():
                    newb, moved, gained = apply_move(current, d)
                    if not moved:
                        continue
                    # Lookahead one spawn
                    val = heuristic(add_random_tile(newb)) + gained
                    if val > best_val:
                        best_val = val
                        best_state = newb
                        best_gain = gained
                
                if best_state is None:
                    break
                new_state, gained = best_state, best_gain

            total_score += gained
            # After player move, simulate random spawn
            current = add_random_tile(new_state)

        # Add heuristic value of final board
        total_score += heuristic(current)
        return float(total_score)

    def _backpropagate(self, node, reward):
        """Update node statistics up to root."""
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur = cur.parent


class RandomPlayoutAgent(Agent):
    """Monte Carlo agent: evaluate moves by random playouts (no tree)."""
    
    def __init__(self, num_simulations=400, rollout_depth=60, rollout_epsilon=0.2):
        """Initialize with playout parameters."""
        super().__init__()
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth
        self.rollout_epsilon = rollout_epsilon
        # Reuse MCTS simulate function
        self.mcts = ImprovedMCTSAgent(num_simulations=0)

    def next_move(self, game_grid):
        """Evaluate each move by random playouts and pick the best."""
        base_board = [row[:] for row in game_grid.matrix]
        results = {}
        
        for d in MOVE_FUNCS.keys():
            try:
                newb, moved, gained = apply_move(base_board, d)
            except Exception:
                continue
            
            if not moved:
                continue
            
            # Run playouts from this move
            total = 0.0
            start_state = add_random_tile(newb)
            for _ in range(self.num_simulations):
                total += self.mcts._simulate(start_state)
            
            avg = gained + total / max(1, self.num_simulations)
            results[d] = avg
        
        if not results:
            return None
        
        return max(results.keys(), key=lambda k: results[k])
