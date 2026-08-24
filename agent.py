# agent.py
import random
import heapq
from collections import deque


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)

class SimpleReflexAgent:
    def sense_and_act(self, percept: dict) -> str:
        # Condition-Action Rule 1: IF food_here THEN stay/collect it.
        if percept['food_here']:
            return 'Up'

        # Condition-Action Rule 2: IF wall_ahead THEN turn.
        if percept['wall_ahead']:
            return 'Left'

        # Condition-Action Rule 3 (ELSE): move forward.
        return 'Up'

class ModelBasedAgent:
    def __init__(self):
        # Internal memory: what happened last cycle, and where we are in
        # our "try a different direction" cycle when stuck.
        self.last_percept = None
        self.last_action = None
        self.turn_options = ['Left', 'Right', 'Down', 'Up']
        self.turn_index = 0

    def sense_and_act(self, percept: dict) -> str:
        # --- Update internal state first (Transition + Sensor Model) ---
        stuck_in_loop = (
            percept['wall_ahead']
            and self.last_percept is not None
            and self.last_percept == percept
        )

        # --- Apply IF-THEN rules that now query memory, not just the percept ---
        if percept['food_here']:
            action = 'Up'
        elif percept['wall_ahead']:
            if stuck_in_loop:
                self.turn_index = (self.turn_index + 1) % len(self.turn_options)
            action = self.turn_options[self.turn_index]
        else:
            action = 'Up'
            self.turn_index = 0  # reset once moving freely again

        # --- Record this cycle for next time ---
        self.last_percept = percept
        self.last_action = action
        return action


class SearchAgent:
    """
    Practical 03: Goal-Based / Planning Agent.

    Unlike SimpleReflexAgent / ModelBasedAgent (which react cell-by-cell),
    this agent is given the full world model via the percept and forms a
    complete plan (a sequence of actions) BEFORE it ever moves, using
    uninformed search (BFS, DFS or UCS) over the grid's state space.
    """

    # Action -> (dx, dy). Kept identical to VisualGridHuntGame's movement logic.
    ACTIONS = {
        'Up': (0, 1),
        'Down': (0, -1),
        'Left': (-1, 0),
        'Right': (1, 0),
    }

    def __init__(self):
        self.plan = []               # The queued sequence of actions to execute
        self.active_algo = 'BFS'     # 'BFS' | 'DFS' | 'UCS' - swap to compare strategies

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _in_bounds(self, pos, grid_size):
        x, y = pos
        width, height = grid_size
        return 0 <= x < width and 0 <= y < height

    def _get_successors(self, pos, walls, grid_size):
        """Yield (action, next_state) pairs for every legal move from pos."""
        for action, (dx, dy) in self.ACTIONS.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if self._in_bounds(nxt, grid_size) and nxt not in walls:
                yield action, nxt

    def _closest_food(self, agent_pos, all_food):
        """Pick a search goal: the food pellet with the smallest Manhattan distance."""
        if not all_food:
            return None
        return min(
            all_food,
            key=lambda f: abs(f[0] - agent_pos[0]) + abs(f[1] - agent_pos[1])
        )

    # ------------------------------------------------------------------ #
    # Step 1.2 - Uninformed search strategies
    # ------------------------------------------------------------------ #
    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Breadth-First Search - FIFO frontier. Explores shallowest nodes first."""
        walls = set(walls)
        start_pos, goal_pos = tuple(start_pos), tuple(goal_pos)

        if start_pos == goal_pos:
            return []

        frontier = deque([(start_pos, [])])   # (state, path_of_actions)
        reached = {start_pos}

        while frontier:
            state, path = frontier.popleft()   # FIFO -> shallowest first

            for action, nxt in self._get_successors(state, walls, grid_size):
                if nxt in reached:
                    continue
                new_path = path + [action]
                if nxt == goal_pos:
                    return new_path
                reached.add(nxt)               # graph search: mark reached on discovery
                frontier.append((nxt, new_path))

        return None  # Goal unreachable

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Depth-First Search - LIFO frontier (stack). Explores deepest nodes first."""
        walls = set(walls)
        start_pos, goal_pos = tuple(start_pos), tuple(goal_pos)

        if start_pos == goal_pos:
            return []

        frontier = [(start_pos, [])]           # list used as a LIFO stack
        reached = {start_pos}

        while frontier:
            state, path = frontier.pop()       # LIFO -> deepest first

            if state == goal_pos:
                return path

            for action, nxt in self._get_successors(state, walls, grid_size):
                if nxt not in reached:
                    reached.add(nxt)
                    frontier.append((nxt, path + [action]))

        return None  # Goal unreachable

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        """Uniform-Cost Search - Priority Queue ordered by path cost g(n).
        Every step costs 1 here, so UCS degenerates to BFS, but the frontier
        is still a genuine priority queue as required by the practical."""
        walls = set(walls)
        start_pos, goal_pos = tuple(start_pos), tuple(goal_pos)

        if start_pos == goal_pos:
            return []

        counter = 0  # tie-breaker so heapq never compares tuples of paths
        frontier = [(0, counter, start_pos, [])]   # (g_cost, tie_breaker, state, path)
        heapq.heapify(frontier)
        best_cost = {start_pos: 0}

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)

            if state == goal_pos:
                return path

            # A cheaper route to this state may have been queued/reached since
            if cost > best_cost.get(state, float('inf')):
                continue

            for action, nxt in self._get_successors(state, walls, grid_size):
                new_cost = cost + 1  # uniform step cost
                if new_cost < best_cost.get(nxt, float('inf')):
                    best_cost[nxt] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, nxt, path + [action]))

        return None  # Goal unreachable

    # ------------------------------------------------------------------ #
    # Step 1.3 - Forming and executing an offline plan
    # ------------------------------------------------------------------ #
    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            agent_pos = percept['agent_pos']
            walls = percept['walls']
            grid_size = percept['grid_size']
            all_food = percept['all_food']

            goal = self._closest_food(agent_pos, all_food)

            if goal is None:
                # No food left to plan towards - just hold position.
                return random.choice(list(self.ACTIONS.keys()))

            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(agent_pos, goal, walls, grid_size) or []
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(agent_pos, goal, walls, grid_size) or []
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(agent_pos, goal, walls, grid_size) or []
            else:
                raise ValueError(f"Unknown active_algo: {self.active_algo}")

            if not self.plan:
                # Goal unreachable - fall back to a random legal-ish move so the
                # simulation doesn't stall forever.
                return random.choice(list(self.ACTIONS.keys()))

        return self.plan.pop(0)