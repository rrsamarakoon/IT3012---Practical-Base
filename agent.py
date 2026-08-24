# agent.py
import random
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