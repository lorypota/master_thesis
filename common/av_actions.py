STEP = 5
MAX_ACTION = 30
CAPACITY = 100


def available_actions(state):
    min_action = max(-MAX_ACTION, -(state[0] // STEP) * STEP)
    max_action = min(MAX_ACTION, ((CAPACITY - state[0]) // STEP) * STEP)
    return list(range(min_action, max_action + STEP, STEP))
