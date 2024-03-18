# Add two tuples together, intended for applying policy action primitives to grid indices
def add_tuples(t1, t2):
    return tuple(x + y for x, y in zip(t1, t2))

def is_out_of_bounds(state, state_space):
    x, y = state
    if x < 0 or y < 0 or x >= len(state_space) or y >= len(state_space[0]):
        return True
    return False