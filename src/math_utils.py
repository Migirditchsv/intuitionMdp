# Add two tuples together, intended for applying policy action primitives to grid indices
def add_tuples(t1, t2):
    return tuple(x + y for x, y in zip(t1, t2))