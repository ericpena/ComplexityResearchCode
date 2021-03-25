import numpy as np

def t_neighbor_sum(bm):

    size = bm.shape[0]
    neighbors = np.zeros((size, size))
    for i in range(size):
        for j in range(size):

            neighbors[i, j] = int((bm[i, (j-1) % size] \
                + bm[i, (j+1) % size] \
                + bm[(i-1) % size, j] \
                + bm[(i+1) % size, j] \
                + bm[(i-1) % size, (j-1) % size] \
                + bm[(i-1) % size, (j+1) % size] \
                + bm[(i+1) % size, (j-1) % size] \
                + bm[(i+1) % size, (j+1) % size]))
    
    locale = neighbors
    return locale

def update(bm, locale, B, S):
    """
    This will update the bm configuration of the cellular automata
    one iteration given the rules of Game of Life

    Args:
        self (Agent): A single chorosome that represents an IC

    Returns:
        None
    """
    
    size = bm.shape[0]
    for i in range(size):
        for j in range(size):

            if bm[i][j] == 0 and locale[i][j] in B:
                    bm[i][j] = 1
            else:
                if locale[i][j] not in S:
                    bm[i][j] = 0           

    locale = t_neighbor_sum(bm)
    
    return bm, locale

def run_rule(bm, steps, B, S):

    # size = bm.shape[0]
    # neighbors = np.zeros((size, size))
    neighbors = t_neighbor_sum(bm)

    
    for _ in range(steps):
        bm, neighbors = update(bm, neighbors, B, S)
    
    return bm