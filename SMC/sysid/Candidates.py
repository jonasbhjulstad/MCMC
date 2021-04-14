import numpy as np
from itertools import product
def order_combinations(order_max, N_terms):
    r = [range(order_max) for _ in range(N_terms)]
    return product(*r)

def volterra_terms(orders):
    N_factors = len(orders)
    return lambda x: np.multiply([x[i]**order for i, order in enumerate(order)])