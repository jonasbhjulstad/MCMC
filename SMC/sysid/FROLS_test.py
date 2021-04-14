import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Integrator.ERK import RK4_M
from models.Epidemiological import SIR_y, SIR_stochastic
from sysid.FROLS import FROLS
from sysid.Candidates import volterra_terms, order_combinations

#%%
if __name__ == '__main__':

    alpha = 1./9
    N_pop = 5.3e6
    R0 = 1.2
    beta = alpha*R0

    dt = 1.
    M = 1000
    N = 360

    I0 = 100000
    N_sto = 100
    x0 = [N_pop-I0, I0, 0]
    X_det = np.zeros((N+1, 4))
    X_det[0,:] = x0 + [0]

    lag_max_y = 5
    order_max = 2

    candidate_combs = order_combinations(order_max, lag_max_y)
    candidate_terms = [volterra_terms(order_list) for order_list in candidate_combs]
#%%

    for i in range(N):
        X_det[i+1,:] = RK4_M(SIR_y, X_det[i,:], dt, M, arg=[alpha, beta])

    fig, ax = plt.subplots(3)

    _ = [x.plot(X_det[:,i], marker='o', color='k', markersize=.9) for i,x in zip(range(3), ax)]

    plt.show()
# %%
