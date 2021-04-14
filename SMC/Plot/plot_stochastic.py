import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Integrator.ERK import RK4_M
from models.Epidemiological import SIR_y, SIR_stochastic, SIR_stochastic_dispersed


if __name__ == '__main__':

    alpha = 1./9
    N_pop = 5.3e6
    R0 = 1.2
    beta = alpha*R0

    dt = 1.

    param = {'alpha': 1./9, 'beta': beta, 'N_pop': N_pop, 'dt': dt, 'r_p_I': 2., 'gamma_p_R': .5}
    M = 10
    N = 360

    I0 = 100000
    N_sto = 100
    x0 = [N_pop-I0, I0, 0]
    X_sto = np.zeros((N_sto, N+1,3))
    for k in range(N_sto):
        X_sto[k,0,:] = x0
    X_det = np.zeros((N+1, 4))
    X_det[0,:] = x0 + [0]
    for i in range(N):
        for k in range(N_sto):
            X_sto[k,i+1,:] = SIR_stochastic_dispersed(X_sto[k,i,:], param)
        X_det[i+1,:] = RK4_M(SIR_y, X_det[i,:], dt, M, arg=[alpha, beta])


    
    fig, ax = plt.subplots(3)

    _ = [[x.plot(Xk[:,i]) for i,x in zip(range(3), ax)] for Xk in X_sto]
    _ = [x.plot(X_det[:,i], marker='o', color='k', markersize=.9) for i,x in zip(range(3), ax)]

    plt.show()