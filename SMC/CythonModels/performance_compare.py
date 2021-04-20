import numpy as np
from numpy import Inf
import matplotlib.pyplot as plt
from scipy.stats import norm, nbinom
from numpy.random import poisson
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Integrator.ERK import RK4_M
from models.Epidemiological import SIR_y, SIR_stochastic
import time
import CythonModels.BB_SIR as CS
if __name__ == '__main__':
    N_pop = 1e8
    I0 = 1e7
    x0 = [N_pop - I0, I0, 0.0]
    alpha = .1/9
    R0 = 1.2
    beta = 1.2*alpha

    M = 5
    N = 50
    T = 28
    DT = T/N
    X = []
    x_plot = []

    y = []
    xk = x0
    scale = 100
    param = {'alpha': alpha, 'beta': beta, 'N_pop': N_pop, 'dt': DT, 'nu_R': 1e-6, 'nu_I': 1e-6}
    start = time.process_time()
    for k in range(100):
        xk = x0
        for i in range(N):
            xk,_ = SIR_stochastic(xk, param)
            X.append(xk)
            y.append(xk[1])
    print(time.process_time()-start)

    start = time.process_time()
    for k in range(100):
        xk = x0
        for i in range(N):
            xk,_ = SIR_stochastic(xk, param)
            X.append(xk)
            y.append(xk[1])
    print(time.process_time()-start)