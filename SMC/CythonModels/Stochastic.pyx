from scipy.stats import nbinom
import numpy as np
from scipy.stats import betabinom, nbinom, poisson, binom, norm
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Inference.distributions import negBin, betaBin, negBin_sampler, betaBin_sampler
import matplotlib.pyplot as plt



def SIR_stochastic(x, param, dispersed=True, y=[]):
    alpha = param['alpha']
    beta = param['beta']
    N_pop = param['N_pop']
    dt = param['dt']
    nu_R = param['nu_R']
    nu_I = param['nu_I']


    S = x[0]
    I = x[1]
    R = x[2]

    p_I = 1-np.exp(-beta*I/N_pop*dt)
    p_R = 1-np.exp(-alpha*dt)

    if dispersed:
        samplers = [betaBin_sampler(p_I, nu_I, S), betaBin_sampler(p_R, nu_R, I)]
    else:
        samplers = [poisson(S*p_I), binom.rvs(int(I), p_R)]

    k_SI, k_IR = [S.rvs() if hasattr(S, 'rvs') else S for S in samplers]


    delta_S = -k_SI
    delta_I = k_SI - k_IR
    delta_R = k_IR

    Xk_1 = [S + delta_S, I + delta_I, R + delta_R]
    ll_kSI = []
    if np.any(y):
        #ll_kSI = binom.logpmf(Xk_1[1], y, 1-np.exp(-beta*y/N_pop*dt))
        ll_kSI = norm.logpdf(Xk_1[1], y, 10000)
    return Xk_1, ll_kSI



