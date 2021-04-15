from scipy.stats import nbinom
import numpy as np
from numpy.random import binomial
from scipy.stats import betabinom, nbinom, poisson
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Inference.distributions import negBin, betaBin, negBin_sampler, betaBin_sampler
import matplotlib.pyplot as plt

def SIR_y(x, param):
    S = x[0]
    I = x[1]
    R = x[2]
    alpha = param['alpha']
    beta = param['beta']
    N_pop = param['N_pop']
    return np.array([-beta * S * I/N_pop, beta * S * I/N_pop - alpha * I, alpha * I, beta/N_pop * S * I])
def grad_SIR_y(x, theta):
    S = x[0]
    I = x[1]
    R = x[2]
    alpha = theta[0]
    beta = theta[1]
    return np.array([[-beta*I, -beta*S, 0, 0],
                     [beta*I - alpha, beta*S, 0, 0],
                     [0, alpha, 0, 0],
                    [beta*I, beta*S, 0, 0]])
def SIR(x, theta):
    S = x[0]
    I = x[1]
    R = x[2]
    alpha = theta[0]
    beta = theta[1]
    return np.array([-beta * S * I / N_pop, beta * S * I / N_pop - alpha * I, alpha * I])

def reed_frost(x, p):
    print('prob:',1-(1-p)**x[1])
    I = binomial(x[0], 1-(1-p)**x[1])
    S = x[0] - I
    return np.array([S, I])

def SIR_stochastic(x, param):
    alpha = param['alpha']
    beta = param['beta']
    N_pop = param['N_pop']
    dt = param['dt']

    S = x[0]
    I = x[1]
    R = x[2]

    p_I = 1-np.exp(-beta*I/N_pop*dt)
    p_R = 1-np.exp(-alpha*dt)

    k_SI = poisson.rvs(S*p_I)
    k_IR = binomial(int(I), p_R)

    delta_S = -k_SI
    delta_I = k_SI - k_IR
    delta_R = k_IR

    return [S + delta_S, I + delta_I, R + delta_R]

def SIR_stochastic_dispersed(x, param):
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

    # p, r_p_I = coeff_NegBin(S*p_I, nu_I)

    # x = np.linspace(0,2*S*p_I, 10000, dtype=int)
    # ax = plt.subplot()
    # ax.plot(x, [nbinom.pmf(xk, S*p_I, p) for xk in x])
    # ax.plot(x, [poisson.pmf(xk, S*p_I) for xk in x])
    # plt.show()

    # print(max([nbinom.pmf(xk, S*p_I, 1-p) for xk in x]))
    # k_SI = negBin(S*p_I, nu_I)
    bB = betaBin_sampler(p_I, nu_I, S)
    nB = negBin_sampler(S*p_I, nu_I)

    # x = np.linspace(0,2*S*p_I, 10000, dtype=int)
    # ax = plt.subplot()
    # ax.plot(x, [nB.pmf(xk) for xk in x])
    # ax.plot(x, [bB.pmf(xk) for xk in x])
    # plt.show()

    k_SI = betaBin(p_I, nu_I, S)

    # a, b = coeffs_BetaBin(p_R, nu_R, I)
    # k_IR = betabinom.rvs(int(I*p_R), a, b
    # )
    k_IR = betaBin(p_R, nu_R, I)

    delta_S = -k_SI
    delta_I = k_SI - k_IR
    delta_R = k_IR

    return [S + delta_S, I + delta_I, R + delta_R]



def simulate(param):
    N_pop = 5.3e6
    I0 = 2e5
    parms = [2./9, 1./9, 0.00, N_pop, 0.1]
    tf = 200
    tl = param['N'] + 1
    t = np.linspace(0,tf,tl)
    S = np.zeros(tl)
    I = np.zeros(tl)
    R = np.zeros(tl)
    Y = np.zeros(tl)
    u = [N_pop-I0,I0,0,0]
    S[0],I[0],R[0],Y[0] = u
    for j in range(1,tl):
        u = SIR_stochastic(u,param,t[j])
        S[j],I[j],R[j],Y[j] = u
    return {'t':t,'S':S,'I':I,'R':R,'Y':Y}