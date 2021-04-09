from scipy.stats import nbinom
import numpy as np
from numpy.random import binomial

N_pop = 1000

def SIR_y(x, theta):
    S = x[0]
    I = x[1]
    R = x[2]
    alpha = theta[0]
    beta = theta[1]
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
def SIR_stochastic(u,param,t):
    bet = param['beta']
    gamm = param['alpha']
    dt = param['dt']
    N_pop = param['N_pop']

    S,I,R,Y=u
    lambd = bet*(I)/N_pop
    ifrac = 1.0 - np.exp(-lambd*dt)
    rfrac = 1.0 - np.exp(-gamm*dt)
    infection = np.random.binomial(S,ifrac)
    recovery = np.random.binomial(I,rfrac)
    return [S-infection,I+infection-recovery,R+recovery,Y+infection]


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