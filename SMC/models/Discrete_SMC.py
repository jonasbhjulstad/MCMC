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
from Epidemiological import SIR_y, SIR_stochastic

def chain_sim(f,y,x0,param,N_particle):
    rng = np.random.default_rng()
    Ny = len(y)
    Nx = len(x0)
    X = np.zeros((N_particle,Ny, Nx))
    X[:,0,:] = np.tile(x0, (N_particle, 1))
    X_prop = np.zeros((N_particle,Ny, Nx))
    avg_weights = np.zeros(Ny)
    weights = np.zeros(N_particle)

    #Initial:
    for s in range(N_particle):
        X[s,0,:], weights[s] = f(x0, param, y=y[0])
    avg_weights[0] = sum(weights)/Ny
    
    for t in range(Ny):
        for s in range(N_particle):
            X_prop[s,t+1, :], weights[s]= f(X[s,t, :], param,y=y[t])
            # weights[s] = f_ll(y[t+1], X_prop[s,t+1, 1], param)
        avg_weights[t] = sum(weights)/Ny
        ind = rng.choice(N_particle, p=weights/sum(weights), size=N_particle)
        X[:,0:t+1,:] = np.concatenate([X[ind,0:t, :], X_prop[ind,t+1:t+1,:]], axis=0)
    return X, X_prop, avg_weights

if __name__ == '__main__':
    N_pop = 1e8
    I0 = 1e7
    x0 = [N_pop - I0, I0, 0.0,0.0]
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
    for i in range(N):
        for j in range(M):
            xk = RK4_M(SIR_y, xk, DT, M,arg=param)
            x_plot.append(xk)
        yk = xk[1]
        X.append(xk)
        y.append(yk)
    
    x0 = x0[:3]
    Ny = len(y)
    N_ODE_params = 2
    tlen = 5
    mcmclen = 5000
    N_particle = 200


    N_mcmc = 10
    Theta = np.zeros((N_mcmc, N_ODE_params))
    ll = np.zeros(N_mcmc)
    ll[0] = -np.Inf
    X_sample = np.zeros((N_mcmc, Ny))

    for n in range(N_mcmc):
        Theta_prop = np.abs(Theta[n-1] + np.random.normal(size=N_ODE_params, loc=[alpha, beta], scale=1))
        X, X_prop, avg_weights = chain_sim(SIR_stochastic, y, x0, param, N_particle)
        ll_prop = sum(avg_weights)
        prob_update = min(1, np.exp(ll_prop-ll[n]))
        #Metropolis-Hastings:
        if (np.random.uniform(0, 1, 1) < prob_update):
            sig_mcmc[n+1] = Theta_prop
            X_sample[n+1,:] = X[:,np.random.choice(nparticle, 1)].ravel()
            ll[n+1] = ll_prop
        else:
            Theta[n+1,:] = Theta[n,:]
            X_sample[n+1, :] = X_sample[n,:]
            ll[n+1] = ll[n]

    sig_mean = np.mean(sig_mcmc)
    CI_dev = 1.96*np.std(sig_mcmc)/np.sqrt(len(sig_mcmc))
    CI_sig = [sig_mean - CI_dev, sig_mean + CI_dev]
