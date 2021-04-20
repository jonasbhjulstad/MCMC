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
import multiprocessing as mp
from tqdm import tqdm

def chain_sim(f,y,x0,param,N_particle):
    rng = np.random.default_rng()
    Ny = len(y)
    Nx = len(x0)
    X = np.zeros((N_particle,Ny, Nx))
    X[:,0,:] = np.tile(x0, (N_particle,1))
    X_prop = np.zeros((N_particle,Ny, Nx))
    avg_weights = np.zeros(Ny)
    weights = np.zeros(N_particle)

    #Initial:
    for s in range(N_particle):
        X[s,0,:], weights[s] = f(x0, param, y=y[0])
    avg_weights[0] = sum(weights)/Ny
    
    for t in range(Ny-1):
        for s in range(N_particle):
            X_prop[s,t+1, :], weights[s]= f(X[s,t, :], param,y=y[t+1])
            # weights[s] = f_ll(y[t+1], X_prop[s,t+1, 1], param)
        avg_weights[t] = sum(weights)/Ny
        ind = rng.choice(N_particle, p=weights/sum(weights), size=N_particle)
        X[:,0:t+1,:] = np.concatenate([X[ind,0:t, :], X_prop[ind,t:t+1,:]], axis=1)
    return X, X_prop, avg_weights

def mcmc_single(theta_0, x0, f, N_particle, N_mcmc, f_prop=[]):
    Theta = np.zeros((N_mcmc, len(theta_0)))
    Theta[0,:] = theta_0
    ll = np.zeros(N_mcmc)
    ll[0] = -np.Inf
    X_sample = np.zeros((N_mcmc, Ny, Nx))

    for n in tqdm(range(N_mcmc-1)):
        Theta_prop = np.abs(Theta[n] + np.random.normal(size=N_ODE_params, loc=[0,0], scale=.2))
        X, X_prop, avg_weights = chain_sim(f, y, x0, param, N_particle)
        ll_prop = sum(avg_weights)
        prob_update = min(1, np.exp(ll_prop-ll[n]))
        print(np.exp(ll_prop-ll[n]))
        #Metropolis-Hastings:
        if (np.random.uniform(0, 1, 1) < prob_update):
            Theta[n+1,:] = Theta_prop
            X_sample[n+1,:] = X[np.random.choice(N_mcmc,1),:,:]
            ll[n+1] = ll_prop
        else:
            print('Keeping step')
            Theta[n+1,:] = Theta[n,:]
            X_sample[n+1, :,:] = X_sample[n,:,:]
            ll[n+1] = ll[n]
    return Theta, X_sample, ll

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
    Nx = len(x0)
    N_ODE_params = 2
    tlen = 5
    mcmclen = 5000
    N_particle = 200
    N_mcmc = 30
    f = SIR_stochastic
    theta_0 = [alpha-alpha/10, beta+beta/10]

    Theta, X_sample, ll = mcmc_single(theta_0, x0, f, N_particle, N_mcmc)


    # sig_mean = np.mean(sig_mcmc)
    # CI_dev = 1.96*np.std(sig_mcmc)/np.sqrt(len(sig_mcmc))
    # CI_sig = [sig_mean - CI_dev, sig_mean + CI_dev]

    
    fig1, ax1 = plt.subplots(2,2)
    ax1[0,0].plot(Theta[:,0])
    ax1[0,1].hist(Theta[:,0], bins=100)
    ax1[0,0].plot(Theta[:,1])
    ax1[0,1].hist(Theta[:,1], bins=100)


    fig2, ax2 = plt.subplots()
    ax2.plot(X_sample[:,:,1])
    plt.show()
