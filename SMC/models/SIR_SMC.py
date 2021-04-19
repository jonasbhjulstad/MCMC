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
from Epidemiological import SIR_y
alpha = .1/9
N_pop = 5.3e6
R0 = 1.2
beta = R0 * alpha


def BF_sim(f, param, theta):
    y = param['y']
    x0 = param['x0']


    tlen = len(y)
    Nx = len(x0)
    xt = np.zeros((tlen, Nx))
    xt_prop = np.zeros((tlen, Nx))
    avg_wt = np.zeros(tlen)
    avg_wt[0] = 1
    for t in range(1, tlen):
        xt_prop[t, :] = RK4_M(f,xt[t-1,:],param['DT'],param['M'],arg=param)
        weight = norm.logpdf(y[t], xt_prop[t,1], scale=scale)
        avg_wt[t] = sum(weights)/size
        if sum(weights) == 0:
            weights += 1

        ind = np.random.choice(size, p=weights/sum(weights), size = size)
        xt[0:t+1, :] = np.concatenate([xt[0:t, ind], xt_prop[t:t+1, ind]], axis=0)
    return xt, xt_prop, avg_wt


def BF_multisim(f, param):
    res = [BF_sim(f, param, theta) for theta in param['thetas']]
    # returns: X_list, X_prop_list, avg_wt_list 
    return zip(*res)



if __name__ == '__main__':
    I0 = 200000.0
    x0 = [N_pop - I0, I0, 0.0,0.0]

    M = 5
    N = 50
    T = 28
    DT = T/N
    X = []
    x_plot = []
    y = []
    xk = x0
    scale = 100
    param = {'alpha': alpha, 'beta': beta, 'N_pop': N_pop}
    for i in range(N):
        for j in range(M):
            xk = RK4_M(SIR_y, xk, DT, M,arg=param)
            x_plot.append(xk)
        yk = norm.rvs(loc=xk[1], scale=scale)
        X.append(xk)
        y.append(yk)


    np.random.seed(1000)

    N_theta=100
    thetas = np.vstack([norm.rvs(loc=alpha, scale=.1*alpha, size=N_theta), norm.rvs(loc=beta, scale=.1*beta, size=N_theta)]).T

    param = {'DT': DT, 'x0': x0, 'thetas': thetas, 'y': y, 'M': M, 'scale': scale}
    X_list, X_prop_list, avg_wt_list = BF_multisim(SIR_y, param)

    fig0, ax0 = plt.subplots(2)
    ax0[0].plot(xt_prior, label='prior')
    ax0[1].plot(xt_posterior, label='posterior')
    _ = [x.grid for x in ax0]

    plt.show() 

    #PMCMC

    np.random.seed(22)
    tlen = 5
    mcmclen = 5000
    nparticle = 200
    merror = 0.1

    sig_mcmc = np.zeros(mcmclen)
    sig_mcmc[0] = 1
    ll = np.zeros(mcmclen)
    ll[0] = -Inf
    x_mcmc = np.zeros((mcmclen, tlen))
    prop_sd = .5

    for n in range(1,mcmclen):
        sig_prop = np.abs(sig_mcmc[n-1] + np.random.normal(size=1, loc=0, scale=prop_sd))
        xt, xt_prop, avg_wt = BF_sim(y, sig_prop, merror, nparticle)
        ll_prop = sum(np.log(avg_wt))
        prob_update = min(1, np.exp(ll_prop-ll[n-1]))

        if (np.random.uniform(0, 1, 1) < prob_update):
            sig_mcmc[n] = sig_prop
            x_mcmc[n,:] = xt[:,np.random.choice(nparticle, 1)].ravel()
            ll[n] = ll_prop
        else:
            sig_mcmc[n] = sig_mcmc[n-1]
            x_mcmc[n, :] = x_mcmc[n-1,:]
            ll[n] = ll[n-1]

    sig_mean = np.mean(sig_mcmc)
    CI_dev = 1.96*np.std(sig_mcmc)/np.sqrt(len(sig_mcmc))
    CI_sig = [sig_mean - CI_dev, sig_mean + CI_dev]

    fig1, ax1 = plt.subplots(2)
    ax1[0].plot(sig_mcmc)
    ax1[1].hist(sig_mcmc, bins=100)
    _ = [ax1[1].axhline(x=x) for x in CI_sig]


    fig2, ax2 = plt.subplots()
    ax2.plot(x_mcmc.T)
    plt.show()