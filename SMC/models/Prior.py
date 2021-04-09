import numpy as np
from numpy import Inf
import matplotlib.pyplot as plt
from scipy.stats import norm, nbinom, poisson
import multiprocessing as mp
from Epidemiological import *
import math
import pandas as pd

alpha = 1. / 9
N_pop = 5.3e6
R0 = 1.2
beta = R0 * alpha


def RK4(f, X, DT, arg=[]):
    if arg != []:
        k1 = f(X, arg)
        k2 = f(X + DT / 2 * k1, arg)
        k3 = f(X + DT / 2 * k2, arg)
        k4 = f(X + DT * k3, arg)
    else:
        k1 = f(X)
        k2 = f(X + DT / 2 * k1)
        k3 = f(X + DT / 2 * k2)
        k4 = f(X + DT * k3)
    return X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def F(f, X, DT, M):
    for i in range(M):
        xk = RK4(f, X, DT)


def BF_sim(yt, sig, merror, size):
    tlen = len(yt)
    xt = np.zeros((tlen, size))
    xt_prop = np.zeros((tlen, size))
    avg_wt = np.zeros(tlen)
    avg_wt[0] = 1
    for t in range(1, tlen):
        xt_prop[t, :] = np.random.normal(loc=xt[t - 1], scale=sig, size=size)
        weights = norm.cdf(yt[t] + 0.5, loc=xt_prop[t, :], scale=merror) - \
                  norm.cdf(yt[t] - .5, loc=xt_prop[t, :], scale=merror)

        avg_wt[t] = sum(weights) / size
        if sum(weights) == 0:
            weights += 1

        ind = np.random.choice(size, p=weights / sum(weights), size=size)
        xt[0:t + 1, :] = np.concatenate([xt[0:t, ind], xt_prop[t:t + 1, ind]], axis=0)
    return xt, xt_prop, avg_wt


def prior_sim(param):
    y = param['y']
    DT = param['DT']
    x0 = param['x0']
    M = param['M']
    theta = param['theta']
    beta = theta[0]
    alpha = theta[1]

    Ny = len(y)
    X = np.zeros((len(x0), Ny))
    X[:, 0] = x0
    xk = x0
    for i in range(Ny-1):
        for j in range(M):
            xk = RK4(SIR_y, xk, DT, arg=theta)
        X[:, i+1] = xk
    q_i = 1.0 - np.exp(-beta * y / N_pop)
    weights = nbinom.pmf(X[-1, :].astype(int), y.astype(int), q_i)
    avg_weight = sum(weights) / Ny

    return X, avg_weight


def multi_prior_sim(param, thetas):
    sol = []
    for theta in thetas.T:
        param['theta'] = theta
        sol.append(prior_sim(param))
    return {'X': [s[0] for s in sol], 'weight': [s[1] for s in sol]}

def simulate_traj(f, param, x0, tspan):
    X = np.zeros((len(x0), N+1))
    X[:,0] = x0
    for i, dt in enumerate(np.diff(tspan)):
        param['dt'] = dt
        X[:,i+1] = f(X[:,i], param)
    return X


if __name__ == '__main__':

#%%
    pool = mp.Pool(mp.cpu_count())
    I0 =2e6
    x0 = [N_pop - I0, I0, 0,0]

    M = 10
    N = 20000
    T = 200
    DT = T / N / M
    tgrid = np.linspace(0, T, N + 1)
    X_list = []
    Nk = 100
    param = {'beta': beta, 'N_pop': N_pop,'N': N, 'alpha': alpha, 'N_pop': N_pop, 'dt': np.diff(tgrid)[0]}

    # X = simulate_traj(SIR_stochastic, param, x0, tgrid)
    sir_out = pd.DataFrame(simulate(param))
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")
    # sline = plt.plot("t","S","",data=sir_out,color="red",linewidth=2)
    iline = plt.plot("t","I","",data=sir_out,color="green",linewidth=2)
    # rline = plt.plot("t","R","",data=sir_out,color="blue",linewidth=2)
    plt.xlabel("Time",fontweight="bold")
    plt.ylabel("Number",fontweight="bold")
    legend = plt.legend(title="Population",loc=5,bbox_to_anchor=(1.25,0.5))
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_linewidth(0)

    # ax = plt.subplot()
    # ax.plot(tgrid, X[1,:])
    # plt.show()

#%%
    N_thetas = int(20 / mp.cpu_count()) * mp.cpu_count()
    R0s = np.abs(np.random.normal(loc=R0, scale=.01 * R0, size=N_thetas))

    betas = R0s * alpha
    alphas = np.ones(N_thetas) * alpha

    thetas = np.vstack([alphas, betas])
    N_slices = int(thetas.shape[0] / mp.cpu_count())

    param = {'x0': x0, 'M': M, 'DT': DT, 'y': sir_out['I'].values}

#%%
    param['theta'] = [alpha, beta]
    X, w = prior_sim(param)

    plt.plot(tgrid, X[1,:].T)
    plt.show()
    # res = [pool.apply(multi_prior_sim, args=(param, thetas)) for theta_slice in
    #        np.split(thetas.T, mp.cpu_count())]
    # X_list = np.concatenate([r['X'] for r in res])
    # weights = np.concatenate([r['weight'] for r in res])
    #
    # ax = plt.subplot()


    # _ = [ax.plot(tgrid, X_list[i,1, :]) for i in range(X_list.shape[0])]
    # plt.show()

# if __name__ == '__main__' and is_BF_sim:
#     I0 = 200000.0
#     x0 = [N_pop - I0, I0, 0.0]
#
#     M = 5
#     N = 50
#     T = 28
#     DT = T / M / N
#     X = []
#     x_plot = []
#     y = []
#     xk = x0
#     for i in range(N):
#         for j in range(M):
#             xk = RK4(SIR, xk, DT)
#             x_plot.append(xk)
#         yk = poisson(xk[1])
#         X.append(xk)
#         y.append(yk)
#
#     np.random.seed(1000)
#
#     sig = 0.5
#     merror = 0.1
#     prior_p = np.concatenate([np.zeros((3, 10000)), np.random.normal(loc=0, scale=sig, size=(4, 10000))], axis=0)
#     xt_prior = np.cumsum(prior_p, axis=0)
#
#     xt_posterior, xt_prop, avg_wt = BF_sim(y, sig, merror, 10000)
#
#     fig0, ax0 = plt.subplots(2)
#     ax0[0].plot(xt_prior, label='prior')
#     ax0[1].plot(xt_posterior, label='posterior')
#     _ = [x.grid for x in ax0]
#
#     plt.show()
#
#     # PMCMC
#
#     np.random.seed(22)
#     tlen = 5
#     mcmclen = 5000
#     nparticle = 200
#     merror = 0.1
#
#     sig_mcmc = np.zeros(mcmclen)
#     sig_mcmc[0] = 1
#     ll = np.zeros(mcmclen)
#     ll[0] = -Inf
#     x_mcmc = np.zeros((mcmclen, tlen))
#     prop_sd = .5
#
#     for n in range(1, mcmclen):
#         sig_prop = np.abs(sig_mcmc[n - 1] + np.random.normal(size=1, loc=0, scale=prop_sd))
#         xt, xt_prop, avg_wt = BF_sim(yt, sig_prop, merror, nparticle)
#         ll_prop = sum(np.log(avg_wt))
#         prob_update = min(1, np.exp(ll_prop - ll[n - 1]))
#
#         if (np.random.uniform(0, 1, 1) < prob_update):
#             sig_mcmc[n] = sig_prop
#             x_mcmc[n, :] = xt[:, np.random.choice(nparticle, 1)].ravel()
#             ll[n] = ll_prop
#         else:
#             sig_mcmc[n] = sig_mcmc[n - 1]
#             x_mcmc[n, :] = x_mcmc[n - 1, :]
#             ll[n] = ll[n - 1]
#
#     import scipy.stats as st
#
#     sig_mean = np.mean(sig_mcmc)
#     CI_dev = 1.96 * np.std(sig_mcmc) / np.sqrt(len(sig_mcmc))
#     CI_sig = [sig_mean - CI_dev, sig_mean + CI_dev]
#
#     fig1, ax1 = plt.subplots(2)
#     ax1[0].plot(sig_mcmc)
#     ax1[1].hist(sig_mcmc, bins=100)
#     _ = [ax1[1].axhline(x=x) for x in CI_sig]
#
#     fig2, ax2 = plt.subplots()
#     ax2.plot(x_mcmc.T)
#     plt.show()
#
#     a = 1
