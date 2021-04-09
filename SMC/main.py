import numpy as np
from numpy import Inf
import matplotlib.pyplot as plt
from scipy.stats import norm
def BF_sim(yt, sig, merror, size):
    tlen = len(yt)
    xt = np.zeros((tlen, size))
    xt_prop = np.zeros((tlen, size))
    avg_wt = np.zeros(tlen)
    avg_wt[0] = 1
    for t in range(1, tlen):
        xt_prop[t, :] = np.random.normal(loc=xt[t-1], scale=sig, size=size)
        weights = norm.cdf(yt[t] + 0.5, loc=xt_prop[t,:], scale=merror) -\
        norm.cdf(yt[t]-.5, loc=xt_prop[t,:], scale=merror)

        avg_wt[t] = sum(weights)/size
        if sum(weights) == 0:
            weights += 1

        ind = np.random.choice(size, p=weights/sum(weights), size = size)
        xt[0:t+1, :] = np.concatenate([xt[0:t, ind], xt_prop[t:t+1, ind]], axis=0)
    return xt, xt_prop, avg_wt


if __name__ == '__main__':
    yt = [0,1,1,1,2]
    np.random.seed(1000)

    sig=0.5
    merror=0.1
    prior_p = np.concatenate([np.zeros((1,10000)), np.random.normal(loc=0, scale=sig, size=(4,10000))], axis=0)
    xt_prior = np.cumsum(prior_p, axis=0)

    xt_posterior, xt_prop, avg_wt = BF_sim(yt, sig, merror, 10000)

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
        xt, xt_prop, avg_wt = BF_sim(yt, sig_prop, merror, nparticle)
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

    import scipy.stats as st
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