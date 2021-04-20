
import cython

cimport cython

import numpy as np
from numpy.random import binomial, normal, poisson
cimport numpy as np




@cython.wraparound
@cython.boundscheck
cpdef SIR_stochastic(np.ndarray[np.float64_t, ndim=1] x, param,np.ndarray[np.float64_t, ndim=1] y):
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


@cython.wraparound
@cython.boundscheck
cpdef chain_sim(f,np.ndarray[np.float64_t, ndim=1] y,np.ndarray[np.float64_t, ndim=1] x0, param, N_particle):
    rng = np.random.default_rng()
    Ny = len(y)
    Nx = len(x0)
    cdef np.ndarray[np.float64_t, ndim=3] X = np.zeros((N_particle,Ny, Nx), dtype='float')
    X[:,0,:] = np.tile(x0, (N_particle,1))
    cdef np.ndarray[np.float64_t, ndim=3] X_prop = np.zeros((N_particle,Ny, Nx), dtype='float')
    cdef np.ndarray[np.int64_t, ndim=1] avg_weights = np.zeros((Ny), dtype='float')
    cdef np.ndarray[np.int64_t, ndim=1] weights = np.zeros((N_particle), dtype='float')


    #Initial:
    for s in range(N_particle):
        X[s,0,:], weights[s] = f(x0, param, y=y[0])
    avg_weights[0] = sum(weights)/Ny

    for t in range(Ny-1):
        for s in range(N_particle):
            X_prop[s,t+1, :], weights[s]= f(X[s,t, :], param,y=y[t+1])
        avg_weights[t] = sum(weights)/Ny
        ind = rng.choice(N_particle, p=weights/sum(weights), size=N_particle)
        X[:,0:t+1,:] = np.concatenate([X[ind,0:t, :], X_prop[ind,t:t+1,:]], axis=1)
    return X, X_prop, avg_weights

cpdef my_loglike(theta, np.ndarray[np.float64_t, ndim=1] x,
                 np.ndarray[np.float64_t, ndim=1] data, sigma):
    """
    A Gaussian log-likelihood function for a model with parameters given in theta
    """

    model = my_model(theta, x)

    return -(0.5/sigma**2)*np.sum((data - model)**2)