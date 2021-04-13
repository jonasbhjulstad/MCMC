import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from pymc3 import *
import theano.tensor as tt
import sys
from scipy.integrate import odeint
sys.path.append(r'/home/deb/Documents/MCMC/SMC/models')
import pandas as pd
from Epidemiological import SIR_y, simulate
import pickle as pck
az.style.use("arviz-darkgrid")
alpha = 1./9
R0 = 1.2
beta = R0*alpha
N_pop = 5.3e6

def SIR_t_y(x, t, theta):
    alpha = theta[0]
    beta = theta[1]
    S, I, R = x[0], x[1], x[2]
    return [-beta * S * I/N_pop, beta * S * I/N_pop - alpha * I, alpha * I, beta/N_pop * S * I]


if __name__ == '__main__':


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
    data = sir_out['I']

    S_noise = np.random.normal(sir_out['I'], 1)
    I_noise = np.random.normal(sir_out['I'], 1)
    R_noise = np.random.normal(sir_out['I'], 1)
    Y_noise = np.random.normal(sir_out['I'], 1)
    # y = np.concatenate(np.nan())

    na = np.empty(I_noise.shape)
    na[:] = 0

    a = 1

    obs = np.vstack([na, I_noise, na, na]).T


    ode_model = pm.ode.DifferentialEquation(func=SIR_t_y, times=tgrid, n_states=4, n_theta=2, t0=0)

    with pm.Model() as model:
        # Specify prior distributions for soem of our model parameters
        alpha_pm = pm.Normal("alpha", mu=alpha, sigma=.2*alpha)
        beta_pm = pm.Normal("beta", mu=beta, sigma=.2*beta)

        # If we know one of the parameter values, we can simply pass the value.
        ode_solution = ode_model(y0=[N_pop, 0, 0, 0], theta=[alpha_pm, beta_pm])
        # The ode_solution has a shape of (n_times, n_states)

        Y = pm.Normal("Y", mu=ode_solution, sigma=1, observed=obs)

        prior = pm.sample_prior_predictive()
        trace = pm.sample(2000, tune=1000, cores=4, init='adapt_diag')
        posterior_predictive = pm.sample_posterior_predictive(trace)

        data = az.from_pymc3(trace=trace, prior=prior, posterior_predictive=posterior_predictive)
        with open('data.pck', 'wb') as f:
            pck.dump(data,f)

        az.plot_trace(data)
        plt.show()