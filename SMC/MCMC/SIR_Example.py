import pymc3 as pm
from pymc3.ode import DifferentialEquation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import arviz as az
import pandas as pd
import sys

def SIR(y, t, p):
    S = y[0]
    I = y[1]
    R = y[2]
    
    beta = p[0]
    gamma = p[1]
    
    ds = -beta*I*S/N
    di = beta*I*S/N-gamma*I
    dr = gamma*I
    
    return [ds, di, dr]

N = 763
i0 = 1
R0 = 0
y0 = [N - i0, i0, R0]
t0 = 0 
ts = np.arange(1,15,1)
X = [y0]
for i in range(len(ts)-1):
    X.append(SIR(X[-1], 0, [1./9, 1.2*1./9]))

data = [x[1] for x in X]  

sir_model = DifferentialEquation(
    func=SIR,
    times=ts,
    n_states=3,
    n_theta=2,
    t0=t0,
)

with pm.Model() as model:
    y = pm.Data("y", data)
   
    beta = pm.Bound(pm.Normal, lower=0.0)('beta', mu=2, sigma=1)
    gamma = pm.Bound(pm.Normal, lower=0.0)('gamma', mu=0.4, sigma=0.5)
    phi_inv = pm.Exponential("phi_inv", lam=5)
     
    R0 = pm.Deterministic('R0', beta / gamma)
    phi = pm.Deterministic('phi', 1. / phi_inv)

    sir_curves = sir_model(y0=y0, theta=[beta, gamma])

    Y = pm.NegativeBinomial('Y', mu=sir_curves[:,1], alpha=phi, observed=y)
    prior = pm.sample_prior_predictive()

    with model:    
        trace = pm.sample(draws=100, tune=50, step=pm.NUTS(), chains=2, cores=1)
    posterior_predictive = pm.sample_posterior_predictive(trace)

    data = az.from_pymc3(trace=trace, prior=prior, posterior_predictive=posterior_predictive)
    az.plot_trace(trace)

    a = 1