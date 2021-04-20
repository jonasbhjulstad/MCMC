import pymc3 as pm
from pymc3.ode import DifferentialEquation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import arviz as az
import pandas as pd
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from models.Epidemiological import SIR_stochastic
import theano.tensor as tt
from theano.compile.ops import as_op
import theano
from Integrator.ERK import RK4_M
from models.Epidemiological import SIR_y
class SIR_Th:
    def __init__(self, N, x0):
        self._x0 = x0
        self.N = N
        self.Nx = len(x0)

    def _simulate(self, param):

        X = np.zeros((self.N, self.Nx))
        Xk = self._x0
        for i in range(N):
            Xk = SIR_stochastic(Xk, param)[0]
            X[i,:] = Xk            
        return X

    def simulate(self, x):
        return self._simulate(x)

N_pop = 1e8
I0 = 1e7
x0 = [N_pop - I0, I0, 0.0,0.0]
alpha = 1./9
R0 = 1.2
beta = R0*alpha
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
    yk = xk[1]
    X.append(xk)
    y.append(yk)


x0 = x0[:3]
ode_model = SIR_Th(N, x0)

@as_op(itypes=[tt.dscalar, tt.dscalar], otypes=[tt.dvector])
def th_forward_model(a, b):
    param = {'alpha': a, 'beta': b, 'N_pop': N_pop, 'dt': DT, 'nu_R': 1e-6, 'nu_I': 1e-6, 'dt': DT}

    th_states = ode_model.simulate(param)
    return th_states[:,1]




draws = 1000
with pm.Model() as model:

    a = pm.Bound(pm.Normal, lower=1e-6)("a", mu=alpha, sigma=.1*alpha)
    b = pm.Bound(pm.Normal, lower=1e-6)("b", mu=beta, sigma=.1*beta)

    param['alpha'] = a
    param['beta'] = b


    forward = th_forward_model(a,b)


    Y_obs = pm.Normal("Y_obs", mu=forward, sigma=10000, observed=y)

    #startsmc = {v.name: np.random.uniform(1e-3, 2, size=draws) for v in model.free_RVs}

    trace_FN = pm.sample_smc(draws)