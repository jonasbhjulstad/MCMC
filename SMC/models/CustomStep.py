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

class CustomStepper(BlockedStep):
    def __init__(self, var, counts: np.ndarray, concentration):
        self.vars = [var]
        self.counts = counts
        self.name = var.name
        self.conc_prior = concentration

    def step(self, point: dict):
        # Since our concentration parameter is going to be log-transformed
        # in point, we invert that transformation so that we
        # can get conc_posterior = conc_prior + counts
        conc_posterior = np.exp(point[self.conc_prior.transformed.name]) + self.counts
        draw = sample_dirichlet(conc_posterior)

        # Since our new_p is not in the transformed / unconstrained space,
        # we apply the transformation so that our new value
        # is consistent with PyMC3's internal representation of p
        point[self.name] = stick_breaking.forward_val(draw)

        return point