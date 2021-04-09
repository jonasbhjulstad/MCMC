from scipy.stats import nbinom
import numpy as np
if __name__ == '__main__':
    n = 0.4
    p_obs = 0.6
    I0 = 200000
    moments = nbinom.stats(.6*I0, p_obs, moments='mvsk')



