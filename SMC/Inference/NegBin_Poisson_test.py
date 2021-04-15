from scipy.stats import nbinom, poisson, binom, betabinom
import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from matplotlib import cm
from matplotlib.colors import Colormap
from Inference.distributions import coeffs_BetaBin
if __name__ == '__main__':
    p = 0.5
    lbd = 1e3
    n = lbd/p
    lbd_small = 5
    n_small = lbd_small/p

    x = np.linspace(0,2*lbd, 1000, dtype=int)
    x_small = np.linspace(0,2*lbd_small, 1000, dtype=int)

    g_mean = 1/(n-1)
    nu_list = np.linspace(1, 20,20)

    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(x, [poisson.pmf(xk, mu=lbd) for xk in x], color='k')
    ax[0,1].plot(x, [binom.pmf(xk, n, p) for xk in x], color='k')

    ax[1,0].plot(x_small, [poisson.pmf(xk, mu=lbd_small) for xk in x_small], color='k')
    ax[1,1].plot(x_small, [binom.pmf(xk, n_small, p) for xk in x_small], color='k')

    colormap = cm.get_cmap('Greys', len(nu_list))
    colors = colormap(np.linspace(1,.4,len(nu_list)))

    for nu_k, color in zip(reversed(nu_list), reversed(colors)):
        rk = lbd/nu_k
        pk = lbd/(rk+lbd)
        ax[0,0].plot(x, [nbinom.pmf(xk, rk, 1-pk) for xk in x], color=color, linestyle='--')
        gk = nu_k/(n-1)
        a, b = coeffs_BetaBin(p, gk)
        ax[0,1].plot(x, [betabinom.pmf(xk, n, a, b) for xk in x], color=color, linestyle='--')

        rk = lbd_small/nu_k
        pk = lbd_small/(rk+lbd_small)
        ax[1,0].plot(x_small, [nbinom.pmf(xk, rk, 1-pk) for xk in x_small], color=color, linestyle='--')
        gk = nu_k/(n_small-1)
        a, b = coeffs_BetaBin(p, gk)
        ax[1,1].plot(x_small, [betabinom.pmf(xk, n_small, a, b) for xk in x_small], color=color, linestyle='--')
    _ = [[x.grid(True) for x in row] for row in ax]
    titles = [[r'Poisson/Negative Binomial, $\lambda$ = {}, $p$ = {}'.format(lbd, p), r'Binomial/Beta-Binomial, $\lambda$ = {}, $p$ = {}'.format(lbd, p)],
    [r'Poisson/Negative Binomial, $\lambda$ = {}, $p$ = {}'.format(lbd_small, p), r'Binomial/Beta-Binomial, $\lambda$ = {}, $p$ = {}'.format(lbd_small, p)]]
    _ = [[x.set_title(t) for x, t in zip(row, trow)] for row, trow in zip(ax, titles)]

    ax[0,0].set_ylabel(r'$P(X=k)$')
    ax[1,0].set_ylabel(r'$P(X=k)$')
    ax[1,0].set_xlabel(r'$k$')
    ax[1,1].set_xlabel(r'$k$')
    ax[0,0].set_xticklabels([])
    ax[0,1].set_xticklabels([])
    plt.show()

    fig.savefig(r'../Figures/Poisson_Binomial_Comparison.eps', format='eps')