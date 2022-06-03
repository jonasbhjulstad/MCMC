import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

if __name__ == '__main__':
    nu_list = [0, 1,2,3,4]
    N_MCMC = 1000
    ll = 100

    model = 'SEIAR'
    params = []
    weights = []
    for nu in nu_list:
        params.append(pd.read_csv("../Data/" + model + "/param_" + str(nu) + "_" + str(ll) +
                      "_" + str(N_MCMC) + ".csv", names=['alpha', 'beta', 'gamma', 'p', 'mu'], header=None))
        # weights.append(pd.read_csv("../Data/" + model + "/weight_" + str(nu) +  "_" + str(ll) + "_" + str(N_MCMC) + ".csv", names=['Weights'], header=None))

    discard = False
    if discard:

        a = 1000
        print("Discarding first %i".format(a))
        for i in range(len(params)):
            params[i] = params[i][1000:]

    R0 = 1.2
    alpha = 1./9
    beta = alpha*R0
    p = 0.5
    gamma = 1./3
    mu = alpha
    true_param = [alpha, beta, gamma, p, mu]
    init_param = [2 * alpha, 2 * beta, 2 * gamma, 2*p, 2*mu]
    fig, ax = plt.subplots(5)
    ax[0].axvline(alpha, color='k', linestyle='--', label=r'$\theta_1$')
    ax[1].axvline(beta, color='k', linestyle='--')
    _ = [x.axvline(tp, color='k', linestyle='--')
         for x, tp in zip(ax, true_param)]

    ax[0].axvline(2*alpha, color='k', label=r'$\theta_0$')
    _ = [x.axvline(tp, color='k') for x, tp in zip(ax, init_param)]
    # ax[1].axvline(2*beta, color='k', marker='o')

    colormap = cm.get_cmap('Greys', len(nu_list))
    colors = colormap(np.linspace(.35, .75, len(nu_list)))
    ymax = 0

    N_bin = 100
    for param, nu, color in zip(reversed(params), reversed(nu_list), colors):

        x0, bin0, p0 = ax[0].hist(
            param['alpha'], bins=N_bin, density=True, color=color, label=r'$\nu=%i$' % nu)
        x1, bin1, p1 = ax[1].hist(
            param['beta'], bins=N_bin, density=True, color=color)
        x2, bin2, p2 = ax[2].hist(
            param['gamma'], bins=N_bin, density=True, color=color)
        x3, bin3, p3 = ax[3].hist(
            param['p'], bins=N_bin, density=True, color=color)
        x4, bin4, p4 = ax[4].hist(
            param['mu'], bins=N_bin, density=True, color=color)
        for item0, item1, item2, item3, item4 in zip(p0, p1, p2, p3, p4):
            x0_norm = item0.get_height()/sum(x0)
            x1_norm = item1.get_height()/sum(x1)
            x2_norm = item2.get_height()/sum(x2)
            x3_norm = item3.get_height()/sum(x3)
            x4_norm = item4.get_height()/sum(x4)
            item0.set_height(x0_norm)
            item1.set_height(x1_norm)
            item2.set_height(x2_norm)
            item3.set_height(x3_norm)
            item4.set_height(x4_norm)
            ymax = max(ymax, x0_norm, x1_norm, x2_norm, x3_norm, x4_norm)

    ax[0].legend()
    _ = [x.set_ylim([0, ymax]) for x in ax]
    _ = [x.set_title(t)
         for t, x in zip(['alpha', 'beta', 'gamma', 'p', 'mu'], ax)]
    _ = [x.grid() for x in ax]
    # ax[1].legend()
    plt.show()
