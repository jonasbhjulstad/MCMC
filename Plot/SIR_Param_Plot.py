import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

if __name__ == '__main__':
    # nu_list = np.arange(0,5,2)
    nu_list = [0,1,2,3,4]
    N_MCMC = 50000
    ll = 100

    model = 'SIR'
    params = []

    weights = []
    for nu in nu_list:
        params.append(pd.read_csv("../Data/" + model + "/param_" + str(nu) + "_" + str(ll) + "_" + str(N_MCMC) + ".csv", names=['alpha', 'beta'], header=None))
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
    gamma = 1./3
    true_param = [alpha, beta, gamma]
    init_param = [2 * alpha, 2 * beta, 2 * gamma]
    fig, ax = plt.subplots(2)
    ax[0].axvline(alpha, color='k', linestyle='--', label=r'$\theta$')
    ax[1].axvline(beta, color='k', linestyle='--')
    _ = [x.axvline(tp, color='k', linestyle='--') for x, tp in zip(ax, true_param)]

    ax[0].axvline(2*alpha, color='k', label=r'$\theta_0$')
    _ = [x.axvline(tp, color='k') for x, tp in zip(ax, init_param)]
    # ax[1].axvline(2*beta, color='k', marker='o')

    colormap = cm.get_cmap('Greys', len(nu_list))
    colors = colormap(np.linspace(.35,.75,len(nu_list)))
    ymax = 0

    for param, nu, color in zip(reversed(params), reversed(nu_list), colors):
        
        x0, bin0, p0 = ax[0].hist(param['alpha'], bins=100, density=True, color=color, label=r'$\nu=%i$' %nu)
        x1, bin1, p1 = ax[1].hist(param['beta'], bins=100, density=True, color=color)
        for item0, item1 in zip(p0, p1):
            x0_norm =item0.get_height()/sum(x0)
            x1_norm =item1.get_height()/sum(x1)
            item0.set_height(x0_norm)
            item1.set_height(x1_norm)
            ymax = max(ymax, x0_norm, x1_norm)

    ymax = 0.1
    ax[0].legend()
    _ = [x.set_ylim([0, ymax]) for x in ax]
    _ = [x.set_title(t) for t, x in zip(['alpha', 'beta'], ax)]
    _ = [x.grid() for x in ax]
    # ax[1].legend()
    plt.show()
