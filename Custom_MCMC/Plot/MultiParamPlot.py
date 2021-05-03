import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    nu_list = [1,2,3,4,5,6,7,8]
    params = []
    weights = []
    for nu in nu_list:
        params.append(pd.read_csv("../Data/param_" + str(nu) + ".csv", names=['alpha', 'beta'], header=None))
        weights.append(pd.read_csv("../Data/weight_" + str(nu) + ".csv", names=['Weights'], header=None))


    discard = False
    if discard:
        a = 1000
        print("Discarding first %i".format(a))
        df_param = df_param[a:]
    
    R0 = 1.2
    alpha = 1./9
    beta = alpha*R0
    fig, ax = plt.subplots(2)

    imax = 0
    for param, nu in zip(reversed(params), reversed(nu_list)):
        x0, bins0, p0 = ax[0].hist(param['alpha'], label=r'$\nu = %i$' %nu, bins=100, density=True, histtype='step')
        x1, bins1, p1 = ax[1].hist(param['beta'], label=r'$\nu = %i' %nu, bins=100, density=True, histtype='step')

        # for i0, i1 in zip(p0, p1):
        #     i0.set_height(i0.get_height()/sum(x0))
        #     i1.set_height(i1.get_height()/sum(x1))
        #     imax = max(imax, i0.get_height(), i1.get_height())

    ax[0].legend()
    ax[0].axvline(alpha, color='k', linestyle='--')
    ax[1].axvline(beta, color='k', linestyle='--')
    # ax[0].set_ylim([0,imax])
    # ax[1].set_ylim([0,imax])
    # ax[1].legend()
    plt.show()
