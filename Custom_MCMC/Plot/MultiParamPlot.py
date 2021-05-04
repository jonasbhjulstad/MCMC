import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    nu_list = [4, 16, 64]
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
    ax[0].axvline(alpha)
    ax[1].axvline(beta)

    for param in params, nu_list):
    
        ax[0].hist(param['alpha'], bins=30)
        ax[1].hist(param['beta'], bins=30)
    # ax[0].legend()
    # ax[1].legend()
    plt.show()
