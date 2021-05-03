import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    print('test')
    
    df_param = pd.read_csv("../Data/param_out.csv", names=['alpha', 'beta'], header=None)
    df_weights = pd.read_csv("../Data/weight_out.csv", names=['Weights'], header=None)

    discard = False
    if discard:
        a = 1000
        print("Discarding first %i".format(a))
        df_param = df_param[a:]
    
    R0 = 1.2
    alpha = 1./9
    beta = alpha*R0
    print(df_weights.iloc[5:20])
    fig, ax = plt.subplots(2)
    ax[0].axvline(alpha)
    ax[1].axvline(beta)
    # df_weights['Weights'] = np. exp(df_weights
    # ['Weights'] - max(df_weights['Weights']))
    ax[0].hist(df_param['alpha'], label=df_param.columns[0], bins=30)
    ax[1].hist(df_param['beta'], label=df_param.columns[1], bins=30)
    ax[0].legend()
    ax[1].legend()
    plt.show()
