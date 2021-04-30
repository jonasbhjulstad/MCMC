import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    print('test')
    df_param = pd.read_csv("../Data/param_out.csv", names=['alpha', 'beta'], header=None)
    df_weights = pd.read_csv("../Data/weight_out.csv", names=['Weights'], header=None)

    df_param = df_param.iloc[100:]
    
    df_weights = df_weights[df_weights['Weights'] < 10]
    print(df_weights)
    # df_weights['Weights'] = np.exp(df_weights['Weights'] - max(df_weights['Weights']))
    plt.hist(df_param, label=df_param.columns, bins=100)
    plt.legend()
    plt.show()
