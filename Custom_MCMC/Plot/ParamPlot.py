import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    print('test')
    df_param = pd.read_csv("../Data/param_out.csv", names=['alpha', 'beta'], header=None)
    df_weights = pd.read_csv("../Data/weight_out.csv", names=['Weights'], header=None)

    print(df_weights)
    plt.hist(df_weights)
    plt.show()
