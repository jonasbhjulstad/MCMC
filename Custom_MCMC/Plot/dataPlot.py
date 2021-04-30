import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
if __name__ == '__main__':
    df = pd.read_csv('../Data/SIR_y.csv', skiprows=2, header=None)
    plt.plot(df)
    plt.show()