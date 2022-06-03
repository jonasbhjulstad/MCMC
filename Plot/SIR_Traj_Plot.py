import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
if __name__ == '__main__':
    I0 = 1000
    alpha = 1./9
    R0 = 1.2
    beta = R0*alpha
    N_pop = 10000

    df = pd.read_csv('../Data/SIR_I0_' + str(I0) + '.csv', skiprows=2, header=None)
    fig, ax = plt.subplots(3)
    for i, (x,s) in enumerate(zip(ax, ['S', 'I', 'R'])):
        x.plot(df[i], color='k', label=s)
        x.set_ylabel(s)
    ax[1].plot(df[1], linestyle='', marker='x', markersize=4.0, color='k', label='measurement')
    ax[2].legend()
    ax[0].set_title(r'SIR Model, $N_{pop} = %.1E, I_0 = %.1E, \alpha = %.2f, \beta = %.2f$' %(N_pop, I0, alpha, beta))
    _ = [x.grid() for x in ax]
    ax[-1].set_xlabel('Time[Days]')


    plt.show()