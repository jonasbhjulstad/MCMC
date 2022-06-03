import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
if __name__ == '__main__':
    I0 = 1000
    alpha = 1./9
    R0 = 1.2
    beta = R0*alpha
    gamma = 1./3
    mu = alpha
    p = 0.5
    N_pop = 10000


    df = pd.read_csv('../Data/SEIAR_I0_' + str(I0) + '.csv', skiprows=2, header=None)
    fig, ax = plt.subplots(5)
    for i, (x,s) in enumerate(zip(ax, ['S', 'E', 'I', 'A', 'R'])):
        x.plot(df[i], color='k')
        x.set_ylabel(s)
    ax[2].plot(df[2], linestyle='', marker='x', markersize=4.0, color='k', label='')
    ax[2].legend()
    ax[0].set_title(r'SEIAR Model, $N_{pop} = %.1E, I_0 = %.1E, \alpha = %.2f, \beta = %.2f, \gamma = %.2f, p = %.1f, \mu = %.2f$' %(N_pop, I0, alpha, beta, gamma, p, mu))
    _ = [x.grid() for x in ax]
    ax[-1].set_xlabel('Time[Days]')


    plt.show()