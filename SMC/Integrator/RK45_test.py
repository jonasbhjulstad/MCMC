from scipy.integrate import Radau
from models.Epidemiological import SIR_y
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    alpha = .1 / 9
    N_pop = 5.3e6
    R0 = 1.2
    beta = R0 * alpha

    I0 = 2000
    x0 = [N_pop - I0, I0, 0, 0]

    M = 10
    N = 5000
    T = 30
    DT = T/N/M
    tgrid = np.linspace(0, T, N + 1)
    X = np.zeros((4, N + 1))
    X_list = []
    X[:, 0] = x0
    p_true = 1 - np.exp(-beta / (alpha*N_pop))
    Nk = 100
    tgrid = np.linspace(0,T,10000)

    solver = Radau(lambda t,x: SIR_y(x, [alpha, beta]),0,x0,T)

    t = []
    y = []
    while solver.status != 'finished':
        t.append(solver.t)
        y.append(solver.y)
        solver.step()

    fig, ax = plt.subplots(4)
    for i, x in enumerate(ax):
        _ = [x.plot(t, [yk[i] for yk in y])]
    plt.show()