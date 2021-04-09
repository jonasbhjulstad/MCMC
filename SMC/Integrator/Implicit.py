import numpy as np
import numpy.linalg as la
from models.Epidemiological import SIR_y, grad_SIR_y

def IRK4_f(k, param):
    xt = param['xt']
    t = param['t']
    dt = param['dt']
    A = param['A']
    c = param['c']
    f = param['f']

    Nx = len(xt)
    Nstage = A.shape[0]
    K = np.reshape(k, (Nx, Nstage))
    Tg = t + dt * c.reshape((-1, 1))
    Xg = xt + dt * K @ A.T
    g = np.reshape(K - f(Tg, Xg), [], 1)
    return g


def IRK4_F(k, param):
    xt = param['xt']
    t = param['t']
    dt = param['dt']
    A = param['A']
    c = param['c']
    dfdx = param['F']

    Nx = len(xt)
    Nstage = A.shape[0]
    K = np.reshape(k, (Nx, Nstage))
    TG = t + dt * c.reshape((-1, 1))
    XG = xt + dt * K @ A.T
    dfdxG = [dfdx(tg, xg).T for tg, xg in zip(TG, XG)]
    G = np.eye(Nx * Nstage) - np.tile(dfdxG, (1, Nstage)) * np.kron(dt * A, np.ones(Nx))
    return G


def newton_rhapson(g, G, k0, param, tol=1e-3):
    gk = g(k0, param)
    ki = k0
    while la.norm(gk) > tol:
        ki = ki - la.inv(G(ki, param)) @ gk
        gk = g(ki, param)
    return ki


def IRK4(butcher, f, F, T, x0):
    Nt = len(T)
    Nx = len(x0)
    dT = np.diff(T)
    x = np.zeros((Nx, Nt))
    x[:, 0] = x0
    param = butcher
    param['f'] = f
    param['F'] = F
    Nstage = param['A'].shape[0]
    xt = x0
    k = np.tile(f(T[0], x0), (Nstage, 1))

    for nt in range(1, Nt):
        param['t'] = T[nt - 1]
        param['dt'] = dT[nt - 1]
        param['xt'] = xt

        k = newton_rhapson(IRK4_f, IRK4_F, k, param)
        K = k.reshape((Nx, Nstage))
        xt += param['dt']* K@param['b']
        x[:,nt] = xt
    return x


if __name__ == '__main__':
    alpha = .1 / 9
    N_pop = 5.3e6
    R0 = 1.2
    beta = R0 * alpha
    I0 = 2000
    x0 = [N_pop - I0, I0, 0]

    M = 10
    N = 5000
    T = 3
    DT = T/N/M
    tgrid = np.linspace(0, T, N + 1)

    c = np.reshape([.5 - np.sqrt(3)/6, .5 + np.sqrt(3)/6], (-1,1))
    b = np.reshape([.5, .5], (1,-1))
    A = np.array([[.25, .25 - np.sqrt(3)/6],
                  [.25 + np.sqrt(3)/6, .25]])
    butcher = {'A': A, 'b': b, 'c': c}
    theta = [alpha, beta]

    x_traj = IRK4(butcher, lambda t, x: SIR_y(x, theta), lambda t, x: grad_SIR_y(x, theta), tgrid,x0)






