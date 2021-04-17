import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Integrator.ERK import RK4_M
from models.Epidemiological import *
from scipy.integrate import odeint
from matplotlib import cm
import multiprocessing as mp
from itertools import repeat

def disperse_sim(par):
    X_list = []
    f = par['f_sim']
    for k in range(par['N_sim']):
        X = [x0]
        for i in range(par['N']):
            X.append(f(X[-1], par))
        X_list.append(X)
    return np.array(X_list)

def stochastic_simulation(parms,nu_list, N_det, tspan, Nk_disp, Nk_sto, x0, f_det):

    N = param['N']
    f_sim = param['f_sim']
    Nx = len(x0)
    tgrid_sto = np.linspace(tspan[0], tspan[1], N + 1)
    tgrid_det = np.linspace(tspan[0], tspan[1], N_det + 1)
    
    fig, ax = plt.subplots(Nx)

    X_sto = [[x0] for i in range(Nk_sto)]
    X_sto_disp = [[x0] for i in range(Nk_disp*Nk_sto)]



    Np = len(parms)
    parallel = False
    if parallel:
        pool = mp.Pool(mp.cpu_count())
        X_disp = pool.map(disperse_sim, parms)
        pool.close()
    else:
        X_disp = []
        for nu in nu_list:
            param['nu'] = nu
            X_disp.append(disperse_sim(param))

    for i in range(N):
        param['nu'] = np.ones((param['nu'].shape))*1e-6
        for Xk in X_sto:
            Xk.append(f_sim(Xk[-1], param, dispersed=False))



    colormap = cm.get_cmap('Greys', len(X_sto_disp))
    colors = colormap(np.linspace(.3,.8,len(X_sto_disp)))
    _ = [[x.plot(tgrid_sto, [xk[i] for xk in Xk], color='k') for i,x in zip(range(Nx), ax)] for Xk in X_sto]
    _ = [[x.plot(tgrid_sto[::int(len(Xk)/50)], [xk[i] for xk in Xk][::int(len(Xk)/50)], color='k', marker='o',linestyle='',markersize=4) for i,x in zip(range(Nx), ax)] for Xk in X_sto]
    X_disp_mean = []
    X_disp_min = []
    X_disp_max = []
    X_disp_std = []
    for X_disp_k in X_disp:
        X_disp_mean.append(np.mean(X_disp_k, axis=0))
        X_disp_min.append(np.min(X_disp_k, axis=0))
        X_disp_max.append(np.max(X_disp_k, axis=0))
        X_disp_std.append(np.std(X_disp_k, axis=0))
    
    

    colormap = cm.get_cmap('Greys', Nk_disp)
    colors = colormap(np.linspace(.2,.6,Nk_disp))

    for X_disk_k, min_k, max_k, mean_k,std_k, color in zip(X_disp[::-1], X_disp_min[::-1], X_disp_max[::-1], X_disp_mean[::-1], X_disp_std[::-1], colors):
        _ = [x.fill_between(tgrid_sto, mean_k[:,i] -std_k[:,i], mean_k[:,i] + std_k[:,i], color=color, alpha=0.6) for i ,x in enumerate(ax)]
        _ = [x.plot(tgrid_sto, mean_k[:,i] -std_k[:,i], color=color,linestyle='--',alpha=0.4) for i ,x in enumerate(ax)]
        _ = [x.plot(tgrid_sto, mean_k[:,i] +std_k[:,i], color=color,linestyle='--',alpha=0.4) for i ,x in enumerate(ax)]
        # _ = [[x.plot(tgrid_sto, xk[:,i], color=color) for i, x in enumerate(ax)] for xk in X_disp_k]
        # _ = [x.fill_between(tgrid_sto, min_k[:,i], max_k[:,i], color=color) for i, x in enumerate(ax)]
        # _ = [x.plot(tgrid_sto, mean_k[:,i]) for i,x in enumerate(ax)]
    # _ = [[x.plot(tgrid_sto, [xk[i] for xk in Xk], color=color) for i,x in zip(range(Nx), ax)] for Xk, color in zip(X_sto_disp, colors)]

    _ = ax[1].plot(tgrid_sto[::int(len(Xk)/50)], [xk[1] for xk in Xk][::int(len(Xk)/50)], color='k', marker='o',linestyle='',markersize=2.8, label='No dispersion')

    X_det = odeint(lambda x, t, p: f_det(x, p), x0, tgrid_det, args=(param,))

    _ = [x.plot(tgrid_det[::int(len(X_det)/50)], X_det[:,i][::int(len(X_det)/50)], color='k', linestyle='', marker='x', markersize=4, label='Deterministic') for i,x in zip(range(3), ax)]
    _ = [x.plot(tgrid_det, X_det[:,i], color='k') for i,x in zip(range(Nx), ax)]
    _ = [x.grid() for x in ax]
    _ = [x.set_xticklabels([]) for x in ax[:-1]]
    _ = [x.set_ylabel(t) for t, x in zip(['S', 'I', 'R'], ax)]
    _ = ax[1].legend()
    ax[-1].set_xlabel('time [days]')
    nu_vals = nu_list[:,0]
    ax[0].set_title(r'SEIR-model, $N_{pop} = %.1E, I_0 = %.1E, \alpha = %.2f$, %i simulations for each stochastic model, $\nu = [%.1f, %.2f, \dots, %.1f]$' %(N_pop, I0, alpha, Nk_sto, nu_vals[0], nu_vals[1], nu_vals[-1]))

    plt.show()


if __name__ == '__main__':

    alpha = 1./9
    N_pop = 1e8
    R0 = 1.2
    beta = alpha*R0

    dt_sto = 1.
    dt_det = 1./100


    N = 360
    N_det = N*100
    M = 500
    Nk_sto = 100
    Nk_disp = 6
    I0 = 1e7
    x0 = [N_pop-I0, 0, I0, 0]
    Nx = len(x0)
    param = {'alpha': 1./9, 'beta': beta, 'gamma': 1./3, 'N_pop': N_pop, 'dt': dt_sto, 'N': N, 'x0': x0, 'N_sim': Nk_sto, 'f_sim': SEIR_stochastic}


    nu_list = np.ones((Nk_disp, Nx-1))*1e-6
    nu_list[:,0] = np.linspace(1e2,1e6,Nk_disp)

    stochastic_simulation(param, nu_list, N_det, [0, 360], Nk_disp, Nk_sto, x0, SEIR)

    a = 1
    # fig.savefig('~/Prosjektoppgave/Images/SIR_Trajectory_Comparison_Dispersion.eps', format='eps')