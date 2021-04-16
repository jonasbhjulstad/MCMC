import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Integrator.ERK import RK4_M
from models.Epidemiological import SIR_y, SIR_stochastic, SIR_stochastic_dispersed
from scipy.integrate import odeint
from matplotlib import cm
import multiprocessing as mp
from itertools import repeat

def disperse_sim(param):
    X_list = np.zeros((param['N']+1, len(param['x0']*param['N_sim'])))
    X_list[0,:] = np.tile(x0, param['N_sim'])
    for k in range(param['N']):
        for i in range(param['N_sim']):
            X_list[k+1,i:i+3] = SIR_stochastic_dispersed(X_list[k, i:i+3], param)
    return X_list


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
    Nk_sto = 10
    Nk_disp = 10
    I0 = 1e7
    x0 = [N_pop-I0, I0, 0]
    param = {'alpha': 1./9, 'beta': beta, 'N_pop': N_pop, 'dt': dt_sto, 'nu_I': 1, 'nu_R': 1e-6, 'N': N, 'x0': x0, 'N_sim': Nk_sto}


    tgrid_sto = np.linspace(0, 360, N + 1)
    tgrid_det = np.linspace(0, 360, N_det + 1)
    
    fig, ax = plt.subplots(3)

    X_sto = [[x0] for i in range(Nk_sto)]
    X_sto_disp = [[x0] for i in range(Nk_disp*Nk_sto)]


    nu_vals = np.linspace(0.1, 0.2,Nk_disp)
    nu_list = np.repeat(nu_vals, Nk_sto)


    params = []
    for nu in nu_vals:
        param['nu_I'] = nu
        params.append(param)

    Np = len(params)
    parallel = False
    if parallel:
        pool = mp.Pool(mp.cpu_count())
        X_disp = pool.map(disperse_sim, params)
        pool.close()
    else:
        X_disp = [disperse_sim(param) for param in params]

    for i in range(N):
        for Xk in X_sto:
            Xk.append(SIR_stochastic(Xk[-1], param))



    colormap = cm.get_cmap('Greys', len(X_sto_disp))
    colors = colormap(np.linspace(.3,.8,len(X_sto_disp)))
    _ = [[x.plot(tgrid_sto, [xk[i] for xk in Xk], color='k') for i,x in zip(range(3), ax)] for Xk in X_sto]
    _ = [[x.plot(tgrid_sto[::int(len(Xk)/50)], [xk[i] for xk in Xk][::int(len(Xk)/50)], color='k', marker='o',linestyle='',markersize=2.8) for i,x in zip(range(3), ax)] for Xk in X_sto]
    X_disp = [np.array_split(Xk_disp, Nk_sto, axis=1) for Xk_disp in X_disp]
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
    colors = colormap(np.linspace(.3,.8,Nk_disp))

    for min_k, max_k, mean_k,std_k, color in zip(X_disp_min[::-1], X_disp_max[::-1], X_disp_mean[::-1], X_disp_std[::-1], colors[::-1]):
        _ = [x.fill_between(tgrid_sto, mean_k[:,i] -std_k[:,i], mean_k[:,i] + std_k[:,i], color=color, alpha=0.4) for i ,x in enumerate(ax)]
        # _ = [x.fill_between(tgrid_sto, max_k[:,i], color=color) for i, x in enumerate(ax)]
        _ = [x.plot(tgrid_sto, mean_k[:,i]) for i,x in enumerate(ax)]
    # _ = [[x.plot(tgrid_sto, [xk[i] for xk in Xk], color=color) for i,x in zip(range(3), ax)] for Xk, color in zip(X_sto_disp, colors)]

    _ = ax[1].plot(tgrid_sto[::int(len(Xk)/50)], [xk[1] for xk in Xk][::int(len(Xk)/50)], color='k', marker='o',linestyle='',markersize=2.8, label='No dispersion')

    # ax[1].plot(tgrid_sto, [xk[1] for xk in X_sto_disp[-1]], color=colors[-1], linestyle='--', label='Dispersed trajectories')


    # X_det = np.zeros((N_det+1, 4))
    # X_det[0,:] = x0 + [0]
    # for i in range(N_det):
    #     X_det[i+1,:] = RK4_M(SIR_y, X_det[i,:], dt_det, M, arg=[alpha, beta])

    X_det = odeint(lambda x, t, p: SIR_y(x, p), x0 + [0], tgrid_det, args=(param,))

    _ = [x.plot(tgrid_det[::int(len(X_det)/50)], X_det[:,i][::int(len(X_det)/50)], color='k', linestyle='', marker='x', markersize=2.8, label='Deterministic') for i,x in zip(range(3), ax)]
    _ = [x.plot(tgrid_det, X_det[:,i], color='k') for i,x in zip(range(3), ax)]
    _ = [x.grid() for x in ax]
    _ = [x.set_xticklabels([]) for x in ax[:-1]]
    _ = [x.set_ylabel(t) for t, x in zip(['S', 'I', 'R'], ax)]
    _ = ax[1].legend()
    ax[-1].set_xlabel('time [days]')
    ax[0].set_title(r'SIR-model, $N_{pop} = %.1E, I_0 = %.1E, \alpha = %.2f$, %i simulations for each stochastic model, $\nu = [%.1f, %.2f, \dots, %.1f]$' %(N_pop, I0, alpha, Nk_sto, nu_vals[0], nu_vals[1], nu_vals[-1]))

    plt.show()

    a = 1
    # fig.savefig('~/Prosjektoppgave/Images/SIR_Trajectory_Comparison_Dispersion.eps', format='eps')