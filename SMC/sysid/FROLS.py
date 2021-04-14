
import numpy as np
import numpy.linalg as la
def FROLS(X, y, rho, fun_dict):
    
    ND = len(fun_dict)
    N_samples, Nx = np.shape(X)
    sigma = y.T @ y
    s_max = 10
    
    g = np.zeros((1,s_max))
    g_m = np.zeros((s_max, ND))
    q_s = np.zeros((N_samples, s_max))
    g_s = np.zeros((1, s_max))
    
    ERRs = []
    ERRs_max = np.zeros((1,s_max))
    ells = np.zeros((1,s_max))
    
# =============================================================================
#      Step 1: s = 1
# =============================================================================
    for i in range(ND):
        q = X[:,i]
        g_m[0,i] = y @ q / q.T @ q
        ERRs.append(g_m[0,i]**2*q.T @ q/sigma)
    
    ERRs_max[0,0] = np.amax(ERRs)
    ells[0,0] = int(np.where(ERRs == ERRs_max[0,0])[0])
    print(ells)
    g[0,0] = g_m[0,int(ells[0,0])]
    q_s[:,0] = X[:,int(ells[0,0])]
    
    
# =============================================================================
#     %% Step 2: s >= 2
# =============================================================================
    s = 2
    ESR = 10
    while((ESR >= rho) and s < s_max):
        g_m = np.zeros((1, ND))
        q_m = np.zeros((N_samples, ND))
        ERRs = np.zeros((1,ND))
        
        for i in range(ND):
            if i not in ells:

                p_m = X[:,i]
        
                q_m[:,i] = p_m - sum_of_projections(p_m, q_s[:,1:s-1])
                
                g_m[i] = y.T@q_m[:,i]@ la.inv(q_m[:,i].T@q_m[:,i])
                
                ERRs[i] = g_m[:,i]**2@(q_m[:,i].T@q_m[:,i]/sigma)
            else:
                ERRs[i] = -np.Inf
            

        ERRs_max[s] = np.amax(ERRs)
        ell = np.where(ERRs == ERRs_max[s])
        q_s[:,s] = q_m[:,ell]
        g_s[s] = g_m[ell]
        
        ESR = 1- sum(ERRs_max)
        s = s+1
    return g, q, ERRs_max
    
def sum_of_projections(p, q):
    N_samples, N_q = np.shape(q)
    proj_sum = []
    for r in range(N_q):
        proj_sum = proj_sum + p.T @ q[:,r]/(q[:,r].T @ q[:,r]) @ q[:,r]
    return proj_sum