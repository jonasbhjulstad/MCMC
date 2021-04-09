import matplotlib.pyplot as plt
from numpy.random import normal, uniform
import numpy as np
import numpy.linalg as la
def HMC(U, grad_U, epsilon, L, current_q):
    q = current_q
    p = normal(loc=0, scale=1, size=(len(current_q), 1))
    current_p = p
    p = p - epsilon*grad_U(q) /2

    for i in range(L):
        q = q + epsilon*p

        if i < (L-1):
            p = p-epsilon*grad_U(q)
    p = p-epsilon * grad_U(q)/2
    p = -p
    current_U = U(current_q)
    current_K = current_p.T@current_p/2
    proposed_U = U(q)
    proposed_K = current_p.T @ current_p/2

    exp_k = la.norm(np.exp(current_U - proposed_U + current_K - proposed_K))
    if (uniform() < la.norm(np.exp(current_U - proposed_U + current_K - proposed_K))):
        return q, p
    else:
        return current_q, current_p

    def leapfrog():
        for i in range(L):
            q = q + epsilon * p

            if i != L:
                p = p - epsilon * grad_U(q)
        p = p-epsilon * grad_U(q)/2
        p = -p

if __name__ == '__main__':
    Sigma = [[1, .95],
             [.95,1]]
    U = lambda q: q.T @ la.inv(Sigma) @ q/2
    grad_U = lambda q: la.inv(Sigma) @ q

    q0 = np.array([[-1.5, 1.55]]).T
    p0 = np.array([[-1,1]]).T
    epsilon = .01
    L = 25
    q_list = [q0]
    p_list = [p0]
    qk = q0
    for i in range(10000):
        qk, pk = HMC(U, grad_U, epsilon, L, qk)
        q_list.append(qk)
        p_list.append(pk)

    q_list = np.array(q_list)
    p_list = np.array(p_list)

    fig, ax = plt.subplots(2)
    ax[0].plot(q_list[:,0,:], q_list[:,1,:])
    ax[1].plot(p_list[:,0,:], p_list[:,1,:])

    plt.show()