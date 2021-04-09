n_states = 2
n_odeparams = 4
n_ivs = 2
import numpy as np
from Epidemiological import SIR_y, grad_SIR_y
from casadi import *
from scipy.integrate import odeint

def compile_SIR_functions():
    X = MX.sym('X', 4)
    theta = MX.sym('theta', 2)
    Xdot = SIR_y(X, theta)
    F = Function('Xdot', [X, theta], [Xdot])
    grad_F_X = Function('grad_F_X', [X, theta], [jacobian(Xdot, X)])
    grad_F_theta = Function('grad_F_theta', [X, theta], [jacobian(Xdot, theta)])

    C = CodeGenerator('SIR.c')
    C.add(F)
    C.add(grad_F_X)
    C.add(grad_F_theta)
    exec('gcc -fPIC -shared SIR.c -o SIR.so')

    


class SIRModel:
    def __init__(self, n_states, n_odeparams, n_ivs, x0=None):
        self._n_states = n_states
        self._n_odeparams = n_odeparams
        self._n_ivs = n_ivs
        self._x0 = x0
        compile_SIR_functions()
        self.F = external('f', './SIR.so')
        self.grad_F_X = external('grad_F_X', './SIR.so')
        self.grad_F_theta = external('grad_F_theta', './SIR.so')

    def simulate(self, parameters, times):
        return self._simulate(parameters, times, False)

    def simulate_with_sensitivities(self, parameters, times):
        return self._simulate(parameters, times, True)

    def _simulate(self, theta, times, sensitivities):

        def r(x, t, theta):
            return self.f(x, theta)

        if sensitivities:

            def jac(x):
                return self.grad_F_X(x, theta)

            def dfdp(x):

                return self.grad_F_X(x, theta)

            def rhs(y_and_dydp, t, p):
                y = y_and_dydp[0 : self._n_states]
                dydp = y_and_dydp[self._n_states :].reshape(
                    (self._n_states, self._n_odeparams + self._n_ivs)
                )
                dydt = r(y, t, p)
                d_dydp_dt = np.matmul(jac(y), dydp) + dfdp(y)
                return np.concatenate((dydt, d_dydp_dt.reshape(-1)))

            y0 = np.zeros((2 * (n_odeparams + n_ivs)) + n_states)
            y0[6] = 1.0  # \frac{\partial  [X]}{\partial Xt0} at t==0, and same below for Y
            y0[13] = 1.0
            y0[0:n_states] = [Xt0, Yt0]
            result = odeint(rhs, y0, times, (theta,), rtol=1e-6, atol=1e-5)
            values = result[:, 0 : self._n_states]
            dvalues_dp = result[:, self._n_states :].reshape(
                (len(times), self._n_states, self._n_odeparams + self._n_ivs)
            )
            return values, dvalues_dp
        else:
            values = odeint(r, [Xt0, Yt0], times, (theta,), rtol=1e-6, atol=1e-5)
            return values

if __name__ == '__main__':
    test = SIRModel()