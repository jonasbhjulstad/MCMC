def RK4(f, X, DT, arg=[]):
    if arg != []:
        k1 = f(X, arg)
        k2 = f(X + DT / 2 * k1, arg)
        k3 = f(X + DT / 2 * k2, arg)
        k4 = f(X + DT * k3, arg)
    else:
        k1 = f(X)
        k2 = f(X + DT / 2 * k1)
        k3 = f(X + DT / 2 * k2)
        k4 = f(X + DT * k3)
    return X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def RK4_M(f, X, DT, M, arg=[]):
    xk = X
    for i in range(M):
        xk =  RK4(f, xk, DT/M, arg=arg)
    return xk