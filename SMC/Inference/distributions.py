from scipy.stats import nbinom, betabinom
def negBin(lbd, nu):
    r = lbd/nu
    if lbd < 1.0:
        return lbd
    else:
        return nbinom.rvs(r, lbd/(r+lbd)) 

def betaBin(mu, nu, n):
    if n < 5.0:
        return n
    elif mu == 0.0:
        return mu
    else:
        gamma = nu/(n-1)
        a = (1/gamma - 1)*mu,
        b = (gamma-1)*(mu-1)/gamma
        return  betabinom.rvs(n, a, b)

def negBin_sampler(lbd, nu):
    r = lbd/nu
    return nbinom(r, lbd/(r+lbd)) 

def betaBin_sampler(mu, nu, n):
    gamma = nu/(n-1)
    a = (1/gamma - 1)*mu,
    b = (gamma-1)*(mu-1)/gamma
    return  betabinom(n, a, b)
