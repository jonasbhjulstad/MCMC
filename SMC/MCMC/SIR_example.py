class SIR_I:

# instance attribute
    def __init__(self, x0):
        self.x = x0
        self.theta = [0,0]

    def F(x, t, theta):
        alpha = theta[0]
        beta = theta[1]
        S, I, R = x[0], x[1], x[2]
        return [-beta * S * beta * S * I/N_pop - alpha * I, alpha * I, beta/N_pop * S * I]
