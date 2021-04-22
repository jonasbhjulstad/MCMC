#include <array>
#include <iostream>
#include <armadillo>
#include <gmtl>


template <int M>
vec RK4_M(vec (*f)(vec, const double*), vec X, const double* param, const double DT)
{
    double dt = DT/M;
    for (int i; i < M; i++)
    {
        vec k1 = f(X, param);
        vec k2 = f(X + dt/2*k1, param);
        vec k3 = f(X + dt/2*k2, param);
        vec k4 = f(X + dt*k3, param);
        X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    }
    return X;
}

vec SIR(vec X, const double* param)
{
    const double alpha = param[0];
    const double beta = param[1];
    vec Xdot(3);
    Xdot[0] = -beta*X[0]*X[1];
    Xdot[1] = beta*X[0]*X[1] - alpha*X[1];
    Xdot[2] = alpha*X[1];
    return Xdot;
}


int main()
{
    int N = 100;
    double N_pop = 1e8;
    double I0 = 1e7;
    double S0 = N_pop-I0;
    mat X_traj(3, N, fill::zeros);
    X_traj.col(0) = {S0, I0, 0}; 
    const double tspan[2] = {0, 28};
    const double DT = (tspan[1] - tspan[0])/N;
    const int M = 8;

    const double R0 = 1.2;
    const double alpha = 1./9;
    const double beta = R0*alpha;
    const double param[3] = {alpha, beta, N_pop};
    std::cout << "Initialized.." << endl;
    for (int i; i < N; i++)
    {
        X_traj.col(i) = RK4_M<M>(SIR, X_traj.col(i), &param[0], DT);
        std::cout << X_traj(i,1) << endl;
        std::cout << "Running.." << endl;
    }

}