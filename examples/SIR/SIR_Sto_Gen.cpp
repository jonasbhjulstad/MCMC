#include <array>
#include <iostream>
#include <armadillo>
// #include <gmtl>

template <int Nx>
class State{
    public:
    double X[Nx];

    void set_values(double* x)
    {
        for (int i; i < Nx; i++)
        {
            std::cout << x[i] << std::endl;
            X[i] = x[i];
        }
    }

    double& operator[](int i)
    {
        return X[i];
    }

    void operator=(State Y)
    {
        for (int i; i < Nx; i++)
        {
            X[i] = Y[i];
        }
    }


    State operator+(State y)
    {
        State Z;
        for (int i; i < Nx; i++)
        {
            Z[i] += X[i] + y[i];
        }
        return Z;
    }

    State operator*(State y)
    {
        State Z;
        for (int i; i < Nx; i++)
        {
            Z[i]*=X[i] + y[i];
        }
        return Z;
    }

    template<typename T>
    State mul(T Y)
    {
        State Z;
        Z.set_values(&X[0]);
        for (int i; i < Nx; i++)
        {
           Z[i] *= Y; 
        }
        return Z;
    }

    template<typename T>
    State add(T Y)
    {
        State Z;
        for (int i; i < Nx; i++)
        {
            Z[i] = X[i] + Y;
        }
        return Z;
    }


};


template <int Nx, int M>
void RK4_M(State<Nx> (*f)(State<Nx>, const double*), State<Nx>* X, const double* param, const double DT)
{
    double dt = DT/M;
    State<Nx> Xk;
    Xk.set_values(X->X);
    for (int i; i < M; i++)
    {
        State<Nx> k1 = f(Xk, param);
        State<Nx> k2 = f(Xk + k1.mul(dt/2), param);
        State<Nx> k3 = f(Xk + k2.mul(dt/2), param);
        State<Nx> k4 = f(Xk + k3.mul(dt), param);
        Xk = Xk + (k1 +  k2.mul(2) + k3.mul(2) + k4).mul(DT / 6);
    }
    *(X+1) = Xk;
}

State<3> SIR(State<3> X, const double* param)
{
    const double alpha = param[0];
    const double beta = param[1];
    return (State<3>) {-beta*X[0]*X[1], beta*X[0]*X[1] - alpha*X[1], alpha*X[1]};
}


int main()
{
    const int Nx = 3;
    int N = 100;
    double N_pop = 1e8;
    double I0 = 1e7;
    double S0 = N_pop-I0;
    State<Nx>* X_traj = (State<Nx>*)malloc(sizeof(State<Nx>)*(N+1));
    double x0[3] = {S0, I0, 0.0};
    X_traj[0].set_values(&x0[0]); 
    const double tspan[2] = {0, 28};
    const double DT = (tspan[1] - tspan[0])/N;
    const int M = 8;

    const double R0 = 1.2;
    const double alpha = 1./9;
    const double beta = R0*alpha;
    const double param[3] = {alpha, beta, N_pop};
    std::cout << "Initialized.." << std::endl;
    for (int i; i < N; i++)
    {
        RK4_M<Nx,M>(SIR, &X_traj[i], &param[0], DT);
        std::cout << X_traj[i][1] << std::endl;
        std::cout << "Running.." << std::endl;
    }

}