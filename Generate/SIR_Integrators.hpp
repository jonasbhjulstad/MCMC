#ifndef SIR_INTEGRATORS_HPP
#define SIR_INTEGRATORS_HPP
#include "Integrator_Models.hpp"
#include <algorithm>
namespace MCMC::Integrators {

struct SIR_Stochastic : public Model_Integrator<3> {
public:
  static constexpr size_t Nx = 3;
  oneapi::dpl::default_engine engine;
  realtype alpha, beta, N_pop, dt;

  SIR_Stochastic(realtype alpha, realtype beta, realtype N_pop, realtype dt)
      : alpha(alpha), beta(beta), N_pop(N_pop), dt(dt) {}

  std::array<realtype, 3>step(const std::array<realtype, 3>&x) {
    realtype p_I = 1 - std::exp(-beta * x[1] / N_pop * dt);
    realtype p_R = 1 - std::exp(-alpha * dt);

    realtype K_SI;
    realtype K_IR;

    MCMC::binomial_distribution SI_dist(x[0], p_I);
    MCMC::binomial_distribution IR_dist(x[1], p_R);

    K_SI = SI_dist(engine);
    K_IR = IR_dist(engine);

    std::array<realtype, 3> delta_x = {-K_SI, K_SI - K_IR, K_IR};

    std::array<realtype, 3> x_next;
    for (int i = 0; i < Nx; i++) {
      x_next[i] = std::max({x[i] + delta_x[i], 0.});
    }
    return x_next;
  }
};

// Simple function that calculates the differential equation.
static int SIR_eval_f(realtype t, N_Vector x, N_Vector x_dot, void *param) {
  realtype *x_data = N_VGetArrayPointer(x);
  realtype *x_dot_data = N_VGetArrayPointer(x_dot);
  realtype S = x_data[0];
  realtype I = x_data[1];
  realtype R = x_data[2];
  realtype alpha = ((realtype *)param)[0];
  realtype beta = ((realtype *)param)[1];
  realtype N_pop = ((realtype *)param)[2];

  x_dot_data[0] = -beta * S * I / N_pop;
  x_dot_data[1] = beta * S * I / N_pop - alpha * I;
  x_dot_data[2] = alpha * I;

  return 0;
}

// Jacobian function vector routine.
static int SIR_eval_jac(N_Vector v, N_Vector Jv, realtype t, N_Vector x,
                        N_Vector fx, void *param, N_Vector tmp) {
  realtype *x_data = N_VGetArrayPointer(x);
  realtype *Jv_data = N_VGetArrayPointer(Jv);
  realtype *v_data = N_VGetArrayPointer(v);
  realtype S = x_data[0];
  realtype I = x_data[1];
  realtype R = x_data[2];
  realtype alpha = ((realtype *)param)[0];
  realtype beta = ((realtype *)param)[1];
  realtype N_pop = ((realtype *)param)[2];

  Jv_data[0] = -beta * I / N_pop * v_data[0] - beta * S / N_pop * v_data[1];
  Jv_data[1] = beta * I / N_pop * v_data[0] + beta * S / N_pop * v_data[1] -
               alpha * v_data[1];
  Jv_data[2] = alpha * v_data[1];

  return 0;
}

struct SIR_Deterministic : public CVODE_Integrator<3, SIR_Deterministic> {
  realtype param[3];
  SIR_Deterministic(const std::array<realtype, 3>&x0, realtype alpha, realtype beta,
                    realtype N_pop, realtype dt)
      : CVODE_Integrator<3, SIR_Deterministic>(x0, dt) {
    param[0] = alpha;
    param[1] = beta;
    param[2] = N_pop;
    assert(this->initialize_solver(SIR_eval_f, SIR_eval_jac, (void*) param) == EXIT_SUCCESS);
  }
};
} // namespace MCMC::Integrators
#endif // SIR_INTEGRATORS_HPP