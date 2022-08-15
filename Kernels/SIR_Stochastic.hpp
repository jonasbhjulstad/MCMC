#ifndef SIR_Stochastic_H
#define SIR_Stochastic_H
#include "SMC_Model.hpp"
#include "SMC_Particle.hpp"
#include <MCMC_Distributions.hpp>
#include <SIR_Integrators.hpp>
#include <oneapi/dpl/cmath>

namespace MCMC {
template <typename realtype, size_t _Nt, class _Engine>
class SIR_Stochastic : public SMC::Model<SIR_Stochastic<realtype, _Nt, _Engine>,
                                         3, 2, realtype> {
public:
  static constexpr size_t Nx = 3;
  static constexpr size_t Np = 2;
  static constexpr size_t Nt = _Nt;

private:
  realtype dt;
  realtype N_pop;
  realtype nu_I;
  realtype nu_R;
  const realtype ll_std;
  const std::array<realtype, Np * Np> prop_std;
  const std::array<realtype, Nt> y;
  const std::array<realtype, Nx> x0;
  MCMC::multivariate_normal_distribution<realtype, Np> dist_prop;

public:
  SIR_Stochastic(const std::array<realtype, Nt> &y,
                 const std::array<realtype, Nx> &x0, realtype dt,
                 realtype N_pop, const std::array<realtype, Np * Np> &prop_std,
                 realtype ll_std, realtype nu_I, realtype nu_R, _Engine &engine,
                 sycl::queue& queue)
      : y(y), x0(x0), dt(dt), N_pop(N_pop), ll_std(ll_std), prop_std(prop_std),
        dist_prop(prop_std, queue) {}

  // Calculates the next state and likelihood for that state
  void advance(const size_t &t, std::array<realtype, Nx> &particle_state,
               const std::array<realtype, Np> &param, _Engine &engine) const {
    using namespace oneapi::dpl;
    realtype alpha = param[0];
    realtype beta = param[1];

    realtype p_I = 1 - std::exp(-beta * particle_state[1] / N_pop * dt);
    realtype p_R = 1 - std::exp(-alpha * dt);

    realtype K_SI;
    realtype K_IR;

    MCMC::binomial_distribution SI_dist(particle_state[0], p_I);
    MCMC::binomial_distribution IR_dist(particle_state[1], p_R);

    // K_SI = BinomialSample((uint)particle_state[0], p_I);
    // K_IR = BinomialSample((uint)particle_state[1], p_R);
    K_SI = SI_dist(engine);
    K_IR = IR_dist(engine);

    std::array<realtype, 3> delta_x = {-K_SI, K_SI - K_IR, K_IR};

    for (int i = 0; i < Nx; i++) {
      particle_state[i] += delta_x[i];
    }
  }
  // Update the parameter proposal:
  void proposal_sample(const std::array<realtype, Np> &param_old,
                       std::array<realtype, Np> &res,
                       _Engine& engine) const {

    res = dist_prop(engine, param_old);
  }

  void reset(std::array<realtype, Nx> &state) {
    state[0] = x0[0];
    state[1] = x0[1];
    state[2] = x0[2];
  }

  realtype log_likelihood(const std::array<realtype, Nx> &state,
                          const size_t &t) const {
    return normal_distribution_pdf(state[1], y[t], ll_std);
  }
};
} // namespace MCMC
#endif