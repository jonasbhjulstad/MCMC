#ifndef SIR_MODEL_H
#define SIR_MODEL_H
#include <oneapi/dpl/random>
#include "Particle.hpp"
#include "SMC_Model.hpp"


template <size_t N_observations, typename realtype, class _RNG_Engine>
class SIR_Model : public SMC::Model<SIR_Model<N_observations, realtype, _RNG_Engine>, 3, 3> {
private:
  static constexpr size_t Nx = 3;
  const realtype ll_std;
  const std::vector<realtype> prop_std;
  const std::vector<realtype> y;
  const std::vector<realtype> x0;
  realtype dt;
  realtype N_pop;
  realtype nu_I;
  realtype nu_R;
  _RNG_Engine& engine;

public:
  SIR_Model(const std::vector<realtype> &y, const std::vector<realtype> &x0,
            realtype dt, realtype N_pop, const realtype *prop_std,
            realtype ll_std, realtype nu_I, realtype nu_R, _RNG_Engine& engine)
      : y(y), x0(x0), dt(dt), N_pop(N_pop), ll_std(ll_std), prop_std(prop_std), engine(engine) {
  }

  // Calculates the next state and likelihood for that state
  void advance(const size_t &t, std::vector<realtype> &particle_state,
               const std::vector<realtype> &param) {
    realtype alpha = param[0];
    realtype beta = param[1];

    realtype p_I = 1 - exp(-beta * particle_state[1] / N_pop * dt);
    realtype p_R = 1 - exp(-alpha * dt);

    realtype K_SI;
    realtype K_IR;

    K_SI = BinomialSample((uint)particle_state[0], p_I);
    K_IR = BinomialSample((uint)particle_state[1], p_R);
    realtype delta_x[Nx] = {-K_SI, K_SI - K_IR, K_IR};

    for (int i = 0; i < Nx; i++) {
      particle_state[i] += delta_x[i];
    }
  }
  // Update the parameter proposal:
  void proposal_sample(const std::vector<realtype> &param_old,
                       std::vector<realtype> &res) {
    MultivariateNormalSample<2>(rng, param_old, prop_sigma,
                                                res);
  }

  void reset(std::vector<realtype>& state)
  {
    if (state.size() < Nx)
    {
      state.resize(Nx);
    }
    state[0] = x0[0];
    state[1] = x0[1];
    state[2] = x0[2];
  }

  realtype log_likelihood(const std::vector<realtype>& state, const size_t& t)
  {
    oneapi::dpl::normal_distribution<realtype> normal_dist(y[t], ll_std);
    return normal_dist()
  } 
};
#endif