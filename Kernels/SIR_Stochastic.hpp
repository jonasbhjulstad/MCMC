#ifndef SIR_MODEL_H
#define SIR_MODEL_H
#include <oneapi/dpl/cmath>
#include <MCMC_Distributions.hpp>
#include <Eigen/Dense>
#include "SMC_Particle.hpp"
#include "SMC_Model.hpp"


template <typename realtype, class _RNG_Engine>
class SIR_Model : public SMC::Model<SIR_Model<realtype, _RNG_Engine>, realtype> {
private:
  using Vec = Eigen::Vector<realtype, 2>;
  using Mat = Eigen::Matrix<realtype, 2, 2>;
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
  MCMC::multivariate_normal_distribution<realtype> dist_prop;
public:
  SIR_Model(const std::vector<realtype> &y, const std::vector<realtype> &x0,
            realtype dt, realtype N_pop, const Mat& prop_std,
            realtype ll_std, realtype nu_I, realtype nu_R, _RNG_Engine& engine)
      : y(y), x0(x0), dt(dt), N_pop(N_pop), ll_std(ll_std), prop_std(prop_std), engine(engine) {
  }

  // Calculates the next state and likelihood for that state
  void advance(const size_t &t, std::vector<realtype> &particle_state,
               const std::vector<realtype> &param) {
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

    realtype delta_x[Nx] = {-K_SI, K_SI - K_IR, K_IR};

    for (int i = 0; i < Nx; i++) {
      particle_state[i] += delta_x[i];
    }
  }
  // Update the parameter proposal:
  void proposal_sample(const std::vector<realtype> &param_old,
                       std::vector<realtype> &res) {
    
    Vec result = dist_prop(engine, Eigen::Map<Vec>(param_old.data()));
    res = std::vector(result.data(), result.data() + result.size());
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
    gsl_ran_gaussian_pdf(y[t] - state, ll_std);
  } 
};
#endif