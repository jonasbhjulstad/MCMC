#ifndef SMC_RANDOM_WALK_MODEL_HPP
#define SMC_RANDOM_WALK_MODEL_HPP
#include <oneapi/dpl/random>
#include <gsl/gsl_randist.h>
#include "SMC_Model.hpp"
template <typename _RNG_Engine> 
struct RandomWalk : public SMC::Model<RandomWalk<_RNG_Engine>, double>  
{
  _RNG_Engine &engine;
  oneapi::dpl::normal_distribution<double> sample_dist;
  double prop_std;
  double ll_std;

  RandomWalk(_RNG_Engine &engine, double sample_std, double prop_std, double ll_std)
      : engine(engine), sample_dist(0.0, sample_std), prop_std(prop_std), ll_std(ll_std) {}

  void advance(const size_t &t, std::vector<double> &particle_state,
               const std::vector<double> &param) {
    particle_state[0] += sample_dist(engine);
  }

  void proposal_sample(const std::vector<double> &param_old,
                       std::vector<double> &res) {
    oneapi::dpl::normal_distribution prop_dist(param_old[0], prop_std);
    res[0] = prop_dist(engine);
  }

  void reset(std::vector<double> &particle_state) {
    particle_state[0] = 0;
  }

  double log_likelihood(const std::vector<double> &state, const size_t &t) {
    return gsl_ran_gaussian_pdf(state[0], ll_std);
  }

  size_t get_Nx() { return 1; }
};


#endif // SMC_RANDOM_WALK_MODEL_HPP