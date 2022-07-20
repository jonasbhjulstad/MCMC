#ifndef SMC_TEST_MODEL_HPP
#define SMC_TEST_MODEL_HPP
#include <iostream>
#include <numeric>
#include <vector>
#include <limits>

#include <SMC_Model.hpp>
#include <SMC_Sampler.hpp>
#include <oneapi/dpl/random>
static constexpr size_t N_states = 2;
static constexpr size_t N_parameters = 2;
static constexpr size_t N_particles = 10;
static constexpr size_t Nt = 10;
static const std::vector<double> param_init = {1.0, 1.0};
using _Engine = oneapi::dpl::ranlux48;
static _Engine engine;

class TestModel : public SMC::Model<TestModel, double> {
public:
  using SMC_Model = SMC::Model<TestModel, double>;

  void advance(const size_t &t, std::vector<double>& particle_state,
               const std::vector<double> &param) {
    particle_state[0] += param[0];
    particle_state[1] += param[1];
  }

  void proposal_sample(const std::vector<double> &param_old,
                       std::vector<double> &res) {
    res[0] = param_old[0] + 0.1;
    res[1] = param_old[1] + 0.1;
  }

  void reset(std::vector<double> &particle_state) {
    particle_state[0] = 0;
    particle_state[1] = 0;
  }

  realtype log_likelihood(const std::vector<double>& state, const size_t & t)
  {
    return 1;
  }

  size_t get_Nx()
  {
    return N_states;
  }
};


#endif