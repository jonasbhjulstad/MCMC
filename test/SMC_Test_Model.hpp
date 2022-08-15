#ifndef SMC_TEST_MODEL_HPP
#define SMC_TEST_MODEL_HPP
#include <SMC_Model.hpp>
#include <SMC_Sampler.hpp>
#include <iostream>
#include <limits>
#include <numeric>
#include <oneapi/dpl/random>
#include <vector>
static constexpr size_t N_particles = 10;
static constexpr size_t Nt = 10;
static constexpr size_t N_MCMC = 1000;
using _Engine = oneapi::dpl::ranlux48;
static _Engine engine;
struct TestModel : public SMC::Model<TestModel, 2, 2, double> {
  static constexpr size_t Nx = 2;
  static constexpr size_t Np = 2;
  static constexpr size_t N_particles = 10;
  static constexpr size_t Nt = 10;

  TestModel() : SMC::Model<TestModel, 2, 2, double>() {}

  void advance(const size_t &t, std::array<realtype, 2> &particle_state, const std::array<realtype, 2> &param) const {
    particle_state[0] += param[0];
    particle_state[1] += param[1];
  }

  void proposal_sample(const std::array<realtype, 2> &param_old, std::array<realtype, 2> &res) {
    res[0] = param_old[0] + 0.1;
    res[1] = param_old[1] + 0.1;
  }

  void reset(std::array<realtype, 2> &particle_state) const {
    particle_state[0] = 0;
    particle_state[1] = 0;
  }

  realtype log_likelihood(const std::array<realtype, 2> &state, const size_t &t) const { return 1; }

  size_t get_Nx() { return 2; }
};

#endif