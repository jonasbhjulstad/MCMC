#include <SIR_Integrators.hpp>
#include <gtest/gtest.h>

static constexpr size_t Nt = 10;
static constexpr size_t Nx = 3;

TEST(Integrator_Model_Test, Stochastic_SIR) {
  double alpha = .1;
  double beta = .1;
  double N_pop = 100;
  double I0 = 10;
  double dt = .1;
  MCMC::Integrators::SIR_Stochastic model(alpha, beta, N_pop, dt);
  std::array<realtype, Nx> x0 = {N_pop - I0, I0, 0};
  std::vector<std::array<realtype, Nx>> trajectory =
      model.run_trajectory(x0, Nt);
  // check that all elements of trajectory are positive
  for (auto &x : trajectory) {
    for (auto &y : x) {
      EXPECT_GE(y, 0);
    }
  }
}

TEST(Integrator_Model_Test, Deterministic_SIR) {
  double alpha = .1;
  double beta = .1;
  double N_pop = 100;
  double I0 = 10;
  double dt = .1;
  size_t Nt = 10;

  std::array<realtype, Nx> x0 = {N_pop - I0, I0, 0};
  MCMC::Integrators::SIR_Deterministic model(x0, alpha, beta, N_pop, dt);
  std::vector<std::array<realtype, Nx>> trajectory = model.run_trajectory(x0, Nt);
  // check that all elements of trajectory are positive
  for (auto &x : trajectory) {
    //sum x
    realtype sum = 0;
    for (auto &y : x) {
      sum += y;
    }
    EXPECT_NEAR(sum, N_pop, 1e-5);
    EXPECT_NE(x[1], 0);
    for (auto &y : x) {
      EXPECT_GE(y, 0);
    }
  }
}