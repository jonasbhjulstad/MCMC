#include "SMC_Test_Model.hpp"
#include <CL/sycl.hpp>
#include <MCMC_Sampler.hpp>
#include <SIR_Integrators.hpp>
#include <SIR_Stochastic.hpp>
#include <gtest/gtest.h>
#include <array>


TEST(MCMC_SamplerTest, Advance) {
  TestModel model;
  realtype threshold = .9;
  std::array<realtype, 2> param_init = {1.0, 1.0};
  SMC::Sampler<TestModel, Nt, N_particles, _Engine> smc_sampler(model, threshold, engine);
  MCMC::Sampler<TestModel, N_particles, N_MCMC, realtype, _Engine> mcmc_sampler(param_init, Nt,
                                                         smc_sampler);
  EXPECT_EQ(mcmc_sampler.log_sum_weights.size(), N_particles);
  mcmc_sampler.advance();

  EXPECT_EQ(mcmc_sampler.log_sum_weights[0], Nt);
}

TEST(MCMC_SamplerTest, SIR_Chain) {
  realtype alpha = .1;
  realtype beta = .1;
  realtype N_pop = 100;
  realtype I0 = 10;
  realtype dt = .1;
  realtype nu_I = 0;
  realtype nu_R = 0;
  std::array<std::array<double, 2>, 2> prop_std = {1.0, 0., 0., 1.};
  realtype ll_std = N_pop/10;

  size_t Nt = 10;
  constexpr size_t Nx = 3;
  size_t N_MCMC = 1000;

  std::array<realtype, Nx> x0 = {N_pop - I0, I0, 0};
  MCMC::Integrators::SIR_Deterministic integrator(x0, alpha, beta, N_pop, dt);

  auto traj = integrator.run_trajectory(x0, Nt);
  std::array<realtype, Nt> y;
  for (int t = 0; t < Nt; t++)
  {
    y[t] = traj[t][1];
  }

  sycl::queue queue(sycl::default_selector{});

  using Model = MCMC::SIR_Stochastic<realtype, decltype(engine)>;
  Model model(y, x0, dt, N_pop, prop_std, ll_std, nu_I, nu_R, engine, queue);

  std::array<realtype, 2> param_init = {1., 1.};
  realtype threshold = .9;
  SMC::Sampler<Model, Nt, N_particles, _Engine> smc_sampler(model, threshold, N_particles,
                                               Nt, engine);
  MCMC::Sampler<Model, N_particles, N_MCMC, realtype, _Engine> mcmc_sampler(param_init,
                                                         smc_sampler);
  EXPECT_EQ(mcmc_sampler.log_sum_weights.size(), N_MCMC);

  std::vector<Vec> result = mcmc_sampler.run_chain();
}
