#include "SMC_Test_Model.hpp"
#include <MCMC_Sampler.hpp>
#include <CL/sycl.hpp>
#include <gtest/gtest.h>


TEST(MCMC_SamplerTest, Advance) {
  TestModel model;
  double threshold = .9;
  SMC::Sampler<TestModel, _Engine> smc_sampler(model, threshold, N_particles, Nt, engine);
  MCMC::Sampler<TestModel, double, _Engine> mcmc_sampler(param_init, Nt, smc_sampler);
  EXPECT_EQ(mcmc_sampler.log_sum_weights.size(), N_particles);
  mcmc_sampler.advance();

  EXPECT_EQ(mcmc_sampler.log_sum_weights[0], Nt);  
}

TEST(MCMC_SamplerTest, run_chain)
{
  
}

