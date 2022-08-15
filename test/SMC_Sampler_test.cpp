#include "SMC_Test_Model.hpp"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>



TEST(SMC_SamplerTest, ParticleReset) {
  TestModel model;
  double threshold = .9;
  SMC::Sampler<TestModel, Nt, N_particles, _Engine> sampler(model, threshold,
                                           engine);

  auto particles = sampler.get_particles();
  for (auto &particle : particles) {
    EXPECT_EQ(particle.log_weight, 0);
  }
}

TEST(SMC_SamplerTest, NormalizeAccumulate) {
  TestModel model;
  double threshold = .9;
  SMC::Sampler<TestModel, Nt, N_particles, _Engine> sampler(model, threshold, engine);

  auto particles = sampler.get_particles();
  sampler.normalize_accumulate_weights();
  double maxWeight = -std::numeric_limits<double>::infinity();

  for (auto &particle : particles) {
    maxWeight = std::max(maxWeight, particle.log_weight);
  }
  particles = sampler.get_particles();

  for (auto &particle : particles) {
    EXPECT_LE(particle.log_weight, maxWeight);
  }
}

TEST(SMC_SamplerTest, Advance) {
  TestModel model;
  double threshold = .9;
  SMC::Sampler<TestModel, Nt, N_particles, _Engine> sampler(model, threshold,
                                                        engine);
  std::array<double, 2> param = {1., 1.};
  sampler.advance(param);
  auto particles = sampler.get_particles();

  for (auto &particle : particles) {
    EXPECT_EQ(particle.state[0], .1);
    EXPECT_EQ(particle.state[1], .1);
  }
}
