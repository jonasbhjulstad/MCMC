#include <CL/sycl.hpp>
#include <MCMC_Sampler.hpp>
#include <SIR_Integrators.hpp>
#include <SIR_Stochastic.hpp>
using _Engine = oneapi::dpl::ranlux48;
static _Engine engine;
static constexpr size_t Nt = 10;
static constexpr size_t N_particles = 1000;
static constexpr size_t N_MCMC = 1000;
using Model = MCMC::SIR_Stochastic<realtype, Nt, decltype(engine)>;
using SMC_Sampler = SMC::Sampler<Model, Nt, N_particles, _Engine>;
using MCMC_Sampler = MCMC::Sampler<Model, N_particles, N_MCMC, realtype, _Engine>;
int main() {
  // Deterministic SIR-ODE parameters:
  realtype alpha = .1;
  realtype R0 = 1.4;
  realtype beta = alpha * R0;
  realtype dt = 5.;


  // Initial population sizes
  realtype N_pop = 10000;
  realtype I0 = 100;
  std::array<realtype, 3> x0 = {N_pop - I0, I0, 0};
  // Initial parameters (alpha, beta)
  std::array<realtype, 2> param_init = {1.0,1.0};
  // Standard deviation of proposal parameter distribution
  std::array<realtype, 2*2> prop_std = {1.,0., 0.,1.};
  // Standard deviation of likelihood parameter distribution
  realtype ll_std = N_pop / 10;

  // Stochastic-SIR overdispersion:
  realtype nu_I = 0;
  realtype nu_R = 0;


  // Get deterministic trajectory
  MCMC::Integrators::SIR_Deterministic integrator(x0, alpha, beta, N_pop, dt);

  std::cout << "Computing deterministic trajectory.." << std::endl;
  auto traj = integrator.run_trajectory(x0, Nt);
  std::array<realtype, Nt> y;
  std::ofstream y_file("trajectory.csv");
  for (int t = 0; t < Nt; t++) {
    y[t] = traj[t][1];
    y_file << y[t] << "\n";
  }
  y_file.close();


  cl::sycl::queue queue;


  Model model(y, x0, dt, N_pop, prop_std, ll_std, nu_I, nu_R, engine, queue);

  realtype threshold = .9;
  SMC_Sampler smc_sampler(model, threshold);
  MCMC_Sampler mcmc_sampler(param_init, smc_sampler, queue);
  std::cout << "Running MCMC-Chain.." << std::endl;
  auto result = mcmc_sampler.run_chain(true);

  // Write result to csv file
  std::ofstream ofs("result.csv");
  for (int i = 0; i < result.size(); i++) {
    ofs << result[i][0] << "," << result[i][1] << "\n";
  }
  ofs.close();
  return 0;

  return EXIT_SUCCESS;
}