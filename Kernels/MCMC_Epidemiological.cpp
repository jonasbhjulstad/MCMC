#ifndef MCMC_EPIDEMIOLOGICAL_cpp
#define MCMC_EPIDEMIOLOGICAL_cpp
#include "MCMC_Sampler.cpp"
#include "SIR_Stochastic.hpp"
void SIR_compute(const ulong *seed, const realtype *x_init,
                 const realtype *y_obs, const realtype *param_init, realtype dt,
                 realtype N_pop, const realtype *prop_std, realtype ll_std,
                 realtype nu_I_init, realtype nu_R_init,
                 realtype resample_threshold, realtype *param_res,
                 realtype *log_sum_weight_res) {


  typedef SIR_Model<N_OBSERVATIONS> Model;
  constexpr size_t N_param = 2;
  constexpr size_t Nx = 2;
  PRNG_GENERATOR rng(seed[gID]);
  realtype param[N_param * (N_MCMC_ITERATIONS + 1)];
  realtype param_prop[N_param];
  realtype log_sum_weights[N_MCMC_ITERATIONS + 1];
  copy_vec(param, param_init, N_param);

  Model model(N_MCMC_ITERATIONS, y_obs, x_init, dt, N_pop, prop_std, ll_std,
              nu_I_init, nu_R_init);

  SMC::Particle<Nx> particles[N_PARTICLES];

  for (size_t i = 0; i < N_MCMC_ITERATIONS; i++) {
    realtype *param_prev = &param[i * N_param];
    realtype *param_current = &param[(i + 1) * N_param];
    realtype &log_sum_weight_prev = log_sum_weights[i];
    realtype &log_sum_weight_current = log_sum_weights[i + 1];
    MCMC::advance<PRNG_GENERATOR, Model, Nx, N_PARTICLES, N_param>(
        rng, model, particles, param_prev, param_current, param_prop,
        log_sum_weight_prev, log_sum_weight_current, N_OBSERVATIONS,
        resample_threshold);
  }
  // printf("First params %f, %f\n", param[0], param[1]);
  copy_vec(&param_res[(N_param * N_MCMC_ITERATIONS + 1) * gID], param,
           N_param * (N_MCMC_ITERATIONS + 1));
  copy_vec(&log_sum_weight_res[(N_MCMC_ITERATIONS + 1) * gID], log_sum_weights,
           N_MCMC_ITERATIONS + 1);

  log_sum_weight_res[0] = 1000;
}


#endif