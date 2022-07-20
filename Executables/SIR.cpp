#include <iostream>
#include <numeric>
#include <vector>
#include <numeric>

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "oneapi/mkl/rng/device.hpp"

#include "Parameter_Configuration.hpp"
#include <MCMC_Sampler.cpp>
#include <SIR_Stochastic.hpp>
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

}

static const auto seed = 7777;
using namespace oneapi;
template <size_t Nt, size_t N_chains, typename realtype>
void run_chain(sycl::queue& q, MCMC::Sampler<SIR_Model, Nt>& mcmc_sampler)
{
    mkl::rng::philox4x32_10_t rng(seed);
    mkl::rng::uniform distr;

    size_t wg_size = std::min(q.get_device().get_info<sycl::info::device::max_work_group_size>(), n_points);
    size_t max_compute_units = q.get_device().get_info<sycl::info::device::max_compute_units>();
    size_t wg_num = (N_chains > wg_size * max_compute_units) ? max_compute_units : 1;

    size_t N_chain_threads = N_chains / (wg_size * wg_num);
    std::vector<realtype> log_sum_weights(N_chains*Nt);

    {
        sycl::buffer<size_t, 1> seed_buffer(N_chains);
        sycl::buffer<realtype, 1> llsum_buffer(log_sum_weights);
        q.submit([&](sycl::handler& h) {
            auto llsum_acc = llsum_buffer.template get_access<sycl::access::mode::write>(h);
            auto seed_acc = seed_buffer.template get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::nd_range<1>(wg_size*wg_num, wg_size),
            [=](sycl::nd_item<1> item)
            {
                realtype 
            })
        })
    }

}


int main()
{

    SIR_Model model(y, x0, dt, N_pop, prop_std, ll_std, nu_I, nu_R);
      Sampler(SMC::Model<Impl_Model> &model, const realtype threshold) {

    SMC::Sampler smc_sampler(model, threshold);
    MCMC::Sampler<SIR_Model, Nt> mcmc_sampler(model, smc_sampler);

}