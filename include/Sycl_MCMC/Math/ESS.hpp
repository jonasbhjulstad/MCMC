#ifndef Sycl_MCMC_ESS_HPP
#define Sycl_MCMC_ESS_HPP
#include <CL/sycl.hpp>
namespace Sycl_MCMC::Math
{
    template <typename T, typename dType = float>
    dType ESS(sycl::queue& q, sycl::buffer<T, 1>& log_weights)
    {
        auto f_expsum = [](auto x, auto y) { return x + sycl::exp(y); };
        auto expsum = sycl::reduce(q, log_weights, 0.0, f_expsum);

        auto f_expsum2 = [](auto x, auto y) { return x + sycl::exp(2 * y); };
        auto expsum2 = sycl::reduce(q, log_weights, 0.0, f_expsum2);
        return expsum * expsum / expsum2;
    }
}
#endif