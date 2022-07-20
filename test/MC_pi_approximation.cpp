#include <iostream>
#include <numeric>
#include <vector>
#include <numeric>

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "oneapi/mkl/rng/device.hpp"
static constexpr size_t seed = 7777;
static const auto pi = 3.1415926535897932384626433832795;

static constexpr size_t N_samples = 120000000;
double estimate_pi(sycl::queue& q, size_t N_points)
{
    using namespace oneapi;
    double estimated_pi;
    size_t N_under_curve = 0;


    size_t N_per_thread = 32;
    constexpr size_t vec_size = 2;

    {
        sycl::buffer<size_t, 1> count_buf(&N_under_curve, 1);
        q.submit([&](sycl::handler& h)
        {
            auto count_acc = count_buf.template get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(N_points/(N_per_thread*vec_size/2)), [=](sycl::item<1>item){
                size_t id_global = item.get_id(0);
                sycl::vec<float, vec_size> r;
                sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::relaxed,
                sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_counter {count_acc[0]};
                
                
                mkl::rng::device::uniform distr;
                mkl::rng::device::philox4x32x10<vec_size> engine(seed, id_global*N_per_thread*vec_size);
                size_t count = 0;
                for (int i = 0; i < N_per_thread; i++)
                {
                    r = mkl::rng::device::generate(distr, engine);
                    if (sycl::length(r) < 1.0)
                    {
                        count++;
                    }
                }

                atomic_counter.fetch_add(count);

            });
        });

        
    }

    estimated_pi = N_under_curve/((double) N_points)*4.0;

    return estimated_pi;
}

int main()
{


    auto exception_handler = [&](sycl::exception_list exceptions)
    {
        for (std::exception_ptr const& e: exceptions)
        {
            try
            {
                std::rethrow_exception(e);
            }
            catch( sycl::exception const& e)
            {
                std::cout << "Caught asynchronous SYCL exception:\n"
                          << e.what() << std::endl;
                std::terminate();
            }
        }

    };
    double estimated_pi;
    try {
        sycl::queue q(sycl::default_selector{}, exception_handler);

        estimated_pi = estimate_pi(q, N_samples);
    }
    catch(...)
    {
        std::cout << "Failure" << std::endl;
        std::terminate();
    }

        // Printing results
    // std::cout << "Estimated value of Pi = " << estimated_pi << std::endl;
    // std::cout << "Exact value of Pi = " << pi << std::endl;
    // std::cout << "Absolute error = " << fabs(pi-estimated_pi) << std::endl;
    // std::cout << std::endl;
    assert(fabs(pi-estimated_pi) < 0.01);
    return 0;
    
}
