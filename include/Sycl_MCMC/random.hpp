#ifndef Sycl_MCMC_RANDOM_HPP
#define Sycl_MCMC_RANDOM_HPP

#include <random>
#include <array>
#include <tinymt/tinymt.h>
namespace Sycl_MCMC::random
{

    struct uniform_real_distribution;
    template <typename dType = float>
    struct bernoulli_distribution;
    template <typename dType = float>
    struct normal_distribution;
    template <typename dType = float>
    struct binomial_distribution;
    template <typename uI_t = uint32_t>
    struct poisson_distribution;

    template <typename Derived, typename dType = float>
    struct distribution
    {
        template <typename RNG = tinymt::tinymt32>
        dType operator()(RNG &rng)
        {
            return static_cast<Derived *>(this)->operator()(rng);
        }
    };
    template <size_t N_distributions, typename RNG = tinymt::tinymt32, typename dType = float, typename ... Dist_Types>
    struct distributionOp: public distribution<distributionOp<N_distributions, RNG, dType, Dist_Types...>, RNG, dType>
    {
        using Dist_t = std::variant<Dist_Types...>;
        std::array<Dist_t, N_distributions> distributions;
        dType operator()(RNG &rng)
        {
            return std::accumulate(distributions.begin(), distributions.end(), [&](dType sum, Dist_t& dist){
                return sum + std::visit([&](auto& dist){return dist(rng);}, dist);
            });
        }
    };

    template <typename dType = float>
    struct uniform_real_distribution: public distribution<uniform_real_distribution<dType>, dType>
    {
        uniform_real_distribution() = default;
        uniform_real_distribution(dType a, dType b) : a(a), b(b) {}
        dType a = 0;
        dType b = 1;
        template <typename RNG = tinymt::tinymt32>
        dType operator()(RNG &rng)
        {
            auto val = rng();
            // convert val to random uniform float
            return ((dType)val / (dType)rng.max()) * (b - a) + a;
        }
        void set_a(dType a) { this->a = a; }
        void set_b(dType b) { this->b = b; }
    };
    template <typename dType = float>
    struct bernoulli_distribution: public distribution<bernoulli_distribution<dType>, dType>
    {
        bernoulli_distribution(dType p) : p(p) {}
        dType p = 0;
        template <typename RNG = tinymt::tinymt32>
        dType operator()(RNG &rng)
        {
            auto val = rng();
            dType uniform = (val / (dType)rng.max());
            return uniform < p;
        }
    };

    template <typename dType = float>
    struct normal_distribution: public distribution<normal_distribution<dType>, dType>
    {
        normal_distribution() = default;
        normal_distribution(dType mean, dType stddev) : mean(mean), stddev(stddev) {}
        dType mean = 0;
        dType stddev = 1;
        template <typename RNG = tinymt::tinymt32>
        dType operator()(RNG &rng)
        {
            auto val = rng();
            // convert val to random uniform float
            dType uniform = (val / (dType)rng.max());
            return std::sqrt(-2 * std::log(uniform)) * std::cos(2 * M_PIf * uniform) * stddev + mean;
        }
    };
    template <typename dType = float>
    struct binomial_distribution: public distribution<binomial_distribution<dType>, dType>
    {
        binomial_distribution(dType n, dType p) : n(n), dist(p) {}
        bernoulli_distribution<T> dist;
        dType n;
        template <typename RNG = tinymt::tinymt32>
        dType operator()(RNG &rng)
        {
            T count = 0;
            for (T i = 0; i < n; i++)
            {
                count += dist(rng);
            }
            return count;
        }
        void set_trials(dType n)
        {
            this->n = n;
        }
        void set_probability(dType p)
        {
            dist.p = p;
        }
    };

    template <typename dType>
    struct poisson_distribution: public distribution<binomial_distribution<dType>, dType>
    {
        poisson_distribution(dType lambda) : lambda(lambda) {}
        dType lambda;
        template <typename RNG = tinymt::tinymt32>
        dType operator()(RNG &rng)
        {
            T count = 0;
            T p = 1;
            T L = std::exp(-lambda);
            uniform_real_distribution<dType> dist(0, 1);
            while (p > L)
            {
                p *= dist(rng);
                count++;
            }
            return count - 1;
        }
    };

    // multinomial distribution with static size
    template <size_t N, typename dType = float, typename uI_t = uint32_t>
    struct multinomial_distribution: : public distribution<binomial_distribution<dType>, dType>
    {
        multinomial_distribution(std::array<T, N> p) : p(p) {}
        std::array<T, N> p;
        template <typename RNG = tinymt::tinymt32>
        std::array<T, N> operator()(RNG &rng)
        {
            std::array<T, N> counts;
            T sum = 0;
            for (size_t i = 0; i < N; i++)
            {
                counts[i] = binomial_distribution<T>(1, p[i])(rng);
                sum += counts[i];
            }
            for (size_t i = 0; i < N; i++)
            {
                counts[i] /= sum;
            }
            return counts;
        }

        template <typename RNG = tinymt::tinymt32>
        std::array<uI_t, N> operator()(sycl::queue &q, sycl::buffer<T, 1> &p, sycl::buffer<RNG, 1> rng_buf)
        {
            std::array<T, N> counts;
            q.submit([&](sycl::handler &cgh)
                     {
                auto p_acc = p.get_access<sycl::access::mode::read>(cgh);
                auto rng_acc = rng_buf.get_access<sycl::access::mode::read_write>(cgh);
                cgh.parallel_for<class multinomial_distribution>(sycl::range<1>(N), [=](sycl::id<1> i) {
                    counts[i] = binomial_distribution<T>(1, p_acc[i])(rng_acc[i]);
                }); });
            q.wait();
            T sum = 0;
            for (size_t i = 0; i < N; i++)
            {
                sum += counts[i];
            }
            for (size_t i = 0; i < N; i++)
            {
                counts[i] /= sum;
            }
            return counts;
        }
    };

    typedef tinymt::tinymt32 default_rng;
#endif
}
#endif