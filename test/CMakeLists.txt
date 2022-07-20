add_executable(MC_pi_approximation MC_pi_approximation.cpp)
target_link_libraries(MC_pi_approximation PUBLIC MKL::mkl_sycl MKL::mkl_intel_ilp64 MKL::mkl_tbb_thread MKL::mkl_core)

add_executable(t2 t2.cpp)
target_link_libraries(t2 PUBLIC MKL::mkl_sycl MKL::mkl_intel_ilp64 MKL::mkl_tbb_thread MKL::mkl_core oneDPL)

add_executable(SMC_Sampler SMC_Sampler_test.cpp)
target_link_libraries(SMC_Sampler PUBLIC MKL::mkl_sycl MKL::mkl_intel_ilp64 MKL::mkl_tbb_thread MKL::mkl_core GTest::gtest_main)
target_include_directories(SMC_Sampler PRIVATE "../Kernels")

add_executable(MCMC_Sampler MCMC_Sampler_test.cpp)
target_link_libraries(MCMC_Sampler PUBLIC MKL::mkl_sycl MKL::mkl_intel_ilp64 MKL::mkl_tbb_thread MKL::mkl_core GTest::gtest_main)
target_include_directories(MCMC_Sampler PRIVATE "../Kernels")


include(GoogleTest)
gtest_discover_tests(SMC_Sampler)
gtest_discover_tests(MCMC_Sampler)