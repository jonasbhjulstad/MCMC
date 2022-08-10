add_executable(MC_pi_approximation MC_pi_approximation.cpp)
target_link_libraries(MC_pi_approximation PUBLIC MKL::mkl_sycl MKL::mkl_intel_ilp64 MKL::mkl_tbb_thread MKL::mkl_core)

add_executable(t2 t2.cpp)
target_link_libraries(t2 PUBLIC MKL::mkl_sycl MKL::mkl_intel_ilp64 MKL::mkl_tbb_thread MKL::mkl_core oneDPL)

add_executable(SMC_Sampler SMC_Sampler_test.cpp)
target_link_libraries(SMC_Sampler PUBLIC MKL::mkl_sycl MKL::mkl_intel_ilp64 MKL::mkl_tbb_thread MKL::mkl_core GTest::gtest_main Eigen3::Eigen)
target_include_directories(SMC_Sampler PRIVATE "../Kernels")

add_executable(MCMC_Sampler MCMC_Sampler_test.cpp)
target_link_libraries(MCMC_Sampler PUBLIC MKL::mkl_sycl MKL::mkl_intel_ilp64 MKL::mkl_tbb_thread MKL::mkl_core GTest::gtest_main Eigen3::Eigen)
target_include_directories(MCMC_Sampler PRIVATE "../Kernels")

add_executable(SMC_Resampling SMC_Resampling_test.cpp)
target_link_libraries(SMC_Resampling PUBLIC MKL::mkl_sycl MKL::mkl_intel_ilp64 MKL::mkl_tbb_thread MKL::mkl_core GTest::gtest_main)
target_include_directories(SMC_Resampling PRIVATE "../Kernels")

add_executable(Integrator_Model Integrator_Model_test.cpp)
target_link_libraries(Integrator_Model PUBLIC MKL::mkl_sycl MKL::mkl_intel_ilp64 MKL::mkl_tbb_thread MKL::mkl_core GTest::gtest_main SUNDIALS::cvode SUNDIALS::nvecserial Eigen3::Eigen)
target_include_directories(Integrator_Model PRIVATE "../Kernels" "../Generate")


add_executable(MCMC_runtest MCMC_testrun.cpp)
target_link_libraries(MCMC_runtest PUBLIC MKL::mkl_sycl MKL::mkl_intel_ilp64 MKL::mkl_tbb_thread MKL::mkl_core GSL::gsl Eigen3::Eigen)
target_include_directories(MCMC_runtest PRIVATE "../Kernels")



include(GoogleTest)
gtest_discover_tests(SMC_Sampler)
gtest_discover_tests(MCMC_Sampler)
gtest_discover_tests(SMC_Resampling)