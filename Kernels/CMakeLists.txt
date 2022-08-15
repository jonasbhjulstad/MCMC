
add_library(MCMC_Kernels INTERFACE)
target_include_directories(MCMC_Kernels INTERFACE "../Generate/")
target_link_libraries(MCMC_Kernels INTERFACE Eigen3::Eigen)
