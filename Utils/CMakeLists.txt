configure_file(CLCPP_MCMC_FilePaths.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/CLCPP_MCMC_FilePaths.hpp)
add_library(CLCPP_MCMC_Utils INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/CLCPP_MCMC_FilePaths.hpp)
target_include_directories(CLCPP_MCMC_Utils INTERFACE ${CMAKE_CURRENT_BINARY_DIR})