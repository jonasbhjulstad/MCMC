find_package(GSL REQUIRED)    # See below (2)
add_executable(SIR_EX run_SIR.cpp)
# add_executable(SEIR_EX run_SEIR.cpp)
# add_executable(SEIAR_EX run_SEIAR.cpp)

target_link_libraries(SIR_EX PUBLIC GSL::gsl GSL::gslcblas CLG_Utils CSV_Reader CLCPP_MCMC_Utils OpenCL::OpenCL OpenCL::Headers)
target_include_directories(SIR_EX PUBLIC ${CLCPP_PRNG_INCLUDE_DIR} ../include)
# target_link_libraries(SEIR_EX LINK_PUBLIC GSL::gsl GSL::gslcblas MCMC)
# target_link_libraries(SEIAR_EX LINK_PUBLIC GSL::gsl GSL::gslcblas MCMC)