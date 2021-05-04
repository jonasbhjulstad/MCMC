add_library(SIR SIR_Stochastic.cc SIR_Stochastic.hh)
target_include_directories(SIR PUBLIC ../MCMC_Sampler ../MCMC_Model/)