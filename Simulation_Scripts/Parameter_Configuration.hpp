#ifndef PARAMETER_CONFIGURATION_H
#define PARAMETER_CONFIGURATION_H
#include <string>
#include <fstream>
#include <ios>
#include <sstream>
#define N_OBSERVATIONS_MAX 1000

const char dataDir[] = "/home/deb/Downloads/MCMC/Custom_MCMC/Data/";

//ODE-parameters:
const float N_pop = 1e4;
const float I0 = 1000;
const float p_alpha = 1.0 / 9;
const float R0 = 1.2;
const float p_beta = R0 * p_alpha;
const float x0_SIR[] = {N_pop - I0, I0, 0};        // SIR
const float x0_SEIR[] = {N_pop - I0, 0, I0, 0, 0}; // SEIR/SEIAR
const size_t N_param_SIR = 2;

//SEIR:
const float p_gamma = 1. / 3;
//SEIAR:
const float p_p = 0.5;
const float p_mu = p_alpha;

//Storage for observations:
float **data_ptr;
float y[N_OBSERVATIONS_MAX];
long N_observations;
float dt;

//Distribution parameters:
const float prop_std[] = {0.1f * p_alpha, 0.1f * p_beta, 0.1f * p_gamma, 0.1f * p_p, 0.1f * p_mu};
const float ll_std = 100;
const float nu_E = 1e-5;
const float nu_I = 1e-5;
const float nu_R = 1e-5;
const long N_nu = 5;

//nu_I/nu_E/nu_E is swapped out with elements of
//nu_list every iteration for SIR/SEIR/SEIAR
float nu_list[N_nu] = {0, 1, 2, 3, 4};

//Initial Parameter proposal:
const float param_prop[] = {2.f * p_alpha, 2.f * p_beta, 2.f * p_gamma, 1.5f * p_p, 2.f * p_mu};

long N_particles = 100;
long N_MCMC = 500;

#endif