#ifndef PARAMETER_CONFIGURATION_H
#define PARAMETER_CONFIGURATION_H
#include <string>
#include <fstream>
#include <ios>
#include <sstream>
#define N_OBSERVATIONS_MAX 1000

const char dataDir[] = "/home/deb/Downloads/MCMC/Custom_MCMC/Data/";

//ODE-parameters:
double N_pop = 1e4;
double I0 = 1000;
double p_alpha = 1.0 / 9;
double R0 = 1.2;
double p_beta = R0 * p_alpha;
double x0_SIR[] = {N_pop - I0, I0, 0};        // SIR
double x0_SEIR[] = {N_pop - I0, 0, I0, 0, 0}; // SEIR/SEIAR
//SEIR:
double p_gamma = 1. / 3;
//SEIAR:
double p_p = 0.5;
double p_mu = p_alpha;

//Storage for observations:
double **data_ptr;
double y[N_OBSERVATIONS_MAX];
long N_observations;
double dt;

//Distribution parameters:
double prop_std[] = {0.1 * p_alpha, 0.1 * p_beta, 0.1 * p_gamma, 0.1 * p_p, 0.1 * p_mu};
double ll_std = 100;
double nu_E = 1e-5;
double nu_I = 1e-5;
double nu_R = 1e-5;
const long N_nu = 5;

//nu_I/nu_E/nu_E is swapped out with elements of
//nu_list every iteration for SIR/SEIR/SEIAR
double nu_list[N_nu] = {0, 1, 2, 3, 4};

//Initial Parameter proposal:
double propParam[] = {2 * p_alpha, 2 * p_beta, 2 * p_gamma, 1.5 * p_p, 2 * p_mu};

long N_particles = 100;
long N_MCMC = 500;

#endif