#ifndef SIMFUNCTIONS_H
#define SIMFUNCTIONS_H

#include "MCMC_Sampler.hpp"
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix_double.h>

extern long lIterates;
extern long lNumber;
extern long lChainLength;
extern double dSchedule;
extern double dThreshold;

struct pSIR
{
    double X[3];
};


void fMove(long lTime, smc::particle<pSIR>  & pFrom, smc::rng *pRng);

double logWeightFactor(long lTime, const pSIR & pState);
double prior_logLikelihood(const gsl_vector* param);
double particle_logLikelihood(long lTime, double I);
void f_SIR(long lTime, smc::particle<pSIR> &pState, double *param, smc::rng *pRng);
void proposal_sample(double *&res, const double *oldParam, smc::rng *pRng);

smc::particle<pSIR> fInitialise(smc::rng *pRng);

///The number of grid elements to either side of the current state for the single state move
#define GRIDSIZE 12
///The value of alpha at the specified time
#define ALPHA(T) (double(T)*double(dSchedule) / double(lIterates))
///The terminal version of alpha
#define FTIME    (ALPHA(lIterates))
///The exceedance threshold which we are interested in.
#define THRESHOLD dThreshold
///The number of steps in the Markov chain
#define PATHLENGTH lChainLength

#endif
